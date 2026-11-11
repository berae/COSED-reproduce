"""Compare dense masked supervision against the real update(), without training."""
import contextlib
import argparse
import datetime
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

import train
from data import ROOT, MIX, SOURCES, CLASSES, Corpus, ancestor_map, wave
from model import COSED

SEED = 193
CHECKPOINT = ROOT / 'runs/seed31415926_grufp32/last.pt'
AUTOCAST = torch.autocast


def sample_rows():
    corpus = object.__new__(Corpus)
    corpus.ancestors = ancestor_map()
    corpus.rows = []
    for source, name in enumerate(SOURCES):
        lines = (MIX / (name + '.jsonl')).read_text().splitlines()
        for line in random.Random(SEED + source).sample(lines, 4):
            corpus.rows.append((source, json.loads(line)))
    rows = [corpus.get(i, SEED + i) for i in range(len(corpus.rows))]
    random.Random(SEED).shuffle(rows)
    return corpus, rows


def dense_supervision(corpus, rows):
    labels = [CLASSES]
    for source in range(1, 5):
        texts = set()
        for row in rows:
            if row['src'] == source:
                texts.update(row['phrases'] if source in (1, 3) else [row['cap']])
        labels.append(sorted(texts))
    widths = [len(x) for x in labels]
    offsets = np.cumsum([0] + widths)
    count = int(offsets[-1])
    frame_y = np.zeros((len(rows), 250, count), np.float32)
    clip_y = np.zeros((len(rows), count), np.float32)
    frame_mask = np.zeros_like(frame_y, dtype=bool)
    clip_mask = np.zeros_like(clip_y, dtype=bool)
    for i, row in enumerate(rows):
        source = row['src']
        start, end = offsets[source:source + 2]
        if source in (2, 4):
            clip_mask[i, start:end] = True
            clip_y[i, start + labels[source].index(row['cap'])] = 1
        else:
            frame_mask[i, :, start:end] = True
            for phrase, segments in row['phrases'].items():
                indices = corpus.ancestors[phrase] if source == 0 else [labels[source].index(phrase)]
                for onset, offset in segments:
                    first, last = max(0, int(onset * 25)), min(250, int(np.ceil(offset * 25)))
                    for index in indices:
                        frame_y[i, first:last, start + index] = 1
    class_mask = frame_mask.copy()
    class_mask[:, :, widths[0]:] = False
    caption_mask = frame_mask & ~class_mask
    tensors = [torch.from_numpy(v).cuda() for v in
               (frame_y, clip_y, class_mask, caption_mask, clip_mask)]
    stats = []
    for source in range(5):
        start, end = offsets[source:source + 2]
        y = clip_y[:, start:end] if source in (2, 4) else frame_y[:, :, start:end]
        mask = clip_mask[:, start:end] if source in (2, 4) else frame_mask[:, :, start:end]
        stats.append(dict(source=SOURCES[source], queries=widths[source],
                          valid=int(mask.sum()), positive=float(y[mask].sum()),
                          negative=float((1-y[mask]).sum())))
    rng = random.Random(120)
    prompts = []
    for source, names in enumerate(labels):
        for name in names:
            template = rng.choice(train.TEMPLATES) if source == 0 else 'sound of {}'
            prompts.append(template.format(name.replace('_', ' ').lower()))
    return labels, offsets, tensors, stats, prompts


def model():
    torch.manual_seed(31415926)
    result = COSED(initial_scale=.1).cuda()
    state = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    assert (state['epoch'], state['step']) == (10, 3570)
    result.load_state_dict(state['model'])
    result.eval()
    result.temporal.train()  # cuDNN requires training mode for backward.
    result.text.training = True  # checkpoint activations; child dropout remains off.
    return result


def fix_mel(m, rows):
    audio = torch.stack([wave(row) for row in rows]).cuda()
    with torch.no_grad():
        spec = m.audio.mel_forward(audio)
        differences = {}
        for size in [1, 2, 8]:
            split = torch.cat([m.audio.mel_forward(audio[i:i+size])
                               for i in range(0, len(rows), size)])
            differences[str(size)] = dict(max_abs=float((split-spec).abs().max()),
                                          changed_fraction=float((split!=spec).float().mean()))
    def cached(x):
        indices = []
        for item in x:
            matches = [i for i, original in enumerate(audio) if torch.equal(item, original)]
            assert len(matches) == 1
            indices.append(matches[0])
        return spec[indices]
    m.audio.mel_forward = cached
    return differences


def disable_autocast(*args, **kwargs):
    kwargs['enabled'] = False
    return AUTOCAST(*args, **kwargs)


def dense_loss(m, rows, tensors, prompts, mixed):
    frame_y, clip_y, class_mask, caption_mask, clip_mask = tensors
    with AUTOCAST('cuda', dtype=torch.bfloat16, enabled=mixed):
        q = m.text(prompts)
        x = torch.stack([wave(row) for row in rows]).cuda()
        frames, clips = m.encode(m.audio.mel_forward(x))
        frame_z = m.scores(frames, q, False)
        clip_z = m.scores(clips, q, True)
        values = []
        for z, y, mask, weight in [
            (frame_z, frame_y, class_mask, .9),
            (frame_z, frame_y, caption_mask, .1),
            (clip_z, clip_y, clip_mask, .1),
        ]:
            z, y = z[mask], y[mask]
            p = z.sigmoid()
            positive = y * (1-p) * F.softplus(-z)
            negative = (1-y) * p * F.softplus(z)
            if len(values) == 0:
                values.append(weight * (positive + negative).mean())
            else:
                values.append(weight * (.05 * positive.sum() / y.sum()
                                        + .95 * negative.sum() / (1-y).sum()))
        total = sum(values)
    total.backward()
    return float(total), [float(v) for v in values]


def gradients(m):
    return {k: p.grad.detach().cpu().clone() for k, p in m.named_parameters()
            if p.grad is not None}


def compare(m, reference):
    totals = {}
    for name, param in m.named_parameters():
        expected = reference.get(name)
        if expected is None and param.grad is None:
            continue
        if expected is None:
            expected = torch.zeros_like(param, device='cpu')
        actual = torch.zeros_like(expected) if param.grad is None else param.grad.detach().cpu()
        key = name.split('.')[0]
        numbers = totals.setdefault(key, np.zeros(5, np.float64))
        a, b = actual.double(), expected.double()
        numbers[:4] += [(a-b).square().sum().item(), b.square().sum().item(),
                        a.square().sum().item(), (a*b).sum().item()]
        numbers[4] = max(numbers[4], (a-b).abs().max().item())
    return {k: dict(relative_l2=float(np.sqrt(v[0]/max(v[1], 1e-40))),
                    cosine=float(v[3]/max(np.sqrt(v[1]*v[2]), 1e-40)),
                    max_abs=float(v[4])) for k, v in totals.items()}


def controlled_update(m, corpus, rows, rank, world, micro, mixed, reference):
    bank = train.query_bank(rows)
    labels, offsets, tensors, stats, prompts = dense_supervision(corpus, rows)
    assert [bank[i] for i in range(5)] == labels
    frame_y, clip_y, class_mask, caption_mask, clip_mask = tensors
    for i, row in enumerate(rows):
        source = row['src']
        expected = (clip_y[i] if source in (2, 4) else frame_y[i])[..., offsets[source]:offsets[source+1]]
        assert torch.equal(train.targets(row, bank, corpus.ancestors).cuda(), expected)
    captured = {}
    real_clip = torch.nn.utils.clip_grad_norm_
    def capture(parameters, *args, **kwargs):
        if rank == 0:
            captured.update(compare(m, reference))
        return real_clip(parameters, *args, **kwargs)
    precision = contextlib.nullcontext() if mixed else patch('torch.autocast', disable_autocast)
    with precision, patch('train.augment', lambda spec, rng: spec), \
         patch.object(random.Random, 'random', lambda self: 1.0), \
         patch('torch.nn.utils.clip_grad_norm_', capture):
        loss, norm = train.update(m, torch.optim.SGD(m.parameters(), lr=0),
                                  rows, corpus, rank, world, 1, micro, 119)
    # update uses seed+step=120, exactly matching the reference prompt choices.
    return dict(loss=loss, preclip_norm=norm, gradients=captured)


def distributed(rank, rendezvous, directory, fixed_mel):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', rank=rank, world_size=4,
                            init_method='file://' + rendezvous,
                            timeout=datetime.timedelta(minutes=10))
    corpus, rows = sample_rows()
    m = model()
    if fixed_mel:
        fix_mel(m, rows)
    reference = torch.load(Path(directory)/'fp32.pt', weights_only=True, mmap=True)
    result = controlled_update(m, corpus, rows, rank, 4, 2, False, reference)
    if rank == 0:
        (Path(directory)/'distributed.json').write_text(json.dumps(result))
        print('four_rank_fp32', json.dumps(result), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-mel', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(4)
    corpus, rows = sample_rows()
    labels, offsets, tensors, stats, prompts = dense_supervision(corpus, rows)
    report = dict(source_counts=stats, samples=[dict(source=SOURCES[r['src']],
                  path=r['path'], start=r['start']) for r in rows], tests={})
    report['denominators'] = dict(
        class_mass=stats[0]['valid'],
        frame_caption_positive=sum(stats[i]['positive'] for i in (1,3)),
        frame_caption_negative=sum(stats[i]['negative'] for i in (1,3)),
        clip_caption_positive=sum(stats[i]['positive'] for i in (2,4)),
        clip_caption_negative=sum(stats[i]['negative'] for i in (2,4)))
    # A deliberately microbatch-only pool illustrates which negative columns would be lost.
    local_counts = [set() for _ in range(5)]
    for start in range(0, len(rows), 2):
        bank = train.query_bank(rows[start:start+2])
        for source in range(1, 5):
            local_counts[source].add(len(bank[source]))
    report['micro_only_query_counts'] = [sorted(v) for v in local_counts]
    report['source_hashes'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ['train.py','model.py','data.py']}
    print('supervision', json.dumps(report), flush=True)
    with tempfile.TemporaryDirectory(prefix='cosed_grad_audit_') as directory:
        m = model()
        if args.fixed_mel:
            report['mel_batch_dependence'] = fix_mel(m, rows)
            print('mel_batch_dependence', report['mel_batch_dependence'], flush=True)
        for mixed in [False, True]:
            label = 'bf16' if mixed else 'fp32'
            m.zero_grad(set_to_none=True)
            loss, blocks = dense_loss(m, rows, tensors, prompts, mixed)
            reference = gradients(m)
            report['tests']['dense_'+label] = dict(loss=loss, blocks=blocks)
            print('dense_'+label, loss, blocks, flush=True)
            if not mixed:
                torch.save(reference, Path(directory)/'fp32.pt')
            for micro in [2, 8]:
                result = controlled_update(m, corpus, rows, 0, 1, micro, mixed, reference)
                report['tests'][label+'_micro'+str(micro)] = result
                print(label+'_micro'+str(micro), json.dumps(result), flush=True)
            del reference
        del m, tensors
        torch.cuda.empty_cache()
        torch.multiprocessing.start_processes(distributed, args=(directory+'/init', directory, args.fixed_mel),
                                              nprocs=4, start_method='spawn', join=True)
        report['tests']['four_rank_fp32'] = json.loads((Path(directory)/'distributed.json').read_text())
    destination = ROOT/'log/gradient_audit.json'
    if args.fixed_mel:
        previous = json.loads(destination.read_text())
        previous['fixed_mel_control'] = report
        report = previous
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination)


if __name__ == '__main__':
    main()
