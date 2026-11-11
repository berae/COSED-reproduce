"""Read-only representation audit; identical inputs for init, online and EMA."""
import datetime
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from data import ROOT, MIX, SOURCES, CLASSES, Corpus, ancestor_map, wave
from model import COSED


def stats(raw, groups=None):
    raw = raw.double().cpu()
    x = F.normalize(raw, dim=-1)
    centered = x - x.mean(0)
    eig = torch.linalg.svdvals(centered).square()
    mass = eig / eig.sum().clamp_min(1e-30)
    cos = x @ x.T
    select = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
    result = dict(n=len(x), norm_mean=raw.norm(dim=-1).mean().item(),
                  centered_variance=centered.square().sum(1).mean().item(),
                  effective_rank=(-(mass * mass.clamp_min(1e-30).log()).sum()).exp().item(),
                  top_pc_share=mass[0].item())
    def quantiles(v):
        return dict(zip(['p05', 'median', 'p95'], v.quantile(torch.tensor([.05,.5,.95],dtype=v.dtype)).tolist()))
    result['pair_cosine'] = quantiles(cos[select])
    if groups is not None:
        same = groups[:, None] == groups[None, :]
        result['within_clip_cosine'] = quantiles(cos[select & same])
        result['cross_clip_cosine'] = quantiles(cos[select & ~same])
    return result


def sample_rows():
    corpus = object.__new__(Corpus)
    corpus.ancestors = ancestor_map()
    corpus.rows = []
    for source, name in enumerate(SOURCES):
        lines = (MIX / (name + '.jsonl')).read_text().splitlines()
        corpus.rows.extend((source, json.loads(line)) for line in random.Random(193 + source).sample(lines, 10))
    return [corpus.get(i, 193 + i) for i in range(len(corpus.rows))]


@torch.inference_mode()
def inspect(model, specs, queries, captions):
    q = model.text(queries).float().cpu()
    result = {'class_queries': stats(q), 'caption_queries': stats(model.text(captions))}
    collected = {k: [] for k in ('encoder_frames', 'temporal_frames', 'clip_embeddings')}
    indices = torch.linspace(0, 249, 10).round().long()
    means = []
    for spec in specs:
        frames = model.audio(spec.cuda())
        temporal = model.temporal(frames.float())[0]
        clip = model.clip(frames.mean(1))
        collected['encoder_frames'].append(frames[0, indices].cpu())
        collected['temporal_frames'].append(temporal[0, indices].cpu())
        collected['clip_embeddings'].append(clip.cpu())
        means.append(F.normalize(temporal[0],dim=-1).mean(0).cpu())
    groups = torch.arange(len(specs)).repeat_interleave(10)
    for key, values in collected.items():
        result[key] = stats(torch.cat(values), None if key == 'clip_embeddings' else groups)
    sim = torch.stack(means).double() @ q.double().T
    result['clip_mean_frame_query_cosine'] = dict(mean=sim.mean().item(),
        between_class_std_mean=sim.std(1).mean().item(), between_audio_std_mean=sim.std(0).mean().item())
    result['scale'] = model.scale.cpu().tolist()
    result['bias'] = model.bias.cpu().tolist()
    return result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(31415926)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = COSED(initial_scale=.1).cuda().eval()  # Historical run initialization.
    rows = sample_rows()
    # Per-clip Mel once, shared verbatim by every state. No augmentation/dropout.
    with torch.inference_mode():
        specs = [model.audio.mel_forward(wave(row).cuda()[None]).cpu() for row in rows]
    captions = sorted(set(row['cap'] for row in rows if row['cap']))
    queries = ['sound of ' + name.replace('_', ' ').lower() for name in CLASSES]
    checkpoint = ROOT / 'runs/seed31415926_grufp32/last.pt'
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    assert (state['epoch'], state['step']) == (10, 3570)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    result = dict(updated=datetime.datetime.now().isoformat(), checkpoint=str(checkpoint),
        checkpoint_sha256=digest.hexdigest(),
        epoch=state['epoch'], step=state['step'],
        protocol='FP32 eval; 10 samples/source; fixed per-clip Mel; 10 evenly spaced frames/clip; direction-normalized centered variance and covariance entropy effective rank',
        samples=rows, class_queries=queries, caption_queries=captions, states={})
    for name, key in [('initialization', None), ('online', 'model'), ('ema', 'ema')]:
        if key:
            model.load_state_dict(state[key], strict=True)
        print('Inspecting', name, flush=True)
        result['states'][name] = inspect(model, specs, queries, captions)
        print(json.dumps(result['states'][name]), flush=True)
    target = ROOT / 'log/representation_audit.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, default=str))
    temporary.replace(target)
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
