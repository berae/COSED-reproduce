"""Score saved predictions: DESED PSDS1, TACOS raw-caption PSDS1, MAESTRO mpAUC."""
import argparse
from pathlib import Path
import json
import importlib.metadata
import hashlib
import tempfile
import os
from functools import lru_cache
import numpy as np
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval.intersection_based import psds
from sed_scores_eval.segment_based import auroc

ROOT = Path.home() / 'projects/ov-sed/benchmark'
SOURCE = 'https://dcase.community/challenge2023/task-sound-event-detection-with-weak-labels-and-synthetic-soundscapes'
PROTOCOLS = {
    'psds1': dict(dtc_threshold=0.7, gtc_threshold=0.7,
                  cttc_threshold=None, alpha_ct=0, alpha_st=1,
                  max_efpr=100, unit_of_time='hour'),
}
MODELS = ['dasm', 'finelap', 'flexsed', 'mga-clap', 'openflam']
MAESTRO_CLASSES = ['birds_singing', 'car', 'people talking', 'footsteps',
    'children voices', 'wind_blowing', 'brakes_squeaking', 'large_vehicle',
    'cutlery and dishes', 'metro approaching', 'metro leaving']
MAESTRO_SOURCE = 'https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/local/sed_trainer_pretrained.py'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



def merge_events(events, limit):
    result = []
    for label in sorted({e[2] for e in events}):
        merged = []
        for start, end, _ in sorted(e for e in events if e[2] == label):
            start, end = max(0., start), min(end, limit)
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end, label])
        result.extend(tuple(e) for e in merged)
    return result


def load_prediction(folder, clip, field="scores"):
    with np.load(folder / (clip + '.npz')) as p:
        times = np.round(p['timestamps'], 6)
        keep = np.diff(times) > 0
        values = p[field][keep]
        labels = list(p['labels'])
        edges = np.r_[times[:-1][keep], times[-1]]
    assert edges[0] == 0 and np.all(np.diff(edges) > 0)
    assert np.isfinite(values).all()
    return edges, values, labels


@lru_cache(maxsize=1)
def semantic_unions(manifest_text):
    """Same fixed direct-neighbor policy as the prior audit; CPU only."""
    from sentence_transformers import SentenceTransformer
    manifest = json.loads(manifest_text)
    items = manifest['items']
    encoder_path = Path.home() / 'models/ov-sed/diagnostics/all-MiniLM-L6-v2'
    encoder = SentenceTransformer(str(encoder_path), device='cpu')
    queries = sorted({q for i in items for q in i['queries']})
    embeddings = encoder.encode(queries, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False)
    lookup = dict(zip(queries, embeddings))
    unions = {}
    for item in items:
        q = item['queries']
        matrix = np.array([lookup[x] for x in q])
        similarity = matrix @ matrix.T
        for j, label in enumerate(q):
            neighbors = {q[k] for k in range(len(q)) if similarity[j,k] > .8} | {label}
            unions[(item['id'],j)] = merge_events(
                [(a,b,'event') for a,b,c in item['events'] if c in neighbors], item['duration'])
    return unions


def score_tacos(model, manifest, policy='exact'):
    folder = ROOT / 'runs-extra' / model / 'tacos-test'
    unions = semantic_unions(json.dumps(manifest, sort_keys=True)) if policy == 'semantic' else None
    scores, truth, durations = {}, {}, {}
    for item in manifest['items']:
        edges, values, labels = load_prediction(folder, item['id'])
        assert labels == item['queries']
        for j, label in enumerate(labels):
            key = item['id'] + '__' + str(j)
            scores[key] = create_score_dataframe(values[:,j:j+1], edges, ['event'])
            truth[key] = (unions[(item['id'],j)] if unions is not None else
                merge_events([(a,b,'event') for a,b,c in item['events'] if c == label], float(edges[-1])))
            durations[key] = float(edges[-1])
    value = psds(scores, truth, durations, **PROTOCOLS['psds1'], num_jobs=1)[0]
    return dict(status='complete', psds1=float(value), files=len(manifest['items']),
        pairs=len(scores), protocol=PROTOCOLS['psds1'], annotation_policy=policy,
        score_field='scores', encoder='sentence-transformers/all-MiniLM-L6-v2' if unions is not None else None,
        cosine_threshold=.8 if unions is not None else None,
        merge_method='direct neighbors, not transitive; retain original queries' if unions is not None else 'exact caption union',
        manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        pooling='all file-caption pairs pooled as one event class',
        comparability='Local protocol; COSED exact encoder/query/split equivalence unconfirmed.')


def score_tag(model, manifest, policy='official'):
    # FineLAP TAG PSDS consumes unfiltered scores; legacy scores remain untouched.
    field = 'raw' if model == 'finelap' and policy == 'official' else 'scores'
    folder = ROOT / 'runs-extra' / model / 'tag-test'
    scores, truth, durations = {}, {}, {}
    for item in manifest['items']:
        edges, values, labels = load_prediction(folder, item['id'], field)
        for j,label in enumerate(labels):
            key = item['id']+'__'+str(j)
            scores[key] = create_score_dataframe(values[:,j:j+1], edges, ['event'])
            truth[key] = merge_events([(a,b,'event') for a,b,c in item['events'] if c==label],float(edges[-1]))
            durations[key] = float(edges[-1])
    parameters = dict(dtc_threshold=.5, gtc_threshold=.5, alpha_st=0, max_efpr=800)
    value = psds(scores, truth, durations, **parameters, num_jobs=1)[0]
    return dict(status='complete', grounding_psds=float(value), files=len(manifest['items']),
        pairs=len(scores), protocol=parameters, score_field=field, postprocessing_policy=policy,
        note='official means FineLAP unfiltered TAG PSDS branch only; other models retain saved scores; not full official replication',
        manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())


def mean_segments(edges, values):
    """Duration-weighted 1-second averages, as in DCASE _get_segment_scores."""
    boundaries = np.r_[np.arange(0., edges[-1], 1.), edges[-1]]
    averages = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        weights = np.maximum(0., np.minimum(edges[1:], end) - np.maximum(edges[:-1], start))
        assert weights.sum() > 0
        averages.append((values * weights[:, None]).sum(axis=0) / weights.sum())
    return boundaries, np.asarray(averages)


def score_maestro(model, manifest):
    from sklearn.metrics import roc_curve
    folder = ROOT / 'runs-extra' / model / 'maestro-development'
    scores, truth, durations = {}, {}, {}
    all_scores, all_targets = [], []
    for item in manifest['items']:
        edges, values, labels = load_prediction(folder, item['id'])
        values = values[:, [labels.index(c) for c in MAESTRO_CLASSES]]
        boundaries, averages = mean_segments(edges, values)
        key = item['id']
        scores[key] = create_score_dataframe(averages, boundaries, MAESTRO_CLASSES)
        events = [(a, b, c) for a, b, c, confidence in item['soft_events']
                  if confidence > 0.5 and c in MAESTRO_CLASSES]
        truth[key] = merge_events(events, float(edges[-1]))
        durations[key] = float(edges[-1])
        targets = np.zeros_like(averages, dtype=bool)
        for a, b, c in truth[key]:
            targets[:, MAESTRO_CLASSES.index(c)] |= (boundaries[:-1] < b) & (boundaries[1:] > a)
        all_targets.append(targets)
        all_scores.append(averages)
    result, official_curves = auroc(scores, truth, durations, segment_length=1.0,
                   max_fpr=0.1, num_jobs=1)
    y, x = np.concatenate(all_targets), np.concatenate(all_scores)
    independently = {}
    for j, c in enumerate(MAESTRO_CLASSES):
        assert 0 < y[:, j].sum() < len(y), c
        fpr, tpr, _ = roc_curve(y[:, j], x[:, j], drop_intermediate=False)
        order = np.lexsort((tpr, fpr))
        fpr, tpr = fpr[order], tpr[order]
        official_tpr, official_fpr = official_curves[c][:2]
        assert np.allclose(tpr, official_tpr, atol=1e-12)
        assert np.allclose(fpr, official_fpr, atol=1e-12)
        # Mirror the released library's second x-sort, including equal-FPR
        # handling, after independently validating every ROC point above.
        order = np.argsort(fpr)
        fpr, tpr = fpr[order], tpr[order]
        independently[c] = float((np.diff(np.minimum(fpr, 0.1)) * tpr[:-1]).sum() / 0.1)
        assert abs(independently[c] - result[c]) < 1e-8, (c, independently[c], result[c])
    assert abs(np.mean(list(independently.values())) - result['mean']) < 1e-8
    return dict(status='complete', mpauc=float(result['mean']),
        per_class={c: float(result[c]) for c in MAESTRO_CLASSES},
        files=len(manifest['items']), segments=len(y), classes=MAESTRO_CLASSES,
        positive_segments={c: int(y[:, j].sum()) for j, c in enumerate(MAESTRO_CLASSES)},
        protocol=dict(segment_length=1.0, max_fpr=0.1, soft_label_threshold='>0.5',
            aggregation='duration-weighted mean per 1s segment; staircase pAUC divided by max_fpr; macro average'),
        source=MAESTRO_SOURCE, independent_sklearn_check='passed',
        comparability='All 49 development recordings; not official validation split or private evaluation; not confirmed COSED split.')


def load_inputs(model, dataset, manifest):
    folder = ROOT / 'runs-source-audit-v2' / model / dataset
    scores, truth, durations = {}, {}, {}
    for item in manifest['items']:
        clip = item['id']
        with np.load(folder / (clip + '.npz')) as pred:
            times = np.round(pred['timestamps'], 6)
            keep = np.diff(times) > 0
            labels = list(pred['labels'])
            assert set(labels) == set(manifest['classes'])
            assert np.isfinite(pred['scores']).all()
            scores[clip] = create_score_dataframe(
                pred['scores'][keep],
                np.r_[times[:-1][keep], times[-1]], labels)
            limit = float(pred['timestamps'][-1])
        events = []
        for label in sorted({e[2] for e in item['events']}):
            merged = []
            for start, end, _ in sorted(e for e in item['events'] if e[2] == label):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end, label])
            events.extend(merged)
        truth[clip] = [(start, min(end, limit), label)
                       for start, end, label in events if start < limit]
        durations[clip] = limit
    return scores, truth, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['desed-public', 'desed-validation'],
        choices=['desed-public', 'desed-validation', 'tacos-test', 'tag-test', 'maestro-development'])
    parser.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    parser.add_argument('--tacos-policy', choices=['exact','semantic','both'], default='both')
    parser.add_argument('--tag-policy', choices=['official','legacy','both'], default='both')
    args = parser.parse_args()
    report = dict(source=SOURCE, protocols=PROTOCOLS,
        library_version=importlib.metadata.version('sed_scores_eval'),
        caveats=['Official metric parameters on local subsets, not an official submission.',
                 'Validation missing 100/1168 files; splits evaluated separately.',
                 'Model-specific postprocessing retained; training exposure not controlled.'],
        datasets={})
    for dataset in dict.fromkeys(args.datasets):
        manifest = json.loads((ROOT / 'manifests' / (dataset + '.json')).read_text())
        result = dict(files=len(manifest['items']), expected=manifest['expected'],
                      missing=len(manifest['missing']), models={})
        report['datasets'][dataset] = result
        for model in dict.fromkeys(args.models):
            if dataset in ['tacos-test', 'tag-test']:
                selected = args.tacos_policy if dataset == 'tacos-test' else args.tag_policy
                policies = (['exact','semantic'] if dataset == 'tacos-test' else ['legacy','official']) if selected == 'both' else [selected]
                result['models'][model] = {}
                for policy in policies:
                    values = (score_tacos if dataset == 'tacos-test' else score_tag)(model, manifest, policy)
                    values['library_version'] = report['library_version']
                    folder = ROOT / 'runs-extra' / model / dataset
                    # A named result never overwrites the historical metrics.json.
                    save_json(folder / ('metrics.'+policy+'.json'), values)
                    result['models'][model][policy] = values
                    print(dataset, model, policy, values.get('psds1',values.get('grounding_psds')), flush=True)
                continue
            if dataset == 'maestro-development':
                values = (score_tacos if dataset == 'tacos-test' else score_maestro)(model, manifest)
                values['library_version'] = report['library_version']
                folder = ROOT / 'runs-extra' / model / dataset
                save_json(folder / 'metrics.json', values)
                status_path = folder / 'status.json'
                status = json.loads(status_path.read_text())
                metric_name = 'psds1' if dataset == 'tacos-test' else 'mpauc'
                status.update(status='complete', scoring_status='complete',
                              **{metric_name: values[metric_name]})
                save_json(status_path, status)
                result['models'][model] = values
                print(dataset, model, values.get('psds1', values.get('mpauc')), flush=True)
                continue
            scores, truth, durations = load_inputs(model, dataset, manifest)
            values = {}
            for name, parameters in PROTOCOLS.items():
                value, per_class, _, _ = psds(scores, truth, durations,
                    **parameters, time_decimals=6, num_jobs=1)
                values[name] = float(value)
                values[name + '_per_class'] = {k: float(v) for k, v in per_class.items()}
            previous = json.loads((ROOT / 'runs-source-audit-v2' / model / dataset / 'metrics.json').read_text())['psds1']
            assert abs(values['psds1'] - previous) < 1e-9
            result['models'][model] = values
            print(dataset, model, values['psds1'], flush=True)
    if all(ds.startswith('desed-') for ds in args.datasets):
        target = ROOT / 'analysis/desed/official_psds.json'
        # Preserve already-scored models/splits when an explicit subset is rerun.
        if target.exists():
            existing = json.loads(target.read_text())
            for ds, values in report['datasets'].items():
                existing['datasets'].setdefault(ds, values)['models'].update(values['models'])
            report['datasets'] = existing['datasets']
        save_json(target, report)


if __name__ == '__main__':
    main()
