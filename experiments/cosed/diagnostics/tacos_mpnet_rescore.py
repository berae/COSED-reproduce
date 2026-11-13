"""Re-score TACOS semantic PSDS1 with the author's merge encoder (all-mpnet-base-v2).

Author clarification (2026-11-12): COSED merges near-duplicate TACOS captions with
sentence-transformers/all-mpnet-base-v2. Cosine > 0.8 and direct-neighbour merging are retained local settings, not explicitly confirmed in that email. The frozen scorer used
all-MiniLM-L6-v2. Only the encoder changes; the direct-neighbour policy, thresholds
and the saved per-file scores are reused. summary.json is left untouched.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FROZEN = Path('/work/zhanghc/Myllm/FineLAP/dependencies/benchmark')
ENCODER = Path('/work/zhanghc/hfstore/models/all-mpnet-base-v2/e8c3b32edf5434bc2275fc9bab85f82640a19130')
ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, help='runs/<run>/benchmark_<hash> directory')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bench = (ROOT / a.bench).resolve()

    sys.path.insert(0, '/work/zhanghc/Myllm/FineLAP/dependencies/scoring')
    scorer = load('benchscore', FROZEN / 'scripts/score_benchmarks.py')
    scorer.ROOT = bench
    manifest = json.loads((FROZEN / 'manifests/tacos-test.json').read_text())

    # Sanity check: the frozen MiniLM unions must reproduce the stored semantic score.
    frozen = json.loads((FROZEN / 'tacos_semantic_unions.json').read_text())
    scorer.semantic_unions = lambda _: {(k.rsplit('__', 1)[0], int(k.rsplit('__', 1)[1])): v
                                        for k, v in frozen.items()}
    baseline = scorer.score_tacos('cosed', manifest, 'semantic')['psds1']
    stored = json.loads((bench / 'summary.json').read_text())['tacos-test']['semantic']['psds1']
    assert abs(baseline - stored) < 1e-9, (baseline, stored)

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(ENCODER), device='cpu')
    items = manifest['items']
    queries = sorted({q for i in items for q in i['queries']})
    emb = encoder.encode(queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    lookup = dict(zip(queries, emb))
    unions, merged_pairs = {}, 0
    for item in items:
        q = item['queries']
        sim = np.array([lookup[x] for x in q]) @ np.array([lookup[x] for x in q]).T
        for j, label in enumerate(q):
            neighbors = {q[k] for k in range(len(q)) if sim[j, k] > .8} | {label}
            merged_pairs += len(neighbors) > 1
            unions[(item['id'], j)] = scorer.merge_events(
                [(s, e, 'event') for s, e, c in item['events'] if c in neighbors], item['duration'])
    scorer.semantic_unions = lambda _: unions

    result = scorer.score_tacos('cosed', manifest, 'semantic')
    result.update(encoder='sentence-transformers/all-mpnet-base-v2',
                  encoder_revision=ENCODER.name,
                  queries_with_merged_neighbors=int(merged_pairs),
                  minilm_psds1_recomputed=baseline,
                  note='Author-specified merge encoder; saved scores from ' + str(bench.relative_to(ROOT)))
    out = ROOT / a.out
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('psds1', 'minilm_psds1_recomputed', 'pairs',
                                             'queries_with_merged_neighbors')}))


if __name__ == '__main__':
    main()
