"""Counterfactual diagnostics of saved DESED scores; never changes benchmark results.

Run on rabbit05 in the finelap environment. All thresholds in the component
diagnostic are enumerated, not tuned. Ground-truth gating is an oracle probe,
not a deployable model. Single-event reachability is NOT a PSDS upper bound.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import numpy as np

ROOT = Path.home() / 'projects/ov-sed/benchmark'
MODELS = ['dasm', 'finelap', 'flexsed', 'mga-clap', 'openflam']
CLASSES = ['Speech', 'Dishes', 'Dog', 'Vacuum_cleaner']


def merge(events, limit=10.):
    out = []
    for a, b in sorted(events):
        a, b = max(0., a), min(limit, b)
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(events, edges):
    out = np.zeros(len(edges)-1)
    for a, b in events:
        out += np.maximum(0., np.minimum(edges[1:], b)-np.maximum(edges[:-1], a))
    return np.minimum(np.diff(edges), out)


def weighted_auc(values, positive_time, negative_time):
    """Exact score-ranking AUC of piecewise-constant scores, including mixed bins."""
    if positive_time.sum() <= 0 or negative_time.sum() <= 0:
        return None
    order = np.argsort(values, kind='stable')
    starts = np.r_[0, np.where(np.diff(values[order]) != 0)[0]+1]
    pos = np.add.reduceat(positive_time[order], starts)
    neg = np.add.reduceat(negative_time[order], starts)
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))


def threshold_components(values):
    """All distinct positive components achievable with one scalar threshold.

    Activate equal-valued frames together. A union-find visits each possible
    component when it is created; unchanged components need not be repeated.
    """
    n = len(values)
    parent = np.arange(n)
    left = np.arange(n)
    right = np.arange(n)+1
    active = np.zeros(n, dtype=bool)
    found = set()

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(-values, kind='stable')
    groups = np.r_[0, np.where(np.diff(values[order]) != 0)[0]+1, n]
    for g0, g1 in zip(groups[:-1], groups[1:]):
        changed = order[g0:g1]
        active[changed] = True
        for i in changed:
            for j in (i-1, i+1):
                if j < 0 or j >= n or not active[j]:
                    continue
                a, b = root(i), root(j)
                if a != b:
                    parent[b] = a
                    left[a] = min(left[a], left[b])
                    right[a] = max(right[a], right[b])
        for i in changed:
            r = root(i)
            found.add((int(left[r]), int(right[r])))
    return np.array(sorted(found), dtype=int)


def can_match(a, b, intervals):
    if len(intervals) == 0:
        return False
    intersection = np.maximum(0., np.minimum(b, intervals[:, 1])-np.maximum(a, intervals[:, 0]))
    return bool(np.any((intersection >= .7*(b-a)-1e-9) &
                       (intersection >= .7*np.diff(intervals, axis=1)[:, 0]-1e-9)))


def grid_can_match(a, b, edges):
    # The nearest inside/outside boundary choices suffice for a single GT interval.
    starts = np.unique(np.clip([np.searchsorted(edges, a, side='right')-1,
                               np.searchsorted(edges, a, side='left')], 0, len(edges)-1))
    ends = np.unique(np.clip([np.searchsorted(edges, b, side='right')-1,
                             np.searchsorted(edges, b, side='left')], 0, len(edges)-1))
    candidates = np.array([[edges[i], edges[j]] for i in starts for j in ends if j > i])
    return can_match(a, b, candidates)


def median_or_none(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def event_summary(rows):
    if not rows:
        return {'events': 0}
    return dict(events=len(rows), clips=len({r['clip'] for r in rows}),
                grid_matchable=float(np.mean([r['grid_matchable'] for r in rows])),
                raw_matchable=float(np.mean([r['raw_matchable'] for r in rows])),
                scores_matchable=float(np.mean([r['scores_matchable'] for r in rows])),
                raw_local_auc_median=median_or_none([r['raw_local_auc'] for r in rows]),
                scores_local_auc_median=median_or_none([r['scores_local_auc'] for r in rows]))


def analyze(dataset, model, manifest, official):
    from sed_scores_eval.base_modules.scores import create_score_dataframe
    from sed_scores_eval.intersection_based import psds
    from sklearn.metrics import average_precision_score
    from scipy.ndimage import median_filter
    folder = ROOT/'runs-source-audit-v2'/model/dataset
    truth, durations = {}, {}
    frames = {k: {} for k in ['raw', 'scores', 'oracle_negative_clip_gate']}
    clip_rows = {c: [] for c in CLASSES}
    event_rows = {c: [] for c in CLASSES}
    digest = hashlib.sha256()
    dt = None
    windows = dict(dasm=16, finelap=3, flexsed=9, **{'mga-clap': 3}, openflam=3)
    for item in manifest['items']:
        file = folder/(item['id']+'.npz')
        digest.update(item['id'].encode()); digest.update(hashlib.sha256(file.read_bytes()).digest())
        with np.load(file) as p:
            audio_duration = float(p['timestamps'][-1])
            edges = np.round(p['timestamps'], 6)
            keep = np.diff(edges) > 0
            indices = [list(p['labels']).index(c) for c in CLASSES]
            raw = np.asarray(p['raw'][:, indices], dtype=float)
            smooth = np.asarray(p['scores'][:, indices], dtype=float)
            assert np.array_equal(smooth, median_filter(raw, size=(windows[model], 1)))
            raw, smooth = raw[keep], smooth[keep]
            edges = np.r_[edges[:-1][keep], edges[-1]]
        assert edges[0] == 0 and np.all(np.diff(edges) > 0)
        assert np.isfinite(raw).all() and np.isfinite(smooth).all()
        dt = float(np.median(np.diff(edges)))
        gt = {c: merge([(a,b) for a,b,label in item['events'] if label == c], audio_duration) for c in CLASSES}
        truth[item['id']] = [(a,b,c) for c, events in gt.items() for a,b in events]
        durations[item['id']] = audio_duration
        gated = smooth.copy()
        for j, c in enumerate(CLASSES):
            if not gt[c]:
                gated[:, j] = -1e6
        for key, values in [('raw',raw), ('scores',smooth), ('oracle_negative_clip_gate',gated)]:
            frames[key][item['id']] = create_score_dataframe(values, edges, CLASSES)
        for j, c in enumerate(CLASSES):
            intervals = gt[c]
            pos = overlap(intervals, edges)
            neg = np.maximum(0., np.diff(edges)-pos)
            clip = dict(clip=item['id'], positive=bool(intervals), positive_seconds=float(pos.sum()))
            comps = {}
            for key, values in [('raw',raw[:,j]), ('scores',smooth[:,j])]:
                clip[key+'_max'] = float(values.max())
                clip[key+'_auc'] = weighted_auc(values,pos,neg)
                if pos.sum() > 0 and np.any(pos == 0):
                    delta = float(values[pos>0].max()-values[pos==0].max())
                    clip[key+'_peak_position'] = 'inside' if delta > 1e-10 else ('outside' if delta < -1e-10 else 'tie')
                else:
                    clip[key+'_peak_position'] = None
                if intervals:
                    comps[key] = edges[threshold_components(values)]
            clip_rows[c].append(clip)
            if not intervals:
                continue
            others = merge([(a,b) for a,b,label in item['events'] if label != c], edges[-1])
            for a,b in intervals:
                context_start,context_end=max(0,a-1),min(edges[-1],b+1)
                local = overlap([[context_start,context_end]],edges)
                local_targets = [[max(x,context_start),min(y,context_end)] for x,y in intervals
                                 if min(y,context_end)>max(x,context_start)]
                bg = np.maximum(0.,local-overlap(local_targets,edges))
                event_pos = overlap([[a,b]],edges)
                other_fraction = sum(max(0,min(b,y)-max(a,x)) for x,y in others)/(b-a)
                row = dict(clip=item['id'], onset=a, offset=b, duration=b-a,
                           other_label_overlap=other_fraction, grid_matchable=grid_can_match(a,b,edges))
                for key, values in [('raw',raw[:,j]), ('scores',smooth[:,j])]:
                    row[key+'_matchable'] = can_match(a,b,comps[key])
                    row[key+'_local_auc'] = weighted_auc(values,event_pos,bg)
                event_rows[c].append(row)
    metrics = {}
    for key in frames:
        result = psds(frames[key],truth,durations,dtc_threshold=.7,gtc_threshold=.7,
                      cttc_threshold=None,alpha_ct=0,alpha_st=1,max_efpr=100,
                      unit_of_time='hour',num_jobs=1)
        metrics[key] = {c: float(v) for c,v in result[1].items()}
        print(dataset,model,key,metrics[key],flush=True)
    for c in CLASSES:
        assert abs(metrics['scores'][c]-official[model]['psds1_per_class'][c]) < 1e-10, (dataset,model,c,metrics['scores'][c],official[model]['psds1_per_class'][c])
    summary = {}
    for c, rows in event_rows.items():
        clips = clip_rows[c]
        y = [r['positive'] for r in clips]
        s = {'events': event_summary(rows), 'duration_bins': {}, 'clip': {}}
        for lo,hi in [(0,.25),(.25,.5),(.5,1),(1,11)]:
            s['duration_bins'][f'{lo:g}-{hi:g}'] = event_summary([r for r in rows if lo <= r['duration'] < hi])
        for key in ['raw','scores']:
            positions = [r[key+'_peak_position'] for r in clips if r[key+'_peak_position'] is not None]
            s['clip'][key] = dict(ap=float(average_precision_score(y,[r[key+'_max'] for r in clips])),
                positive_clips=sum(y), peak_eligible=len(positions),
                peak_inside=sum(v=='inside' for v in positions), peak_outside=sum(v=='outside' for v in positions),
                peak_tied=sum(v=='tie' for v in positions),
                auc_positive_clip_median=median_or_none([r[key+'_auc'] for r in clips if r['positive']]))
        summary[c] = s
    return dict(files=len(manifest['items']),frame_step_seconds=dt,median_frames=windows[model],
                prediction_digest=digest.hexdigest(),identity=json.loads((folder/'identity.json').read_text()),
                psd_area=metrics,summary=summary,clips=clip_rows,events=event_rows)


def self_test():
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(123)
    for n in range(1,45):
        v=rng.integers(0,6,n).astype(float)
        expected=set()
        for th in np.unique(v):
            changes=np.diff(np.r_[False,v>=th,False].astype(int))
            expected.update(zip(np.where(changes==1)[0],np.where(changes==-1)[0]))
        assert expected==set(map(tuple,threshold_components(v)))
        p=rng.random(n);neg=rng.random(n)
        expected_auc=roc_auc_score(np.r_[np.ones(n),np.zeros(n)],np.r_[v,v],sample_weight=np.r_[p,neg])
        assert abs(weighted_auc(v,p,neg)-expected_auc)<1e-12
        edges=np.linspace(0,10,n+1)
        for _ in range(30):
            a,b=sorted(rng.uniform(0,10,2))
            all_intervals=np.array([(x,y) for i,x in enumerate(edges) for y in edges[i+1:]])
            assert grid_can_match(a,b,edges)==can_match(a,b,all_intervals)
    print('PASS: threshold components, weighted AUC, and grid feasibility against brute force')


def summarize(result):
    """Cheap, reproducible robustness checks from the retained event-level rows."""
    result['summary_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['definitions']['bootstrap'] = ('Paired source-recording cluster bootstrap; source key is '
        'clip.rsplit("_",2)[0]. Event-weighted post-minus-raw difference, 3000 draws, '
        'seed 20261019, percentile 95% interval. Exploratory, no multiplicity correction.')
    for ds in result['datasets'].values():
        for m in ds['models'].values():
            rows = m['events']['Dishes']
            short = [r for r in rows if .25 <= r['duration'] < .5]
            grouped = {}
            for r in short:
                group = r['clip'].rsplit('_', 2)[0]
                grouped.setdefault(group, []).append(int(r['scores_matchable'])-int(r['raw_matchable']))
            groups = [grouped[k] for k in sorted(grouped)]
            sums = np.array([sum(g) for g in groups]); counts = np.array([len(g) for g in groups])
            rng = np.random.default_rng(20261019)
            sampled = rng.integers(0, len(groups), size=(3000, len(groups)))
            changes = sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
            isolated = [r for r in short if all(
                q is r or q['clip'] != r['clip'] or
                q['onset'] >= r['offset']+1-1e-9 or q['offset'] <= r['onset']-1+1e-9
                for q in rows)]
            m['short_dishes_robustness'] = dict(events=len(short), source_groups=len(groups),
                paired_change=float(sums.sum()/counts.sum()),
                percentile_95=[float(x) for x in np.quantile(changes,[.025,.975])],
                isolated_at_least_1s=event_summary(isolated))
            for events in m['events'].values():
                assert all(not r['raw_matchable'] or r['grid_matchable'] for r in events)
                assert all(not r['scores_matchable'] or r['grid_matchable'] for r in events)
    profiles = {}
    selections = {'desed-public':['Dishes','Dog','Vacuum_cleaner'],
                  'realdesed-test':['cutlery_dishes','light_switch','footsteps','vacuum_cleaner'],
                  'urbansed-test':['dog_bark','gun_shot']}
    for dataset, classes in selections.items():
        path = ROOT/'manifests'/(dataset+'.json')
        manifest = json.loads(path.read_text())
        profiles[dataset] = {'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'classes':{}}
        for c in classes:
            durations=[]; positive_clips=0
            for item in manifest['items']:
                events=merge([(a,b) for a,b,label in item['events'] if label==c],float(item['duration']))
                positive_clips += bool(events)
                durations.extend(b-a for a,b in events)
            d=np.array(durations)
            profiles[dataset]['classes'][c]=dict(events=len(d),positive_clips=positive_clips,
                median_seconds=float(np.median(d)),fraction_below_half_second=float(np.mean(d<.5)),
                fraction_below_one_second=float(np.mean(d<1)))
    result['cross_dataset_annotation_profiles']=profiles


def plot_summary(result):
    if any('mga-clap' not in result['datasets'].get(ds,{}).get('models',{})
           for ds in ['desed-public','desed-validation']):
        print('SKIP figure: both splits and MGA-CLAP are required')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':200})
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2),sharey=True)
    bins=['0.25-0.5','0.5-1','1-11']
    for ax,dataset,label in zip(axes,['desed-public','desed-validation'],['Public (692 clips)','Validation (1,068 clips)']):
        summary=result['datasets'][dataset]['models']['mga-clap']['summary']['Dishes']['duration_bins']
        for key,name,color,style in [('grid_matchable','Grid feasibility','#7a7a7a','--'),
            ('raw_matchable','Raw scores','#2874a6','-'),('scores_matchable','3-frame median','#c45d21','-')]:
            ax.plot(range(3),[100*summary[b][key] for b in bins],style,marker='o',color=color,label=name)
        ax.set_xticks(range(3),[f'{s}\nn={summary[b]["events"]}' for b,s in zip(bins,['0.25–0.5 s','0.5–1 s','≥1 s'])])
        ax.set_title(label);ax.set_ylim(0,105);ax.set_yticks(range(0,101,20))
        ax.grid(axis='y',alpha=.15);ax.set_xlabel('Merged ground-truth event duration')
    axes[0].set_ylabel('Events with a qualifying interval (%)')
    axes[1].legend(loc='lower right',frameon=False,fontsize=9)
    fig.suptitle('MGA-CLAP / Dishes: short-event responses are lost after smoothing',fontsize=12)
    fig.text(.5,.025,'Each event may choose its own threshold; DTC ≥ 0.7 and GTC ≥ 0.7.\nDiagnostic only: not event recall or a PSDS upper bound. Events <0.25 s omitted.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.09,1,.94))
    for suffix in ['png','pdf']:
        path=ROOT/'analysis/desed/research'/('bottlenecks.'+suffix)
        fig.savefig(path,bbox_inches='tight')
        print('SAVED',path)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['desed-public','desed-validation'])
    parser.add_argument('--models',nargs='+',default=MODELS,choices=MODELS)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--summarize',action='store_true',help='Only refresh robustness checks and plots from existing JSON')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    output=ROOT/'analysis/desed/research/bottlenecks.json'
    if args.summarize:
        result=json.loads(output.read_text())
        summarize(result)
        plot_summary(result)
        with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False,encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,allow_nan=False)
            temporary=Path(f.name)
        temporary.replace(output)
        print('SAVED',output,flush=True)
        return
    result={'date':'2026-10-19','classes':CLASSES,'datasets':{},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions':{x:importlib.metadata.version(x) for x in ['numpy','scipy','sed_scores_eval']},
            'definitions':{
                'psd_area':'Standard per-class PSDS1 curve area, full split and negatives retained; no single-threshold tuning.',
                'oracle_negative_clip_gate':'GT-known negative clips set to -1e6; full durations retained. Diagnostic only, not deployable or an additive error attribution.',
                'matchability':'For each merged GT event independently, does ANY threshold component satisfy one-to-one DTC>=.7 and GTC>=.7? Different events may choose different thresholds; other false positives ignored. Not event recall and not a formal PSDS upper bound.',
                'grid_matchable':'Same one-to-one criterion allowing ANY interval on the native frame grid, ignoring learned scores.',
                'local_auc':'Duration-weighted native-bin score AUC for event vs 1s context on both sides, excluding other same-class GT from negatives. Fractional GT overlap weights; not the earlier 100ms-grid AUC.',
                'peak_position':'Strict max in any GT-overlapping frame vs max in frames with zero GT overlap; tied plateaus kept separate.',
                'scope':'Observational cross-model comparisons. Raw-vs-scores is a controlled postprocessing intervention on saved outputs. No training, no test-set configuration selection.'}}
    official=json.loads((ROOT/'analysis/desed/official_psds.json').read_text())['datasets']
    for dataset in args.datasets:
        path=ROOT/'manifests'/(dataset+'.json');manifest=json.loads(path.read_text())
        result['datasets'][dataset]={'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'models':{}}
        for model in args.models:
            result['datasets'][dataset]['models'][model]=analyze(dataset,model,manifest,official[dataset]['models'])
    summarize(result)
    plot_summary(result)
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False,encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False)
        temporary=Path(f.name)
    temporary.replace(output)
    print('SAVED',output,flush=True)


if __name__=='__main__':
    main()
