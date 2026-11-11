"""Read-only benchmark class/query audit. Canonical metrics are never overwritten."""
from pathlib import Path
import sys, json, hashlib
from collections import defaultdict
import numpy as np
ROOT=Path.home()/'projects/ov-sed/benchmark'
sys.path.insert(0,str(ROOT/'scripts'))
import score_benchmarks as scoring
from sed_scores_eval.intersection_based import psds
from sed_scores_eval.base_modules.scores import create_score_dataframe
OUT=ROOT/'analysis/classwise'
OUT.mkdir(exist_ok=True)
MODELS=scoring.MODELS
DATASETS=['desed-public','desed-validation','realdesed-test','urbansed-test','maestro-development','tag-test','tacos-test']

def profile(items, labels, soft=False, fixed=False):
    result={}
    for c in labels:
        lengths=[]; positive=0; total=0.
        for x in items:
            lim=min(10.,x['duration']) if fixed else x['duration']
            raw=[(a,b,k) for a,b,k,v in x['soft_events'] if v>.5] if soft else x['events']
            e=scoring.merge_events([e for e in raw if e[2]==c],lim)
            positive+=bool(e)
            lengths.extend(b-a for a,b,_ in e)
            total+=lim
        result[c]=dict(positive_files=positive,merged_intervals=len(lengths),target_seconds=sum(lengths),median_seconds=float(np.median(lengths)) if lengths else None,under_1s=float(np.mean(np.array(lengths)<1)) if lengths else None,total_audio_seconds=total)
    return result

report={'date':'2026-10-19','models':MODELS,'datasets':{}}
for ds in DATASETS:
    path=ROOT/'manifests'/f'{ds}.json'; manifest=json.loads(path.read_text());items=manifest['items']
    r=dict(files=len(items),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),models={})
    report['datasets'][ds]=r
    if ds in ['tag-test','tacos-test']:
        groups=defaultdict(list)
        for item in items:
            for j,c in enumerate(item['queries']):groups[c].append((item,j))
        r.update(scoring_variant='legacy' if ds=='tag-test' else 'exact',score_field='scores',kind='exact_query',pairs=sum(map(len,groups.values())),unique_queries=len(groups),singleton_queries=sum(len(v)==1 for v in groups.values()),profile={})
        for c,rows in groups.items():
            r['profile'][c]=profile([i for i,j in rows],[c])[c]
        for m in MODELS:
            folder=ROOT/'runs-extra'/m/ds
            arrays={x['id']:scoring.load_prediction(folder,x['id']) for x in items}
            params=dict(scoring.PROTOCOLS['psds1']) if ds=='tacos-test' else dict(dtc_threshold=.5,gtc_threshold=.5,alpha_st=0,max_efpr=800)
            scores={};gt={};dur={};byquery={}
            for c,rows in groups.items():
                keys=[]
                for item,j in rows:
                    edges,values,labels=arrays[item['id']];assert labels==item['queries']
                    key=item['id']+'__'+str(j);keys.append(key)
                    scores[key]=create_score_dataframe(values[:,j:j+1],edges,['event'])
                    gt[key]=scoring.merge_events([(a,b,'event') for a,b,k in item['events'] if k==c],float(edges[-1]))
                    dur[key]=float(edges[-1])
                # Query-conditional score; no additional absent-query recordings.
                byquery[c]=float(psds({k:scores[k] for k in keys},{k:gt[k] for k in keys},{k:dur[k] for k in keys},**params,num_jobs=1)[0])
            total=float(psds(scores,gt,dur,**params,num_jobs=1)[0])
            old=json.loads((folder/'metrics.json').read_text());expected=old.get('psds1',old.get('grounding_psds'))
            assert abs(total-expected)<1e-9,(ds,m,total,expected)
            r['models'][m]={'overall':total,'per_class':byquery}
            print(ds,m,'queries',len(byquery),'verified',flush=True)
    else:
        labels=scoring.MAESTRO_CLASSES if ds=='maestro-development' else manifest['classes']
        r.update(kind='fixed_class',profile=profile(items,labels,soft=ds=='maestro-development',fixed=ds.startswith('desed') or ds=='urbansed-test'))
        for m in MODELS:
            if ds=='maestro-development':
                old=json.loads((ROOT/'runs-extra'/m/ds/'metrics.json').read_text())
                r['models'][m]={'overall':old['mpauc'],'per_class':old['per_class']}
                for c in labels:r['profile'][c]['positive_segments']=old['positive_segments'][c]
            else:
                scores,gt,dur=scoring.load_inputs(m,ds,manifest)
                value,pc,*_=psds(scores,gt,dur,**scoring.PROTOCOLS['psds1'],time_decimals=6,num_jobs=1)
                old=json.loads((ROOT/'runs-source-audit-v2'/m/ds/'metrics.json').read_text())
                assert abs(value-old['psds1'])<1e-9,(ds,m,value,old)
                r['models'][m]={'overall':float(value),'per_class':{c:float(v) for c,v in pc.items()}}
            print(ds,m,'verified',flush=True)
    (OUT/'statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE',flush=True)
