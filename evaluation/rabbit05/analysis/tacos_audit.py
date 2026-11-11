"""Audit TACOS completeness and caption-union scoring, without changing canonical results."""
from pathlib import Path
import json,sys,csv,collections,hashlib
import numpy as np
import soundfile as sf
ROOT=Path.home()/'projects/ov-sed/benchmark';OUT=ROOT/'analysis/tacos';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import score_benchmarks as sc
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval.intersection_based import psds
data=Path.home()/'datasets/cosed-benchmarks/tacos/data/15379789'
manifest=json.loads((ROOT/'manifests/tacos-test.json').read_text());items=manifest['items']
test=[r['filename'] for r in csv.DictReader((data/'test_split.csv').open())]
expected=set(test);assert len(test)==len(expected)==2000
assert {Path(i['path']).name for i in items}==expected
annotation=collections.defaultdict(list)
for r in csv.DictReader((data/'annotations_strong.csv').open()):
    if r['filename'] in expected:annotation[r['filename']].append((float(r['onset']),float(r['offset']),r['text']))
report={'test_files':len(items),'missing_files':0,'annotation_rows':sum(map(len,annotation.values())),'models':{},'audio_errors':[]}
max_duration_error=0.;boundary_clips=0
for i in items:
    audio,sr=sf.read(i['path'],always_2d=True)
    assert len(audio)>0 and np.isfinite(audio).all()
    max_duration_error=max(max_duration_error,abs(len(audio)/sr-i['duration']))
    raw=annotation[Path(i['path']).name]
    assert set(i['queries'])=={e[2] for e in raw}
    clean=[]
    for a,b,c in raw:
        if a<0 or b>i['duration']:boundary_clips+=1
        clean.append((max(0,a),min(b,i['duration']),c))
    assert len(clean)==len(i['events'])
    assert all(c==z and abs(a-x)<1e-6 and abs(b-y)<1e-6 for (a,b,c),(x,y,z) in zip(sorted(clean),sorted(i['events'])))
report.update(max_audio_duration_error=max_duration_error,boundary_clipped_rows=boundary_clips,pairs=sum(len(i['queries']) for i in items))
for m in sc.MODELS:
    folder=ROOT/'runs-extra'/m/'tacos-test'
    assert {f.stem for f in folder.glob('*.npz')}=={i['id'] for i in items}
    errors=[];dts=[];mismatch=0
    for i in items:
        with np.load(folder/(i['id']+'.npz')) as p:
            t=p['timestamps'];v=p['scores'];raw=p['raw']
            assert list(p['labels'])==i['queries']
            assert v.shape==raw.shape==(len(t)-1,len(i['queries']))
            assert np.isfinite(v).all() and np.isfinite(raw).all() and np.all(np.diff(t)>0) and t[0]==0
            errors.append(abs(float(t[-1])-i['duration']));dts.append(float(np.median(np.diff(t))))
            # Independently reconstruct frame coverage using the published 10/5 window schedule.
            duration=float(t[-1]);starts=list(np.arange(0,max(1,duration-10+1e-8),5.))
            if starts[-1]+10<duration:starts.append(starts[-1]+5)
            centers=(t[:-1]+t[1:])/2
            coverage=sum(((centers>=s)&(centers<min(duration,s+10))).astype(int) for s in starts)
            assert np.all(coverage>0)
    report['models'][m]={'prediction_files':len(items),'max_end_time_error':max(errors),'median_frame_step':float(np.median(dts))}
print('COMPLETENESS',json.dumps(report),flush=True)
previous=json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else {}
if 'semantic' in previous: report['semantic']=previous['semantic']
if '--semantic' not in sys.argv:
    sc.save_json(OUT/'audit.json',report)
    sys.exit()
from sentence_transformers import SentenceTransformer
modelpath=Path.home()/'models/ov-sed/diagnostics/all-MiniLM-L6-v2'
encoder=SentenceTransformer(str(modelpath),device='cpu')
queries=sorted({q for i in items for q in i['queries']})
emb=encoder.encode(queries,normalize_embeddings=True,show_progress_bar=False,batch_size=64)
lookup={q:emb[j] for j,q in enumerate(queries)}
unions={};changed_pairs=0;changed_files=0;examples=[]
for i in items:
    q=i['queries'];matrix=np.array([lookup[x] for x in q]);sim=matrix@matrix.T;changed=False
    for j,label in enumerate(q):
        near={q[k] for k in range(len(q)) if sim[j,k]>.8};near.add(label)
        union=sc.merge_events([(a,b,'event') for a,b,c in i['events'] if c in near],i['duration'])
        exact=sc.merge_events([(a,b,'event') for a,b,c in i['events'] if c==label],i['duration'])
        unions[(i['id'],j)]=union
        if union!=exact:
            changed_pairs+=1;changed=True
            if len(examples)<10:examples.append(dict(file=i['id'],query=label,neighbors=sorted(near),exact=exact,merged=union))
    changed_files+=changed
report['semantic']={'encoder':'sentence-transformers/all-MiniLM-L6-v2','cosine_threshold':.8,'method':'direct neighbors, not transitive components; keep original queries and predictions','changed_pairs':changed_pairs,'changed_files':changed_files,'examples':examples,'scores':{}}
for m in sc.MODELS:
    folder=ROOT/'runs-extra'/m/'tacos-test'
    value=sc.score_tacos(m,manifest,'semantic')['psds1']
    baseline=json.loads((folder/'metrics.json').read_text())['psds1']
    report['semantic']['scores'][m]={'raw':baseline,'merged':value,'difference':value-baseline}
    print(m,report['semantic']['scores'][m],flush=True)

sc.save_json(OUT/'audit.json',report)
print('DONE',flush=True)
