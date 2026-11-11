"""Local Omni event-span benchmark; resumes saved audio/windows, no fake frame scores."""
import argparse, base64, io, json, time, math, hashlib, sys, urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
ROOT=Path.home()/'projects/ov-sed/benchmark'
sys.path.insert(0,str(ROOT/'scripts'))
import score_benchmarks as sc
DATASETS=['desed-public','realdesed-test','urbansed-test','tag-test','tacos-test','maestro-development']
BASE='http://127.0.0.1:8901/v1'
PROMPT='Listen to this {duration:.3f}-second audio. Detect every audible occurrence matching the queries below. Give separate start/end times for separate occurrences, relative to this window. Events may overlap. Do not infer sounds from the scene. Use only listed query_id integers. Return ONLY a valid JSON object with an events array. Each event object must have query_id (integer), start (number of seconds), and end (number of seconds). If none are audible, return an empty events array. Times must satisfy 0 <= start < end <= {duration:.3f}. Determine all event times from the audio. Do not provide confidence scores or explanations. Queries:\n{queries}'


def request(url,body=None):
    req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)

def parse(text,labels,duration):
    text=text.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    try:obj=json.loads(text)
    except json.JSONDecodeError:
        # Only repair a missing final outer brace. No event/label/time is invented.
        if text.endswith(']') and text.count('{')==text.count('}')+1:obj=json.loads(text+'}')
        else:raise
    if not isinstance(obj,dict) or not isinstance(obj.get('events'),list):raise ValueError('Missing events list')
    events=[]
    for e in obj['events']:
        j=e['query_id'];a=float(e['start']);b=float(e['end'])
        if isinstance(j,bool) or not isinstance(j,int) or not 0<=j<len(labels):raise ValueError('Invalid query id')
        if not math.isfinite(a+b) or not 0<=a<b<=duration+.05:raise ValueError('Invalid timestamp')
        events.append([a,min(b,duration),labels[j]])
    return events

def audio(item,labels,folder,model):
    path=folder/(item['id']+'.json')
    saved=json.loads(path.read_text()) if path.exists() else dict(clip_id=item['id'],labels=labels,audio_path=item['path'],windows=[],status='running')
    if saved['status']=='complete':return saved
    x,sr=sf.read(item['path'],always_2d=True)
    x=librosa.resample(x.mean(axis=1),orig_sr=sr,target_sr=16000);sr=16000
    starts=list(range(0,max(1,len(x)-10*sr+1),5*sr))
    if starts[-1]+10*sr<len(x):starts.append(starts[-1]+5*sr)
    known={w['offset'] for w in saved['windows'] if w['status']=='complete'}
    for start in starts:
        if start in known:continue
        chunk=x[start:start+10*sr];duration=len(chunk)/sr;buf=io.BytesIO()
        sf.write(buf,chunk,sr,format='WAV',subtype='PCM_16')
        prompt=PROMPT.format(duration=duration,queries='\n'.join(f'{j}: {label}' for j,label in enumerate(labels)))
        body=dict(model=model,modalities=['text'],temperature=0,seed=42,max_tokens=1536,messages=[dict(role='user',content=[dict(type='input_audio',input_audio=dict(data=base64.b64encode(buf.getvalue()).decode(),format='wav')),dict(type='text',text=prompt)])])
        attempts=[];events=None;t0=time.time()
        for retry in range(3):
            try:
                reply=request(BASE+'/chat/completions',body);attempts.append(dict(reply=reply))
                if reply['choices'][0].get('finish_reason')=='length':raise ValueError('Output truncated')
                content=reply['choices'][0]['message']['content']
                events=parse(content,labels,duration)
                plain=content.strip()
                if plain.startswith('```'):plain=plain.split('\n',1)[1].rsplit('```',1)[0].strip()
                try:json.loads(plain);attempts[-1]['format_repair']=None
                except json.JSONDecodeError:attempts[-1]['format_repair']='appended missing final outer brace only'
                break
            except Exception as e:attempts.append(dict(error=repr(e)));time.sleep(2)
        window=dict(offset=start,start=start/sr,duration=duration,seconds=time.time()-t0,status='complete' if events is not None else 'failed',events=events,attempts=attempts)
        saved['windows']=[w for w in saved['windows'] if w['offset']!=start]+[window]
        sc.save_json(path,saved)
        if events is None:raise RuntimeError('Window failed: '+str(path))
    events=[(a+w['start'],b+w['start'],c) for w in saved['windows'] for a,b,c in w['events']]
    saved.update(status='complete',events=sc.merge_events(events,len(x)/sr),duration=len(x)/sr,inference_seconds=sum(w['seconds'] for w in saved['windows']))
    sc.save_json(path,saved);return saved

def score_spans(ds,manifest,folder):
    from sed_scores_eval.base_modules.scores import create_score_dataframe
    from sed_scores_eval.intersection_based import fscore
    scores={};truth={};tp=fp=fn=0
    clip_tp=clip_fp=clip_fn=clip_tn=0
    clip_by_class={};exact_audio=0
    unions=sc.semantic_unions(json.dumps(manifest,sort_keys=True)) if ds=='tacos-test' else None
    for item in manifest['items']:
        p=json.loads((folder/(item['id']+'.json')).read_text());assert p['status']=='complete'
        duration=p['duration']
        audio_correct=True
        for j,label in enumerate(p['labels']):
            events=[(a,b,'event') for a,b,c in p['events'] if c==label]
            if unions is not None:gt=unions[(item['id'],j)]
            elif ds=='maestro-development':gt=sc.merge_events([(a,b,'event') for a,b,c,v in item['soft_events'] if c==label and v>.5],duration)
            else:gt=sc.merge_events([(a,b,'event') for a,b,c in item['events'] if c==label],duration)
            clip_tp+=int(bool(events) and bool(gt));clip_fp+=int(bool(events) and not gt)
            clip_fn+=int(not events and bool(gt));clip_tn+=int(not events and not gt)
            clip_by_class.setdefault(label,[]).append((int(bool(gt)),int(bool(events))))
            audio_correct &= bool(events)==bool(gt)
            if ds=='maestro-development':
                edges=np.r_[np.arange(0.,duration,1.),duration];yp=np.zeros(len(edges)-1,bool);yg=yp.copy()
                for a,b,c in events:yp|=(edges[:-1]<b)&(edges[1:]>a)
                for a,b,c in gt:yg|=(edges[:-1]<b)&(edges[1:]>a)
                tp+=int((yp&yg).sum());fp+=int((yp&~yg).sum());fn+=int((~yp&yg).sum());continue
            edges=np.unique([0.,duration]+[z for a,b,c in events for z in (a,b)])
            v=np.zeros((len(edges)-1,1));centers=(edges[:-1]+edges[1:])/2
            for a,b,c in events:v[(centers>=a)&(centers<b),0]=1
            key=item['id']+'__'+str(j);scores[key]=create_score_dataframe(v,edges,['event']);truth[key]=gt
        exact_audio+=int(audio_correct)
    clip=dict(tp=clip_tp,fp=clip_fp,fn=clip_fn,tn=clip_tn,recall=clip_tp/(clip_tp+clip_fn) if clip_tp+clip_fn else None,precision=clip_tp/(clip_tp+clip_fp) if clip_tp+clip_fp else None,f1=2*clip_tp/max(1,2*clip_tp+clip_fp+clip_fn),fpr=clip_fp/(clip_fp+clip_tn) if clip_fp+clip_tn else None,note='Audio-class/query presence; repeated events collapsed; only supplied queries evaluated.')
    from sklearn.metrics import average_precision_score
    class_ap={};excluded_ap=[]
    for label,pairs in clip_by_class.items():
        y,v=np.array(pairs).T
        if y.sum()==0:excluded_ap.append(label);continue
        class_ap[label]=float(average_precision_score(y,v))
    clip.update(accuracy=(clip_tp+clip_tn)/max(1,clip_tp+clip_fp+clip_fn+clip_tn),exact_audio_accuracy=exact_audio/len(manifest['items']),
        exact_audio_correct=exact_audio,audio_count=len(manifest['items']),binary_map=float(np.mean(list(class_ap.values()))) if class_ap and clip_fp+clip_tn else None,
        binary_ap_by_class=class_ap if clip_fp+clip_tn else {},ap_classes_without_positive=excluded_ap,
        map_note='Macro average of sklearn non-interpolated AP per class with positive GT; scores are binary 0/1 generated presence, not native confidence. No evaluable mAP for positive-only grounding queries. All equal-score predictions are tied, no arbitrary tie ordering.')
    if ds=='maestro-development':return dict(metric='generated-span micro 1s segment F1, NOT mpAUC',f1=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn,clip=clip)
    criterion=.5 if ds=='tag-test' else .7
    result=fscore(scores,truth,threshold=.5,dtc_threshold=criterion,gtc_threshold=criterion,num_jobs=1)
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        return x.tolist() if hasattr(x,'tolist') else x
    return dict(metric='generated-span pooled intersection F1, NOT PSDS',dtc=criterion,gtc=criterion,result=clean(result),clip=clip,note='Binary masks encode text-generated intervals; no model frame probabilities or confidence sweep.')

def compare_clip(root):
    """Threshold-free AP on identical complete-audio cohorts, preserving score provenance."""
    from sklearn.metrics import average_precision_score
    result={}
    for ds in [DATASETS[0],DATASETS[1],DATASETS[2],DATASETS[5]]:
        manifest=json.loads((ROOT/'manifests'/f'{ds}.json').read_text())
        labels=sc.MAESTRO_CLASSES if ds=='maestro-development' else manifest['classes']
        items=[];truth=[];omni=[]
        for item in manifest['items']:
            p=json.loads((root/ds/(item['id']+'.json')).read_text())
            if p['status']!='complete':continue
            assert set(p['labels'])==set(labels)
            present={c for a,b,c,v in item['soft_events'] if v>.5 and b>a} if ds=='maestro-development' else {c for a,b,c in item['events'] if b>a}
            predicted={c for a,b,c in p['events'] if b>a}
            items.append(item);truth.append([c in present for c in labels]);omni.append([c in predicted for c in labels])
        y=np.asarray(truth);keep=y.any(axis=0)
        def ap(values):
            values=np.asarray(values);assert values.shape==y.shape and np.isfinite(values).all()
            per={c:float(average_precision_score(y[:,j],values[:,j])) for j,c in enumerate(labels) if keep[j]}
            return dict(map=float(np.mean(list(per.values()))),ap_by_class=per)
        models={'qwen3-omni-30b':dict(ap(omni),score_type='binary generated presence')}
        previous=json.loads((root/ds/'metrics.complete-subset.json').read_text())
        assert abs(models['qwen3-omni-30b']['map']-previous['clip']['binary_map'])<1e-12
        for model in sc.MODELS:
            folder=ROOT/('runs-extra' if ds=='maestro-development' else 'runs-source-audit-v2')/model/ds
            values=[]
            for item in items:
                t,x,classes=sc.load_prediction(folder,item['id']);x=x[:,[classes.index(c) for c in labels]]
                if ds=='maestro-development':t,x=sc.mean_segments(t,x)
                values.append(x.max(axis=0))
            models[model]=dict(ap(values),score_type='maximum saved frame score; MAESTRO maximum duration-weighted 1s score')
        result[ds]=dict(scored=len(items),total=len(manifest['items']),clip_ids=[i['id'] for i in items],classes=labels,models=models)
        print(ds,{m:round(v['map'],6) for m,v in models.items()},flush=True)
    sc.save_json(root/'clip-comparison.json',dict(protocol='Clip-class AP, sklearn non-interpolated AP macro averaged over positive classes; identical Omni-complete audio cohort. Omni binary scores and OVSED continuous scores are distinct representations. No thresholds selected.',datasets=result))
    order=sc.MODELS+['qwen3-omni-30b']
    lines=['','## 与五个 OVSED 模型比较：Clip mAP','', '| 模型 | DESED Public | RealDESED | UrbanSED | MAESTRO Dev |','|---|---:|---:|---:|---:|']
    for model in order:lines.append('| '+model+' | '+' | '.join(f"{d['models'][model]['map']:.2%}" for d in result.values())+' |')
    lines+=['','- 四套数据均只使用 Omni 完整成功音频，分别为 '+', '.join(f"{ds}: {d['scored']}/{d['total']}" for ds,d in result.items())+'。同音频、同类别、同 GT、同 AP 实现。','- OVSED 取各类逐帧最高分；MAESTRO 先一秒加权再取最高分。Omni 只有生成类别的0/1输出。两种分数表示的信息量不同，不能将差值完全解释为模型能力差距；该表不是统一输出条件的受控优劣结论。','- 这里是 Clip 级类别存在性 mAP，不检查起止时间，不是 PSDS、MAESTRO 原 mpAUC 或定位 mAP。AP 不需选固定阈值。','- TAG/TACOS 原查询均为阳性，无有效 Clip mAP 对比，不加入表中。','- 本次不为 OVSED 临时指定0.5或按测试GT选择最佳阈值，因此没有伪造对应的Precision/准确率比较；这些固定操作点指标需要另设验证集选阈值协议。','- 逐类AP、音频ID与分数来源：runs-extra/qwen3-omni-30b/clip-comparison.json。','复算：~/venvs/finelap/bin/python scripts/run_omni_benchmark.py --compare-clip','']
    path=ROOT/'analysis/omni_results.md';old=path.read_text().split('\n## 与五个 OVSED 模型比较：Clip mAP')[0];path.write_text(old+'\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');parser.add_argument('--score-complete-only',action='store_true');parser.add_argument('--compare-clip',action='store_true');args=parser.parse_args()
    root=ROOT/'runs-extra/qwen3-omni-30b';root.mkdir(parents=True,exist_ok=True)
    if args.compare_clip:
        compare_clip(root);return
    if args.score_complete_only:
        for ds in DATASETS:
            manifest=json.loads((ROOT/'manifests'/f'{ds}.json').read_text());items=manifest['items'];valid=[];excluded=[]
            for item in items:
                path=root/ds/(item['id']+'.json')
                if path.exists() and json.loads(path.read_text()).get('status')=='complete':valid.append(item)
                else:excluded.append(item['id'])
            subset=dict(manifest,items=valid)
            result=score_spans(ds,subset,root/ds)
            result['coverage']=dict(total=len(items),scored=len(valid),excluded=len(excluded),excluded_ids=excluded,policy='Exclude incomplete audio entirely, including its successful windows; not a full-test-set score.')
            sc.save_json(root/ds/'metrics.complete-subset.json',result)
            print(ds,json.dumps({k:v for k,v in result.items() if k!='coverage'}),flush=True)
        return
    model=request(BASE+'/models')['data'][0]['id']
    config=dict(model=model,prompt=PROMPT,window_seconds=10,hop_seconds=5,temperature=0,seed=42,max_tokens=1536,merge='union overlapping same-query intervals',note='Generated event spans; no native frame scores, PSDS or mpAUC. No GT included in prompts.')
    config['identity']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (root/'config.json').exists():assert json.loads((root/'config.json').read_text())['identity']==config['identity']
    sc.save_json(root/'config.json',config)
    manifests={ds:json.loads((ROOT/'manifests'/f'{ds}.json').read_text()) for ds in DATASETS}
    state=dict(status='running',started=time.time(),completed=0,failed=[])
    def one(ds,item):
        folder=root/ds;folder.mkdir(exist_ok=True)
        labels=item.get('queries',sc.MAESTRO_CLASSES if ds=='maestro-development' else manifests[ds]['classes'])
        pred=audio(item,labels,folder,model);state['completed']+=1;state['current']=dict(dataset=ds,clip=item['id']);state['updated']=time.time()
        sc.save_json(root/'status.json',state)
        print(ds,item['id'],len(pred['events']),'events',round(pred['inference_seconds'],2),'seconds',flush=True)
    try:
        for ds in DATASETS:one(ds,manifests[ds]['items'][0])
        state['smoke_all_six']='passed';sc.save_json(root/'status.json',state)
        if not args.smoke_only:
            for ds in DATASETS:
                for item in manifests[ds]['items'][1:]:
                    try:one(ds,item)
                    except Exception as e:
                        state['failed'].append(dict(dataset=ds,clip=item['id'],error=repr(e)))
                        sc.save_json(root/'status.json',state)
                try:sc.save_json(root/ds/'metrics.json',score_spans(ds,manifests[ds],root/ds))
                except Exception as e:sc.save_json(root/ds/'metrics.json',dict(status='scoring_failed_or_incomplete',error=repr(e)))
        state.update(status='complete_with_failures' if state['failed'] else 'complete',finished=time.time());sc.save_json(root/'status.json',state)
    except Exception as e:
        state.update(status='failed',error=repr(e));sc.save_json(root/'status.json',state);raise

if __name__=='__main__':main()
