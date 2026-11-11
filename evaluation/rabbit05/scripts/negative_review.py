"""Six-benchmark failure-candidate census. No deployment threshold or official error counts.
Reuses saved predictions; never changes benchmark metrics or NPZ files.
"""
from pathlib import Path
import sys,json,csv,hashlib,html,argparse
from collections import Counter,defaultdict
import numpy as np
R=Path.home()/'projects/ov-sed/benchmark'
sys.path.insert(0,str(R/'scripts'))
import score_benchmarks as sc
OUT=R/'negative-review'
MODELS=sc.MODELS
DATASETS=['desed-public','realdesed-test','urbansed-test','tag-test','tacos-test','maestro-development']
NAMES={'desed-public':'DESED Public','realdesed-test':'RealDESED Test','urbansed-test':'UrbanSED Test','tag-test':'TAG Test','tacos-test':'TACOS Test','maestro-development':'MAESTRO Development'}
TYPES=['目标/背景排序倒置','背景峰值超过目标','连续区间覆盖不足','区间外扩/拖尾','区间覆盖与纯度均不足','断续响应','无标注目标高响应','低响应阳性秒段','高响应阴性秒段']

def components(values,edges):
    """All distinct connected superlevel components, equal scores activated together."""
    n=len(values);parent=np.full(n,-1,int);left=np.arange(n);right=np.arange(n)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[b]=a;left[a]=min(left[a],left[b]);right[a]=max(right[a],right[b])
    order=np.argsort(-values,kind='stable');out=[];k=0
    while k<n:
        end=k+1
        while end<n and values[order[end]]==values[order[k]]:end+=1
        indices=order[k:end]
        for i in indices:parent[i]=i
        for i in indices:
            if i and parent[i-1]>=0:union(i,i-1)
            if i+1<n and parent[i+1]>=0:union(i,i+1)
        for root in sorted({find(i) for i in indices}):out.append((edges[left[root]],edges[right[root]+1],float(values[order[k]])))
        k=end
    return np.array(out)

def intersections(t,events):
    w=np.zeros(len(t)-1)
    for a,b,*_ in events:w+=np.maximum(0,np.minimum(t[1:],b)-np.maximum(t[:-1],a))
    return np.minimum(w,np.diff(t))

def auc(v,pos,neg):
    if pos.sum()==0 or neg.sum()==0:return None
    o=np.argsort(v,kind='stable');x=v[o];p=pos[o];n=neg[o]
    starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1]
    ps=np.add.reduceat(p,starts);ns=np.add.reduceat(n,starts)
    return float((ps*(np.cumsum(ns)-ns+.5*ns)).sum()/(ps.sum()*ns.sum()))

def newrow(ds,m,item,label,a,b,types,**kw):
    row=dict(dataset=ds,model=m,clip_id=item['id'],target_class=str(label),onset=float(a),offset=float(b),
      candidate_types=types,audio_path=item['path'],review_status='待人工复核',**kw)
    row['case_id']=hashlib.sha256(json.dumps([ds,m,item['id'],str(label),float(a),float(b),types],ensure_ascii=False).encode()).hexdigest()[:16]
    return row

def runs(mask):
    c=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(c==1),np.flatnonzero(c==-1))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');parser.add_argument('--render-output-view',action='store_true');parser.add_argument('--render-only',action='store_true');parser.add_argument('--refresh-background',action='store_true');parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS);args=parser.parse_args()
    if args.serve:
        serve_outputs();return
    if args.render_output_view:
        render_outputs();return
    if args.refresh_background:
        refresh_background();render();render_outputs();return
    if args.render_only:
        render();render_outputs();return
    OUT.mkdir(exist_ok=True);allrows=[];coverage=[]
    for ds in args.datasets:
        manifestpath=R/'manifests'/f'{ds}.json';manifest=json.loads(manifestpath.read_text());items=manifest['items']
        unions=sc.semantic_unions(json.dumps(manifest,sort_keys=True)) if ds=='tacos-test' else None
        datasetrows=[]
        for m in MODELS:
            start_count=len(datasetrows);n_queries=n_events=0;negatives=[];positive_peaks=defaultdict(list);mae=[]
            folder=R/('runs-extra' if ds in ['tag-test','tacos-test','maestro-development'] else 'runs-source-audit-v2')/m/ds
            field='raw' if ds=='tag-test' and m=='finelap' else 'scores'
            for item in items:
                t,x,labels=sc.load_prediction(folder,item['id'],field)
                if ds=='maestro-development':
                    t,x=sc.mean_segments(t,x[:,[labels.index(c) for c in sc.MAESTRO_CLASSES]]);labels=sc.MAESTRO_CLASSES
                for j,label in enumerate(labels):
                    n_queries+=1;v=x[:,j]
                    if ds=='maestro-development':
                        gt=sc.merge_events([(a,b,c) for a,b,c,confidence in item['soft_events'] if c==label and confidence>.5],float(t[-1]))
                        y=intersections(t,gt)>0
                        mae.append((item,label,t,v,y,gt));continue
                    gt=unions[(item['id'],j)] if unions is not None else sc.merge_events([e for e in item['events'] if e[2]==label],float(t[-1]))
                    if not gt:
                        k=int(v.argmax());negatives.append((item,label,float(v[k]),float(t[k]),float(t[k+1])));continue
                    positive_peaks[label].append(float(v.max()))
                    bg=np.maximum(0,np.diff(t)-intersections(t,gt));tree=components(v,t);beg,end,levels=tree.T
                    for a,b,*_ in gt:
                        n_events+=1;pos=intersections(t,[(a,b)]);av=auc(v,pos,bg);flags=[]
                        inter=np.maximum(0,np.minimum(end,b)-np.maximum(beg,a));iou=inter/((end-beg)+(b-a)-inter)
                        k=int(np.argmax(iou));pa,pb=beg[k],end[k];purity=float(inter[k]/(pb-pa));recall=float(inter[k]/(b-a))
                        criterion=.5 if ds=='tag-test' else .7
                        feasible=bool(np.any((inter/(end-beg)>=criterion)&(inter/(b-a)>=criterion)))
                        if av is not None and av<.5:flags.append('目标/背景排序倒置')
                        target_peak=float(v[pos>0].max());bgidx=np.flatnonzero(intersections(t,gt)<=1e-8)
                        bgpeak=None;bgtime=None
                        if len(bgidx):
                            z=int(bgidx[np.argmax(v[bgidx])]);bgpeak=float(v[z]);bgtime=[float(t[z]),float(t[z+1])]
                            if bgpeak>target_peak:flags.append('背景峰值超过目标')
                        # Per-event best-IoU component is an oracle diagnostic, not an operating point.
                        if not feasible:
                            lowp=purity+1e-9<criterion;lowr=recall+1e-9<criterion
                            flags.append('区间覆盖与纯度均不足' if lowp and lowr else '区间外扩/拖尾' if lowp else '连续区间覆盖不足')
                            intersecting=[(float(t[u]),float(t[w])) for u,w in runs(v>=levels[k]) if t[u]<b and t[w]>a]
                            if len(intersecting)>1:flags.append('断续响应')
                        if flags:
                            others=sorted({str(c) for aa,bb,c in item['events'] if c!=label and aa<b and bb>a})
                            datasetrows.append(newrow(ds,m,item,label,a,b,flags,within_event_auc=av,target_peak=target_peak,
                                background_peak=bgpeak,background_peak_interval=bgtime,gt_intervals=gt,
                                diagnostic_interval=[float(pa),float(pb)],best_iou=float(iou[k]),coverage=recall,purity=purity,
                                oracle_score_cut=float(levels[k]),single_component_feasible=feasible,criterion=criterion,
                                overlapping_labels=others,score_field=field))
            # Cross-recording candidate rule: negative peak exceeds median positive-clip peak of same model/class.
            for item,label,peak,a,b in negatives:
                vals=positive_peaks.get(label,[])
                if vals and peak>=np.median(vals):
                    datasetrows.append(newrow(ds,m,item,label,a,b,['无标注目标高响应'],target_peak=peak,
                        positive_clip_peak_median=float(np.median(vals)),gt_intervals=[],score_field=field))
            if mae:
                pools=defaultdict(lambda:[[],[]])
                for item,label,t,v,y,gt in mae:
                    pools[label][0].extend(v[y].tolist());pools[label][1].extend(v[~y].tolist())
                for item,label,t,v,y,gt in mae:
                    pos,neg=pools[label]
                    for mask,kind,ref in [(y & (v<=np.median(neg)),'低响应阳性秒段',float(np.median(neg))) if neg else (y&False,'低响应阳性秒段',None),
                                          (~y & (v>=np.median(pos)),'高响应阴性秒段',float(np.median(pos))) if pos else (y&False,'高响应阴性秒段',None)]:
                        for u,w in runs(mask):
                            datasetrows.append(newrow(ds,m,item,label,t[u],t[w],[kind],segment_count=int(w-u),
                                segment_scores=[float(z) for z in v[u:w]],opposite_label_score_median=ref,
                                gt_intervals=gt,score_field=field))
            coverage.append(dict(dataset=ds,model=m,files=len(items),queries=n_queries,events=n_events,candidates=len(datasetrows)-start_count,
                                 score_field=field,manifest_sha256=hashlib.sha256(manifestpath.read_bytes()).hexdigest()))
            print(ds,m,'files',len(items),'candidates',len(datasetrows)-start_count,flush=True)
        for row in datasetrows:
            row['severity']=round((1-(row.get('within_event_auc') if row.get('within_event_auc') is not None else .5))+(1-row.get('best_iou',.5)),6)
        datasetrows.sort(key=lambda q:(-q['severity'],q['clip_id'],q['model'],q['onset']))
        allrows.extend(datasetrows)
        sc.save_json(OUT/'candidates.json',allrows)
    summary=dict(date='2026-10-22',coverage=coverage,rows=len(allrows),types=dict(Counter(k for row in allrows for k in row['candidate_types'])),
                 note='Diagnostic candidates, not official FP/FN counts. All saved files scanned; no handpicked model thresholds. Oracle intervals use test GT only to diagnose shape; not deployment or scoring. Multiple types per row.')
    sc.save_json(OUT/'summary.json',summary)
    fields=['case_id','dataset','model','clip_id','target_class','onset','offset','candidate_types','audio_path','gt_intervals','within_event_auc','best_iou','diagnostic_interval','coverage','purity','background_peak_interval','single_component_feasible','score_field','review_status']
    for name,rows in [('candidates',allrows)]+[(ds,[q for q in allrows if q['dataset']==ds]) for ds in args.datasets]:
        with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print('COMPLETE',len(allrows),flush=True)
    render();render_outputs()

def render():
    import soundfile as sf
    import librosa
    rows=json.loads((OUT/'candidates.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    selected={};counts=Counter()
    for ds in DATASETS:
        for kind in TYPES:
            eligible=[q for q in rows if q['dataset']==ds and kind in q['candidate_types']];used=set()
            for m in sorted(MODELS,key=lambda m:counts[(ds,m)]):
                q=next((q for q in eligible if q['model']==m and q['clip_id'] not in used),None)
                if q:selected[q['case_id']]=q;used.add(q['clip_id']);counts[(ds,m)]+=1
                if len(used)>=2:break
    media=OUT/'listen';media.mkdir(exist_ok=True);curves={}
    for cid,q in selected.items():
        info=sf.info(q['audio_path']);a=max(0,q['onset']-2);b=min(info.duration,max(q['offset']+2,a+6),a+16)
        audio,sr=sf.read(q['audio_path'],start=int(a*info.samplerate),stop=int(b*info.samplerate),always_2d=True)
        audio=librosa.resample(audio.mean(axis=1),orig_sr=sr,target_sr=16000)
        sf.write(media/(cid+'.wav'),audio,16000,subtype='PCM_16')
        ds=q['dataset'];folder=R/('runs-extra' if ds in ['tag-test','tacos-test','maestro-development'] else 'runs-source-audit-v2')/q['model']/ds
        t,x,labels=sc.load_prediction(folder,q['clip_id'],q['score_field'])
        if ds=='maestro-development':t,x=sc.mean_segments(t,x)
        v=x[:,labels.index(q['target_class'])];ix=np.flatnonzero((t[:-1]<b)&(t[1:]>a))
        bundle={}
        for model in MODELS:
            ff=R/('runs-extra' if ds in ['tag-test','tacos-test','maestro-development'] else 'runs-source-audit-v2')/model/ds
            tt,xx,ll=sc.load_prediction(ff,q['clip_id'],'raw' if ds=='tag-test' and model=='finelap' else 'scores')
            if ds=='maestro-development':tt,xx=sc.mean_segments(tt,xx)
            vv=xx[:,ll.index(q['target_class'])];ii=np.flatnonzero((tt[:-1]<b)&(tt[1:]>a))
            bundle[model]=dict(t=[round(float(z),6) for z in tt[np.r_[ii,ii[-1]+1]]],v=[round(float(z),7) for z in vv[ii]])
        curves[cid]=dict(start=a,end=b,models=bundle,query=q['target_class'],gt=q['gt_intervals'],audio='listen/'+cid+'.wav')
    questions=[];counts_by_audio=defaultdict(list)
    for row in rows:counts_by_audio[(row['dataset'],row['clip_id'])].append(row)
    for ds in DATASETS:
        manifest=json.loads((R/'manifests'/f'{ds}.json').read_text())
        unions=sc.semantic_unions(json.dumps(manifest,sort_keys=True)) if ds=='tacos-test' else None
        for item in manifest['items']:
            rr=counts_by_audio[(ds,item['id'])]
            if ds=='maestro-development':
                labels=sc.MAESTRO_CLASSES
                gt={c:sc.merge_events([(a,b,c) for a,b,k,v in item['soft_events'] if k==c and v>.5],item['duration']) for c in labels}
            else:
                labels=item.get('queries',manifest.get('classes',[]))
                gt={c:(unions[(item['id'],j)] if unions is not None else sc.merge_events([e for e in item['events'] if e[2]==c],item['duration'])) for j,c in enumerate(labels)}
            question_id=ds+'/'+item['id']
            questions.append(dict(id=question_id,dataset=ds,clip_id=item['id'],audio_path=item['path'],duration=item['duration'],
                gt=gt,types=sorted({k for row in rr for k in row['candidate_types']}),
                flagged_models=[m for m in MODELS if any(row['model']==m for row in rr)],
                excerpts=[cid for cid,q in selected.items() if q['dataset']==ds and q['clip_id']==item['id']]))
        columns=['audio_id','dataset','duration_seconds','audio_path','has_candidate','candidate_types']+MODELS
        with (OUT/(ds+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
            for q in [q for q in questions if q['dataset']==ds]:
                rr=counts_by_audio[(ds,q['clip_id'])]
                row=dict(audio_id=q['clip_id'],dataset=ds,duration_seconds=q['duration'],audio_path=q['audio_path'],has_candidate=bool(rr),candidate_types='；'.join(q['types']))
                row.update({m:'；'.join(sorted({k for r in rr if r['model']==m for k in r['candidate_types']})) or '未触发当前筛查规则' for m in MODELS})
                w.writerow(row)
    summary['count_unit']='dataset + audio_id';summary['audio_total']=len(questions);summary['audio_flagged']=sum(bool(q['flagged_models']) for q in questions)
    summary['audio_by_dataset']={ds:dict(total=sum(q['dataset']==ds for q in questions),flagged=sum(q['dataset']==ds and bool(q['flagged_models']) for q in questions),
        by_model={m:sum(q['dataset']==ds and m in q['flagged_models'] for q in questions) for m in MODELS}) for ds in DATASETS}
    page=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六个 Benchmark 失败候选复核</title>
<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 24px;color:#111;background:white}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid;font-size:14px}th{border-bottom:1px solid}td,th{padding:8px;text-align:center;vertical-align:top}tr{border-bottom:1px solid #ddd}button,select,input{padding:7px;margin:4px;font:inherit}button{cursor:pointer}#detail{border:1px solid #999;padding:18px;margin:20px 0}textarea{width:98%;height:75px;font:inherit}.muted{color:#555}svg{width:100%;height:230px}audio{width:100%}.path{word-break:break-all;font:13px monospace}a{color:#164c80}#detail:empty{display:none}</style>
<h1>六个 Benchmark：按音频逐题复核</h1>
<p><b>一条音频算一道题。</b>主列表和数量均按数据集内音频 ID 去重。点开后先看参考标注，再检查五个模型的响应及诊断细项。</p><p>DESED 仅 Public；MAESTRO 为 Development。FineLAP TAG 用 raw，TACOS 按当前近义合并口径。候选提示供人工复核，未触发规则不代表答对。<a href="README.md">统计定义</a> · <a href="candidates.csv">原诊断细项 CSV</a></p>
<div><select id="ds"></select><select id="model"></select><select id="kind"></select><input id="search" placeholder="音频 ID / 查询 / 案例 ID"><label><input id="only" type="checkbox" checked>仅看有候选的音频</label><label><input id="playable" type="checkbox">仅看可试听代表样本</label><button id="export">导出复核意见</button></div>
<p id="count"></p><p id="csvlinks"></p><div id="detail"></div><div style="overflow-x:auto"><table><thead><tr><th>音频 / Benchmark</th><th>时长</th><th>有候选的模型</th><th>问题类型</th><th>参考答案</th><th>操作</th></tr></thead><tbody id="body"></tbody></table></div><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button>
<script>
const rows=__ROWS__,curves=__CURVES__,names=__NAMES__,questions=__QUESTIONS__,models=['dasm','finelap','flexsed','mga-clap','openflam'];
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let filtered=[],page=0,notes={};try{notes=JSON.parse(localStorage.getItem('sed-audio-review')||'{}')}catch(e){}
const groups={},byid=Object.fromEntries(questions.map(q=>[q.id,q]));rows.forEach(r=>{(groups[r.dataset+'/'+r.clip_id]??=[]).push(r)});
function options(id,all,vals){document.getElementById(id).innerHTML='<option value="">'+all+'</option>'+vals.map(x=>'<option value="'+esc(x)+'">'+esc(names[x]||x)+'</option>').join('')}
options('ds','全部 Benchmark',Object.keys(names));options('model','任意模型有候选',models);options('kind','全部类型',[...new Set(rows.flatMap(r=>r.candidate_types))]);
document.getElementById('csvlinks').innerHTML=Object.entries(names).map(([k,v])=>'<a href="'+k+'.csv">'+v+' 逐音频清单</a>').join(' · ');
function filter(){const d=document.getElementById('ds').value,m=document.getElementById('model').value,k=document.getElementById('kind').value,s=document.getElementById('search').value.toLowerCase(),p=document.getElementById('playable').checked,only=document.getElementById('only').checked;
filtered=questions.filter(q=>(!d||q.dataset===d)&&(!only||q.flagged_models.length)&&(!m||q.flagged_models.includes(m))&&(!k||(groups[q.id]||[]).some(r=>(!m||r.model===m)&&r.candidate_types.includes(k)))&&(!p||q.excerpts.length)&&(!s||(q.id+' '+Object.keys(q.gt).join(' ')).toLowerCase().includes(s)));page=0;draw()}
function draw(){const base=questions.filter(q=>!document.getElementById('ds').value||q.dataset===document.getElementById('ds').value);document.getElementById('count').textContent='当前筛选 '+filtered.length+' 道音频题；该范围共 '+base.length+' 条，其中 '+base.filter(q=>q.flagged_models.length).length+' 条至少一个模型有候选。每条音频只计一次。';
document.getElementById('body').innerHTML=filtered.slice(page*30,(page+1)*30).map(q=>'<tr><td>'+esc(q.clip_id)+'<br>'+names[q.dataset]+'</td><td>'+q.duration.toFixed(2)+' 秒</td><td>'+q.flagged_models.length+'/5<br>'+q.flagged_models.join(', ')+'</td><td>'+q.types.map(esc).join('<br>')+'</td><td>'+Object.entries(q.gt).filter(([k,v])=>v.length).length+' 个有标注的查询</td><td><button data-id="'+esc(q.id)+'">'+(q.excerpts.length?'试听 / 五模型答案':'查看题目 / 五模型答案')+'</button></td></tr>').join('');document.getElementById('page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/30));document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=(page+1)*30>=filtered.length;}
const fmt=x=>x==null?'—':Number(x).toFixed(3);
function trace(c,model){const z=c.models[model],low=Math.min(...z.v),high=Math.max(...z.v),X=t=>65+(t-c.start)/(c.end-c.start)*900,Y=v=>150-(v-low)/(high-low||1)*115;const gt=c.gt.filter(e=>e[0]<c.end&&e[1]>c.start).map(e=>'<rect x="'+X(Math.max(c.start,e[0]))+'" y="25" width="'+(X(Math.min(c.end,e[1]))-X(Math.max(c.start,e[0])))+'" height="125" fill="#c5e6d3"/>').join('');let path='';z.v.forEach((v,i)=>{path+=(i?' L':'M')+X(Math.max(c.start,z.t[i]))+','+Y(v)+' L'+X(Math.min(c.end,z.t[i+1]))+','+Y(v)});return '<p><b>'+model+'</b></p><svg viewBox="0 0 1000 185">'+gt+'<path d="'+path+'" fill="none" stroke="#235b86" stroke-width="1.4"/><text x="2" y="35">'+high.toFixed(3)+'</text><text x="2" y="150">'+low.toFixed(3)+'</text><text x="65" y="178">'+c.start.toFixed(2)+' s</text><text x="890" y="178">'+c.end.toFixed(2)+' s</text></svg>'}
function detail(id){const q=byid[id],rr=groups[id]||[];let content='<h2>'+esc(q.clip_id)+' · '+names[q.dataset]+'</h2><p>一条音频，一道题。时长 '+q.duration.toFixed(3)+' 秒。</p><p class="path">'+esc(q.audio_path)+'</p><h3>参考答案：事件/查询与起止时间</h3><details open><summary>真值（含无标注查询）</summary><table><tr><th>类别 / 查询</th><th>真值时间（秒）</th></tr>'+Object.entries(q.gt).map(([k,v])=>'<tr><td>'+esc(k)+'</td><td>'+(v.length?v.map(e=>fmt(e[0])+'–'+fmt(e[1])).join('；'):'无标注事件')+'</td></tr>').join('')+'</table></details><h3>五个模型的检查结果</h3><p>连续分数是当前模型输出。未设正式操作点，暂不把诊断区间当成模型最终答案；未触发规则也不代表答对。</p>';
for(const m of models){let rs=rr.filter(r=>r.model===m);for(const [label,events] of Object.entries(q.gt)){for(const event of events){if(!rs.some(r=>r.target_class===label&&Math.abs(r.onset-event[0])<1e-5&&Math.abs(r.offset-event[1])<1e-5)){rs.push({target_class:label,onset:event[0],offset:event[1],candidate_types:['未触发当前筛查规则（不等于漏检或答对）'],best_iou:null,within_event_auc:null,diagnostic_interval:null,unflagged:true})}}}rs.sort((a,b)=>a.target_class.localeCompare(b.target_class)||a.onset-b.onset);const flagged=rs.filter(r=>!r.unflagged);content+='<details '+(rs.length?'open':'')+'><summary><b>'+m+'</b>：'+(flagged.length?[...new Set(flagged.flatMap(r=>r.candidate_types))].map(esc).join('；'):'未触发当前筛查规则，待复核')+'</summary>'+(rs.length?'<table><tr><th>查询</th><th>全部 GT / 背景检查时间</th><th>现象</th><th>最佳区间 IoU / AUC</th><th>诊断区间</th></tr>'+rs.map(r=>'<tr><td>'+esc(r.target_class)+'</td><td>'+fmt(r.onset)+'–'+fmt(r.offset)+'</td><td>'+r.candidate_types.map(esc).join('；')+'</td><td>'+fmt(r.best_iou)+' / '+fmt(r.within_event_auc)+'</td><td>'+esc(JSON.stringify(r.diagnostic_interval))+'</td></tr>').join('')+'</table>':'<p>完整答案仍需结合该模型原预测检查。</p>')+'</details>';}
if(q.excerpts.length){content+='<h3>同一音频片段：对照五模型响应</h3>';for(const cid of q.excerpts){const c=curves[cid];content+='<details open><summary>'+esc(c.query)+' · '+fmt(c.start)+'–'+fmt(c.end)+' 秒</summary><audio controls src="'+c.audio+'"></audio><p>绿色为真值，纵轴各自独立，无阈值线。</p>'+models.map(m=>trace(c,m)).join('')+'</details>';}}
content+='<h3>这道题的人工复核</h3><p>笔记按音频保存；可导出后和我讨论。</p><textarea id="note" placeholder="真值是否合理？哪些模型答对、哪些答错？具体听到了什么？">'+esc(notes[id]||'')+'</textarea>';document.getElementById('detail').innerHTML=content;document.getElementById('note').oninput=e=>{notes[id]=e.target.value;try{localStorage.setItem('sed-audio-review',JSON.stringify(notes))}catch(e){}};document.getElementById('detail').scrollIntoView({behavior:'smooth'});}
for(const id of ['ds','model','kind','playable','only'])document.getElementById(id).onchange=filter;document.getElementById('search').oninput=filter;document.getElementById('body').onclick=e=>{if(e.target.dataset.id)detail(e.target.dataset.id)};document.getElementById('prev').onclick=()=>{page--;draw()};document.getElementById('next').onclick=()=>{page++;draw()};document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='audio-review.json';a.click();URL.revokeObjectURL(a.href)};filter();
</script></html>'''
    gt_pool={}; compact=[]
    fields=['case_id','dataset','model','clip_id','target_class','onset','offset','candidate_types','audio_path','gt_ref','within_event_auc','best_iou','coverage','purity','diagnostic_interval','background_peak_interval']
    for row in rows:
        q=dict(row);gt=q.pop('gt_intervals',[]);key=hashlib.sha256(json.dumps(gt).encode()).hexdigest()[:20]
        gt_pool[key]=gt;q['gt_ref']=key
        for k in ['within_event_auc','best_iou','coverage','purity']:
            if q.get(k) is not None:q[k]=round(q[k],6)
        compact.append([q.get(k) for k in fields])
    page=page.replace('const rows=__ROWS__,','const rows=__ROWS__.map(a=>Object.fromEntries(__FIELDS__.map((k,i)=>[k,a[i]]))),')
    for token,value in [('__ROWS__',compact),('__FIELDS__',fields),('__QUESTIONS__',questions),('__CURVES__',curves),('__NAMES__',NAMES)]:
        page=page.replace(token,json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/'))
    (OUT/'review.html').write_text(page,encoding='utf-8')
    text='''# 六个 Benchmark 失败候选复核

更新：2026-10-22。[可筛选列表、试听与曲线](review.html) · [全量 CSV](candidates.csv)。只读取既有预测，未改权重、推理、主表或正式评分。

## 范围与使用

扫描五模型六套评测数据，DESED 仅 Public，MAESTRO 使用有真值的 Development，并非测试集。FineLAP TAG 用 raw；TACOS 使用当前 MiniLM-L6-v2、同文件余弦 >0.8 直接邻居真值合并，未宣称作者完全等价。

网页可按数据集、模型、类型和查询筛选。每条有稳定 ID、音频路径、时间位置与真值。代表样本附 16 kHz 单声道试听和连续响应曲线，最长 16 秒；其余保留完整服务器路径。笔记可在浏览器保存并导出。尚未人工听音确认。

## 定义

主列表每条音频只出现一次，点开后合并展示全部模型/查询/事件的诊断细项。旧 92,954 是修正前的细项数，不是题目数；已排除跨 GT 边界帧触发的背景峰值标记。全量扫描并列出符合规则的候选，不等于穷尽官方失败集合。PSDS 跨阈值计分，没有固定操作点就没有唯一的 FP/FN 清单。候选数不用于模型排名；一条可有多个类型，同音频可有多模型、多事件记录。

| 类型 | 筛选依据 | 限制 |
|---|---|---|
| 目标/背景排序倒置 | 单 GT 事件与同音频非目标时间的时长加权 AUC <0.5 | 排除其他同类事件；不是漏检率 |
| 背景峰值超过目标 | 完全处于全部同类 GT 之外的帧，其最高分 > 当前事件最高帧分数 | 跨 GT 边界的帧不参与背景峰值；高峰仍只是复核线索，不等于误报 |
| 连续区间覆盖不足 | 没有单连续区间同时满足覆盖与纯度要求；最佳 IoU 区间覆盖不足 | 是形态诊断，不是官方漏检 |
| 区间外扩/拖尾 | 同上，最佳 IoU 区间纯度不足 | 包括提前启动、拖尾、桥接相邻事件 |
| 区间覆盖与纯度均不足 | 同上，两项都不足 | 可能错位，暂不归因 |
| 断续响应 | 单区间不可行且最佳 IoU 所在分数水平有多个区间与 GT 相交 | 官方匹配可累积多个检测区间，不一定失败 |
| 无标注目标高响应 | 整条无该标签，最高分 ≥ 同模型同类阳性音频最高分中位数 | 测试集排序诊断，不是部署阈值；无标注不等于声音不存在 |
| 低响应阳性秒段 | MAESTRO 正秒段分数 ≤ 同模型同类负秒段分数中位数 | 沿用软标签 >0.5、一秒加权；相邻候选秒段合并 |
| 高响应阴性秒段 | MAESTRO 负秒段分数 ≥ 同模型同类正秒段分数中位数 | 不评价亚秒边界，不是误报率 |

连续区间枚举全部唯一分数对应的超水平连通区间。覆盖=与单个 GT 的交集/GT 时长；纯度=交集/连续区间时长。TAG 参考 0.5，其余事件任务参考 0.7；只借用比例作诊断，未修改官方计分。最佳 IoU 区间由测试 GT 辅助选择，是逐事件乐观诊断，不是统一阈值。AUC 用原始网格精确时长，部分边界帧可以同时有正负权重，不等同此前 100 ms 网格的片内 AUC。

TAG/TACOS 只扫描官方清单已有音频–查询配对，未另造 absent-query 负例。全阳性样本无背景，AUC 留空。所有结论待听辨，不预先标注具体原因。

## 分数据集完整列表

| Benchmark | 总音频题数 | 诊断细项（辅助） | 有候选的音频题数 | 逐音频清单 |
|---|---:|---:|---:|---|
'''
    for ds in DATASETS:
        rr=[q for q in rows if q['dataset']==ds];n=next(c['files'] for c in summary['coverage'] if c['dataset']==ds)
        text+=f"| {NAMES[ds]} | {n} | {len(rr)} | {len({q['clip_id'] for q in rr})} | [CSV]({ds}.csv) |\n"
    text+='\n## 按音频分类数量（同一题可有多种问题）\n\n| Benchmark | 类型 | 音频题数 |\n|---|---|---:|\n'
    for ds in DATASETS:
        for k in TYPES:
            n=sum(q['dataset']==ds and k in q['types'] for q in questions)
            if n:text+=f'| {NAMES[ds]} | {k} | {n} |\n'
    text+='\n## 试听起点\n\n代表样本优先选严重现象并尽量覆盖不同模型与录音；非随机抽样，不估算错误率。\n'
    for ds in DATASETS:
        text+='\n### '+NAMES[ds]+'\n\n| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |\n|---|---|---|---|---|---|\n'
        for cid,q in selected.items():
            if q['dataset']==ds:
                label=q['target_class'].replace('|','\\|').replace('\n',' ')
                text+=f"| {cid} | {q['model']} | {label} | {q['onset']:.3f}–{q['offset']:.3f} | {'；'.join(q['candidate_types'])} | [wav](listen/{cid}.wav) |\n"
    text+='\n## 复算\n\n在 rabbit05 benchmark 根目录运行：\n\n```bash\nOMP_NUM_THREADS=4 ~/venvs/finelap/bin/python scripts/negative_review.py\n# 仅更新页面与试听\n~/venvs/finelap/bin/python scripts/negative_review.py --render-only\n```\n\n覆盖见 summary.json；完整候选见 candidates.json / CSV。旧三个划分每类五条的筛查由本轮替代。原始预测和旧协议成绩未改。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8');summary['representative_excerpt_count']=len(selected);summary['representative_audio_count']=len({(q['dataset'],q['clip_id']) for q in selected.values()});sc.save_json(OUT/'summary.json',summary)
    print('RENDERED',len(rows),'rows',len(selected),'audio excerpts',flush=True)

def refresh_background():
    rows=json.loads((OUT/'candidates.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    grouped=defaultdict(list)
    for row in rows:grouped[(row['dataset'],row['model'],row['clip_id'])].append(row)
    result=[];removed_flags=0
    for (ds,m,clip),rr in grouped.items():
        relevant=[r for r in rr if '背景峰值超过目标' in r['candidate_types']]
        if relevant:
            folder=R/('runs-extra' if ds in ['tag-test','tacos-test','maestro-development'] else 'runs-source-audit-v2')/m/ds
            t,x,labels=sc.load_prediction(folder,clip,rr[0]['score_field']);cache={}
            for row in relevant:
                label=row['target_class']
                if label not in cache:
                    ids=np.flatnonzero(intersections(t,row['gt_intervals'])<=1e-8)
                    if len(ids):
                        j=labels.index(label);k=int(ids[np.argmax(x[ids,j])]);cache[label]=(float(x[k,j]),[float(t[k]),float(t[k+1])])
                    else:cache[label]=(None,None)
                peak,interval=cache[label];row['background_peak']=peak;row['background_peak_interval']=interval
                if peak is None or peak<=row['target_peak']:
                    row['candidate_types'].remove('背景峰值超过目标');removed_flags+=1
        result.extend(r for r in rr if r['candidate_types'])
    result.sort(key=lambda r:(DATASETS.index(r['dataset']),-r['severity'],r['clip_id'],r['model'],r['onset']))
    summary.update(rows=len(result),types=dict(Counter(k for row in result for k in row['candidate_types'])),background_peak_policy='only frames with zero overlap with ALL same-query GT intervals; crossing frames excluded')
    for c in summary['coverage']:c['candidates']=sum(r['dataset']==c['dataset'] and r['model']==c['model'] for r in result)
    sc.save_json(OUT/'candidates.json',result);sc.save_json(OUT/'summary.json',summary)
    fields=['case_id','dataset','model','clip_id','target_class','onset','offset','candidate_types','audio_path','gt_intervals','within_event_auc','best_iou','diagnostic_interval','coverage','purity','background_peak_interval','single_component_feasible','score_field','review_status']
    with (OUT/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in result:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print('BACKGROUND CORRECTION removed flags',removed_flags,'retained rows',len(result),flush=True)



def output_detail(ds,clip,label):
    manifest=json.loads((R/'manifests'/f'{ds}.json').read_text());item=next(i for i in manifest['items'] if i['id']==clip)
    assert ds in DATASETS
    allowed=item.get('queries',sc.MAESTRO_CLASSES if ds=='maestro-development' else manifest['classes'])
    if label not in allowed:raise ValueError('Unknown query')
    if ds=='tacos-test':gt=sc.semantic_unions(json.dumps(manifest,sort_keys=True))[(clip,allowed.index(label))]
    elif ds=='maestro-development':gt=sc.merge_events([(a,b,label) for a,b,c,v in item['soft_events'] if c==label and v>.5],item['duration'])
    else:gt=sc.merge_events([e for e in item['events'] if e[2]==label],item['duration'])
    result=dict(dataset=ds,clip_id=clip,query=label,duration=item['duration'],gt=gt,models={})
    for m in MODELS:
        folder=R/('runs-extra' if ds in ['tag-test','tacos-test','maestro-development'] else 'runs-source-audit-v2')/m/ds
        t,x,labels=sc.load_prediction(folder,clip,'raw' if ds=='tag-test' and m=='finelap' else 'scores')
        if ds=='maestro-development':t,x=sc.mean_segments(t,x)
        v=x[:,labels.index(label)];bg=np.maximum(0,np.diff(t)-intersections(t,gt));fully=intersections(t,gt)<=1e-8
        per_gt=[]
        for a,b,*_ in gt:
            w=intersections(t,[(a,b)]);ii=w>0
            per_gt.append(dict(onset=a,offset=b,mean=float((w*v).sum()/w.sum()),peak=float(v[ii].max())))
        result['models'][m]=dict(timestamps=t.tolist(),scores=v.tolist(),per_gt=per_gt,
            background_mean=float((bg*v).sum()/bg.sum()) if bg.sum() else None,
            background_peak=float(v[fully].max()) if fully.any() else None,
            score_field='raw' if ds=='tag-test' and m=='finelap' else 'scores')
    return result


def serve_outputs():
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from urllib.parse import urlparse,parse_qs
    from functools import lru_cache
    import io,soundfile as sf,librosa
    @lru_cache(maxsize=40)
    def detail(ds,clip,q):return json.dumps(output_detail(ds,clip,q),ensure_ascii=False,separators=(',',':')).encode()
    @lru_cache(maxsize=3)
    def audio(ds,clip):
        manifest=json.loads((R/'manifests'/f'{ds}.json').read_text());item=next(i for i in manifest['items'] if i['id']==clip)
        x,sr=sf.read(item['path'],always_2d=True);x=librosa.resample(x.mean(axis=1),orig_sr=sr,target_sr=16000)
        b=io.BytesIO();sf.write(b,x,16000,format='WAV',subtype='PCM_16');return b.getvalue()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u=urlparse(self.path);q=parse_qs(u.query)
            try:
                if u.path in ['/','/review.html']:
                    data=(OUT/'review.html').read_bytes();kind='text/html; charset=utf-8'
                elif u.path=='/detail':
                    ds=q['dataset'][0];assert ds in DATASETS
                    data=detail(ds,q['clip'][0],q['query'][0]);kind='application/json; charset=utf-8'
                elif u.path=='/audio':
                    ds=q['dataset'][0];assert ds in DATASETS
                    data=audio(ds,q['clip'][0]);kind='audio/wav'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Access-Control-Allow-Origin','*');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(data)
            except (KeyError,ValueError,AssertionError,StopIteration):self.send_error(400)
            except (BrokenPipeError,ConnectionResetError):pass
    print('Full output review: http://127.0.0.1:18765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()


def render_outputs():
    # Retain the audio question index and manual notes; replace diagnostic-centric presentation.
    previous=(OUT/'review.html').read_text(encoding='utf-8')
    mark='questions=';begin=previous.index(mark)+len(mark)
    questions,_=json.JSONDecoder().raw_decode(previous[begin:])
    questions=[{k:q[k] for k in ['id','dataset','clip_id','duration','gt','audio_path']} for q in questions]
    cat=output_detail('desed-public','-0H7vrzA80U_4_14','Cat')
    page=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐音频模型输出对照</title>
<style>body{font:16px/1.6 system-ui;margin:28px auto;max-width:1300px;padding:0 20px;color:#111}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid}th{border-bottom:1px solid}td,th{text-align:center;padding:8px}button,input,select{font:inherit;padding:7px;margin:4px}button{cursor:pointer}.muted{color:#555}svg{width:100%;height:190px;display:block}audio{width:100%}textarea{width:98%;height:90px;font:inherit}.path{font:12px monospace;word-break:break-all}details{margin:15px 0}#content{margin:25px 0}.status{padding:10px;background:#f4f4f4}</style>
<h1>逐条音频：真值与五模型实际分数</h1>
<p>选择音频和类别，查看完整时长的响应。绿色区间为全部同类 GT；蓝线是已保存的逐帧分数。所有模型纵轴统一为 0–1，不归一化原始分数，不设检测阈值。MAESTRO 显示评测用的一秒加权分数。若有超出 0–1 的分数，会在对应图旁注明。</p>
<div><select id="dataset"></select><input id="search" placeholder="搜索音频 ID"><select id="question"></select><button id="prev">上一题</button><button id="next">下一题</button></div>
<div><label>类别 / 文本查询 <select id="query"></select></label><button id="load">查看完整输出</button><button id="export">导出人工笔记</button></div><p id="position"></p><div id="content"></div>
<script>
const questions=__QUESTIONS__,names=__NAMES__,embedded=__EXAMPLE__,models=['dasm','finelap','flexsed','mga-clap','openflam'];
const api=location.protocol==='file:'?'http://127.0.0.1:18765':'';const $=id=>document.getElementById(id),esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let visible=[],active=null,request=0,notes={};try{notes=JSON.parse(localStorage.getItem('sed-audio-review')||'{}')}catch(e){}
$('dataset').innerHTML=Object.entries(names).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('');
function chooseList(){const s=$('search').value.toLowerCase();visible=questions.filter(q=>q.dataset===$('dataset').value&&q.clip_id.toLowerCase().includes(s));$('question').innerHTML=visible.map(q=>'<option value="'+esc(q.id)+'">'+esc(q.clip_id)+'</option>').join('');chooseQuestion()}
function chooseQuestion(){active=visible.find(q=>q.id===$('question').value)||visible[0];if(!active){$('content').textContent='没有匹配的音频';return}const labels=Object.keys(active.gt);$('query').innerHTML=labels.map(c=>'<option value="'+esc(c)+'">'+esc(c)+(active.gt[c].length?' · GT '+active.gt[c].length+' 段':' · 无标注')+'</option>').join('');const first=labels.find(c=>active.gt[c].length);if(first)$('query').value=first;$('position').textContent=(visible.indexOf(active)+1)+' / '+visible.length+' · '+active.duration.toFixed(3)+' 秒';load()}
function plot(m,d){const z=d.models[m],X=t=>75+t/d.duration*900,Y=v=>145-v*115;let low=Infinity,high=-Infinity;for(const v of z.scores){low=Math.min(low,v);high=Math.max(high,v)}const gt=d.gt.map(e=>'<rect x="'+X(e[0])+'" y="30" width="'+(X(e[1])-X(e[0]))+'" height="115" fill="#c5e6d3"/>').join('');let path='';z.scores.forEach((v,i)=>{path+=(i?' L':'M')+X(z.timestamps[i])+','+Y(v)+' L'+X(z.timestamps[i+1])+','+Y(v)});const ticks=[0,.25,.5,.75,1].map(v=>'<line x1="75" x2="975" y1="'+Y(v)+'" y2="'+Y(v)+'" stroke="#ddd" stroke-width=".6"/><text x="60" y="'+(Y(v)+4)+'" text-anchor="end">'+v.toFixed(2)+'</text>').join('');return '<h3>'+m+'</h3>'+(low<0||high>1?'<p class="muted">原始分数范围 '+low.toFixed(4)+'–'+high.toFixed(4)+'；超出 0–1 的部分未画入坐标框，未修改分数。</p>':'')+'<svg class="curve" viewBox="0 0 1000 180"><defs><clipPath id="score-frame-'+m+'"><rect x="75" y="30" width="900" height="115"/></clipPath></defs>'+gt+ticks+'<path d="'+path+'" clip-path="url(#score-frame-'+m+')" fill="none" stroke="#235b86" stroke-width="1.3"/>'+[0,.25,.5,.75,1].map(f=>'<text x="'+X(f*d.duration)+'" y="172" text-anchor="middle">'+(f*d.duration).toFixed(2)+' s</text>').join('')+'<line class="cursor" x1="75" x2="75" y1="30" y2="145" stroke="#c33" stroke-width="1"/></svg>'}
const fmt=x=>x==null?'—':x.toFixed(3);
function show(d){let s='<h2>'+esc(d.clip_id)+' · '+esc(d.query)+'</h2><audio id="audio" controls preload="none" src="'+api+'/audio?'+new URLSearchParams({dataset:d.dataset,clip:d.clip_id})+'"></audio><p><b>全部 GT：</b>'+(d.gt.map((e,i)=>(i+1)+'. '+fmt(e[0])+'–'+fmt(e[1])+' 秒').join('；')||'没有标注事件')+'</p><p class="path">'+esc(active.audio_path)+'</p>';
s+=models.map(m=>plot(m,d)).join('');s+='<h3>逐段 GT 内实际分数：均值 / 峰值</h3><table><tr><th>GT 时间（秒）</th>'+models.map(m=>'<th>'+m+'</th>').join('')+'</tr>'+d.gt.map((e,i)=>'<tr><td>'+fmt(e[0])+'–'+fmt(e[1])+'</td>'+models.map(m=>{const r=d.models[m].per_gt[i];return '<td>'+fmt(r.mean)+' / '+fmt(r.peak)+'</td>'}).join('')+'</tr>').join('')+'<tr><td>GT 外时间：均值 / 完全背景帧峰值</td>'+models.map(m=>'<td>'+fmt(d.models[m].background_mean)+' / '+fmt(d.models[m].background_peak)+'</td>').join('')+'</tr></table><p class="muted">均值按帧与 GT 重叠时长加权；峰值取与 GT 相交的帧。背景峰值排除跨 GT 边界的帧。分数不可直接跨模型比较。</p><h3>人工观察</h3><textarea id="note" placeholder="每次事件是否有响应？间隙是否持续高分？你听到了什么？">'+esc(notes[active.id]||'')+'</textarea>';$('content').innerHTML=s;const id=active.id;$('note').oninput=e=>{notes[id]=e.target.value;try{localStorage.setItem('sed-audio-review',JSON.stringify(notes))}catch(e){}};$('audio').ontimeupdate=()=>{const x=75+$('audio').currentTime/d.duration*900;document.querySelectorAll('.cursor').forEach(el=>{el.setAttribute('x1',x);el.setAttribute('x2',x)})}}
async function load(){if(!active)return;const token=++request;const q=$('query').value;$('content').innerHTML='<p class="status">读取该音频五个模型的完整分数…</p>';if(active.dataset===embedded.dataset&&active.clip_id===embedded.clip_id&&q===embedded.query){show(embedded);return}try{const res=await fetch(api+'/detail?'+new URLSearchParams({dataset:active.dataset,clip:active.clip_id,query:q}));if(!res.ok)throw Error('HTTP '+res.status);const d=await res.json();if(token===request)show(d)}catch(e){if(token===request)$('content').innerHTML='<p class="status">分数读取失败，请确认服务器连接后重试。</p>'}}
$('dataset').onchange=chooseList;$('search').oninput=chooseList;$('question').onchange=chooseQuestion;$('query').onchange=load;$('load').onclick=load;for(const [id,step] of [['prev',-1],['next',1]])$(id).onclick=()=>{const i=visible.indexOf(active)+step;if(i>=0&&i<visible.length){$('question').value=visible[i].id;chooseQuestion()}};
$('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='audio-review.json';a.click();URL.revokeObjectURL(a.href)};
$('dataset').value='desed-public';chooseList();$('question').value='desed-public/-0H7vrzA80U_4_14';chooseQuestion();$('query').value='Cat';load();
</script></html>'''
    for token,value in [('__QUESTIONS__',questions),('__NAMES__',NAMES),('__EXAMPLE__',cat)]:page=page.replace(token,json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/'))
    (OUT/'review.html').write_text(page,encoding='utf-8');print('FULL OUTPUT VIEW',len(questions),'audio questions',flush=True)

if __name__=='__main__':main()
