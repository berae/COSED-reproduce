"""Official matching counts over thresholds; rates are not heuristic failure flags."""
from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path.home()/'projects/ov-sed/benchmark'
sys.path.insert(0,str(ROOT/'scripts'))
import score_benchmarks as sc
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval.intersection_based.intermediate_statistics import accumulated_intermediate_statistics
OUT=ROOT/'analysis/error_rates'
SETS=['desed-public','realdesed-test','urbansed-test','tag-test','tacos-test','maestro-development']

def inputs(ds,model,manifest):
    if ds in SETS[:3]:
        scores,gt,durations=sc.load_inputs(model,ds,manifest)
        return scores,gt,sum(durations.values())/3600
    folder=ROOT/'runs-extra'/model/ds;scores={};gt={};hours=0
    unions=sc.semantic_unions(json.dumps(manifest,sort_keys=True)) if ds=='tacos-test' else None
    for item in manifest['items']:
        t,x,labels=sc.load_prediction(folder,item['id'],'raw' if ds=='tag-test' and model=='finelap' else 'scores')
        for j,label in enumerate(labels):
            key=item['id']+'__'+str(j)
            scores[key]=create_score_dataframe(x[:,j:j+1],t,['event'])
            gt[key]=unions[(item['id'],j)] if unions is not None else sc.merge_events([(a,b,'event') for a,b,c in item['events'] if c==label],float(t[-1]))
            hours+=float(t[-1])/3600
    return scores,gt,hours

def event_curves(ds,model,manifest):
    scores,gt,hours=inputs(ds,model,manifest)
    criterion=.5 if ds=='tag-test' else .7
    stats,_=accumulated_intermediate_statistics(scores,gt,dtc_threshold=criterion,gtc_threshold=criterion,num_jobs=1)
    arrays={};all_scores=[];by_class={}
    for key,df in scores.items():
        for label in df.columns[2:]:
            v=df[label].to_numpy();by_class.setdefault(label,[[],[]]);by_class[label][0].append(v);by_class[label][1].append(np.minimum(v[:-1],v[1:]));all_scores.append(v)
    # Presentation grid determined from score distribution only, never ground truth.
    v=np.concatenate(all_scores);grid=np.unique(np.r_[np.nextafter(v.min(),-np.inf),np.quantile(v,np.linspace(0,1,201)),np.inf])
    tp=np.zeros(len(grid));fp=tp.copy();n_pred=tp.copy();n_ref=0
    for j,(label,(levels,s)) in enumerate(stats.items()):
        # Library rows describe >= level. For score > grid use the next level.
        idx=np.minimum(np.searchsorted(levels,grid,side='right'),len(levels)-1)
        idx[np.isposinf(grid)]=len(levels)-1
        tp+=s['tps'][idx];fp+=s['fps'][idx];n_ref+=s['n_ref']
        values=np.sort(np.concatenate(by_class[label][0]));adj=np.sort(np.concatenate(by_class[label][1]))
        n_pred+=(len(values)-np.searchsorted(values,grid,side='right'))-(len(adj)-np.searchsorted(adj,grid,side='right'))
        arrays[f'{j}_levels']=levels;arrays[f'{j}_tp']=s['tps'];arrays[f'{j}_fp']=s['fps'];arrays[f'{j}_n_ref']=np.array(s['n_ref'])
        arrays[f'{j}_values']=values;arrays[f'{j}_adjacent_minima']=adj
    assert np.all(fp<=n_pred+1e-6)
    np.savez_compressed(OUT/f'{ds}_{model}.npz',**arrays,classes=np.array(list(stats)))
    return [dict(dataset=ds,model=model,unit='event',threshold=None if np.isinf(t) else float(t),threshold_is_infinity=bool(np.isinf(t)),
        gt_count=int(n_ref),tp=float(a),fn=float(n_ref-a),fp=float(b),prediction_count=int(round(c)),
        miss_rate=float(1-a/n_ref),false_prediction_fraction=float(b/c) if c else None,false_positives_per_hour=float(b/hours),
        negative_segment_fpr=None,dtc=criterion,gtc=criterion) for t,a,b,c in zip(grid,tp,fp,n_pred)]

def segment_curves(model,manifest):
    values=[];targets=[]
    for item in manifest['items']:
        t,x,labels=sc.load_prediction(ROOT/'runs-extra'/model/'maestro-development',item['id'])
        t,x=sc.mean_segments(t,x[:,[labels.index(c) for c in sc.MAESTRO_CLASSES]])
        y=np.zeros_like(x,dtype=bool)
        for a,b,c,confidence in item['soft_events']:
            if confidence>.5 and c in sc.MAESTRO_CLASSES:y[:,sc.MAESTRO_CLASSES.index(c)]|=(t[:-1]<b)&(t[1:]>a)
        values.append(x);targets.append(y)
    v=np.concatenate(values).ravel();y=np.concatenate(targets).ravel();pos=np.sort(v[y]);neg=np.sort(v[~y])
    np.savez_compressed(OUT/f'maestro-development_{model}.npz',positive_scores=pos,negative_scores=neg)
    grid=np.unique(np.r_[np.nextafter(v.min(),-np.inf),np.quantile(v,np.linspace(0,1,201)),np.inf]);rows=[]
    for t in grid:
        tp=len(pos)-np.searchsorted(pos,t,side='right');fp=len(neg)-np.searchsorted(neg,t,side='right')
        rows.append(dict(dataset='maestro-development',model=model,unit='one-second class-segment',threshold=None if np.isinf(t) else float(t),threshold_is_infinity=bool(np.isinf(t)),gt_count=len(pos),tp=int(tp),fn=int(len(pos)-tp),fp=int(fp),prediction_count=int(tp+fp),miss_rate=float(1-tp/len(pos)),false_prediction_fraction=float(fp/(tp+fp)) if tp+fp else None,false_positives_per_hour=None,negative_segment_fpr=float(fp/len(neg)),dtc=None,gtc=None))
    return rows

def clip_curves(ds,model,manifest):
    """One binary answer per audio/class or supplied grounding query, no time matching."""
    values=[];targets=[];owners=[]
    unions=sc.semantic_unions(json.dumps(manifest,sort_keys=True)) if ds=='tacos-test' else None
    folder=ROOT/('runs-source-audit-v2' if ds in SETS[:3] else 'runs-extra')/model/ds
    for owner,item in enumerate(manifest['items']):
        t,x,labels=sc.load_prediction(folder,item['id'],'raw' if ds=='tag-test' and model=='finelap' else 'scores')
        if ds=='maestro-development':
            x=x[:,[labels.index(c) for c in sc.MAESTRO_CLASSES]];labels=sc.MAESTRO_CLASSES
            t,x=sc.mean_segments(t,x)
            present={c for a,b,c,confidence in item['soft_events'] if confidence>.5 and b>a}
        else:
            present={c for a,b,c in item['events'] if b>a}
        y=np.array([bool(unions[(item['id'],j)]) if unions is not None else c in present for j,c in enumerate(labels)],dtype=bool)
        values.extend(np.max(x,axis=0));targets.extend(y);owners.extend([owner]*len(labels))
    return binary_clip_curves(ds,model,np.array(values),np.array(targets),np.array(owners),len(manifest['items']))


def binary_clip_curves(ds,model,v,y,owners,n_audio):
    pos=np.sort(v[y]);neg=np.sort(v[~y]);grid=np.unique(np.r_[np.nextafter(v.min(),-np.inf),np.quantile(v,np.linspace(0,1,201)),np.inf]);rows=[]
    positive_audio=len(np.unique(owners[y]));negative_audio=len(np.unique(owners[~y]))
    for t in grid:
        pred=v>t;tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));fn=len(pos)-tp
        missed_audio=len(np.unique(owners[y&~pred]));extra_audio=len(np.unique(owners[~y&pred]))
        rows.append(dict(dataset=ds,model=model,mode='clip',unit='audio-class presence',threshold=None if np.isinf(t) else float(t),threshold_is_infinity=bool(np.isinf(t)),
            gt_count=len(pos),tp=tp,fn=fn,fp=fp,prediction_count=tp+fp,miss_rate=fn/len(pos) if len(pos) else None,
            false_prediction_fraction=fp/(tp+fp) if tp+fp else None,false_positives_per_hour=None,negative_segment_fpr=fp/len(neg) if len(neg) else None,dtc=None,gtc=None,
            negative_count=len(neg),audio_count=n_audio,positive_audio_count=positive_audio,negative_audio_count=negative_audio,
            missed_audio_count=missed_audio,extra_audio_count=extra_audio,audio_miss_rate=missed_audio/positive_audio if positive_audio else None,
            audio_extra_rate=extra_audio/negative_audio if negative_audio else None))
    return rows


def save_tables(rows):
    for r in rows:r.setdefault('mode','event')
    sc.save_json(OUT/'rates.json',rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    for ds in SETS:
        with (OUT/(ds+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r for r in rows if r['dataset']==ds)
    render(rows)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--clip-only',action='store_true',help='Reuse saved event results; compute only new clip presence curves')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);rows=[]
    if args.clip_only:
        rows=[r for r in json.loads((OUT/'rates.json').read_text()) if r.get('mode','event')!='clip']
    for ds in SETS:
        manifest=json.loads((ROOT/'manifests'/f'{ds}.json').read_text())
        for model in sc.MODELS:
            if not args.clip_only:
                result=segment_curves(model,manifest) if ds=='maestro-development' else event_curves(ds,model,manifest)
                rows.extend(result)
            result=clip_curves(ds,model,manifest);rows.extend(result)
            print(ds,model,len(result),'clip operating points',flush=True)
    save_tables(rows)
    print('COMPLETE',len(rows),flush=True)

def render(rows):
    page=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OVSED 漏检与误报</title>
<style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#111}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid}th{border-bottom:1px solid}td,th{text-align:center;padding:9px}select,button,input{font:inherit;margin:5px;padding:6px}svg{width:100%;height:420px}.muted{color:#555}circle{cursor:pointer}a{color:#205b88}</style>
<h1>OVSED：漏检率与误报</h1><p>请选择统计方式和数据集，再点击曲线上的点查看对应阈值。两种方式复用同一批预测，不改变原有 PSDS。</p>
<p><b>漏检率</b> = FN / GT；<b>误报占预测的比例</b> = FP / 所有预测事件数。这一比例不是阴性样本的 FPR。没有预测时，误报占比无定义。事件任务同时列每小时误报次数。</p>
<label>统计方式 <select id="mode"><option value="event">事件级：逐个 GT 事件（MAESTRO 为秒段）</option><option value="clip">Clip 级：同类 GT 在一条音频中只算一次</option></select></label><label>数据集 <select id="dataset"></select></label><p id="scope"></p><p id="links"></p><div id="chart"></div><p id="legend"></p>
<table><thead><tr><th>模型</th><th>阈值</th><th>漏检率</th><th>误报占预测比例</th><th>FP/小时</th><th>TP / FN / FP</th></tr></thead><tbody id="table"></tbody></table>
<p class="muted">表格初始未选择操作点，不推荐任何默认阈值。图中每个点都是实际可计算的阈值。坐标：横轴误报占预测的比例，纵轴漏检率，均越低越好。各模型分数尺度不同。</p>
<details><summary>口径与完整数据</summary><p>DESED Public、RealDESED、UrbanSED 使用 DTC=GTC=0.7；TAG 为 0.5；TACOS 为 0.7，沿用当前近义真值合并。FineLAP TAG 使用未滤波 raw。事件匹配使用 sed_scores_eval。事件数量不是音频数量：一条音频中多次事件分别计入 GT、FN。</p><p>官方允许多对多匹配，TP 统计检出的 GT，不能直接用 TP+FP 代替预测总数。本页另行精确统计预测连续区间总数，计算 FP/预测数。也不能将这一比例混称为 1−官方 precision。</p><p>MAESTRO 使用十一类、一秒加权分数、软标注 >0.5，统计类别×秒段的 TP/FN/FP；详情另提供标准阴性秒段 FPR。它不是事件级边界统计。</p><p>CSV/页面展示基于分数分布的 201 个分位阈值，另含全部输出/不输出两端点；未用 GT 优选阈值。NPZ 保存全部原生匹配变化点及预测数重建数据，便于任意阈值复算。TAG/TACOS 的 FP/小时分母为文件–查询对的总评测小时；其他事件任务为真实音频小时，不能跨任务直接横比。</p></details>
<details open><summary>Clip 级如何计算</summary><p>例如一条音频有五次猫叫、一次说话，Clip 级只需要回答“有猫叫、有说话”。猫叫只检测到一次也算该类检出，完全忽略起止时间、次数和时长。每类分数取整段最大分数，按 分数 &gt; 阈值 判断出现；MAESTRO 取一秒加权分数的最大值。</p><p>曲线仍以类别答案计数：漏检率 = 未检出的阳性音频–类别对 / 所有阳性音频–类别对；横轴 = 误报类别答案 / 所有预测阳性类别答案。下方同时列出“至少漏一类”的音频数量，一条音频只计一次。标准 FPR 另列为 FP / 阴性音频–类别对。</p><p>TAG/TACOS 只评价清单已有查询，未查询类别不算漏检或误报。若清单没有阴性查询，误报能力不可测；FP=0 不代表没有误报。“所有 GT 类别检出”也不代表起止时间正确。</p></details>
<div id="detail"></div><script>
const rows=__DATA__,models=['dasm','finelap','flexsed','mga-clap','openflam'],colors=['#2463a5','#b14232','#34844b','#8b5b9c','#a47716'];let selected={};const $=x=>document.getElementById(x),pct=x=>x==null?'—':(100*x).toFixed(2)+'%',num=x=>x==null?'—':Number(x).toFixed(2);
$('dataset').innerHTML=[...new Set(rows.map(r=>r.dataset))].map(ds=>'<option>'+ds+'</option>').join('');
function table(){const clip=$('mode').value==='clip';$('table').innerHTML=models.map(m=>{const r=selected[m];return '<tr><td>'+m+'</td><td>'+(r?(r.threshold_is_infinity?'∞':r.threshold.toPrecision(6)):'点击曲线选择')+'</td><td>'+pct(r?.miss_rate)+'</td><td>'+(r&&clip&&r.negative_count===0?'不可测':pct(r?.false_prediction_fraction))+'</td><td>'+num(r?.false_positives_per_hour)+'</td><td>'+(r?[r.tp,r.fn,r.fp].map(x=>Number(x).toFixed(0)).join(' / '):'—')+'</td></tr>'}).join('');$('detail').innerHTML=Object.entries(selected).map(([m,r])=>'<p><b>'+m+'</b>：'+(clip?'阳性音频–类别对 '+r.gt_count+'，阴性对 '+r.negative_count+'；标准 FPR '+pct(r.negative_segment_fpr)+'。<br>按音频计数：至少漏一类 '+r.missed_audio_count+' / '+r.positive_audio_count+'（'+pct(r.audio_miss_rate)+'）；至少多报一类 '+r.extra_audio_count+' / '+r.negative_audio_count+'（'+pct(r.audio_extra_rate)+'）。前者分母为含 GT 的音频，后者为有阴性类别可检查的音频。':'GT '+r.gt_count+'，预测 '+r.prediction_count+'；'+(r.negative_segment_fpr!=null?'阴性秒段 FPR '+pct(r.negative_segment_fpr):'事件匹配 DTC/GTC '+r.dtc+'/'+r.gtc))+'</p>').join('')}
function draw(){selected={};const ds=$('dataset').value,mode=$('mode').value;$('scope').textContent=mode==='clip'?'Clip 级：同一音频同一类别只算一个答案，不检查时间。下方另列按音频去重的漏检、误报数量。':'保留原统计：逐个 GT 事件匹配；MAESTRO 为类别×一秒段。';let svg='<svg viewBox="0 0 1000 430"><path d="M70 20 V370 H970" fill="none" stroke="black"/><text x="75" y="15">漏检率 ↑</text>';
for(let k=0;k<=5;k++){const f=k/5;svg+='<text x="'+(70+900*f)+'" y="394" text-anchor="middle">'+(100*f)+'%</text><text x="60" y="'+(375-350*f)+'" text-anchor="end">'+(100*f)+'%</text>'}
models.forEach((m,j)=>{const data=rows.filter(r=>r.dataset===ds&&r.model===m&&(r.mode||'event')===mode);const pts=data.filter(r=>r.false_prediction_fraction!=null);svg+='<polyline points="'+pts.map(r=>(70+900*r.false_prediction_fraction)+','+(370-350*r.miss_rate)).join(' ')+'" fill="none" stroke="'+colors[j]+'" stroke-width="1.2" opacity=".65"/>';pts.forEach(r=>{svg+='<circle cx="'+(70+900*r.false_prediction_fraction)+'" cy="'+(370-350*r.miss_rate)+'" r="3.5" fill="'+colors[j]+'" data-index="'+rows.indexOf(r)+'"><title>'+m+'，阈值 '+r.threshold+'，漏检 '+pct(r.miss_rate)+'，误报占比 '+(mode==='clip'&&r.negative_count===0?'不可测':pct(r.false_prediction_fraction))+'</title></circle>'})});svg+='<text x="520" y="425" text-anchor="middle">误报占预测的比例 →</text></svg>';$('chart').innerHTML=svg;$('legend').innerHTML=models.map((m,j)=>'<span style="color:'+colors[j]+'">● '+m+'</span>').join('　');$('links').innerHTML='<a href="'+ds+'.csv">下载该数据集两种方式的各阈值结果 CSV</a>';table()}
$('dataset').onchange=draw;$('mode').onchange=draw;$('chart').onclick=e=>{const index=e.target.getAttribute('data-index');if(index!==null){const r=rows[Number(index)];selected[r.model]=r;table()}};draw();
</script></html>'''
    (OUT/'review.html').write_text(page.replace('__DATA__',json.dumps(rows,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
    (OUT/'README.md').write_text('''# OVSED 漏检率与误报

[交互查看页](review.html)。按数据集选择，点击曲线上的点读取该模型、该阈值的实际漏检率和误报。

页面顶部可切换：

- 事件级：保留之前的逐事件匹配；MAESTRO 仍按一秒段。
- Clip 级：每条音频中每类只算一次，五次猫叫只需检出一次；不检查事件次数和起止时间。每类取整段最高分，按分数 > 阈值判断出现。MAESTRO 先计算官方一秒加权分数再取最大值，GT 沿用软标注 >0.5。
- Clip 曲线以音频–类别对为单位：FN/阳性对，FP/预测阳性对；下方另列至少漏一类、至少多报一类的音频数及比例，一条音频不重复计数。标准 FPR = FP/阴性对另行展示。
- TAG/TACOS 只检查原有查询；没有阴性查询时，误报不可测，不把未查询的类别当成阴性。TACOS 保持当前近义 GT 合并。

CSV 用 mode=event/clip 区分两种统计。`--clip-only` 复用已保存的旧统计，只补算 Clip 曲线，不重新推理或重算旧事件匹配。正常运行同时更新两种方式。旧 NPZ 原样保留，主评分不变。

- 漏检率 = 官方匹配未检出的 GT 数 / GT 数。
- 误报占预测比例 = 官方 FP 数 / 所有预测连续事件区间数；不是阴性时间 FPR，也不直接用 1−官方 precision。
- FP/小时：固定类别任务除以音频小时；TAG/TACOS 除以文件–查询对评测小时。
- MAESTRO 单独使用类别×一秒段，包含标准 FP/阴性秒段 FPR；不和事件边界指标混称。

页面与 CSV 展示 201 个分数分位阈值及两端点，不用 GT 优选阈值，不提供自选默认操作点。全部原生官方匹配变化点保存在 NPZ；values 和 adjacent_minima 可精确还原任意阈值下连续预测区间数。

匹配参数沿用现有协议，FineLAP TAG 使用 raw；TACOS 使用当前固定 MiniLM 近义合并实现。只读取已有预测，未改变主评分或模型。Omni 不在这一批逐帧阈值统计内。

复算：`OMP_NUM_THREADS=4 ~/venvs/finelap/bin/python analysis/error_rates.py`。日志 `log/20261023/error_rates.log`。
''',encoding='utf-8')

if __name__=='__main__':main()
