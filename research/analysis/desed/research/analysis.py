"""Exploratory DESED validation diagnostics; never changes benchmark scoring."""
from pathlib import Path
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--dataset",choices=["desed-public","desed-validation"],default="desed-public")
DATASET=parser.parse_args().dataset
import hashlib
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.home() / 'projects/ov-sed/benchmark'
OUT = ROOT / 'analysis/desed/research'
if DATASET=='desed-public': OUT=OUT/'public'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'cases').mkdir(exist_ok=True)
MODELS = ['dasm', 'finelap', 'flexsed', 'mga-clap', 'openflam']
manifest_path = ROOT / ('manifests/'+DATASET+'.json')
manifest = json.loads(manifest_path.read_text())
items = manifest['items']
classes = manifest['classes']
official = json.loads((ROOT / 'analysis/desed/official_psds.json').read_text())['datasets'][DATASET]['models']


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        a, b = max(0., a), min(10., b)
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def coverage(intervals, starts, ends):
    return sum((np.maximum(0., np.minimum(ends, b) - np.maximum(starts, a))
                for a, b in intervals), np.zeros_like(starts))


events = [{c: union([(a, b) for a, b, label in item['events'] if label == c])
           for c in classes} for item in items]
presence = np.array([[bool(e[c]) for c in classes] for e in events])
profile = {}
for j, c in enumerate(classes):
    lengths, overlap, target_time = [], 0., 0.
    for e in events:
        others = union([ab for k, intervals in e.items() if k != c for ab in intervals])
        for a, b in e[c]:
            lengths.append(b-a)
            target_time += b-a
            overlap += sum(max(0., min(b, y)-max(a, x)) for x, y in others)
    profile[c] = dict(positive_clips=int(presence[:, j].sum()), events=len(lengths),
        median_duration=float(np.median(lengths)), duration_q25=float(np.quantile(lengths, .25)),
        duration_q75=float(np.quantile(lengths, .75)), fraction_under_1s=float(np.mean(np.array(lengths)<1)),
        annotated_overlap_fraction=overlap/target_time,
        positive_duration_seconds=target_time)

clip_scores = {}
local_aucs = {}
for model in MODELS:
    maxima = np.zeros((len(items), len(classes)))
    aucs = np.full_like(maxima, np.nan)
    for i, item in enumerate(items):
        with np.load(ROOT / 'runs-source-audit-v2' / model / DATASET / (item['id']+'.npz')) as p:
            labels = list(p['labels'])
            values = p['scores'][:, [labels.index(c) for c in classes]]
            t = np.asarray(p['timestamps'])
            maxima[i] = values.max(axis=0)
            # Common 100 ms grid for an auxiliary within-clip separation diagnostic.
            end = min(10., float(t[-1]))
            grid = np.r_[np.arange(0., end, .1), end]
            widths = grid[1:] - grid[:-1]
            weights = np.maximum(0., np.minimum(grid[1:,None],t[None,1:])-np.maximum(grid[:-1,None],t[None,:-1]))
            scores = weights @ values / weights.sum(axis=1,keepdims=True)
        for j,c in enumerate(classes):
            truth = coverage(events[i][c],grid[:-1],grid[1:]) / widths >= .5
            # Require at least 0.5 s of positive and 0.5 s of negative time.
            if widths[truth].sum() >= .5 and widths[~truth].sum() >= .5:
                aucs[i,j] = roc_auc_score(truth,scores[:,j],sample_weight=widths)
    clip_scores[model] = maxima
    local_aucs[model] = aucs
    print('Loaded',model,flush=True)

diagnostics = {}
for model in MODELS:
    diagnostics[model] = {}
    for j,c in enumerate(classes):
        valid = local_aucs[model][:,j]
        valid = valid[np.isfinite(valid)]
        diagnostics[model][c] = dict(
            single_class_psds=float(official[model]['psds1_per_class'][c]),
            clip_ap=float(average_precision_score(presence[:,j],clip_scores[model][:,j])),
            positive_clip_prevalence=float(presence[:,j].mean()),
            within_clip_auc_median=float(np.median(valid)) if len(valid) else None,
            within_clip_auc_eligible=len(valid))

result = dict(dataset=DATASET,files=len(items),date='2026-10-19',
    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    profile=profile,diagnostics=diagnostics,
    caveats=['Exploratory diagnostics only; standard PSDS1 unchanged.',
        'Clip AP uses maximum saved frame score; no event boundary evaluation.',
        'Within-clip AUC uses a 100 ms grid with GT coverage >= 50%, excludes clips with <0.5 s positive or negative time.',
        'Within-clip AUC is not PSDS or an event success rate. Cannot subtract AP/AUC from PSDS to attribute errors.',
        'Annotation overlap is not acoustic SNR. Training data exposure and label quality remain uncontrolled.'])
(OUT/'diagnostics.json').write_text(json.dumps(result,indent=2))

# One common-scale figure of existing standard single-class scores.
fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
matrix=np.array([[diagnostics[m][c]['single_class_psds'] for m in MODELS] for c in classes])
im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(5),MODELS);ax.set_yticks(range(len(classes)),classes)
for i in range(len(classes)):
    for j in range(5):
        ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.65 else '#142534')
ax.set_title(f'{DATASET} (n={len(items)}): standard single-class PSD areas')
fig.colorbar(im,ax=ax,label='Single-class PSD area (not overall PSDS1)')
fig.savefig(OUT/'class_scores.png',dpi=160);plt.close(fig)

# Matched recordings across all five models; deterministic extremes for browsing only.
cases=[]
for c in (['Speech','Dishes'] if DATASET=='desed-public' else ['Speech','Dishes','Vacuum_cleaner','Electric_shaver_toothbrush']):
    j=classes.index(c)
    stack=np.stack([local_aucs[m][:,j] for m in MODELS])
    eligible=np.where(np.isfinite(stack).all(axis=0))[0]
    ranked=sorted(eligible,key=lambda i:(float(np.median(stack[:,i])),items[i]['id']))
    if not ranked:
        continue
    selected=[('low',ranked[0]),('high',ranked[-1])]
    if DATASET=='desed-public':
        gap=max(eligible,key=lambda i:(float(stack[:,i].max()-stack[:,i].min()),items[i]['id']))
        if gap not in [x[1] for x in selected]:selected.append(('disagreement',gap))
    for kind,i in selected:
        item=items[i];case_id=f'{c}_{kind}'
        # Audio/Mel preprocessing is solely for inspection, never for inference.
        audio,sr=sf.read(item['path'],always_2d=True)
        audio=audio[:round(10*sr)].mean(axis=1)
        audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        sf.write(OUT/'cases'/(case_id+'.wav'),audio,16000,subtype='PCM_16')
        mel=librosa.feature.melspectrogram(y=audio,sr=16000,n_fft=512,hop_length=160,n_mels=80,fmin=0,fmax=8000)
        db=librosa.power_to_db(mel,ref=np.max,top_db=80)
        fig,axes=plt.subplots(8,1,figsize=(12,14),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,2,.6,1,1,1,1,1]})
        axes[0].plot(np.arange(len(audio))/16000,audio,lw=.5);axes[0].set_ylabel('Waveform')
        librosa.display.specshow(db,sr=16000,hop_length=160,x_axis='time',y_axis='mel',ax=axes[1],vmin=-80,vmax=0)
        axes[1].set_ylabel('Mel (Hz)')
        for a,b in events[i][c]:axes[2].axvspan(a,b,color='#16856d',alpha=.7)
        axes[2].set_yticks([]);axes[2].set_ylabel('GT')
        for k,m in enumerate(MODELS):
            with np.load(ROOT/'runs-source-audit-v2'/m/DATASET/(item['id']+'.npz')) as p:
                v=p['scores'][:,list(p['labels']).index(c)];t=p['timestamps']
            axes[k+3].stairs(v,t,color='#285b85',lw=.9)
            for a,b in events[i][c]:axes[k+3].axvspan(a,b,color='#16856d',alpha=.12)
            axes[k+3].set_ylabel(m);axes[k+3].grid(alpha=.2)
        axes[-1].set_xlim(0,10);axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{c} | {kind} within-clip separation | {item["id"]}\nScores have independent y-scales; Mel relative to this clip peak. Not a confirmed TP/FP case.')
        fig.savefig(OUT/'cases'/(case_id+'.png'),dpi=120);plt.close(fig)
        cases.append(dict(id=case_id,clip_id=item['id'],class_name=c,selection=kind,
            median_within_clip_auc=float(np.median(stack[:,i])),
            by_model={m:float(local_aucs[m][i,j]) for m in MODELS},
            intervals=events[i][c],overlapping_labels=[k for k,v in events[i].items() if v],human_review='pending'))
(OUT/'cases/index.json').write_text(json.dumps(cases,indent=2))
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DESED 科研诊断案例</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#23384b}img{width:100%}article{border-top:1px solid #ccd6de;margin:30px 0}audio{width:100%}.note{background:#fff3d9;padding:15px}</style><h1>DESED Validation：时间分离度对照案例</h1><p style="background:#fff3d9;padding:15px"><b>划分说明：</b>本页所有案例来自 DESED Validation，仅用于辅助诊断；不是 Public Test 案例。DESED 主成绩只使用 692 条公开测试音频。</p><p class="note">这 8 条是按五个模型的片内 AUC 中位数选出的高低两端，仅用于探索；不是随机样本，也不是已确认的成功/失败事件。没有实际听辨确认。主指标仍是标准 PSDS1。</p><p>音频用于试听，已单声道化并重采样到 16 kHz；Mel 图逐音频峰值归一化，不能比较绝对响度。模型曲线保持原输出，各自独立纵轴，不能比较跨模型分数大小。</p>']
case_notes = {}

if DATASET=='desed-public':
    case_notes={}
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DESED Public Test 样本分析</title><style>body{font:16px/1.6 Georgia,serif;max-width:1050px;margin:30px auto;padding:20px;color:#111}img,audio{width:100%}article{border-top:1px solid #111;margin-top:35px}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid}th,td{text-align:center;padding:8px}th{border-bottom:1px solid}</style><h1>DESED Public Test：Speech / Dishes</h1><p>692 条公开测试音频。案例按片内时间 AUC 的共同高分、共同低分与模型差异挑选，供探索，不代表总体错误比例。音频尚待人工听辨；不同模型纵轴独立。GT 为绿色区间。</p>']
parts.append('<p><b>修订：</b>已撤下无验证集依据的固定阈值及预测区间；撤回依赖这些区间的边界、误报、漏报横向判断。保留响应曲线、真值和片内 AUC。正式操作点需在验证集按同一规则选定，再固定用于测试集。</p>')
for c in cases:
    parts += [f'<article id="{c["id"]}"><h2>{html.escape(c["class_name"])} · {c["selection"]}</h2>{case_notes.get(c["id"], "")}<p>{html.escape(c["clip_id"])} · 片内 AUC 中位数 {c["median_within_clip_auc"]:.3f} · 人工复核待完成</p><audio controls preload="none" src="cases/{c["id"]}.wav"></audio><img loading="lazy" alt="Waveform, log Mel, GT and five model responses" src="cases/{c["id"]}.png"></article>']
parts.append('</html>');(OUT/'cases.html').write_text(''.join(parts),encoding='utf-8')
print(json.dumps(result['profile'],indent=2),flush=True)
print('Completed',len(cases),'cases',flush=True)
