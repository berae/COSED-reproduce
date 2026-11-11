"""Five-source labels, clip sampling and AS-S validation for the COSED reconstruction."""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/PretrainedSED'))
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes

CLASSES = list(as_strong_train_classes)
assert len(CLASSES) == len(set(CLASSES)) == 447
SOURCES = ['audioset_strong', 'finelap100k', 'audiocaps', 'tacos', 'clotho']
FACTORS = [1, 1, 2, 5, 4]
DATA = Path('/work/zhanghc/datasets')
AS = Path('/work/zhangyu/AED/audioset_tagging_cnn/datasets')
STRONG = DATA/'AudioSet/metadata/strong'
MIX = Path('/work/zhanghc/Myllm/FineLAP/data/cosed')

def validation():
    with (STRONG/'mid_to_display_name.tsv').open() as f:
        names = dict(csv.reader(f, delimiter='\t'))
    events = defaultdict(list)
    with (STRONG/'audioset_eval_strong.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            events[r['segment_id']].append((float(r['start_time_seconds']),float(r['end_time_seconds']),names[r['label']]))
    audio = {p.stem:p for p in (AS/'audios/eval_segments').rglob('*.wav')}
    with (AS/'metadata/eval_segments.csv').open() as f:
        starts = {('Y'+x[0],round(float(x[1])*1000)) for x in csv.reader((l for l in f if not l.startswith('#')),skipinitialspace=True)}
    items = {}
    for key, labels in events.items():
        vid, start = key.rsplit('_',1)
        vid = 'Y'+vid
        if vid in audio and (vid,int(start)) in starts:
            items[key]=(audio[vid],labels)
    return items, {'Y'+k.rsplit('_',1)[0] for k in events}

def ancestor_map():
    ontology=json.loads((ROOT/'vendor/ontology.json').read_text())
    by_name={x['name']:x['id'] for x in ontology}
    parents=defaultdict(set)
    for x in ontology:
        for child in x['child_ids']:parents[child].add(x['id'])
    if all(k in by_name for k in ('Doorbell','Alarm','Bell')):
        parents[by_name['Doorbell']].discard(by_name['Alarm'])
        parents[by_name['Doorbell']].add(by_name['Bell'])
    with (STRONG/'mid_to_display_name.tsv').open() as f: names=dict(csv.reader(f,delimiter='\t'))
    result={}
    for mid,name in names.items():
        seen=set();stack=[mid]
        while stack:
            v=stack.pop()
            if v not in seen:
                seen.add(v);stack.extend(parents[v])
        result[name]=[i for i,c in enumerate(CLASSES) if any(names.get(x)==c for x in seen)]
    return result

class Corpus:
    def __init__(self, heldout):
        self.rows=[];self.indices=[];self.starts=[];self.counts=[]
        with (DATA/'TACOS/downloads/metadata.csv').open() as f:
            speech={r['filename'] for r in csv.DictReader(f) if r['subclass'].lower()=='speech'}
        for source,(name,factor) in enumerate(zip(SOURCES,FACTORS)):
            start=len(self.rows);self.starts.append(start)
            with (MIX/(name+'.jsonl')).open() as f:
                for line in f:
                    r=json.loads(line);key=r['audio_id']
                    if source==0 and 'Y'+key.rsplit('_',1)[0] in heldout:continue
                    if source==2 and key in heldout:continue
                    if source==3 and key in speech:continue
                    self.rows.append((source,r))
            self.counts.append(len(self.rows)-start)
            self.indices.extend(list(range(start,len(self.rows)))*factor)
        self.ancestors=ancestor_map()

    def get(self,index,seed):
        rng=random.Random(seed);source,row=self.rows[index]
        path=row['audio_path'];info=sf.info(path);sr=info.samplerate
        start=rng.randrange(max(0,info.frames-10*sr)+1)
        duration=min(10,(info.frames-start)/sr)
        phrases=defaultdict(list)
        for p in row.get('phrases',[]):
            for a,b in p['segments']:
                a=max(0,a-start/sr);b=min(duration,b-start/sr)
                if b>a:phrases[p['phrase']].append((a,b))
        cap=row.get('caption','')
        if isinstance(cap,list):cap=rng.choice(cap)
        return dict(src=source,path=path,start=start,sr=sr,duration=duration,phrases=phrases,cap=cap)

def wave(row):
    x,sr=sf.read(row['path'],start=row['start'],frames=row['sr']*10,always_2d=True,dtype='float32')
    x=torch.from_numpy(x[:,0].copy())
    if sr!=16000:x=AF.resample(x,sr,16000)
    return torch.nn.functional.pad(x[:160000],(0,max(0,160000-len(x))))

def query_bank(rows):
    queries={0:CLASSES}
    for src in range(1,5):
        texts=set()
        for r in rows:
            if r['src']==src:texts.update(r['phrases'] if src in (1,3) else [r['cap']])
        queries[src]=sorted(texts)
    return queries

def targets(row,queries,ancestors):
    src=row['src'];labels=queries[src];n=len(labels)
    if src in (2,4):
        y=torch.zeros(n);y[labels.index(row['cap'])]=1;return y
    y=torch.zeros(250,n)
    for name,segments in row['phrases'].items():
        indices=ancestors.get(name,[]) if src==0 else [labels.index(name)]
        for a,b in segments:
            y[max(0,int(a*25)):min(250,int(np.ceil(b*25))),indices]=1
    return y

