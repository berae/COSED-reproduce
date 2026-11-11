"""Evaluate AS-S-selected EMA using the existing frozen six-benchmark protocols."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import torch
from torch.nn import functional as F
import torchaudio.functional as AF
import soundfile as sf
import librosa

from data import ROOT
from model import COSED
from train import save_json

FROZEN=Path('/work/zhanghc/Myllm/FineLAP/dependencies/benchmark')
SPLITS=['desed-public','realdesed-test','urbansed-test','tag-test','tacos-test','maestro-development','desed-validation']
sys.path.insert(0,'/work/zhanghc/Myllm/FineLAP/dependencies/scoring')
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--checkpoint', choices=['best','last'], default='best')
    ap.add_argument('--online', action='store_true', help='Evaluate the online weights of last.pt for diagnosis')
    ap.add_argument('--dataset', choices=SPLITS)
    ap.add_argument('--allow-interrupted', action='store_true')
    args=ap.parse_args();run=ROOT/'runs'/args.run
    if args.online and args.checkpoint!='last':
        ap.error('--online requires --checkpoint last')
    if not args.allow_interrupted and json.loads((run/'status.json').read_text())['stage']!='completed':
        ap.error('Incomplete training run; use --allow-interrupted for checkpoint diagnostics.')
    weight=run/(args.checkpoint+'.pt')
    sha=hashlib.sha256()
    with weight.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            sha.update(chunk)
    digest=sha.hexdigest()
    out=run/('benchmark_'+digest[:12]+('_online' if args.online else ''));manifests=out/'manifests';manifests.mkdir(parents=True,exist_ok=True)
    state=torch.load(weight,map_location='cpu',weights_only=True)
    if args.checkpoint=='last':
        key='model' if args.online else 'ema'
        print('Evaluating',key,'at epoch',state['epoch'],'step',state['step'],flush=True)
        state=state[key]
    model=COSED().cuda().eval();model.load_state_dict(state,strict=True)
    del state
    core=module('benchcore',FROZEN/'runners/core.py');extra=module('benchextra',FROZEN/'runners/grounding.py');scorer=module('benchscore',FROZEN/'scripts/score_benchmarks.py')
    class Adapter:
        sr=16000;median=9
        def __init__(self,name):self.cache={}
        def window(self,audio,labels):
            key=tuple(labels)
            if key not in self.cache:self.cache[key]=model.text(['sound of '+s.replace('_',' ').lower() for s in labels])
            x=torch.as_tensor(audio[:160000],device='cuda');x=F.pad(x,(0,160000-len(x)))
            fr,_=model.encode(model.audio.mel_forward(x[None]))
            return model.scores(fr,self.cache[key],False).sigmoid()[0].cpu().numpy()
    for m in (core,extra,scorer):m.ROOT=out
    core.Adapter=extra.Adapter=Adapter;extra.score=lambda *a,**k:None
    def local_audio(path,sr,mono):
        p=Path('/work/zhanghc/datasets/cosed-benchmarks')/Path(path).relative_to('/home/zhanghc/datasets/cosed-benchmarks')
        x,old=sf.read(p,dtype='float32',always_2d=True)
        return AF.resample(torch.from_numpy(x[:,0].copy()),old,sr).numpy(),sr
    librosa.load=local_audio
    unions=json.loads((FROZEN/'tacos_semantic_unions.json').read_text())
    scorer.semantic_unions=lambda _:{(k.rsplit('__',1)[0],int(k.rsplit('__',1)[1])):v for k,v in unions.items()}
    dest=out/'summary.json';results=json.loads(dest.read_text()) if dest.exists() else {}
    for name in ([args.dataset] if args.dataset else SPLITS):
        if name in results:continue
        raw=(FROZEN/'manifests'/(name+'.json')).read_bytes();(manifests/(name+'.json')).write_bytes(raw)
        manifest=json.loads(raw);is_extra=name in ('tacos-test','tag-test','maestro-development')
        (extra if is_extra else core).run(SimpleNamespace(dataset=name,model='cosed',limit=args.limit))
        if args.limit:continue
        if name=='tag-test':v=scorer.score_tag('cosed',manifest,'official')
        elif name=='tacos-test':v={p:scorer.score_tacos('cosed',manifest,p) for p in ('exact','semantic')}
        elif name=='maestro-development':v=scorer.score_maestro('cosed',manifest)
        else:v=json.loads((out/'runs-source-audit-v2/cosed'/name/'metrics.json').read_text())
        results[name]=v;save_json(dest,results)
if __name__=='__main__':main()

