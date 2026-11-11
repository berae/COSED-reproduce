"""Released-model benchmark runner. Manifests and predictions are resumable.

Run with one model's own environment; do not import multiple model repositories
in one process because several use the same top-level Python module names.
"""
import os, sys, json, time, argparse, hashlib, traceback, math
from pathlib import Path
H = Path.home()
ROOT = H / 'projects/ov-sed/benchmark'
DATA = H / 'datasets/cosed-benchmarks'
MODELS = H / 'models/ov-sed'
REPOS = H / 'projects/ov-sed'

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)

def prepare():
    import pandas as pd
    import soundfile as sf
    from collections import defaultdict
    def save(name, audio, rows, expected=None):
        groups = defaultdict(list)
        for filename, onset, offset, label in rows:
            groups[filename]
            if isinstance(label, str):
                groups[filename].append([float(onset), float(offset), label])
        labels = sorted({r[2] for events in groups.values() for r in events})
        items, missing, repairs = [], [], []
        for filename, events in sorted(groups.items()):
            p = audio / filename
            if not p.exists():
                missing.append(filename); continue
            duration = sf.info(p).duration
            clean = []
            for start, end, label in events:
                if start < 0 or end > duration:
                    if start < -0.1 or start >= duration or end > duration + .5:
                        raise ValueError((name, filename, start, end, duration))
                    repairs.append([filename, start, end, duration])
                    start, end = max(0,start), min(end,duration)
                if end <= start:
                    raise ValueError(('nonpositive interval',filename,start,end))
                clean.append([start,end,label])
            items.append(dict(id=Path(filename).stem,path=str(p),duration=duration,events=clean))
        out=dict(dataset=name,classes=labels,items=items,expected=expected or len(groups),missing=missing,boundary_clips=repairs)
        write_json(ROOT/'manifests'/f'{name}.json',out)
        print(name,len(items),'missing',len(missing),'classes',len(labels),flush=True)
    p=DATA/'desed/data/4560759'
    df=pd.read_csv(p/'metadata/eval/public.tsv',sep='\t')
    save('desed-public',p/'audio/eval/public',df[['filename','onset','offset','event_label']].values.tolist())
    p=DATA/'desed/data/6444477/metadata/validation/validation.tsv'
    df=pd.read_csv(p,sep='\t')
    save('desed-validation',DATA/'desed/local-audioset/validation',df[['filename','onset','offset','event_label']].values.tolist())
    p=DATA/'realdesed/data/20056072/test'
    df=pd.read_csv(p/'annotations.csv')
    save('realdesed-test',p/'audio',df[['filename','onset','offset','class']].values.tolist())
    p=DATA/'urbansed/data/1324404/URBAN-SED_v2.0.0'
    rows=[]
    for f in sorted((p/'annotations/test').glob('*.txt')):
        if f.name.startswith('._'):continue
        for line in f.read_text().splitlines():
            onset,offset,label=line.split('\t')
            rows.append([f.stem+'.wav',onset,offset,label])
    save('urbansed-test',p/'audio/test',rows)

class Adapter:
    def __init__(self,name):
        import torch, yaml
        self.name=name; self.torch=torch; self.cache={}
        torch.set_num_threads(4)
        torch.manual_seed(0)
        repo={'dasm':'DASM','mga-clap':'MGA-CLAP','finelap':'FineLAP','flexsed':'FlexSED','openflam':'OpenFLAM'}[name]
        os.chdir(REPOS/repo);sys.path.insert(0,str(REPOS/repo))
        self.sr={'dasm':32000,'mga-clap':32000,'finelap':16000,'flexsed':16000,'openflam':48000}[name]
        self.median={'dasm':16,'mga-clap':3,'finelap':3,'flexsed':1,'openflam':3}[name]
        if name in ['dasm','mga-clap']:
            sys.path.insert(0,str(REPOS/'MGA-CLAP'))
            from models.ase_model import ASE
            cfg=yaml.safe_load((REPOS/'MGA-CLAP/settings/inference_sed.yaml').read_text())
            self.clap=ASE(cfg).cuda().eval()
            self.clap.load_state_dict(torch.load(MODELS/'MGA-CLAP/model.pt',map_location='cpu')['model'],strict=True)
            if name=='dasm':
                sys.path.insert(0,str(REPOS/'DASM'))
                from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
                cfg=yaml.safe_load((MODELS/'DASM/text_query/config.yaml').read_text())
                self.model=DASM_HTSAT(**cfg['DASM_HTSAT']['init_kwargs'])
                self.model.load_state_dict(torch.load(MODELS/'DASM/text_query/as_full_text_query_best_model.pt',map_location='cpu'),strict=True)
                self.model=self.model.cuda().eval();self.extractor=self.model.get_feature_extractor()
                self.base=self.model.at_query
                if isinstance(self.base,torch.nn.ParameterList):self.base=self.base[0]
        elif name=='finelap':
            from models.finelap import FineLAP
            from models.feature_extractor import EAT_preprocess
            self.preprocess=EAT_preprocess
            cfg=yaml.safe_load(Path('config/finelap_eat_config.yaml').read_text())
            self.model=FineLAP(cfg)
            self.model.load_state_dict(torch.load(MODELS/'FineLAP_Pytorch/finelap_fixed.pt',map_location='cpu',weights_only=False)['model'],strict=True)
            self.model=self.model.cuda().eval()
        elif name=='flexsed':
            from api import FlexSED
            self.wrapper=FlexSED(ckpt_path=str(MODELS/'FlexSED/ckpts/flexsed_as.pt'))
            self.model=self.wrapper.model
        elif name=='openflam':
            import openflam
            self.model=openflam.OpenFLAM(default_ckpt_path=str(MODELS/'hf-cache')).cuda().eval()

    def queries(self, labels):
        torch=self.torch
        key=tuple(labels)
        if key in self.cache:return self.cache[key]
        texts=[x.replace('_',' ').lower() for x in labels]
        if self.name in ['dasm','mga-clap']:
            if self.name=='dasm':texts=['sound of '+x for x in texts]
            _,words,mask=self.clap.encode_text(texts)
            q=torch.nn.functional.normalize(self.clap.msc(words,self.clap.codebook,mask),dim=-1)
            if self.name=='dasm':
                q=torch.cat([self.base,q]); n=len(self.base)
                mask=torch.ones(len(q),len(q),dtype=torch.bool,device='cuda')
                mask[:,:n]=False;mask.fill_diagonal_(False)
                q=(q,mask)
        elif self.name=='finelap':q=self.model.get_global_text_embeds(texts)
        elif self.name=='flexsed':
            texts=['The sound of '+x.capitalize() for x in texts]
            tok=self.wrapper.tokenizer(texts,padding=True,return_tensors='pt')
            q=self.wrapper.clap(**tok).text_embeds.unsqueeze(0).cuda()
        else:q=texts
        self.cache[key]=q;return q

    def window(self,wav,labels):
        import numpy as np
        torch=self.torch; q=self.queries(labels); valid=len(wav)
        if self.name=='mga-clap' and valid<self.sr*10:
            wav=np.tile(wav,math.ceil(self.sr*10/valid))[:self.sr*10]
        else:wav=np.pad(wav,(0,max(0,self.sr*10-valid)))[:self.sr*10]
        x=torch.from_numpy(wav.copy()).float().unsqueeze(0).cuda()
        if self.name=='dasm':
            pad=(torch.arange(500,device='cuda')>=math.ceil(valid/self.sr*50)).unsqueeze(0)
            s,_,_=self.model(input=self.extractor(x),temp_w=.5,pad_mask=pad,query=q[0],query_type='text',tgt_mask=q[1])
            s=s[0,len(self.base):].T
        elif self.name=='mga-clap':
            _,frames=self.clap.encode_audio(x)
            s=(torch.nn.functional.normalize(frames,dim=-1)@q.T)[0]
        elif self.name=='finelap':
            feat=self.preprocess(x[0].cpu()).cuda()
            frames=self.model.get_dense_audio_embeds(feat)[0]
            s=torch.sigmoid((frames@q.T)/self.model.temp_local+self.model.b_local)
        elif self.name=='flexsed':
            x=x/(x.abs().max()+1e-9)
            s=self.model(self.model.forward_to_spec(x),q).sigmoid().squeeze(1).T
        else:s=self.model.get_local_similarity(x,q,method='unbiased',cross_product=True)[0].T
        arr=s.detach().cpu().numpy()
        assert arr.ndim==2 and arr.shape[1]==len(labels) and np.isfinite(arr).all(),arr.shape
        return arr

def run(args):
    import numpy as np, librosa, torch
    from scipy.ndimage import median_filter
    manifest=json.loads((ROOT/'manifests'/f'{args.dataset}.json').read_text())
    items=manifest['items'][:args.limit] if args.limit else manifest['items']
    out=ROOT/'runs'/args.model/args.dataset
    if args.limit:out=out/f'smoke-{args.limit}'
    out.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256((Path(__file__).read_text()+json.dumps(manifest,sort_keys=True)).encode()).hexdigest()
    identity_path=out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['sha256']!=identity:
        raise RuntimeError('Runner or manifest changed; use a fresh output directory to avoid mixing predictions')
    write_json(identity_path,dict(sha256=identity))
    start=time.time();a=Adapter(args.model);loaded=time.time()
    status=dict(model=args.model,dataset=args.dataset,total=len(items),completed=0,status='running',pid=os.getpid(),started=start,load_seconds=loaded-start)
    write_json(out/'status.json',status)
    with torch.inference_mode():
        for i,item in enumerate(items):
            target=out/(item['id']+'.npz')
            if not target.exists():
                audio,_=librosa.load(item['path'],sr=a.sr,mono=True)
                labels=item.get('queries',manifest['classes'])
                duration=len(audio)/a.sr
                starts=list(range(0,max(1,len(audio)-10*a.sr+1),5*a.sr))
                if starts[-1]+10*a.sr<len(audio):starts.append(starts[-1]+5*a.sr)
                predictions=[]
                t=time.time()
                for offset in starts:
                    arr=a.window(audio[offset:offset+10*a.sr],labels)
                    predictions.append((offset/a.sr,arr))
                # Native frame spacing for each model; average overlapping windows.
                dt=10/predictions[0][1].shape[0]
                n=math.ceil(duration/dt);centers=(np.arange(n)+.5)*dt
                merged=np.zeros((n,len(labels)),np.float32);counts=np.zeros(n,np.float32)
                for offset,arr in predictions:
                    valid=(centers>=offset)&(centers<min(duration,offset+10))
                    ix=np.minimum(((centers[valid]-offset)/dt).astype(int),len(arr)-1)
                    merged[valid]+=arr[ix];counts[valid]+=1
                # The last partial frame can have its midpoint beyond the clip end.
                if counts[-1]==0:
                    offset,arr=predictions[-1];ix=min(int((duration-1e-8-offset)/dt),len(arr)-1)
                    merged[-1]=arr[ix];counts[-1]=1
                assert (counts>0).all()
                raw=merged/counts[:,None]
                scores=median_filter(raw,size=(a.median,1))
                times=np.minimum(np.arange(n+1)*dt,duration)
                tmp=target.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,scores=scores,raw=raw,timestamps=times,labels=np.array(labels),inference_seconds=time.time()-t)
                tmp.replace(target)
            status.update(completed=i+1,updated=time.time(),elapsed_seconds=time.time()-start)
            write_json(out/'status.json',status)
            if (i+1)%25==0 or i==0:print(json.dumps(status),flush=True)
    status.update(status='inference_complete');write_json(out/'status.json',status)
    if not args.limit: score(args,out,manifest,items)

def score(args,out=None,manifest=None,items=None):
    import numpy as np
    import sed_scores_eval
    from sed_scores_eval.base_modules.scores import create_score_dataframe
    manifest=manifest or json.loads((ROOT/'manifests'/f'{args.dataset}.json').read_text())
    items=items or manifest['items']
    out=out or ROOT/'runs'/args.model/args.dataset
    scores={};gt={};durations={}
    for item in items:
        p=np.load(out/(item['id']+'.npz'))
        scores[item['id']]=create_score_dataframe(p['scores'],p['timestamps'],list(p['labels']))
        gt[item['id']]=[tuple(x) for x in item['events']]
        durations[item['id']]=float(p['timestamps'][-1])
    start=time.time()
    result=sed_scores_eval.intersection_based.psds(scores,gt,durations,dtc_threshold=.7,gtc_threshold=.7,cttc_threshold=None,alpha_ct=0,alpha_st=1,max_efpr=100,num_jobs=4)
    value=float(result[0])
    write_json(out/'metrics.json',dict(psds1=value,files=len(items),expected=manifest['expected'],missing=len(manifest['missing']),metric_seconds=time.time()-start,protocol=dict(dtc=.7,gtc=.7,alpha_st=1,max_efpr=100),comparability='candidate reproduction; exact COSED split/checkpoint equivalence not yet verified'))
    status=json.loads((out/'status.json').read_text());status.update(status='complete',psds1=value);write_json(out/'status.json',status)
    print('PSDS1',value,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','score']);p.add_argument('--model');p.add_argument('--dataset');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args)
    else:score(args)
