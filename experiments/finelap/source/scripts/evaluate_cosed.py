"""Evaluate epoch 10 with frozen existing benchmark runners and separate outputs."""
import os, sys, json, time, hashlib, importlib.util, urllib.request, shutil, fcntl, argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--experiment',default='finelap_cosed_mix_20261029')
parser.add_argument('--gpu',default='0')
cli=parser.parse_args()
assert Path(cli.experiment).name==cli.experiment
os.environ['CUDA_VISIBLE_DEVICES']=cli.gpu
EXP=ROOT/'exps'/cli.experiment
OUT=EXP/'evaluation'
SRC=ROOT/'dependencies/benchmark'
DATASETS=['desed-public','realdesed-test','urbansed-test','tag-test','tacos-test','maestro-development','desed-validation']

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'dependencies/fairseq'));sys.path.insert(0,str(ROOT/'dependencies/scoring'))
    import torch,torchaudio,numpy as np
    torch.set_num_threads(4)
    checkpoint=EXP/'models/epoch_010.pt'
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    identity=dict(checkpoint=str(checkpoint),sha256=h.hexdigest(),epoch=10,
                  runner_hashes={f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in ['runners/core.py','runners/grounding.py','scripts/score_benchmarks.py']})
    p=OUT/'provenance.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:save(p,identity)
    (OUT/'manifests').mkdir(exist_ok=True)
    remote_paths=set()
    local_paths={}
    for ds in DATASETS:
        shutil.copyfile(SRC/'manifests'/f'{ds}.json',OUT/'manifests'/f'{ds}.json')
        remote_paths.update(x['path'] for x in json.loads((SRC/'manifests'/f'{ds}.json').read_text())['items'])
    for path in remote_paths:
        prefix='/home/zhanghc/datasets/cosed-benchmarks/'
        if path.startswith(prefix):
            candidate=Path('/work/zhanghc/datasets/cosed-benchmarks')/path[len(prefix):]
            if candidate.is_file():local_paths[path]=candidate
    if cli.experiment!='finelap_cosed_mix_20261029':
        assert len(local_paths)==len(remote_paths), f'Missing local benchmark files: {len(remote_paths)-len(local_paths)}'
    cache=Path('/work/zhanghc/datasets/cosed-benchmarks/audio');cache.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    original_load=torchaudio.load
    def load(path,*args,**kwargs):
        path=str(path)
        if path in local_paths:
            path=str(local_paths[path])
        elif path in remote_paths:
            local=Path('/work/zhanghc/datasets/TACOS/audio')/Path(path).name
            if '/tacos/' not in path or not local.exists():
                key=hashlib.sha256(path.encode()).hexdigest();local=cache/(key+Path(path).suffix)
                if not local.exists():
                    for attempt in range(4):
                        try:
                            with opener.open('http://10.160.4.167:40918/'+key,timeout=120) as response:
                                expected=int(response.headers['Content-Length'])
                                tmp=local.with_suffix('.part')
                                with tmp.open('wb') as dst:shutil.copyfileobj(response,dst,1024*1024)
                            assert tmp.stat().st_size==expected
                            tmp.replace(local);break
                        except Exception:
                            if attempt==3:raise
                            time.sleep(10)
            path=str(local)
        kwargs['backend']='soundfile'
        return original_load(path,*args,**kwargs)
    torchaudio.load=load
    core=module('benchmark_core',SRC/'runners/core.py')
    extra=module('benchmark_grounding',SRC/'runners/grounding.py')
    scorer=module('benchmark_scoring',SRC/'scripts/score_benchmarks.py')
    for m in [core,extra,scorer]:m.ROOT=OUT
    class Native(core.Adapter):
        def __init__(self,name):
            from models.finelap import FineLAP
            from models.feature_extractor import EAT_preprocess
            self.name=name;self.torch=torch;self.cache={};self.sr=16000;self.median=3;self.preprocess=EAT_preprocess
            torch.manual_seed(0)
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert ck['epoch']==10
            cfg=ck['config'];cfg['model_args']['ckpt_path']=''
            self.model=FineLAP(cfg)
            self.model.load_state_dict(ck['model'],strict=True)
            self.model=self.model.cuda().eval()
    core.Adapter=Native;extra.Adapter=Native
    # TAG's authoritative scorer below uses raw frames; avoid the legacy filtered TAG scorer.
    extra.score=lambda *a,**k:None
    unions=json.loads((SRC/'tacos_semantic_unions.json').read_text())
    scorer.semantic_unions=lambda _: {(k.rsplit('__',1)[0],int(k.rsplit('__',1)[1])):v for k,v in unions.items()}
    report=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
    for ds in DATASETS:
        if ds in report:continue
        save(OUT/'status.json',dict(stage='evaluating',dataset=ds,pid=os.getpid(),updated=time.time(),completed=list(report)))
        args=SimpleNamespace(model='finelap',dataset=ds,limit=0)
        is_extra=ds in ['tag-test','tacos-test','maestro-development']
        (extra if is_extra else core).run(args)
        folder=OUT/('runs-extra' if is_extra else 'runs-source-audit-v2')/'finelap'/ds
        manifest=json.loads((OUT/'manifests'/f'{ds}.json').read_text())
        if ds=='tag-test':
            value={p:scorer.score_tag('finelap',manifest,p) for p in ['official','legacy']}
            for p,v in value.items():save(folder/f'metrics.{p}.json',v)
        elif ds=='tacos-test':
            value={p:scorer.score_tacos('finelap',manifest,p) for p in ['exact','semantic']}
            for p,v in value.items():save(folder/f'metrics.{p}.json',v)
        elif ds=='maestro-development':
            value=scorer.score_maestro('finelap',manifest);save(folder/'metrics.json',value)
        else:value=json.loads((folder/'metrics.json').read_text())
        report[ds]=value;save(OUT/'summary.json',report)
        print('RESULT',ds,json.dumps(value),flush=True)
        torch.cuda.empty_cache()
    save(OUT/'status.json',dict(stage='completed',updated=time.time(),completed=list(report)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'status.json',dict(stage='failed',error=str(e),updated=time.time()));raise
