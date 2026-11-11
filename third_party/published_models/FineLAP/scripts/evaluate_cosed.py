"""Evaluate epoch 10 with frozen existing benchmark runners and separate outputs."""
import os, sys, json, time, hashlib, importlib.util, shutil, fcntl, argparse, subprocess, tarfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'exps/finelap_cosed_mix_20261029'
OUT=EXP/'evaluation'
SRC=ROOT/'dependencies/benchmark'
DATASETS=['realdesed-test','urbansed-test','tag-test','maestro-development','tacos-test','desed-public','desed-validation']

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bootstrap(options):
    """Wait for the detached copy, verify it, check inference parity, then resume."""
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'bootstrap.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    downloads=EXP/'inference_export'
    def state(stage,**kw):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    start=time.time()
    while options.transfer_pid and Path(f'/proc/{options.transfer_pid}/cmdline').exists():
        try:command=Path(f'/proc/{options.transfer_pid}/cmdline').read_bytes()
        except FileNotFoundError:break
        if b'rsync' not in command or b'40920' not in command:break
        sizes=[p.stat().st_size for p in downloads.glob('*epoch_010*') if p.is_file()]
        state('copying_checkpoint',checkpoint_bytes=max(sizes,default=0),gpu=options.gpu)
        if time.time()-start>7200:raise TimeoutError('Initial checkpoint transfer exceeded two hours')
        time.sleep(15)
    for attempt in range(3):
        rc=subprocess.call(['rsync','-rt','--partial-dir=.rsync-partial','--timeout=180','--contimeout=20','--stats','rsync://10.160.4.161:40920/export/',str(downloads)+'/'])
        if rc==0:break
        time.sleep(15)
    else:raise RuntimeError('Checkpoint/results transfer failed')
    state('verifying_transfer')
    meta=json.loads((downloads/'export.json').read_text())
    for name,expected in [('epoch_010_inference.pt',meta['inference_sha256']),('runtime_bundle.tar.gz',meta['bundle_sha256'])]:
        h=hashlib.sha256()
        with (downloads/name).open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        assert h.hexdigest()==expected,(name,'checksum mismatch')
    results=json.loads((downloads/'handoff_ready.json').read_text())
    assert hashlib.sha256((downloads/'completed_results.tar.gz').read_bytes()).hexdigest()==results['sha256']
    with tarfile.open(downloads/'completed_results.tar.gz') as tar:
        for member in tar.getmembers():
            destination=(OUT/member.name).resolve()
            assert destination.is_relative_to(OUT) and not (member.issym() or member.islnk())
            if member.name=='summary.json' and destination.exists():
                incoming=json.load(tar.extractfile(member));existing=json.loads(destination.read_text())
                incoming.update(existing);save(destination,incoming)
            elif not destination.exists():tar.extract(member,OUT)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(options.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    for _ in range(720):
        free=int(subprocess.check_output(['nvidia-smi','--id='+str(options.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=8192:break
        state('waiting_gpu_memory',gpu=options.gpu,free_MiB=free);time.sleep(30)
    else:raise RuntimeError('GPU memory unavailable')
    state('checking_inference_parity',gpu=options.gpu)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe'],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--datasets']+options.datasets,cwd=ROOT,env=env,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true',help='Compare three saved rabbit02 DESED predictions before handoff.')
    parser.add_argument('--bootstrap',action='store_true',help='Finish transfer and validate before starting the queue.')
    parser.add_argument('--gpu',type=int,default=5)
    parser.add_argument('--transfer-pid',type=int,default=0)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    options=parser.parse_args()
    if options.bootstrap:return bootstrap(options)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'dependencies/fairseq'));sys.path.insert(0,str(ROOT/'dependencies/runtime'))
    import torch,torchaudio,numpy as np
    torch.set_num_threads(4)
    checkpoint=EXP/'inference_export/epoch_010_inference.pt'
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    export=json.loads((EXP/'inference_export/export.json').read_text())
    assert h.hexdigest()==export['inference_sha256'], 'Transferred checkpoint checksum mismatch'
    identity=dict(checkpoint=str(checkpoint),sha256=h.hexdigest(),epoch=10,source_checkpoint_sha256=export['source_sha256'],
                  runner_hashes={f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in ['runners/core.py','runners/grounding.py','scripts/score_benchmarks.py']})
    p=OUT/'provenance.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:save(p,identity)
    (OUT/'manifests').mkdir(exist_ok=True)
    remote_paths=set()
    for ds in DATASETS:
        shutil.copyfile(SRC/'manifests'/f'{ds}.json',OUT/'manifests'/f'{ds}.json')
        remote_paths.update(x['path'] for x in json.loads((SRC/'manifests'/f'{ds}.json').read_text())['items'])
    original_load=torchaudio.load
    def load(path,*args,**kwargs):
        path=str(path)
        if not Path(path).is_file():raise FileNotFoundError(path)
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
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
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
    if options.probe:
        torch.cuda.reset_peak_memory_stats()
        core.run(SimpleNamespace(model='finelap',dataset='desed-public',limit=3))
        rows=[]
        for item in json.loads((SRC/'manifests/desed-public.json').read_text())['items'][:3]:
            current=np.load(OUT/'runs-source-audit-v2/finelap/desed-public/smoke-3'/(item['id']+'.npz'))
            reference=np.load(OUT/'reference'/(item['id']+'.npz'))
            assert np.array_equal(current['labels'],reference['labels'])
            assert np.array_equal(current['timestamps'],reference['timestamps'])
            for field in ['raw','scores']:
                error=float(np.abs(current[field]-reference[field]).max())
                rows.append(dict(id=item['id'],field=field,max_abs_error=error))
                assert np.allclose(current[field],reference[field],rtol=1e-4,atol=2e-5),(item['id'],field,error)
        result=dict(status='passed',comparisons=rows,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,torch_version=torch.__version__)
        save(OUT/'handoff_check.json',result);print('HANDOFF_CHECK',json.dumps(result),flush=True)
        return
    report=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
    for ds in options.datasets:
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
