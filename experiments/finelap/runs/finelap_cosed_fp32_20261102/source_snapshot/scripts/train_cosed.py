"""Persistent FineLAP jobs: previous COSED mix or the public-code recipe (--original)."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '--original' in sys.argv
FP32_COSED = '--fp32-cosed' in sys.argv
assert not (ORIGINAL and FP32_COSED)
EXP = ROOT/('exps/finelap_cosed_fp32_20261102' if FP32_COSED else ('exps/finelap_original_20261031' if ORIGINAL else 'exps/finelap_cosed_mix_20261029'))
GPUS = os.environ.get('TRAIN_GPUS', '0,1,2,3' if ORIGINAL else '2,3,4').split(',')
BATCH = int(os.environ.get('TRAIN_BATCH_PER_GPU', '128' if ORIGINAL else '32'))
EPOCHS = int(os.environ.get('TRAIN_EPOCHS', '10'))
TAC_URL = 'http://10.160.4.167:40917/'
TAC_FILES = {
    'annotations_strong.csv': '882d5a5a28f59441f4c7b4ed17ebab05',
    'annotations_weak.csv': '1445727f4d470273a7395ddcb9e34b93',
    'development_split.csv': 'bee28197d13e859dbc36021c2a928f07',
    'test_split.csv': '16f5b5eab84753a571e9944739508854',
    'metadata.csv': '17e8df44f2d217dc7cf4b75767ee5ab3',
    'audio.zip': 'a1b11ed7a88d6b95109decb577c128fd',
}


def state(stage, **extra):
    data = dict(stage=stage, updated=datetime.datetime.now().isoformat(),
                gpus=GPUS, batch_per_gpu=BATCH, global_batch=BATCH*len(GPUS), epochs=EPOCHS, **extra)
    tmp = EXP/'status.json.part'
    tmp.write_text(json.dumps(data, indent=2)+'\n');tmp.replace(EXP/'status.json')
    print(json.dumps(data),flush=True)


def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def download_tacos():
    directory=Path('/work/zhanghc/datasets/TACOS/downloads')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for name,digest in TAC_FILES.items():
        final=directory/name
        if final.exists() and md5(final)==digest:continue
        part=directory/(name+'.lan.part')
        for attempt in range(12):
            try:
                offset=part.stat().st_size if part.exists() else 0
                req=urllib.request.Request(TAC_URL+name,headers={'Range':f'bytes={offset}-'} if offset else {})
                with opener.open(req,timeout=120) as source:
                    append=source.status==206 and offset>0
                    expected=int(source.headers['Content-Length'])+(offset if append else 0)
                    with part.open('ab' if append else 'wb') as dst:
                        while True:
                            b=source.read(1024*1024)
                            if not b:break
                            dst.write(b)
                assert part.stat().st_size==expected
                assert md5(part)==digest, f'MD5 mismatch {name}'
                part.replace(final)
                print('TACOS VERIFIED',name,flush=True)
                break
            except Exception as error:
                print('TACOS RETRY',name,attempt,str(error),flush=True)
                if attempt==11:raise
                time.sleep(30)
    try:
        opener.open(urllib.request.Request(TAC_URL+'shutdown',method='POST'),timeout=10).close()
    except Exception as error:print('Transfer helper cleanup:',str(error),flush=True)


def download_finelap():
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    from huggingface_hub import HfApi, snapshot_download
    destination=Path('/work/zhanghc/datasets/FineLAP-100k')
    revision_file=EXP/'finelap100k_revision.txt'
    for attempt in range(12):
        try:
            if revision_file.exists():revision=revision_file.read_text().strip()
            else:
                revision=HfApi(endpoint='https://hf-mirror.com').dataset_info('AndreasXi/FineLAP-100k').sha
                revision_file.write_text(revision+'\n')
            missing=[f'data/train-{i:05d}-of-00029.parquet' for i in range(29)
                     if not (destination/f'data/train-{i:05d}-of-00029.parquet').exists()]
            if not missing:return
            snapshot_download('AndreasXi/FineLAP-100k',repo_type='dataset',revision=revision,
                local_dir=str(destination),allow_patterns=missing,max_workers=2,endpoint='https://hf-mirror.com')
            return
        except Exception as error:
            print('FINELAP DOWNLOAD RETRY',attempt,str(error),flush=True)
            if attempt==11:raise
            time.sleep(30)


def run(args, log_name, env=None):
    log_path=EXP/log_name
    if ORIGINAL or FP32_COSED:
        directory=ROOT/('log/20261102' if FP32_COSED else 'log/20261031');directory.mkdir(exist_ok=True)
        log_path=directory/(('cosed_fp32_' if FP32_COSED else 'original_')+log_name)
        if not (EXP/log_name).exists():(EXP/log_name).symlink_to(log_path)
    with log_path.open('a') as log:
        p=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (EXP/'child.pid').write_text(str(p.pid)+'\n')
        code=p.wait()
        if code:raise RuntimeError(f'{log_name} exited {code}; inspect {EXP/log_name}')


def wait_for_gpus():
    while True:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={x.split(',')[0].strip():int(x.split(',')[1]) for x in lines.splitlines()}
        if all(used[g]<1000 for g in GPUS):return
        state('waiting_for_free_gpus',used_mib={g:used[g] for g in GPUS})
        time.sleep(60)


def main_original():
    import yaml
    import zipfile
    from collections import Counter
    EXP.mkdir(parents=True,exist_ok=True)
    lock=(EXP/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (EXP/'job.pid').write_text(str(os.getpid())+'\n')
    assert len(GPUS)*BATCH==512,'This run follows the confirmed public-code batch512'
    if any((EXP/'models').glob('*.pt')):raise RuntimeError('Checkpoint exists; refuse implicit restart')
    if '--prepare-pid' in sys.argv:
        pid=int(sys.argv[sys.argv.index('--prepare-pid')+1]);state('preparing_data',prepare_pid=pid)
        while Path(f'/proc/{pid}/cmdline').exists():
            cmd=Path(f'/proc/{pid}/cmdline').read_text().replace('\x00',' ')
            if 'prepare_cosed_data.py' not in cmd:break
            time.sleep(10)
    folder=ROOT/'data/finelap_original'
    mixture=json.loads((folder/'mixture.json').read_text());assert mixture['completed']
    assert len(mixture['source_order'])==10 and all(x['kept']>0 for x in mixture['sources'].values())
    for name,digest in mixture['metadata_sha256'].items():
        assert hashlib.sha256((folder/(name+'.jsonl')).read_bytes()).hexdigest()==digest,name
    probe=json.loads((ROOT/'log/20261031/original_memory_probe.json').read_text())['activation_checkpoint_test']
    assert probe['success'] and probe['gradient_equivalence_passed'] and probe['batch_per_gpu']==BATCH
    config=yaml.safe_load((ROOT/'config/finelap_original.yaml').read_text())
    data=yaml.safe_load((ROOT/'config/data_config/data_original.yaml').read_text())
    assert not config['model_args']['ckpt_path']
    assert config['training']['alpha']==config['training']['beta']==1
    assert not data['data_args']['train_data_args']['random_crop']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(GPUS),OMP_NUM_THREADS='4',
        WANDB_MODE='offline',NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'dependencies/fairseq'))
    wait_for_gpus()
    preflight=EXP/'preflight';preflight.mkdir(exist_ok=True)
    # Include all ten sources and a deliberately elevated strong-label fraction.
    strong={'audioset_strong','desed_strong','finelap100k','urbansed'}
    with (preflight/'metadata.jsonl').open('w') as dst:
        for name in mixture['source_order']:
            with (folder/(name+'.jsonl')).open() as src:
                for i,line in enumerate(src):
                    if i >= (128 if name in strong else 256):break
                    dst.write(line)
    smoke_config=dict(config);smoke_config['training']=dict(config['training'],max_steps=3)
    smoke_data=yaml.safe_load(yaml.safe_dump(data));smoke_data['data_args']['train_data_args']['metadata_files']=[str(preflight/'metadata.jsonl')]
    (preflight/'config.yaml').write_text(yaml.safe_dump(smoke_config,sort_keys=False))
    (preflight/'data.yaml').write_text(yaml.safe_dump(smoke_data,sort_keys=False))
    base=[sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={len(GPUS)}','--master_port=29741','train.py']
    common=['--batch_size',str(BATCH),'--seed','31415926','--lr','0.00005','--alpha','1','--beta','1']
    smoke=base+['--config',str(preflight/'config.yaml'),'--data_config',str(preflight/'data.yaml'),
                '--epochs','1','--exp_dir',str(EXP),'--exp_name','preflight']+common
    state('four_gpu_smoke',counts=mixture['sources'],command=smoke)
    run(smoke,'smoke.log',env)
    assert (preflight/'models/epoch_001.pt').exists(),'Distributed smoke checkpoint missing'
    history=json.loads((preflight/'metrics/train_loss.json').read_text())
    (EXP/'smoke_passed.json').write_text(json.dumps(dict(gpus=GPUS,batch_per_gpu=BATCH,history=history,memory_probe=probe),indent=2)+'\n')
    # Freeze this run's actual code and metadata hashes without duplicating audio or datasets.
    sources=['train.py','scripts/train_cosed.py','utils/prepare_cosed_data.py','models/finelap.py','models/losses.py',
             'models/audio_encoder.py','models/text_encoder.py','data_handling/mix_dataset.py','models/feature_extractor.py',
             'config/finelap_original.yaml','config/data_config/data_original.yaml','data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl']
    with zipfile.ZipFile(EXP/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    provenance=dict(mixture=mixture,memory_probe=probe,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        actual_global_batch=512,gradient_accumulation=False,precision='fp32',gpus=GPUS,
        recipe='public scripts/train.sh; paper batch1024 is intentionally not used per user decision',
        initial_weights='EAT + RoBERTa; no FineLAP checkpoint',epochs=EPOCHS)
    (EXP/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    command=base+['--config','config/finelap_original.yaml','--data_config','config/data_config/data_original.yaml',
                  '--epochs',str(EPOCHS),'--exp_dir',str(ROOT/'exps'),'--exp_name',EXP.name]+common
    state('training',total_samples=mixture['total'],command=command)
    run(command,'train.log',env)
    state('completed',total_samples=mixture['total'])


def main_fp32_cosed():
    import copy
    import yaml
    import zipfile
    EXP.mkdir(parents=True,exist_ok=True)
    lock=(EXP/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (EXP/'job.pid').write_text(str(os.getpid())+'\n')
    assert GPUS==['2','3','4'] and BATCH==32 and EPOCHS==10,'Keep the previous run settings for this precision control'
    if any((EXP/'models').glob('*.pt')):raise RuntimeError('Existing checkpoint: refuse implicit restart')
    baseline=ROOT/'exps/finelap_cosed_mix_20261029'
    old=yaml.safe_load((baseline/'config.yaml').read_text())
    assert old['training']['precision']=='bf16' and old['training']['alpha']==1 and old['training']['beta']==5
    assert old['data_args']['train_data_args']['batch_size']==32 and not old['model_args']['ckpt_path']
    config=copy.deepcopy(old)
    config['training']['precision']='fp32'
    config['training']['activation_checkpointing']=True
    comparison=copy.deepcopy(config);comparison['training']['precision']='bf16';comparison['training'].pop('activation_checkpointing')
    assert comparison==old,'Unexpected change beyond precision and activation recomputation'
    mixture=json.loads((ROOT/'data/cosed/mixture.json').read_text())
    metadata={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old['data_args']['train_data_args']['metadata_files']}
    for name,count in mixture['counts'].items():
        assert sum(1 for _ in (ROOT/'data/cosed'/(name+'.jsonl')).open())==count
    probe=json.loads((ROOT/'log/20261031/original_memory_probe.json').read_text())['activation_checkpoint_test']
    assert probe['gradient_equivalence_passed'] and probe['gradient_max_error']==0
    # All historical effective options are copied; CLI overrides match that run exactly.
    (EXP/'input_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (EXP/'input_data.yaml').write_text(yaml.safe_dump({'data_args':old['data_args']},sort_keys=False))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(GPUS),OMP_NUM_THREADS='4',
        WANDB_MODE='offline',NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'dependencies/fairseq'))
    wait_for_gpus()
    base=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=3','--master_port=29743','train.py']
    common=['--data_config',str(EXP/'input_data.yaml'),'--batch_size','32','--seed','31415926','--lr','0.00005','--alpha','1','--beta','5']
    preflight=EXP/'preflight';preflight.mkdir(exist_ok=True)
    smoke=copy.deepcopy(config);smoke['training']['max_steps']=3
    (preflight/'input_config.yaml').write_text(yaml.safe_dump(smoke,sort_keys=False))
    command=base+['--config',str(preflight/'input_config.yaml'),'--epochs','1','--exp_dir',str(EXP),'--exp_name','preflight']+common
    state('three_gpu_smoke',precision='fp32',command=command)
    run(command,'smoke.log',env)
    assert (preflight/'models/epoch_001.pt').is_file()
    history=json.loads((preflight/'metrics/train_loss.json').read_text())
    (EXP/'smoke_passed.json').write_text(json.dumps(dict(gpus=GPUS,batch_per_gpu=BATCH,history=history),indent=2)+'\n')
    files=['train.py','scripts/train_cosed.py','data_handling/mix_dataset.py','models/finelap.py','models/losses.py',
           'models/audio_encoder.py','models/text_encoder.py','models/feature_extractor.py','data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl']
    with zipfile.ZipFile(EXP/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(ROOT/f,f)
    (EXP/'provenance.json').write_text(json.dumps(dict(baseline=str(baseline),changes=['BF16 autocast -> FP32','activation recomputation, previously loss/gradient-equivalent to plain FP32'],
        configuration_otherwise_identical=True,metadata_sha256=metadata,mixture=mixture,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        initialization='same EAT/RoBERTa checkpoints, from scratch; not resumed from trained FineLAP',
        initializer_sha256=probe.get('initializer_sha256'),packages=probe.get('packages')),ensure_ascii=False,indent=2)+'\n')
    command=base+['--config',str(EXP/'input_config.yaml'),'--epochs','10','--exp_dir',str(ROOT/'exps'),'--exp_name',EXP.name]+common
    state('training',precision='fp32',total_samples=sum(mixture['weighted_counts'].values()),command=command)
    run(command,'train.log',env)
    state('completed',precision='fp32')


def main():
    if FP32_COSED:return main_fp32_cosed()
    if ORIGINAL:return main_original()
    EXP.mkdir(parents=True,exist_ok=True)
    lock=(EXP/'job.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (EXP/'job.pid').write_text(str(os.getpid())+'\n')
    if '--wait-validation' in sys.argv:
        validation_pid=int((EXP/'validation.pid').read_text())
        state('validating_audio',validation_pid=validation_pid)
        while Path(f'/proc/{validation_pid}').exists():
            cmd=Path(f'/proc/{validation_pid}/cmdline').read_bytes().decode(errors='replace')
            if 'prepare_cosed_data.py' not in cmd:break
            time.sleep(15)
        tail=(EXP/'prepare_full.log').read_text()[-3000:]
        assert 'AUDIO_VALIDATION_COMPLETE' in tail, 'Audio validation did not complete; training not started'
    if '--prepared' not in sys.argv:
        state('downloading')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(download_finelap),pool.submit(download_tacos)]
            while not all(f.done() for f in futures):
                for f in futures:
                    if f.done():f.result()
                time.sleep(30)
            for f in futures:f.result()
        state('preparing_full_data')
        run([sys.executable,'-u','utils/prepare_cosed_data.py'],'prepare_full.log')
        state('validating_audio')
        run([sys.executable,'-u','utils/prepare_cosed_data.py','--validate-existing'],'prepare_full.log')
    mixture=json.loads((ROOT/'data/cosed/mixture.json').read_text())
    audit=json.loads((ROOT/'data/cosed/audio_validation.json').read_text())
    assert audit['counts_before']['finelap100k']==100000 and audit['counts_before']['tacos']==10358
    for key,count in mixture['counts'].items():
        assert count==audit['counts_before'][key]-len(audit['removed_ids'][key])
    assert all(mixture['counts'][k]>0 for k in mixture['repeat_factors'])
    if any((EXP/'models').glob('*.pt')):
        raise RuntimeError('Existing training checkpoint found; use explicit resume_from instead of restarting')
    wait_for_gpus()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(GPUS),OMP_NUM_THREADS='4',
        WANDB_MODE='offline',NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'dependencies/fairseq'))
    command=[sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={len(GPUS)}','--master_port=29739',
        'train.py','--config','config/finelap_cosed.yaml','--data_config','config/data_config/data_cosed.yaml',
        '--batch_size',str(BATCH),'--epochs',str(EPOCHS),'--exp_dir',str(ROOT/'exps'),
        '--exp_name',EXP.name,'--seed','31415926','--lr','0.00005','--alpha','1','--beta','5']
    # A successful smoke run is a launch gate, not a full-training checkpoint.
    smoke=ROOT/'exps/finelap_cosed_mix_20261029_smoke/models/last_epoch.pt'
    if not smoke.exists() or not (EXP/'smoke_passed.json').exists():
        raise RuntimeError('Smoke test approval marker missing; full training not started')
    verified=json.loads((EXP/'smoke_passed.json').read_text())
    assert verified['batch_per_gpu']==BATCH and verified['gpus']==GPUS, 'Training settings differ from smoke test'
    state('training',counts=mixture['counts'],repeat_factors=mixture['repeat_factors'],command=command)
    run(command,'train.log',env)
    state('completed',counts=mixture['counts'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('failed',error=str(error))
        raise
