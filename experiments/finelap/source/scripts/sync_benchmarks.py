#!/usr/bin/env python3
"""Copy missing rabbit05 benchmark files; reuse verified rabbit02 data. No deletion."""
import datetime, fcntl, json, os, pathlib, subprocess, time, gzip, hashlib
ROOT=pathlib.Path('/work/zhanghc/Myllm/FineLAP/log/20261030')
DEST=pathlib.Path('/work/zhanghc/datasets/cosed-benchmarks')
SOURCE='rsync://10.160.4.167:40919/benchmarks/'
ROOT.mkdir(parents=True, exist_ok=True)
DEST.mkdir(parents=True, exist_ok=True)
lock=(ROOT/'benchmark_transfer.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state={'source':SOURCE, 'destination':str(DEST), 'pid':os.getpid(), 'started':datetime.datetime.now().isoformat(), 'symlinks':'reuse verified local files and internal links; fetch only missing files'}
def save(stage, **kw):
    state.update(stage=stage, updated=datetime.datetime.now().isoformat(), **kw)
    p=ROOT/'benchmark_transfer.status.json'
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(state,indent=2)); t.replace(p)
filelist=ROOT/'benchmark_missing.files'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
if not filelist.is_file():
    save('waiting_source_hashes')
    hashfile=ROOT/'benchmark_source_hashes.json.gz'
    with (ROOT/'benchmark_transfer.log').open('a') as log:
        for attempt in range(720):
            if hashfile.is_file():break
            rc=subprocess.call(['rsync','-t','--timeout=30','--contimeout=10','rsync://10.160.4.167:40919/transfermeta/benchmark_source_hashes.json.gz',str(hashfile)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if rc==0:break
            time.sleep(30)
        else:
            save('failed',error='Source hashes unavailable after waiting');raise SystemExit(1)
    save('checking_local_content')
    inventory=json.load(gzip.open(ROOT/'benchmark_inventory.json.gz','rt'))
    plan=json.load(gzip.open(ROOT/'benchmark_reuse_plan.json.gz','rt'))
    expected=json.load(gzip.open(ROOT/'benchmark_source_hashes.json.gz','rt'))
    reuse={}; hashed={}; checked=0
    for item in plan:
        for entry in item.get("candidate_hashes",[]):
            hashed[tuple(entry["stat"])]=entry["sha256"]
    for item in plan:
        for candidate in item['candidates']:
            try:
                st=os.stat(candidate); key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
                if key not in hashed: hashed[key]=digest(candidate)
                if hashed[key]==expected.get(item['path']):
                    reuse[item['path']]=candidate; break
            except OSError: continue
        checked+=1
        if checked%500==0: save('checking_local_content',checked=checked,candidates=len(plan),matched=len(reuse))
    missing=[]; internal=0; reused_bytes=0
    for item in inventory:
        p=DEST/item['path']
        if 'internal' in item:
            if item.get('dir') and p.is_dir() and not p.is_symlink() and not any(p.iterdir()):
                p.rmdir()
            if not os.path.lexists(p):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.symlink_to(os.path.relpath(DEST/item['internal'],p.parent),target_is_directory=item.get('dir',False))
            if p.is_symlink(): internal+=1; continue
        if item['path'] in reuse:
            target=pathlib.Path(reuse[item['path']])
            if not os.path.lexists(p):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.symlink_to(os.path.relpath(target,p.parent))
            if p.resolve()==target.resolve(): reused_bytes+=item['size']; continue
        missing.append(item['path'])
    tmp=filelist.with_suffix('.tmp')
    tmp.write_bytes(b''.join(x.encode()+b'\0' for x in missing));tmp.replace(filelist)
    missing_set=set(missing)
    audit={'reused':reuse,'sha256':{k:expected[k] for k in reuse},'internal_links':internal,'reuse_bytes':reused_bytes,'missing_files':len(missing),'missing_bytes':sum(x['size'] for x in inventory if x['path'] in missing_set)}
    with gzip.open(ROOT/'benchmark_reuse_verified.json.gz','wt') as f:json.dump(audit,f)
    save('local_audit_complete',reused_files=len(reuse),reused_bytes=reused_bytes,internal_links=internal,missing_files=len(missing),missing_bytes=audit['missing_bytes'])
if (ROOT/'benchmark_reuse_verified.json.gz').is_file():
    audited=json.load(gzip.open(ROOT/'benchmark_reuse_verified.json.gz','rt'))
    state.update(reused_files=len(audited['reused']),reused_bytes=audited['reuse_bytes'],internal_links=audited['internal_links'],missing_files=audited['missing_files'],missing_bytes=audited['missing_bytes'])
common=['rsync','-rltL','--timeout=180','--contimeout=30','--from0','--files-from='+str(filelist)]
try:
    with (ROOT/'benchmark_transfer.log').open('a', buffering=1) as log:
        for attempt in range(1,4):
            save('transferring',attempt=attempt)
            log.write('\nSTART '+state['updated']+' attempt '+str(attempt)+'\n'); log.flush()
            cmd=common+['--partial-dir=.rsync-partial','--stats','--info=progress2,name0','--no-inc-recursive','--outbuf=L',SOURCE,str(DEST)+'/']
            rc=subprocess.call(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if rc==0: break
            save('retry_wait',exit_code=rc)
            if attempt<3: time.sleep(30)
        if rc: raise RuntimeError('rsync failed with code '+str(rc))
        save('verifying')
        v=subprocess.run(common+['--dry-run','--itemize-changes','--out-format=%i %n%L',SOURCE,str(DEST)+'/'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        log.write('\nVERIFICATION\n'+v.stdout+v.stderr); log.flush()
        if v.returncode or v.stdout.strip(): raise RuntimeError('verification found differences; see log')
        broken=[str(p) for p in DEST.rglob('*') if p.is_symlink() and not p.exists()]
        if broken: raise RuntimeError('Broken local references: '+str(broken[:10]))
        save('completed',exit_code=0,verification='rsync transfer checksum plus size/mtime dry-run; no source differences')
except Exception as e:
    save('failed',error=str(e)); raise
