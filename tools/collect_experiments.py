"""Read-only source/result export. Run on each experiment host; no audio/weights."""
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

TEXT = {'.py','.pyx','.pxd','.c','.cpp','.h','.sh','.md','.txt','.json','.jsonl','.yaml','.yml','.csv','.tsv','.html','.log','.toml','.cfg','.conf','.ini','.ipynb'}
ALLOWED = TEXT | {'.png','.svg','.pdf'}
LIMIT = 20 * 1024**2
SKIP = {'.git','__pycache__','.venv','.agents','.codex','node_modules'}

def main():
    host = sys.argv[1]
    if host == 'rabbit02':
        c = Path('/work/zhanghc/Myllm/COSED')
        f = Path('/work/zhanghc/Myllm/FineLAP')
        groups = [(c/'runs','experiments/cosed/runs'), (c/'log','experiments/cosed/log'),
                  (c/'vendor','third_party/cosed_vendor'), (f/'exps','experiments/finelap/runs'),
                  (f/'log','experiments/finelap/log'), (f/'dependencies/benchmark','evaluation/frozen'),
                  (f/'dependencies/scoring','third_party/scoring')]
        groups += [(f/n,'experiments/finelap/source/'+n) for n in
                   ['models','data_handling','utils','scripts','config','train.py','evaluate.py',
                    'README.md','requirements_train.txt','requirements_infer.txt','resources/training.md']]
    elif host == 'rabbit05':
        groups = [(Path('/home/zhanghc/projects/ov-sed/benchmark'),'evaluation/rabbit05')]
    elif host == 'rabbit05-models':
        root = Path('/home/zhanghc/projects/ov-sed')
        groups = [(root/n, 'third_party/published_models/'+n) for n in
                  ['DASM','FineLAP','FlexSED','MGA-CLAP','OpenFLAM','repositories.json']]
        groups += [(root/'FineLAP/exps', 'experiments/finelap/rabbit05')]
    else:
        raise ValueError(host)
    included=[]; excluded=Counter(); omitted=[]
    output=Path('/tmp/cosed-research-'+host+'.zip')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as dest:
        def add(data, arc, source):
            if len(data)>LIMIT:
                excluded['over_20MiB']+=1;omitted.append(dict(source=source,bytes=len(data)));return
            dest.writestr(arc,data)
            included.append(dict(path=arc,source=source,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        for base,prefix in groups:
            if not base.exists():
                omitted.append(dict(source=str(base),reason='missing'));continue
            paths=[base] if base.is_file() else sorted(base.rglob('*'))
            for p in paths:
                if not p.is_file() or p.is_symlink():continue
                rel=p.relative_to(base) if base.is_dir() else Path()
                if any(part in SKIP for part in rel.parts):continue
                if prefix.startswith('third_party/published_models/'):
                    if any(part in {'checkpoints','weights','datasets','outputs','results','logs','exps','.cache'} for part in rel.parts):continue
                    code_ext = {'.py','.pyx','.pxd','.c','.cpp','.h','.sh','.md','.yaml','.yml','.toml','.cfg','.conf','.ini'}
                    if p.suffix.lower() not in code_ext and p.name not in {'config.json','repositories.json','setup.cfg','setup.py','requirements.txt','requirements_train.txt','requirements_infer.txt'} and not p.name.upper().startswith(('LICENSE','NOTICE','COPYING')):continue
                arc=prefix+('/'+rel.as_posix() if base.is_dir() else '')
                if p.name in ['source.tar.gz','source_before_step_checkpoints.tar.gz','source_snapshot.zip']:
                    # Expose historical code for review rather than burying it in archives.
                    try:
                        if p.suffix=='.zip':
                            with zipfile.ZipFile(p) as a:
                                members=[(x.filename,x.file_size,lambda x=x:a.read(x)) for x in a.infolist() if not x.is_dir()]
                                unpack(members,arc,add,str(p))
                        else:
                            with tarfile.open(p) as a:
                                members=[(x.name,x.size,lambda x=x:a.extractfile(x).read()) for x in a.getmembers() if x.isfile()]
                                unpack(members,arc,add,str(p))
                    except (tarfile.TarError,zipfile.BadZipFile) as e:
                        omitted.append(dict(source=str(p),reason=str(e)))
                    continue
                if p.suffix.lower() not in ALLOWED and not p.name.upper().startswith(('LICENSE','NOTICE','COPYING')):
                    excluded[p.suffix or 'no_suffix']+=1
                    if p.suffix in {'.pt','.pth','.ckpt','.safetensors'}:omitted.append(dict(source=str(p),bytes=p.stat().st_size,reason='model_weights'))
                    continue
                if p.stat().st_size>LIMIT:
                    excluded['over_20MiB']+=1;omitted.append(dict(source=str(p),bytes=p.stat().st_size));continue
                add(p.read_bytes(),arc,str(p))
        inventory=dict(host=host,included=included,excluded_counts=dict(excluded),omitted=omitted,
                       policy='Source/text/results <=20MiB each; no audio, weights, arrays, credentials, git or agent internals.')
        dest.writestr('docs/inventory-'+host+'.json',json.dumps(inventory,indent=2))
    print(json.dumps(dict(host=host,archive=str(output),files=len(included),zip_bytes=output.stat().st_size,excluded=dict(excluded))))

def unpack(members,arc,add,source):
    for name,size,read in members:
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or any(x in SKIP for x in p.parts):continue
        if size>LIMIT or (p.suffix.lower() not in ALLOWED and not p.name.upper().startswith(('LICENSE','NOTICE','COPYING'))):continue
        add(read(),arc.replace('.tar.gz','').replace('.zip','')+'/'+str(p),source+'::'+name)

if __name__=='__main__':main()
