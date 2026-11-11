"""Prepare FineLAP's five-source COSED mixture; never use evaluation rows for training."""
import argparse
import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cosed"
DATA = Path('/work/zhanghc/datasets')
AS = Path('/work/zhangyu/AED/audioset_tagging_cnn/datasets')
FACTORS = {'audioset_strong': 1, 'finelap100k': 1, 'audiocaps': 2, 'tacos': 5, 'clotho': 4}


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def csv_rows(path, **kwargs):
    with path.open() as f:
        return list(csv.DictReader(f, **kwargs))


def write_rows(name, rows):
    target = OUT / (name + '.jsonl')
    tmp = target.with_suffix('.jsonl.part')
    seen = set()
    with tmp.open('w') as f:
        for row in rows:
            assert row['audio_id'] not in seen, (name, row['audio_id'])
            seen.add(row['audio_id'])
            assert row['caption'], (name, row['audio_id'], 'empty caption')
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(target)
    return len(rows)


def prepare_as_and_ac():
    # Cache only a completed index; source metadata and exclusions are re-read on every build.
    ac = csv_rows(OUT / 'verified_audiocaps_train.csv')
    excluded = set()
    for split in ['val', 'test']:
        excluded.update('Y' + r['youtube_id'] for r in csv_rows(OUT / ('verified_audiocaps_' + split + '.csv')))
    for tag in OUT.glob('tag_test*.json'):
        records = json.loads(tag.read_text())
        if isinstance(records, dict):
            records = records.get('data', records.get('audios', []))
        excluded.update(Path(r['audio_id']).stem for r in records)
    strong_root = DATA / 'AudioSet/metadata/strong'
    mapping = dict(csv.reader((strong_root/'mid_to_display_name.tsv').open(), delimiter='\t'))
    groups = defaultdict(lambda: defaultdict(list))
    rejected = []
    for r in csv_rows(strong_root/'audioset_train_strong.tsv', delimiter='\t'):
        a, b = float(r['start_time_seconds']), float(r['end_time_seconds'])
        if b <= a:
            rejected.append(r)
            continue
        groups[r['segment_id']][mapping[r['label']]].append([a, b])
    targets = {'Y' + k.rsplit('_', 1)[0] for k in groups} | {'Y' + r['youtube_id'] for r in ac}
    wc = DATA / 'WavCaps'
    caps = {Path(r['id']).stem: r['caption'] for r in json.loads((wc/'hf_repo/json_files/AudioSet_SL/as_final.json').read_text())['data']}
    index_path = OUT/'audio_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {}
        for directory in [wc/'audio/AudioSet_SL', AS/'audios/balanced_train_segments', AS/'audios/unbalanced_train_segments']:
            for d, _, names in os.walk(directory):
                for name in names:
                    if not name.endswith(('.wav', '.flac')):
                        continue
                    k = Path(name).stem
                    if k in targets and k not in index:
                        index[k] = str(Path(d)/name)
            print('INDEX', directory, len(index), flush=True)
        atomic_json(index_path, index)
    starts = defaultdict(set)
    for file in ['balanced_train_segments.csv', 'unbalanced_train_segments.csv']:
        with (AS/'metadata'/file).open() as f:
            for r in csv.reader((line for line in f if not line.startswith('#')), skipinitialspace=True):
                starts['Y'+r[0]].add(round(float(r[1])*1000))
    rows, missing, filtered, templated = [], [], [], 0
    for clip, phrases in groups.items():
        video, start = clip.rsplit('_', 1)
        key = 'Y' + video
        if key in excluded:
            filtered.append(clip)
            continue
        if key not in index or int(start) not in starts[key]:
            missing.append(clip)
            continue
        caption = caps.get(key)
        if not caption:
            caption = 'The sound of ' + ', '.join(phrases) + '.'
            templated += 1
        rows.append(dict(audio_id=clip, audio_path=index[key], caption=caption,
                         phrases=[dict(phrase=k, segments=v) for k, v in phrases.items()]))
    counts = {'audioset_strong': write_rows('audioset_strong', rows)}
    ac_rows, ac_missing, ac_filtered = {}, [], []
    for r in ac:
        key = 'Y' + r['youtube_id']
        if key in excluded:
            ac_filtered.append(r['audiocap_id'])
            continue
        if key not in index or round(float(r['start_time'])*1000) not in starts[key]:
            ac_missing.append(r['audiocap_id'])
            continue
        if key not in ac_rows:
            ac_rows[key] = dict(audio_id=key, audio_path=index[key], caption=[])
        ac_rows[key]['caption'].append(r['caption'])
    counts['audiocaps'] = write_rows('audiocaps', list(ac_rows.values()))
    atomic_json(OUT/'source_audit.json', dict(counts=counts, as_missing=missing,
        as_excluded=filtered, as_template_captions=templated, invalid_intervals=rejected,
        ac_missing=ac_missing, ac_excluded=ac_filtered, excluded_video_ids=sorted(excluded)))
    return counts


def prepare_clotho():
    root = Path('/work/zhanghc/Myllm/ASU-LLM/data/CLOTHO_v2.1')
    rows = []
    for r in csv_rows(root/'clotho_csv_files/clotho_captions_development.csv'):
        path = root/'clotho_audio_files/development'/r['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(dict(audio_id=r['file_name'], audio_path=str(path),
                         caption=[r['caption_'+str(i)] for i in range(1, 6)]))
    return write_rows('clotho', rows)


def prepare_finelap(smoke=False):
    root = DATA/'FineLAP-100k'
    files = sorted((root/'data').glob('*.parquet'))
    if not smoke and len(files) != 29:
        raise RuntimeError(f'FineLAP-100k incomplete: {len(files)}/29 shards')
    audio_dir = root/'audio'
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['audio_id', 'audio', 'caption', 'phrases']):
            for r in batch.to_pylist():
                dst = audio_dir/(r['audio_id']+'.wav')
                data = r['audio']['bytes']
                if not dst.exists() or dst.stat().st_size != len(data):
                    tmp = dst.with_suffix('.wav.part'); tmp.write_bytes(data); tmp.replace(dst)
                rows.append(dict(audio_id=r['audio_id'], audio_path=str(dst), caption=r['caption'],
                    phrases=[dict(phrase=p['phrase'], segments=[[s['start'], s['end']] for s in p['segments']]) for p in r['phrases']]))
            if smoke and len(rows) >= 32:
                return write_rows('finelap100k_smoke', rows[:32])
        print('EXTRACTED',path.name,len(rows),flush=True)
    assert len(rows) == 100000, len(rows)
    return write_rows('finelap100k', rows)


def prepare_tacos(smoke=False):
    root = DATA/'TACOS'
    src = root/'downloads'
    audio = root/'audio'
    audio.mkdir(exist_ok=True)
    train = {r['filename'] for r in csv_rows(src/'development_split.csv')}
    test = {r['filename'] for r in csv_rows(src/'test_split.csv')}
    assert not train & test
    if smoke:
        train = set(sorted(train)[:8])
    missing = [name for name in train if not (audio/name).is_file()]
    if missing:
        with zipfile.ZipFile(src/'audio.zip') as archive:
            members = {Path(n).name:n for n in archive.namelist() if n.endswith('.mp3')}
            for name in missing:
                if Path(name).name != name:
                    raise ValueError(name)
                data = archive.read(members[name])
                tmp = (audio/name).with_suffix('.mp3.part');tmp.write_bytes(data);tmp.replace(audio/name)
    captions = {r['filename']:r['text'] for r in csv_rows(src/'annotations_weak.csv')}
    groups = defaultdict(lambda:defaultdict(list))
    for r in csv_rows(src/'annotations_strong.csv'):
        if r['filename'] in train and float(r['offset']) > float(r['onset']):
            groups[r['filename']][r['text']].append([float(r['onset']), float(r['offset'])])
    rows = [dict(audio_id=name,audio_path=str(audio/name),caption=captions[name],
                 phrases=[dict(phrase=k,segments=v) for k,v in groups[name].items()]) for name in sorted(train)]
    return write_rows('tacos_smoke' if smoke else 'tacos',rows)


def configure(counts, smoke=False):
    source = yaml.safe_load((ROOT/'config/data_config/data_eat.yaml').read_text())
    train = source['data_args']['train_data_args']
    train.update(batch_size=32, num_workers=4, random_crop=True, strict_audio=True, warn_unknown_phrases=False, audio_backend="soundfile")
    paths=[]
    for name,repeat in FACTORS.items():
        if smoke:
            original = OUT/((name+'_smoke' if name in ['finelap100k','tacos'] else name)+'.jsonl')
            samples=[json.loads(l) for l in original.open()][:8]
            for r in samples:
                info=sf.info(r['audio_path'])
                assert info.frames>0 and info.samplerate>0
            write_rows(name+'_smoke_input',samples)
            paths += [str(OUT/(name+'_smoke_input.jsonl'))]*repeat
        else:
            paths += [str(OUT/(name+'.jsonl'))]*repeat
    train['metadata_files']=paths
    suffix='_smoke' if smoke else ''
    (ROOT/('config/data_config/data_cosed'+suffix+'.yaml')).write_text(yaml.safe_dump(source,sort_keys=False))
    config=yaml.safe_load((ROOT/'config/finelap_eat_config.yaml').read_text())
    config['model_args']['ckpt_path']=''
    config['training']['precision']='bf16'
    config['training']['epochs']=10
    if smoke:config['training']['max_steps']=3
    (ROOT/('config/finelap_cosed'+suffix+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
    summary=dict(counts=counts,repeat_factors=FACTORS,weighted_counts={k:v*FACTORS[k] for k,v in counts.items()},
        smoke_only=smoke, differences=['Full public TACOS development split; COSED speech exclusion IDs unavailable',
        'FineLAP architecture, objectives, phrase bank and 10-epoch optimizer recipe retained',
        'AS captions use WavCaps when available, otherwise deterministic label template',
        'Free-text TACOS queries retained literally; unknown phrases use original FineLAP fallback',
        'TAG test and AudioCaps val/test IDs excluded from both AS and AudioCaps'])
    atomic_json(OUT/('mixture'+suffix+'.json'),summary)
    print(json.dumps(summary,indent=2),flush=True)


def validate_existing():
    import concurrent.futures
    import numpy as np
    from collections import Counter
    rows_by_source={k:[json.loads(l) for l in (OUT/(k+'.jsonl')).open()] for k in FACTORS}
    paths=sorted({r['audio_path'] for rows in rows_by_source.values() for r in rows})
    def check(path):
        try:
            with sf.SoundFile(path) as f:
                duration=f.frames/f.samplerate
                if not 0.1 <= duration <= 3600:
                    raise ValueError(f'invalid duration {duration}')
                total=0
                for block in f.blocks(blocksize=65536,dtype='float32'):
                    if not np.isfinite(block).all():raise ValueError('nonfinite audio')
                    total+=len(block)
                if total != f.frames:raise ValueError(f'decoded {total}, header {f.frames}')
            return path,None
        except Exception as error:return path,str(error)
    bad={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(path,error) in enumerate(pool.map(check,paths),1):
            if error:bad[path]=error
            if i%5000==0:print('DECODE_CHECK',i,len(paths),'bad',len(bad),flush=True)
    removed={k:[r['audio_id'] for r in rows if r['audio_path'] in bad] for k,rows in rows_by_source.items()}
    audit=dict(total_unique_paths=len(paths),bad_paths=bad,removed_ids=removed,counts_before={k:len(v) for k,v in rows_by_source.items()})
    atomic_json(OUT/'audio_validation.json',audit)
    counts={k:write_rows(k,[r for r in rows if r['audio_path'] not in bad]) for k,rows in rows_by_source.items()}
    configure(counts)
    print('AUDIO_VALIDATION_COMPLETE',json.dumps(dict(counts=counts,bad=len(bad))),flush=True)


def prepare_audiosetcaps():
    """Match official captions to existing unbalanced-train WAVs; no audio download."""
    import hashlib
    import re
    from collections import Counter
    folder=ROOT/'data/finelap_original';folder.mkdir(exist_ok=True)
    source=DATA/'AudioSetCaps'
    csv_file=source/'AudioSetCaps_caption.csv'
    metadata=json.loads((source/'source.json').read_text(encoding='utf-8-sig'))
    h=hashlib.sha256()
    with csv_file.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    assert h.hexdigest()==metadata['file']['lfs']['oid'],'AudioSetCaps checksum mismatch'
    index=json.loads((folder/'audio_index.json').read_text())
    known_bad_audio=set()
    previous_validation=ROOT/'data/cosed/audio_validation.json'
    if previous_validation.exists():
        for bad_path in json.loads(previous_validation.read_text()).get('bad_paths',{}):
            key=Path(bad_path).stem
            if key not in index:continue
            try:
                waveform,rate=sf.read(index[key],dtype='float32',always_2d=True)
                if len(waveform)==0:known_bad_audio.add(key)
            except Exception:
                known_bad_audio.add(key)
    def video_id(value):
        stem=Path(str(value)).stem
        if len(stem)>=12 and stem[0]=='Y' and (len(stem)==12 or stem[12]=='_'):return stem[:12]
        if len(stem)>=11 and (len(stem)==11 or stem[11]=='_'):return 'Y'+stem[:11]
        return None
    excluded=set()
    old_audit=json.loads((ROOT/'data/cosed/source_audit.json').read_text())
    excluded.update(old_audit['excluded_video_ids'])
    for row in csv_rows(DATA/'AudioSet/metadata/strong/audioset_eval_strong.tsv',delimiter='\t'):
        excluded.add('Y'+row['segment_id'].rsplit('_',1)[0])
    with (AS/'metadata/eval_segments.csv').open() as f:
        for row in csv.reader((l for l in f if not l.startswith('#')),skipinitialspace=True):excluded.add('Y'+row[0])
    for name in ['desed-public','desed-validation','tag-test']:
        manifest=json.loads((ROOT/'dependencies/benchmark/manifests'/(name+'.json')).read_text())
        for row in manifest['items']:
            key=video_id(Path(row['path']).name)
            if key:excluded.add(key)
    starts={}
    with (AS/'metadata/unbalanced_train_segments.csv').open() as f:
        for row in csv.reader((l for l in f if not l.startswith('#')),skipinitialspace=True):
            key='Y'+row[0];start=round(float(row[1])*1000)
            if key in starts and starts[key]!=start:starts[key]=None
            else:starts[key]=start
    target=folder/'audiosetcaps.jsonl';temporary=target.with_suffix('.jsonl.part')
    rejected_path=folder/'audiosetcaps_excluded.jsonl';rejected_tmp=rejected_path.with_suffix('.jsonl.part')
    count=Counter();seen=set()
    with csv_file.open(newline='',encoding='utf-8-sig') as f,temporary.open('w') as out,rejected_tmp.open('w') as rejected:
        for row in csv.DictReader(f):
            count['input_rows']+=1
            key=row['id'];caption=row['caption'].strip();reason=None
            if not re.fullmatch(r'Y[A-Za-z0-9_-]{11}',key):reason='invalid_id'
            elif key in seen:reason='duplicate_id'
            elif not caption:reason='empty_caption'
            elif key in excluded:reason='heldout_video'
            elif key not in starts:reason='not_unbalanced_train'
            elif starts[key] is None:reason='ambiguous_segment_start'
            elif key not in index:reason='audio_not_indexed'
            elif key in known_bad_audio:reason='known_invalid_audio'
            seen.add(key)
            if reason:
                count[reason]+=1;rejected.write(json.dumps(dict(id=key,reason=reason))+'\n');continue
            out.write(json.dumps(dict(audio_id=key,audio_path=index[key],caption=caption),ensure_ascii=False)+'\n')
            count['matched_train']+=1
            if count['input_rows']%100000==0:print('AUDIOSETCAPS_MATCH',dict(count),flush=True)
    temporary.replace(target);rejected_tmp.replace(rejected_path)
    report=dict(counts=dict(count),source_revision=metadata['revision'],source_sha256=h.hexdigest(),indexed_audio_files=len(index),
                split='AudioSet unbalanced_train only, following original FineLAP config',
                exclusions='AudioSet eval, AudioSet Strong eval, AudioCaps val/test, TAG test, DESED Public/Validation video IDs',
                start_time_check='Caption CSV has no start field; require unique start in official AudioSet unbalanced metadata. Waveform crop not independently verified.',
                limitations=['Audio index matches are not a full decode audit','Original author unbal_train_em_filtered exact filtering is not published; this is a reconstructed input list','No source weighting or training launch implied by this preparation'])
    atomic_json(folder/'audiosetcaps_audit.json',report)
    print('AUDIOSETCAPS_COMPLETE',json.dumps(report),flush=True)


def prepare_original():
    """Rebuild the ten public-code sources, retaining provenance for unavailable author JSONLs."""
    import concurrent.futures
    import hashlib
    import itertools
    from collections import Counter
    folder=ROOT/'data/finelap_original';folder.mkdir(exist_ok=True)
    stats={}; rejected=[]; unknown_phrases=Counter()
    heldout=set(json.loads((OUT/'source_audit.json').read_text())['excluded_video_ids'])
    heldout.update(json.loads(l)['id'] for l in (folder/'audiosetcaps_excluded.jsonl').open()
                   if json.loads(l)['reason']=='heldout_video')
    wavcaps=DATA/'WavCaps'
    bank=[json.loads(l) for l in (ROOT/'data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl').open()]
    alternatives={alt for row in bank for alt in row['alternatives']}
    def rows(path):
        with path.open() as f:
            for line in f:yield json.loads(line)
    def check(row):
        try:
            info=sf.info(row['audio_path'])
            if not 0.1<=info.duration<=3600 or info.channels<1:raise ValueError('invalid audio header/duration')
            return row,None
        except Exception as e:return row,str(e)
    def save(name, iterable, validate=True):
        counts=Counter();path=folder/(name+'.jsonl');temp=path.with_suffix('.jsonl.part')
        iterator=iter(iterable)
        with temp.open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            while True:
                batch=list(itertools.islice(iterator,2048))
                if not batch:break
                for row,error in (pool.map(check,batch) if validate else ((row,None) for row in batch)):
                    counts['input']+=1
                    if error:
                        counts['invalid_audio']+=1;rejected.append(dict(source=name,id=row['audio_id'],reason=error));continue
                    assert row['caption'],(name,row['audio_id'],'empty caption')
                    for phrase in row.get('phrases',[]):
                        if phrase['phrase'] not in alternatives:unknown_phrases[name+':'+phrase['phrase']]+=1
                        assert all(0<=a<b<=10.00001 for a,b in phrase['segments']),(name,phrase)
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');counts['kept']+=1
                if counts['input']%65536==0:print('ORIGINAL_HEADERS',name,dict(counts),flush=True)
        temp.replace(path);stats[name]=dict(counts);print('ORIGINAL_SOURCE',name,dict(counts),flush=True)
        return path
    # Preserve independently decoded AudioCaps, Clotho and the released synthetic corpus.
    save('audiocaps',rows(OUT/'audiocaps.jsonl'),False)
    save('clotho',rows(OUT/'clotho.jsonl'),False)
    save('finelap100k',rows(OUT/'finelap100k.jsonl'),False)
    natural_caps={Path(r['id']).stem:r['caption'] for r in json.loads((wavcaps/'hf_repo/json_files/AudioSet_SL/as_final.json').read_text())['data']}
    strong=[]
    for row in rows(OUT/'audioset_strong.jsonl'):
        key='Y'+row['audio_id'].rsplit('_',1)[0]
        if key in heldout:continue
        if key not in natural_caps:
            rejected.append(dict(source='audioset_strong',id=key,reason='no_WavCaps_caption'));continue
        row['caption']=natural_caps[key];strong.append(row)
    save('audioset_strong',strong,False)
    # DESED real strongly annotated training clips, never Public/Validation or synthetic eval.
    index=json.loads((folder/'audio_index.json').read_text())
    starts=defaultdict(set)
    for split in ['balanced_train_segments','unbalanced_train_segments']:
        with (AS/'metadata'/(split+'.csv')).open() as f:
            for row in csv.reader((l for l in f if not l.startswith('#')),skipinitialspace=True):
                starts['Y'+row[0]].add(round(float(row[1])*1000))
    caps={row['id']:row['caption'] for row in csv_rows(DATA/'AudioSetCaps/AudioSetCaps_caption.csv')}
    groups=defaultdict(lambda:defaultdict(list))
    for row in csv_rows(DATA/'cosed-benchmarks/desed/data/6444477/audioset_strong.tsv',delimiter='\t'):
        a=max(0,float(row['onset']));b=min(10,float(row['offset']))
        if b>a:groups[row['filename']][row['event_label']].append([a,b])
    desed=[]
    for filename,phrases in groups.items():
        key,start,end=Path(filename).stem.rsplit('_',2)
        reason=None
        if key in heldout:reason='heldout_video'
        elif key not in index:reason='missing_audio'
        elif starts[key]!={round(float(start)*1000)}:reason='segment_start_not_unique_match'
        elif key not in natural_caps and key not in caps:reason='no_natural_caption'
        if reason:rejected.append(dict(source='desed_strong',id=filename,reason=reason));continue
        desed.append(dict(audio_id=filename,audio_path=index[key],caption=natural_caps.get(key,caps.get(key)),
                          phrases=[dict(phrase=k,segments=v) for k,v in phrases.items()]))
    save('desed_strong',desed)
    del index,starts,caps,natural_caps
    urban=DATA/'cosed-benchmarks/urbansed/data/1324404/URBAN-SED_v2.0.0'
    urban_rows=[]
    # Paper reports 8k: public train 6k + validate 2k, test 2k excluded.
    for split in ['train','validate']:
        for path in sorted((urban/'annotations'/split).glob('*.txt')):
            if path.name.startswith('._'):continue
            phrases=defaultdict(list)
            for a,b,label in csv.reader(path.open(),delimiter='\t'):
                a=max(0,float(a));b=min(10,float(b))
                if b>a:phrases[label].append([a,b])
            urban_rows.append(dict(audio_id=path.stem,audio_path=str(urban/'audio'/split/(path.stem+'.wav')),
                caption='This audio contains the sounds of '+', '.join(k.replace('_',' ') for k in sorted(phrases))+'.',
                phrases=[dict(phrase=k,segments=v) for k,v in phrases.items()]))
    assert len(urban_rows)==8000,len(urban_rows)
    save('urbansed',urban_rows)
    blacklists={}
    for name in ['blacklist_exclude_test_ac.json','blacklist_exclude_ub8k_esc50_vggsound.json']:
        for source,ids in json.loads((wavcaps/'hf_repo/json_files/blacklist'/name).read_text()).items():
            blacklists.setdefault(source,set()).update(Path(str(x)).stem for x in ids)
    for source,filename in [('FreeSound','fsd_final.json'),('BBC_Sound_Effects','bbc_final.json'),('SoundBible','sb_final.json')]:
        source_index={}
        for directory,_,files in os.walk(wavcaps/'audio'/source):
            for filename_audio in files:
                path=Path(directory)/filename_audio
                if path.suffix.lower() in {'.wav','.flac'}:source_index.setdefault(path.stem,str(path))
        source_rows=[]
        for row in json.loads((wavcaps/'hf_repo/json_files'/source/filename).read_text())['data']:
            key=str(row['id']);reason=None
            if key in blacklists.get(source,set()):reason='official_eval_blacklist'
            elif key not in source_index:reason='missing_audio'
            if reason:rejected.append(dict(source=source,id=key,reason=reason));continue
            source_rows.append(dict(audio_id=key,audio_path=source_index[key],caption=row['caption']))
        save('wavcaps_'+source.lower(),source_rows)
    # Keep stronger/human supervision once; don't count duplicated AudioSet clips again as weak-only.
    preferred={'Y'+row['audio_id'].rsplit('_',1)[0] for row in strong}
    preferred.update(row['audio_id'] for row in rows(folder/'audiocaps.jsonl'))
    preferred.update(Path(row['audio_id']).stem.rsplit('_',2)[0] for row in rows(folder/'desed_strong.jsonl'))
    def weak_rows():
        for row in rows(folder/'audiosetcaps.jsonl'):
            if row['audio_id'] in preferred or row['audio_id'] in blacklists.get('AudioSet',set()):
                rejected.append(dict(source='audiosetcaps_weak',id=row['audio_id'],reason='strong_human_overlap_or_official_eval_blacklist'));continue
            yield row
    save('audiosetcaps_weak',weak_rows())
    order=['audioset_strong','desed_strong','audiocaps','clotho','wavcaps_freesound','wavcaps_bbc_sound_effects','wavcaps_soundbible','finelap100k','audiosetcaps_weak','urbansed']
    config=yaml.safe_load((ROOT/'config/finelap_eat_config.yaml').read_text());config['model_args']['ckpt_path']=''
    config['training'].update(epochs=10,alpha=1,beta=1,precision='fp32',activation_checkpointing=True)
    data=yaml.safe_load((ROOT/'config/data_config/data_eat.yaml').read_text())
    data['data_args']['train_data_args'].update(metadata_files=[str(folder/(n+'.jsonl')) for n in order],
        batch_size=128,num_workers=4,random_crop=False,strict_audio=True,audio_backend='soundfile',warn_unknown_phrases=False)
    (ROOT/'config/finelap_original.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (ROOT/'config/data_config/data_original.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    with (folder/'excluded.jsonl').open('w') as f:
        for row in rejected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=dict(sources=stats,source_order=order,total=sum(s['kept'] for s in stats.values()),repeat_factors={n:1 for n in order},
        metadata_sha256={n:hashlib.sha256((folder/(n+'.jsonl')).read_bytes()).hexdigest() for n in order},
        rejection_counts=dict(Counter(r['source']+':'+r['reason'] for r in rejected)),unknown_phrases=dict(unknown_phrases),
        differences=['Exact author *_em_filtered JSONLs are unpublished; reconstructed public sources',
          'UrbanSED train+validate=8000 inferred from paper count; deterministic event-list captions reconstructed',
          'DESED captions use WavCaps then AudioSetCaps, original author caption source unavailable',
          'AudioSet weak rows exclude duplicated strong/human AudioCaps clips and official WavCaps eval blacklists; exact author filtering unknown',
          'Original first10s/16kHz/phrase-bank/SpecAugment retained; soundfile backend and fail-fast I/O',
          'New audio checked by headers, not exhaustive full decoding; previous decoded sources reused',
          'Activation recomputation for memory; actual global batch512, FP32, no gradient accumulation'],
        completed=True)
    atomic_json(folder/'mixture.json',report);print('ORIGINAL_PREPARED',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--validate-existing',action='store_true');parser.add_argument('--prepare-audiosetcaps',action='store_true');parser.add_argument('--prepare-original',action='store_true');args=parser.parse_args()
    if args.prepare_original:
        prepare_original()
        raise SystemExit(0)
    if args.prepare_audiosetcaps:
        prepare_audiosetcaps()
        raise SystemExit(0)
    if args.validate_existing:
        validate_existing()
        raise SystemExit(0)
    OUT.mkdir(parents=True,exist_ok=True)
    counts=prepare_as_and_ac()
    counts['clotho']=prepare_clotho()
    counts['tacos']=prepare_tacos(args.smoke)
    counts['finelap100k']=prepare_finelap(args.smoke)
    configure(counts,args.smoke)
