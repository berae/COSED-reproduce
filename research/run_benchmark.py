"""Run selected benchmarks using the existing, frozen model runners."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.home() / 'projects/ov-sed/benchmark'
MODELS = ['dasm', 'finelap', 'flexsed', 'mga-clap', 'openflam']
CORE = ['desed-public', 'desed-validation', 'realdesed-test', 'urbansed-test']
EXTRA = ['tag-test', 'tacos-test', 'maestro-development', 'maestro-evaluation']


def command(model, dataset):
    if dataset in EXTRA:
        runner = 'runners/grounding.py'
    elif model == 'flexsed':
        runner = 'runners/core_flexsed.py'
    else:
        runner = 'runners/core.py'
    return [str(Path.home() / 'venvs' / model / 'bin/python'),
            str(ROOT / runner), 'run', '--model', model, '--dataset', dataset]


def save_status(path, records):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(path)


def scoring_command(model, dataset, tacos_policy='both', tag_policy='both'):
    if dataset not in ['tag-test', 'tacos-test', 'maestro-development']:
        return None
    return [str(Path.home() / 'venvs/finelap/bin/python'),
            str(ROOT / 'scripts/score_benchmarks.py'),
            '--models', model, '--datasets', dataset,
            '--tacos-policy', tacos_policy, '--tag-policy', tag_policy]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True, help='One GPU index; confirm it is available.')
    parser.add_argument('--models', nargs='+', choices=MODELS, required=True)
    parser.add_argument('--datasets', nargs='+', choices=CORE + EXTRA, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without launching or writing files.')
    parser.add_argument('--tacos-policy', choices=['exact','semantic','both'], default='both')
    parser.add_argument('--tag-policy', choices=['official','legacy','both'], default='both')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    jobs = [(m, d) for m in dict.fromkeys(args.models) for d in dict.fromkeys(args.datasets)]
    for model, dataset in jobs:
        cmd = command(model, dataset)
        for file in cmd[:2]:
            if not Path(file).is_file():
                parser.error('Missing file: ' + file)
        if args.dry_run:
            print(f'CUDA_VISIBLE_DEVICES={args.gpu} ' + ' '.join(cmd))
        scoring = scoring_command(model, dataset, args.tacos_policy, args.tag_policy)
        if scoring:
            for file in scoring[:2]:
                if not Path(file).is_file():
                    parser.error('Missing scoring file: ' + file)
            if args.dry_run:
                print('Then score: ' + ' '.join(scoring))
    if args.dry_run:
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    status = ROOT / 'state/queues' / f'{stamp}_gpu{args.gpu}.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
    for key, folder in [('HF_HOME', 'hf-home'), ('HF_HUB_CACHE', 'hf-cache'),
                        ('TRANSFORMERS_CACHE', 'hf-cache')]:
        env[key] = str(Path.home() / 'models/ov-sed' / folder)
    records = []
    print('Queue status:', status, flush=True)
    for model, dataset in jobs:
        now = datetime.now()
        logs = ROOT / 'log' / now.strftime('%Y%m%d')
        logs.mkdir(parents=True, exist_ok=True)
        name = f'{now:%H%M%S_%f}_{model}_{dataset}'.replace('-', '_')
        log = logs / (name + '.log')
        record = dict(model=model, dataset=dataset, gpu=args.gpu,
                      started=time.time(), status='running', log=str(log))
        records.append(record)
        save_status(status, records)
        print(model, dataset, '->', log, flush=True)
        try:
            with log.open('w') as output:
                legacy = ROOT / 'runs-extra' / model / dataset / 'metrics.json'
                saved = legacy.read_bytes() if dataset in EXTRA and legacy.exists() else None
                try:
                    result = subprocess.run(command(model, dataset), cwd=ROOT,
                        env=env, stdout=output, stderr=subprocess.STDOUT)
                finally:
                    if saved is not None:
                        tmp = legacy.with_suffix('.restore.tmp')
                        tmp.write_bytes(saved)
                        tmp.replace(legacy)
                scoring = scoring_command(model, dataset, args.tacos_policy, args.tag_policy)
                if result.returncode == 0 and scoring:
                    result = subprocess.run(scoring, cwd=ROOT, env=env,
                        stdout=output, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,
                          status='complete' if result.returncode == 0 else 'failed')
        except KeyboardInterrupt:
            record.update(status='interrupted', finished=time.time())
            save_status(status, records)
            return 130
        except OSError as error:
            record.update(status='failed', error=str(error))
        record['finished'] = time.time()
        save_status(status, records)
    return int(any(record['status'] != 'complete' for record in records))


if __name__ == '__main__':
    sys.exit(main())
