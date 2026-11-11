#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
py=/work/zhanghc/miniconda3/envs/finelapt/bin/python
exp=seed31415926_grufp32_scale10
mkdir -p "runs/$exp" log
exec 9>"runs/$exp/job.lock"
flock -n 9 || exit 1
trap 's=$?; if [ "$s" -ne 0 ]; then printf "%s failed exit=%s\n" "$(date -Is)" "$s" >> "log/$exp.log"; fi' EXIT
"$py" - <<'PY'
import json
from pathlib import Path
p=Path('runs/pilot_1024')
s=json.loads((p/'status.json').read_text())
assert s['stage']=='completed' and s['pilot'] and s['step']==2
assert json.loads(Path('log/provenance.json').read_text())['scorable_classes']==407
assert json.loads(Path('log/text_check.json').read_text())['max_abs_error']<5e-5
assert sum(1 for x in p.glob('benchmark_*/**/smoke-1/*.npz'))==7
PY
if [ ! -f "runs/$exp/source.tar.gz" ]; then
  tar -czf "runs/$exp/source.tar.gz" data.py model.py train.py evaluate.py run.sh log/provenance.json
fi
if ! "$py" -c "import json,pathlib; p=pathlib.Path('runs/$exp/status.json'); exit(0 if p.exists() and json.loads(p.read_text()).get('stage')=='completed' else 1)"; then
  extra=()
  if [ -f "runs/$exp/last.pt" ]; then extra=(--resume); fi
  "$py" -m torch.distributed.run --standalone --nproc_per_node=4 -- train.py --run "$exp" --initial-scale 10 --initial-bias -10 "${extra[@]}" >> "log/$exp.log" 2>&1
fi
CUDA_VISIBLE_DEVICES=0 "$py" -u evaluate.py --run "$exp" >> "log/${exp}_eval.log" 2>&1

