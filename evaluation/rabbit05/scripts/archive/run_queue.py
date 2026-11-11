import subprocess,os,json,time,sys
from pathlib import Path
r=Path.home()/"projects/ov-sed/benchmark"

def experiment_log(name):
 from datetime import datetime
 now=datetime.now()
 folder=r/"log"/now.strftime("%Y%m%d")
 folder.mkdir(parents=True,exist_ok=True)
 return folder/(now.strftime("%H%M%S_%f")+"_"+name.replace("-","_").lstrip("_"))

(r/"state"/"queues").mkdir(parents=True,exist_ok=True)
gpu=sys.argv[1];records=[]
for model in sys.argv[2:]:
 for dataset in ["desed-public","desed-validation","realdesed-test","urbansed-test"]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",HF_HUB_OFFLINE="1",HF_HOME=str(Path.home()/"models/ov-sed/hf-home"),HF_HUB_CACHE=str(Path.home()/"models/ov-sed/hf-cache"),TRANSFORMERS_CACHE=str(Path.home()/"models/ov-sed/hf-cache"),WANDB_MODE="disabled")
  cmd=[str(Path.home()/"venvs"/model/"bin/python"),str(r/("runners/archive/runner-finelap-hf.py" if model=="finelap" else "runners/archive/runner-v1.py")),"run","--model",model,"--dataset",dataset]
  record=dict(model=model,dataset=dataset,started=time.time(),status="running");records.append(record)
  status=r/"state"/"queues"/("queue-gpu"+gpu+("-finelap" if sys.argv[2:]==["finelap"] else "")+".json")
  status.write_text(json.dumps(records,indent=2))
  with open(experiment_log(model+"-"+dataset+".log"),"a") as f:
   p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  record.update(status="complete" if p.returncode==0 else "failed",returncode=p.returncode,finished=time.time())
  status.write_text(json.dumps(records,indent=2))
