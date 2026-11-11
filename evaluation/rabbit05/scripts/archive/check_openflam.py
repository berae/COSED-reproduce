import sys,os,json
from pathlib import Path
import numpy as np,torch
h=Path.home();sys.path.insert(0,str(h/'projects/ov-sed/OpenFLAM'));import openflam
m=openflam.OpenFLAM(default_ckpt_path=str(h/'models/ov-sed/hf-cache')).cuda().eval()
a=np.load(h/'models/ov-sed/OpenFLAM/sanity_data.npy',allow_pickle=True);g=np.load(h/'models/ov-sed/OpenFLAM/sanity_gt.npy',allow_pickle=True)
out=[]
with torch.inference_mode():
 for d,ref in zip(a,g):
  x=torch.from_numpy(d['audio']).cuda()
  item={}
  for key,v in [('global_audio',m.get_global_audio_features(x)),('local_audio',m.get_local_audio_features(x)),('text',m.get_text_features([d['text']]))]:
   actual=v.cpu().numpy();expected=ref[key]
   item[key]={'max_abs_error':float(np.abs(actual-expected).max()),'mean_cosine':float(((actual*expected).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(expected,axis=-1))).mean())}
  out.append(item)
(h/'projects/ov-sed/benchmark/openflam-official-sanity.json').write_text(json.dumps(out,indent=2));print(out)
