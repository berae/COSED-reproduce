"""COSED research reconstruction, data-parallel logical batches with corpus-scoped supervision."""
import argparse
import datetime
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from scipy.ndimage import median_filter
from model import COSED, focal_terms
from data import CLASSES, SOURCES, ROOT, Corpus, query_bank, targets, validation, wave, as_strong_eval_classes

TEMPLATES=['sound of {}','the sound of {}','a recording of {}','audio of {}','{} can be heard','a sound of {}','there is {}','there is a sound of {}','listen to {}','{} is audible','an audio recording of {}','one can hear {}']
def save_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def restore_rng(state):
    random.setstate(state['python'])
    ns=state['numpy'];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])

def save_checkpoint(out,m,ema,opt,epoch,batch_in_epoch,step,best,patience,calibration,rank,world,training_config):
    ns=np.random.get_state()
    rng=dict(python=random.getstate(),numpy=(ns[0],ns[1].tolist(),*ns[2:]),
             torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    rngs=[None]*world
    if world>1:dist.all_gather_object(rngs,rng)
    else:rngs[0]=rng
    if rank==0:
        state=dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            epoch=epoch,batch_in_epoch=batch_in_epoch,step=step,best=best,patience=patience,
            calibration=calibration,rng_states=rngs,training_config=training_config)
        torch.save(state,out/'last.part');(out/'last.part').replace(out/'last.pt')
        print('CHECKPOINT',dict(epoch=epoch,batch_in_epoch=batch_in_epoch,step=step),flush=True)
    if world>1:dist.barrier()

@torch.no_grad()
def parameter_samples(model):
    return {name:p.flatten()[(torch.arange(min(32,p.numel()),device=p.device,dtype=torch.long)*(p.numel()-1))//max(1,min(32,p.numel())-1)].cpu().clone()
            for name,p in model.named_parameters()}

@torch.no_grad()
def report_updates(model,opt,before,step):
    after=parameter_samples(model);report={}
    for name,p in model.named_parameters():
        key='codebook' if name=='text.codebook' else name.split('.')[0]
        r=report.setdefault(key,dict(sampled_values=0,changed_values=0,max_abs_delta=0.,parameters=0,adam_active=0))
        delta=(after[name]-before[name]).abs()
        r['sampled_values']+=delta.numel();r['changed_values']+=int((delta>0).sum())
        r['max_abs_delta']=max(r['max_abs_delta'],float(delta.max()));r['parameters']+=1
        moment=opt.state.get(p,{}).get('exp_avg')
        r['adam_active']+=int(moment is not None and bool(torch.count_nonzero(moment)))
    print('PARAMETER_UPDATES',dict(step=step,modules=report,scale=model.scale.tolist(),bias=model.bias.tolist()),flush=True)

def augment(spec,rng):
    # Approximate strengths/probabilities; independent of target times.
    if rng.random()<.3:
        freq,time_bins=spec.shape[-2:]
        axis=torch.linspace(-1,1,freq,device=spec.device)
        shift=axis+rng.uniform(-.12,.12)*(1-axis.square())
        tt,ff=torch.meshgrid(torch.linspace(-1,1,time_bins,device=spec.device),shift,indexing='xy')
        spec=F.grid_sample(spec,torch.stack([tt,ff],-1).expand(len(spec),-1,-1,-1),align_corners=True,padding_mode='border')
    if rng.random()<.3:
        gain=F.interpolate(torch.randn(len(spec),1,5,device=spec.device)*.08,size=64,mode='linear',align_corners=True)
        spec=spec+gain.unsqueeze(-1)
    if len(spec)>1 and rng.random()<.3:
        mean=spec.mean(dim=-1,keepdim=True)
        std=spec.std(dim=-1,keepdim=True,unbiased=False).clamp_min(1e-5)
        w=rng.random()
        spec=(spec-mean)/std*(std*w+std.roll(1,0)*(1-w))+mean*w+mean.roll(1,0)*(1-w)
    return spec

def update(model,optimizer,rows,corpus,rank,world,step,micro,seed):
    torch.manual_seed(seed+step*world+rank)
    rng=random.Random(seed+step)
    bank=query_bank(rows)
    prompts=[];slices={}
    for src,labels in bank.items():
        start=len(prompts)
        prompts.extend((rng.choice(TEMPLATES) if src==0 else 'sound of {}').format(x.replace('_',' ').lower()) for x in labels)
        slices[src]=slice(start,len(prompts))
    # Build all denominators over the logical batch BEFORE splitting frames across devices.
    ys=[targets(r,bank,corpus.ancestors) for r in rows]
    pos=[0.,0.,0.];mass=[0.,0.,0.]
    for row,y in zip(rows,ys):
        branch=0 if row['src']==0 else (2 if y.ndim==1 else 1)
        pos[branch]+=float(y.sum());mass[branch]+=y.numel()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        live=model.text(prompts)
    q=live.detach().float().requires_grad_()
    loss_value=torch.zeros((),device=q.device)
    # All ranks see the same logical batch. Only the owning rank decodes each waveform.
    mine=list(range(rank,len(rows),world))
    for i in range(0,len(mine),micro):
        indices=mine[i:i+micro]
        audio=torch.stack([wave(rows[j]) for j in indices]).cuda()
        labels=[ys[j] for j in indices]
        sources=[rows[j]['src'] for j in indices]
        mix_groups=[]
        for src in set(sources):
            group=[k for k,s in enumerate(sources) if s==src]
            if len(group)>1 and rng.random()<.5:
                lam=rng.betavariate(.4,.4); mode=rng.choice(['wave','spec']);g=torch.tensor(group,device=q.device)
                mix_groups.append((g,lam,mode))
                mixed=[labels[k]*lam+labels[group[(idx-1)%len(group)]]*(1-lam) for idx,k in enumerate(group)]
                for k,v in zip(group,mixed):labels[k]=v
                if mode=='wave':audio[g]=lam*audio[g]+(1-lam)*audio[g.roll(1)]
        spec=model.audio.mel_forward(audio)
        for g,lam,mode in mix_groups:
            if mode=='spec':spec[g]=lam*spec[g]+(1-lam)*spec[g.roll(1)]
        spec=augment(spec,rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            frames,clips=model.encode(spec)
            objective=frames.sum()*0
            for src in set(sources):
                g=[k for k,s in enumerate(sources) if s==src]
                weak=src in (2,4);y=torch.stack([labels[k] for k in g]).to(q.device)
                logits=model.scores(clips[g] if weak else frames[g],q[slices[src]],weak)
                lp,ln=focal_terms(logits,y)
                if src==0:objective=objective+.9*(lp.sum()+ln.sum())/max(mass[0],1)
                else:
                    branch=2 if weak else 1
                    objective=objective+.1*(.05*lp.sum()/max(pos[branch],1)+.95*ln.sum()/max(mass[branch]-pos[branch],1))
        objective.backward();loss_value+=objective.detach()
    live.backward(q.grad)
    if world>1:
        # Sum locally normalized contributions, preserving the full-batch objective.
        for p in model.parameters():
            if p.grad is None:p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad)
        dist.all_reduce(loss_value)
    if step==0 or (step+1)%100==0:
        norms=torch.stack([p.grad.norm() for p in model.temporal.parameters()])
        active=int((norms>0).sum())
        if rank==0:
            print('BiGRU',dict(step=step+1,active_groups=active,groups=len(norms),grad_norm=norms.norm().item()),flush=True)
        if step==0 and active!=len(norms):
            raise RuntimeError('Incomplete BiGRU gradients on the first logical batch; check precision and frame supervision.')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not loss_value.isfinite():raise FloatingPointError('nonfinite objective')
    optimizer.step()
    return float(loss_value),float(norm)

@torch.no_grad()
def validate(model,items,limit):
    sys.path.insert(0,'/work/zhanghc/Myllm/FineLAP/dependencies/scoring')
    from sed_scores_eval.base_modules.scores import create_score_dataframe
    from sed_scores_eval.intersection_based import psds
    import soundfile as sf
    available=sorted(items)
    if limit:available=random.Random(888).sample(available,min(limit,len(available)))
    model.eval()
    common=set(CLASSES)&set(as_strong_eval_classes)
    gts={};dur={};rows=[]
    for key in available:
        path,ann=items[key];info=sf.info(path);length=min(10,info.duration)
        labels=[(max(0,a),min(b,length),c) for a,b,c in ann if c in common and min(b,length)>max(0,a)]
        if not labels or length<=0:continue
        merged=[]
        for a,b,c in sorted(labels,key=lambda event:(event[2],event[0],event[1])):
            if merged and merged[-1][2]==c and a<=merged[-1][1]:
                merged[-1]=(merged[-1][0],max(b,merged[-1][1]),c)
            else:merged.append((a,b,c))
        gts[key]=merged;dur[key]=length;rows.append((key,dict(path=path,sr=info.samplerate,start=0)))
    classes=sorted({c for lab in gts.values() for _,_,c in lab})
    if not limit:assert len(classes)==407,len(classes)
    q=model.text(['sound of '+c.replace('_',' ').lower() for c in classes])
    scores={}
    for i in range(0,len(rows),12):
        part=rows[i:i+12];x=torch.stack([wave(r) for _,r in part]).cuda()
        fr,_=model.encode(model.audio.mel_forward(x))
        s=model.scores(fr,q,False).sigmoid().cpu().numpy()
        for (key,_),v in zip(part,s):
            n=min(250,int(np.ceil(dur[key]*25)));ts=np.minimum(np.arange(n+1)/25,dur[key])
            scores[key]=create_score_dataframe(median_filter(v[:n],size=(9,1)),ts,classes)
    value=psds(scores,gts,dur,dtc_threshold=.7,gtc_threshold=.7,cttc_threshold=None,alpha_ct=0,alpha_st=1,max_efpr=100,num_jobs=1)[0]
    return dict(psds1=float(value),files=len(rows),classes=len(classes),subset=bool(limit))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True);ap.add_argument('--batch',type=int,default=1024)
    ap.add_argument('--micro',type=int,default=8);ap.add_argument('--steps',type=int,default=0,help='nonzero: pilot only')
    ap.add_argument('--val-limit',type=int,default=0);ap.add_argument('--seed',type=int,default=31415926)
    ap.add_argument('--initial-scale',type=float,default=10.)
    ap.add_argument('--initial-bias',type=float,default=-10.)
    ap.add_argument('--save-every',type=int,default=50,help='Overwrite last.pt every N optimizer steps')
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));torch.cuda.set_device(int(os.getenv('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(hours=12))
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed%2**32);random.seed(a.seed)
    out=ROOT/'runs'/a.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Run exists; choose a new name or --resume')
    if not a.steps and a.val_limit:raise ValueError('Full runs use full available AS-S validation')
    if a.save_every<1:raise ValueError('--save-every must be positive')
    val,heldout=validation();ds=Corpus(heldout)
    calibration=dict(initial_scale=a.initial_scale,initial_bias=a.initial_bias)
    m=COSED(**calibration).cuda();ema=copy.deepcopy(m).eval().requires_grad_(False)
    blocks=[list(m.audio.parameters()),list(m.text.parameters()),list(m.temporal.parameters())+list(m.clip.parameters()),[m.bias,m.scale]]
    rates=[1.96e-5,1.54e-5,4.41e-4,1.47e-3]
    opt=torch.optim.AdamW([dict(params=p,lr=lr) for p,lr in zip(blocks,rates)],weight_decay=.001)
    assert m.text.codebook.requires_grad
    assert sum(p is m.text.codebook for g in opt.param_groups for p in g['params'])==1
    step=0;first=0;start_batch=0;best=-1.;patience=0;per_epoch=len(ds.indices)//a.batch
    training_config=dict(batch=a.batch,micro=a.micro,world=world,seed=a.seed,per_epoch=per_epoch)
    if a.resume:
        st=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        previous=st.get('calibration',dict(initial_scale=.1,initial_bias=-10.))
        if previous!=calibration:
            raise ValueError(f'Checkpoint initialization {previous} differs from requested {calibration}; use a new run for changed initialization.')
        if st.get('training_config',training_config)!=training_config:
            raise ValueError('Resume requires the same batch/micro/world/seed/dataset length')
        m.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer'])
        first,step,best,patience=[st[k] for k in ['epoch','step','best','patience']]
        start_batch=st.get('batch_in_epoch',0)
        if 'rng_states' in st:restore_rng(st['rng_states'][rank])
        del st
    if rank==0:
        save_json(out/'config.json',dict(**vars(a),calibration=calibration,codebook_finetuned=True))
        print('READY',dict(sources=ds.counts,val_files=len(val),params=sum(p.numel() for p in m.parameters()),batch=a.batch,micro=a.micro,world=world),flush=True)
    for epoch in range(first,40):
        m.train();indices=ds.indices.copy();random.Random(a.seed+epoch).shuffle(indices)
        for b in range(start_batch if epoch==first else 0,per_epoch):
            ix=indices[b*a.batch:(b+1)*a.batch]
            if a.steps and b==0:ix[:10]=[ds.starts[s]+j for s in range(5) for j in range(2)]
            rows=[ds.get(j,a.seed+step*a.batch+k) for k,j in enumerate(ix)]
            factor=min(1.,(step+1)/300) if step<300 else .5*(1+np.cos(np.pi*(step-300)/max(1,40*per_epoch-300)))
            for group,base in zip(opt.param_groups,rates):group['lr']=float(factor*base)
            check_update=rank==0 and (b==(start_batch if epoch==first else 0) or (step+1)%a.save_every==0)
            before=parameter_samples(m) if check_update else None
            loss,gn=update(m,opt,rows,ds,rank,world,step,a.micro,a.seed)
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(m.parameters()),.001)
            step+=1
            if check_update:report_updates(m,opt,before,step)
            if rank==0 and (step<=3 or step%10==0):
                status=dict(stage='training',epoch=epoch+1,step=step,loss=loss,grad_norm=gn,updated=time.time());save_json(out/'status.json',status);print(status,flush=True)
            if step==1 or step%a.save_every==0 or b+1==per_epoch or (a.steps and step>=a.steps):
                save_checkpoint(out,m,ema,opt,epoch,b+1,step,best,patience,calibration,rank,world,training_config)
            if a.steps and step>=a.steps:break
        if world>1:dist.barrier()
        if rank==0:
            save_json(out/'status.json',dict(stage='validating',epoch=epoch+1,step=step))
            metrics=validate(ema,val,a.val_limit)
            improved=metrics['psds1']>best;patience=0 if improved else patience+1;best=max(best,metrics['psds1'])
            if improved:
                torch.save(ema.state_dict(),out/'best.part');(out/'best.part').replace(out/'best.pt')
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,step=step,**metrics))+'\n')
            print('VALIDATED',metrics,flush=True)
        save_checkpoint(out,m,ema,opt,epoch+1,0,step,best,patience,calibration,rank,world,training_config)
        stop=torch.tensor(int(patience>=10 or bool(a.steps)),device='cuda')
        if world>1:dist.broadcast(stop,0)
        if stop.item():break
    if rank==0:save_json(out/'status.json',dict(stage='completed',pilot=bool(a.steps),step=step,best_as_psds1=best))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()

