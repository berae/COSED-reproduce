"""Independent COSED model. Public upstream encoder weights; inferred widths documented in README."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import BertConfig,BertModel,BertTokenizer
from data import ROOT
from models.atstframe.ATSTF_wrapper import ATSTWrapper

class MGAText(nn.Module):
    def __init__(self):
        super().__init__()
        folder=ROOT/'weights/mga'
        self.tokenizer=BertTokenizer.from_pretrained(str(folder),local_files_only=True)
        self.bert=BertModel(BertConfig.from_pretrained(str(folder)),add_pooling_layer=False)
        self.project=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,1024),nn.GELU(),nn.LayerNorm(1024),nn.Linear(1024,1024))
        state=torch.load(folder/'mga.pt',map_location='cpu',weights_only=True)
        position_ids=state.pop('text_encoder.text_encoder.embeddings.position_ids',None)
        if position_ids is not None:
            assert torch.equal(position_ids,self.bert.embeddings.position_ids)
        self.bert.load_state_dict({k[len('text_encoder.text_encoder.'):]:v for k,v in state.items() if k.startswith('text_encoder.text_encoder.')},strict=True)
        self.project.load_state_dict({k[len('word_proj.'):]:v for k,v in state.items() if k.startswith('word_proj.')},strict=True)
        self.codebook=nn.Parameter(state['codebook'])
    def embed(self,ids,mask,valid):
        hidden=self.bert(input_ids=ids,attention_mask=mask).last_hidden_state[:,1:]
        words=self.project(hidden)
        logits=((words @ self.codebook.T) * valid[:,:,None] / (32*1000)).amax(1).float()
        z=logits-logits.amax(-1,keepdim=True)
        ordered=z.sort(-1,descending=True).values
        sums=ordered.cumsum(-1)
        k=torch.arange(1,z.size(-1)+1,device=z.device)
        support=(1+k*ordered>sums).sum(-1,keepdim=True)
        weights=(z-(sums.gather(1,support-1)-1)/support).clamp_min(0)
        return F.normalize(weights @ self.codebook,dim=-1)
    def forward(self,queries):
        out=[]
        for chunk in range(0,len(queries),32):
            toks=self.tokenizer(queries[chunk:chunk+32],padding=True,truncation=True,max_length=30,return_special_tokens_mask=True,return_tensors='pt').to(self.codebook.device)
            args=(toks.input_ids,toks.attention_mask,1-toks.special_tokens_mask[:,1:])
            out.append(checkpoint(self.embed,*args,use_reentrant=False) if self.training else self.embed(*args))
        return torch.cat(out)

class COSED(nn.Module):
    def __init__(self, initial_scale=10., initial_bias=-10.):
        super().__init__()
        self.audio=ATSTWrapper()
        ck=torch.load(ROOT/'weights/ATST-F_weak.pt',map_location='cpu',weights_only=True)
        prefix='model.atst.'
        self.audio.atst.load_state_dict({k[len(prefix):]:v for k,v in ck.items() if k.startswith(prefix)},strict=True)
        self.text=MGAText()
        self.temporal=nn.GRU(768,512,num_layers=2,batch_first=True,bidirectional=True)
        self.clip=nn.Sequential(nn.Linear(768,1024),nn.ReLU(),nn.Linear(1024,1024))
        # Author clarification (2026-11-09): direct multiplier starts at 10.
        # Explicit arguments preserve historical scale=0.1 audit reproduction.
        self.scale=nn.Parameter(torch.full((2,), float(initial_scale)))
        self.bias=nn.Parameter(torch.full((2,), float(initial_bias)))
    def encode(self,spec):
        frames=self.audio(spec)
        assert frames.shape[1:]==(250,768),frames.shape
        # cuDNN on rabbit02 casts the GRU to FP16 even under BF16 autocast.
        # Globally normalized frame gradients then underflow before accumulation.
        with torch.autocast(device_type=frames.device.type, enabled=False):
            temporal_frames = self.temporal(frames.float())[0]
            frame_embeddings = F.normalize(temporal_frames, dim=-1)
        return frame_embeddings,F.normalize(self.clip(frames.mean(1)),dim=-1)
    def scores(self,a,q,clip):
        branch=int(clip)
        return (a @ q.T).float()*self.scale[branch]+self.bias[branch]

def focal_terms(logits,y):
    p=logits.sigmoid()
    return -F.logsigmoid(logits)*(1-p)*y,-F.logsigmoid(-logits)*p*(1-y)

