"""Continuous Tm/null adapter around the pinned original D3 promoter CNN."""
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn
from .unconditional import load_author_cnn


class TemperatureEncoder(nn.Module):
    """Lightweight continuous Tm embedding; receives a train-standardized scalar."""

    def __init__(self, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, standardized_tm):
        return self.mlp(standardized_tm)


class TmFeature(nn.Module):
    """Broadcast one scalar into the author's original signal channel."""
    def __init__(self,tm_mean,tm_std):
        super().__init__()
        if not tm_std>0:raise ValueError('Positive training Tm standard deviation required')
        self.register_buffer('tm_mean',torch.tensor(float(tm_mean)))
        self.register_buffer('tm_std',torch.tensor(float(tm_std)))
        self.condition_encoder=TemperatureEncoder(8)
        self.projection=nn.Linear(8,1)
        self.null_embedding=nn.Parameter(torch.zeros(1))

    def forward(self,payload,count,length):
        if payload is None:
            value=self.null_embedding.reshape(1,1).expand(count,1)
        else:
            if tuple(payload.shape)!=(count,2):raise ValueError('Expected [batch,2]: raw Celsius, presence0/1')
            payload=payload.to(self.tm_mean)
            tm,flag=payload[:,:1],payload[:,1:2]
            if not bool(((flag==0)|(flag==1)).all()):raise ValueError('Presence flag must be0 or1')
            present=flag==1
            if not bool(torch.isfinite(tm[present]).all()):raise ValueError('Nonfinite present Tm')
            standardized=torch.where(present,(tm-self.tm_mean)/self.tm_std,torch.zeros_like(tm))
            value=torch.where(present,self.projection(self.condition_encoder(standardized)),self.null_embedding.reshape(1,1))
        return value[:,None,:].expand(count,length,1)


def model_config(*,channels=128,time_embed_dim=210,sequence_length=49):
    return SimpleNamespace(tokens=4,graph=SimpleNamespace(type='uniform'),
        dataset=SimpleNamespace(num_classes=4,sequence_length=sequence_length,signal_dim=1),
        model=SimpleNamespace(conv_channels=channels,cond_dim=time_embed_dim,scale_by_sigma=False))


def make_d3_model(*,source_dir,tm_mean=0.,tm_std=1.,channels=128,time_embed_dim=210,sequence_length=49):
    """No downloads, environment installs, training or generation at construction.

    The generic upstream CNN mixes NLC/NCL signal axes. Use the author's exact
    PromoterConvolutionalModel.forward instead; no rewrite of that forward.
    sequence_length is metadata only. Actual input width is never padded/cropped.
    """
    original=load_author_cnn(Path(source_dir))
    if sequence_length not in [49,512,1024]:raise ValueError('Unknown length regime')
    config=model_config(channels=channels,time_embed_dim=time_embed_dim,sequence_length=sequence_length)
    class TmD3CNN(original.ConvolutionalModel):
        def __init__(self):
            super().__init__(config)
            self.tm_feature=TmFeature(tm_mean,tm_std)

        def forward(self,indices,labels=None,train=True,sigma=None):
            if (indices.ndim!=2 or indices.dtype!=torch.long or not 13<=indices.shape[1]<=sequence_length
                or (sequence_length!=49 and indices.shape[1]!=sequence_length)
                or not bool(((indices>=0)&(indices<4)).all())):
                raise ValueError('Expected exact-length A/C/G/T indices; no PAD/MASK/crop')
            if sigma is None or sigma.numel()!=len(indices) or not bool(torch.isfinite(sigma).all()):
                raise ValueError('One finite noise level per sequence required')
            features=self.tm_feature(labels,len(indices),indices.shape[1])
            return original.PromoterConvolutionalModel.forward(self,indices,features,train,sigma.reshape(-1))
    return TmD3CNN()
