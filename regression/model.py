import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pathlib import Path
import timm
import pandas as pd

def build_model(mname:str,num_classes:int,pretrained=False,chechpoint='none'):
    if chechpoint.lower() =='none':
        model = R_model(mname,num_classes,pretrained)
    else:
        if not Path(chechpoint).is_file():
            raise FileNotFoundError(chechpoint)
        dict_=torch.load(chechpoint)
        name=dict_['hyper_parameters']['args']['mname']
        if name != mname and mname !='default':
            raise RuntimeError(mname)
        model = R_model(name,num_classes,pretrained=False)
        state_dict=dict_['state_dict'];clear_dict={}
        for key in state_dict.keys():
            clear_dict[key.removeprefix('model.')]=state_dict[key]
        model.load_state_dict(clear_dict,strict=True)
    return model
    
class R_model(LightningModule):
    def __init__(self,mname,num_classes,pretrained,in_chans=3):
        super().__init__()
        self.in_chans=in_chans
        in_features=1000
        self.backbone = timm.create_model(mname, num_classes=in_features,
            pretrained=pretrained, in_chans=in_chans)
        self.linear=nn.Linear(in_features=in_features, 
            out_features=num_classes)
        
    def forward(self,x):
        features=self.backbone(x)
        y=self.linear(features)
        return y 
        
if __name__ == '__main__':
    pass
