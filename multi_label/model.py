import torch
from torch import nn
import timm
import pandas as pd


def build_model(mname:str,num_classes:int,pretrained=False,chechpoint='none'):
    if chechpoint.lower() =='none':
        model = M_model(mname,num_classes,pretrained)
    else:
        dict_=torch.load(chechpoint)
        name=dict_['mname']
        if name != mname and mname !='default':
            raise RuntimeError(mname)
        model = M_model(name,num_classes,pretrained=False)
        model.load_state_dict(dict_['state_dict'],strict=True)
    return model
    
class M_model(nn.Module):
    def __init__(self,mname,num_classes,pretrained,in_chans=3):
        super(M_model, self).__init__()
        self.in_chans=in_chans
        in_features=1000
        self.backbone = timm.create_model(mname, num_classes=in_features,
            pretrained=pretrained, in_chans=in_chans)
        self.linear=nn.Linear(in_features=in_features, 
            out_features=num_classes)
        
    def forward(self,x):
        features=self.backbone(x)
        y=self.linear(features)
        return y #torch.sigmoid(y)
        
if __name__ == '__main__':
    pass