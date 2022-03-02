import torch
from torch import nn
import timm
import pandas as pd


def build_model(mname:str,num_classes:int,pretrained=False,chechpoint='none',**kwarg):
    if 'feature' in kwarg.keys():
        feature=kwarg['feature']
    else:
        feature=False
    if chechpoint.lower() =='none':
        model = C_model(mname,num_classes,pretrained)
    else:
        dict_=torch.load(chechpoint)
        name=dict_['mname']
        if name != mname and mname !='default':
            raise RuntimeError(mname)
        
        model = C_model(name,num_classes,pretrained=False,feature=feature)
        model.load_state_dict(dict_['state_dict'],strict=not feature)
    return model
    
class C_model(nn.Module):
    def __init__(self,mname,num_classes,pretrained,in_chans=3,feature=False):
        super(C_model, self).__init__()
        self.in_chans=in_chans
        in_features=1000
        self.backbone = timm.create_model(mname, num_classes=in_features,
            pretrained=pretrained, in_chans=in_chans,features_only=feature)
        self.linear=nn.Linear(in_features=in_features, 
            out_features=num_classes)
        
    def forward(self,x):
        features=self.backbone(x)
        y=self.linear(features)
        return y #torch.sigmoid(y)
        
if __name__ == '__main__':
    pass
