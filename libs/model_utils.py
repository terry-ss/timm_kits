import torch
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from imutils.paths import list_images
import cv2
from enum import IntEnum
from collections import namedtuple
import sys
_cur=Path(__file__).absolute().parent
sys.path.insert(0,str(_cur))
#print(sys.path)
from to_onnx import export2onnx
from general import set_dir

def get_labels(label_path):
    df=pd.read_csv(label_path,header=None)
    if len(df.columns)==1:
        df.columns='label'
    elif len(df.columns)==2:
        df.columns=['label','index']
    else:
        raise RuntimeError('Unknow csv file')
    label=df.label.values
    classes=IntEnum('classLabels',tuple(label),start=0) 
    return classes

class Meter(object):
    def __init__(self,device='cpu'):
        self.preds=torch.tensor([]).to(device)
        self.gts=torch.tensor([]).to(device)
        #self.criterion=criterion
        self.value=0
        self.length=0
    
    def collect(self,value,batch_size):
        self.value+=value
        self.length+=batch_size
     
    @property
    def average(self):
        return self.value/self.length
           
    @torch.no_grad()
    def update(self, preds,gts,ids=None):
        preds=self.avoid_zero_dim(preds)
        gts=self.avoid_zero_dim(gts)       
        self.preds=torch.cat([self.preds,preds])    
        self.gts=torch.cat([self.gts,gts.long()])
        if ids is not None:
            ids=self.avoid_zero_dim(ids)
            self.ids=torch.cat([self.ids,ids])
                  
    @staticmethod
    def avoid_zero_dim(tensor):
        tensor=torch.as_tensor(tensor)
        if not tensor.size():
            tensor=tensor[None]
        return tensor
     
    @staticmethod
    def message(msg,epoch,print_freq):
        logging.info(msg)
        if epoch%print_freq==0:
            print(msg)
    
    @property
    def acc(self):
        return torch.mean((self.preds==self.gts)*1.0).item()
        
        
class Model_Saver():
    def __init__(self,mname,timestamp,
        val_thres=0.8,saved_model_number=2,direction='max',save_info=None):
        if direction=='min':
            self.val_thres=val_thres*-1
        elif direction=='max':
            self.val_thres=val_thres
        else:
            raise ValueError(dirction)
        self.saved_model_number=saved_model_number
        self.df=pd.DataFrame(columns=['fpath','epoch','val'])
        self.mname=mname
        self.timestamp=timestamp
        if save_info is None:
            self.save_info={'save_type':'.pt','mname':mname}
        else:
            self.save_info=save_info
            if save_info['save_type']=='.onnx':
                self.save_info['input_size']=self.get_input_size(args)
            else:
                self.save_info['mname']=mname
        
    def save_model_request(self,model,epoch,val):
        if val>self.val_thres:

            fname=self.mname+'_'+str(epoch)+self.save_info['save_type']
            fpath=f'./output/{self.timestamp}/{fname}'
            fpath=self.save_action(model,fpath,
                save_info=self.save_info)
            ss=dict(fpath=fpath,epoch=epoch, val=val)
            self.df=self.df.append(ss,ignore_index=True)
            self.val_thres=min(self.df.val.tolist())
        if self.df.fpath.count()>self.saved_model_number:
            index=self.df.val.idxmin()
            rm_ind=self.df.loc[self.df.val.idxmin()]
            Path(rm_ind.fpath).unlink()
            self.df.drop(index,inplace=True)
     
    @staticmethod
    def save_action(model,fpath,save_info):
        fpath=str(fpath)
        if fpath.endswith('.pt'):
            save_dict={'state_dict':model.state_dict(),'mname':save_info['mname']}
            torch.save(save_dict,fpath)
        elif fpath.endswith('.onnx'):
            input_size=save_info['input_size']
            fpath=export2onnx(model,input_size,fpath,
                if_simplify=save_info['if_simplify'])
        else:
            raise ValueError(fpath)
        return fpath
    
    @classmethod
    def convert2onnx(cls,model,args):
        #mpath should refer the model.
        fpath=Path(args.mpath).with_suffix('.onnx')
        input_size=cls(None,None).get_input_size(args)
        save_info={'input_size':input_size,'if_simplify':True}
        cls(None,None).save_action(model,fpath,save_info)
    
    @staticmethod
    def get_input_size(args):
        input_size=namedtuple('input','c h w')
        return input_size(*(3, args.shape[0], args.shape[1]))
           
    def __repr__(self):
        return self.df.to_string()
            
class History:
    def __init__(self,mname):
        self.data={'train':{'loss':[],'score':[]},
            'val':{'loss':[],'score':[]}}
        self.mname=mname
        
    def collect(self,loss,score,stage):
        self.data[stage]['loss'].append(loss)
        self.data[stage]['score'].append(score)
        
    def plot(self,save_path,title=None):
        tloss=self.data['train']['loss']
        vloss=self.data['val']['loss']
        tscore=self.data['train']['score']
        vscore=self.data['val']['score']
        fig=plt.figure()
        N=len(tloss)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, N), tloss, '-',label="train_loss")
        ax.plot(np.arange(0, N), vloss, '-',label="val_loss")
        ax2=ax.twinx()
        ax2.plot(np.arange(0, N), vscore, '-r',label="val_score")
        ax2.plot(np.arange(0, N), tscore, '-b',label="train_score")
        if title is None:
            title='Training on '+self.mname
        plt.title(title)
        plt.xlabel("Epoch #")
        ax.set_xlabel("Loss")
        ax2.set_xlabel("Score")
        ax2.set_ylim(0.5,1)
        ax.legend(loc=1)
        ax2.legend(loc=2)
        plt.savefig(save_path)
        
def scan_img(path,model_f,transorm,save_dir):
    pbar=tqdm(list_images(path))
    create=0
    for img_file in pbar:
        
        image=cv2.imread(img_file)
        img_tensor=transorm(image=image)['image'][None].float()
        with torch.no_grad():
            features=model_f.backbone(img_tensor)
        stem=Path(img_file).stem
        for i,feature in enumerate(features):
            gray=torch.sigmoid(torch.mean(feature.squeeze(),dim=0)).numpy()
            gray=np.array(gray*255,dtype=np.uint8)
            cv2.imwrite(f'{save_dir}/{stem}_{i}.jpg',gray)
 
       
def get_features(args,img_dir,model_f,transorm):

    save_dir=Path(args.mpath).parent/'feature'
    set_dir(save_dir)
    scan_img(img_dir,model_f,transorm,save_dir)
    print('feature maps exported')
       