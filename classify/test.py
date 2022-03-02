import torch
import pandas as pd
import numpy as np
import time
import traceback
import torch.utils.data
from pathlib import Path
import os,sys
import cv2
import yaml
from imutils.paths import list_images
from tqdm import tqdm
import argparse
import albumentations as A
from sklearn.metrics import  classification_report

try:
    import pretty_errors
    pretty_errors.configure(display_locals = True)
except ImportError:
    pass
import warnings
#warnings.filterwarnings("ignore",)
from rich import print,console
#from rich.progress import track
console = console.Console()
from model import build_model
from dataset import create_transform,filter_out
from importlib import machinery
_cur=Path(__file__).absolute().parent
machinery.SourceFileLoader('general',str(_cur/'../libs/general.py')).load_module()
machinery.SourceFileLoader('m_utils',str(_cur/'../libs/model_utils.py')).load_module()
from general import clock,create_log,seed_everything,get_args,set_dir
from m_utils import Meter, Model_Saver,History,get_labels,get_features
   
         
def test(args, model):  
    files=list(list_images(args.test_path))
    files=filter_out(files,classLabels)
    trfm_train,trfm_val=create_transform(args) 
    df=pd.DataFrame(columns=('filename','pred','score'))
    pbar=tqdm(files,)
    for file in pbar:
        img=cv2.imread(file)
        y,score=m_detect(model,img,trfm_val)
        if np.max(score)<args.tta:
            y,score=tta(model,img,trfm_train)
        name=Path(file).name
        new={'filename':name,'pred':y,'score':score}
        df=df.append(new, ignore_index=True)
        
    if args.calculate:
        gt=[]
        for file in files:
            label=Path(file).parts[-2]
            label=int(classLabels[label])
            gt.append(label)
        df['gt']=(gt)
    df = df.infer_objects()
    df=df.sort_values(['filename'])
    print(df.tail())
    report_scores(df)
    df.to_csv('output/test_result.csv')

@torch.inference_mode()    
def m_detect(model,img,transform):
    img=transform(image=img)['image']
    img=img[None].float()
    out=model(img);
    score= torch.softmax(out,1)
    y=torch.max(score,1)[1].item()
    score=torch.max(score).item()
    return y,score
 
def tta(model,img,transform):
    ys,scores=[],[]
    for f in range(5):
        _,score=m_detect(model,img,transform)
        scores.append(score)
    #score=np.mean(scores,axis=0).squeeze()
    score=np.mean(scores)
    y=np.argmax(score)
    return y,score
    
def report_scores(df):
    gt=df['gt'];pred=df['pred']
    targets=[x for x in classLabels.__members__]
    repo=classification_report(gt,pred,target_names=targets) 
    print(repo)   

if __name__ == '__main__':
    args=get_args('config.yaml',sys.argv)
    if args.mpath == 'required':
        raise RuntimeError('Need to specify the model path')
    elif not args.mpath.endswith('.pt'):
        raise RuntimeError('Currently support .pt only')
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    mname='default'
    classLabels=get_labels('labels.csv')
    num_classes=len(classLabels)
    model=build_model(mname,num_classes,chechpoint=args.mpath) 
    model.eval()
    #normalizer=False 
    test(args, model)
    if args.if_feature:
        model_f=build_model(mname,num_classes,chechpoint=args.mpath,feature=True) 
        _,trfm_val=create_transform(args) 
        img_dir=args.test_path
        get_features(args,img_dir,model_f,trfm_val)
    Model_Saver.convert2onnx(model,args)
    
