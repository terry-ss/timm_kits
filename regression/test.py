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
import random
try:
    import pretty_errors
    pretty_errors.configure(display_locals = True)
except ImportError:
    pass
import warnings
#warnings.filterwarnings("ignore",)
from rich import print,console
from rich.progress import track
console = console.Console()
from model import build_model
from dataset import create_transform
from importlib import machinery
_cur=Path(__file__).absolute().parent
machinery.SourceFileLoader('general',str(_cur/'../libs/general.py')).load_module()
machinery.SourceFileLoader('m_utils',str(_cur/'../libs/model_utils.py')).load_module()
from general import seed_everything,get_args
from m_utils import get_labels,Model_Saver, get_features
   
         
def test(args, model):  
    files=list(list_images(args.test_path))
    files.sort()
    trfm_train,trfm_val=create_transform(args) 
    df=pd.DataFrame(columns=('filename','pred'))
    for file in track(files):
        img=cv2.imread(file)
        y=m_detect(model,img,trfm_val)
        if random.random()<args.tta:
            y=tta(model,img,trfm_train)
        name=Path(file).name
        new={'filename':name,'pred':y}
        df=df.append(new, ignore_index=True)
        
    if args.calculate:
        gt=[]
        for file in files:
            label=np.array(labels[Path(file).name])
            gt.append(label)
        df['gt']=(gt)
    df = df.infer_objects()
    df=df.sort_values(['filename'])
    print(df.tail())
    report_scores(df)
    df.to_csv('outputs/test_result.csv')

@torch.inference_mode()    
def m_detect(model,img,transform):
    img=transform(image=img)['image']
    img=img[None].float()
    out=model(img);
    y=out.squeeze().tolist()
    return y
 
def tta(model,img,transform):
    ys=[]
    for f in range(5):
        out=m_detect(model,img,transform)
        ys.append(out)
    y=np.mean(ys,axis=0).squeeze()
    return list(y)
    
def report_scores(df):
    gt=np.array(df['gt'].to_list());pred=np.array(df['pred'].to_list())
    mae=np.mean(np.abs(gt-pred))
    std=np.std(np.abs(gt-pred))
    print(f'mae_score is {mae}, std is {std}')
               
    
if __name__ == '__main__':
    seed_everything(42)
    args=get_args('config.yaml',sys.argv)
    if args.mpath == 'required':
        raise RuntimeError('Need to specify the model path')
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    mname='default'
    classLabels=get_labels('data/labels_name.csv')
    with open('data/all_labels.yaml','r') as f:
        labels=yaml.safe_load(f)
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
    
