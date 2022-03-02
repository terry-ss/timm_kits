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
from general import clock,create_log,seed_everything
from m_utils import get_labels,Model_Saver,get_features

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--gpu', dest='gpu',help='gpu ID', type=str)
    parser.add_argument('--shape', dest='shape',help='img resize',nargs='+',type=int )
    parser.add_argument('--mname', dest='mname',help='model name',type=str)
    parser.add_argument('--mpath', dest='mpath',help='model path', type=str)
    parser.add_argument('--tta', dest='tta',help='tta trigger', type=float)
    parser.add_argument('--test',dest='test_path',type=str)
    parser.add_argument('--yaml',dest='yaml_path',type=str)
    parser.add_argument('--calculate',dest='c',help='calculate score or not',type=int)
    parser.add_argument('--thre', dest='thre',help='threshold',type=float)
    parser.add_argument('--if_feature', dest='if_feature',type=int)
    args = parser.parse_args()
    return args    
    
         
def test(args, model):  
    files=list(list_images(args.test_path))
    trfm_train,trfm_val=create_transform(args) 
    df=pd.DataFrame(columns=('filename','pred','score'))
    for file in track(files):
        img=cv2.imread(file)
        y,score=m_detect(model,img,trfm_val,args.thre)
        if max(abs(score-args.thre)[0])*2<args.tta:
            y,score=tta(model,img,trfm_train,args.thre)
        name=Path(file).name
        new={'filename':name,'pred':y,'score':score}
        df=df.append(new, ignore_index=True)
    if args.c:
        with open(args.yaml_path,'r') as f:
            labels=yaml.load(f,Loader=yaml.SafeLoader)
        gt=[]
        for file in files:
            name=Path(file).name
            label=[ x.name in labels[name] for x in classLabels]
            gt.append(label)
        df['gt']=(gt)
    df=df.sort_values(['filename'])
    df.to_csv('output/test_result.csv')
    print(df.tail())
    report_scores(df)

@torch.inference_mode()    
def m_detect(model,img,transform,threshold):
    img=transform(image=img)['image']
    img=img[None].float()
    out=model(img);
    score= torch.sigmoid(out)
    y=score>threshold
    y=np.array(y.tolist());score=np.array(score.tolist())
    return y,score
 
def tta(model,img,transform,threshold):
    ys,scores=[],[]
    for f in range(5):
        _,score=m_detect(model,img,transform,threshold)
        scores.append(score)
    score=np.mean(scores,axis=0)
    y=score>threshold
    return y,score
    
def report_scores(df):
    t=f=0
    for row in df.itertuples():
        t+=np.sum(row.gt==row.pred)
        f+=np.sum(row.gt!=row.pred)
    acc=t/(t+f)
    print(f'accuracy_score is {acc}')
               
    
if __name__ == '__main__':
    args=parse_args()   
    if args.mpath == 'required':
        raise RuntimeError('Need to specify the model path')
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
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
    
