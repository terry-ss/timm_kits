'''
dirs:{
:data: for image and label
  train\, val\:in part train mode, train in train/. in full train mode, train in both.
labels are supposed to be stored in a yaml file,
while the Yaml file is structured as {file_name:[label1,label2,...],...}  

:output: saved model and others
:logs: saving logs and plots

'''

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

import argparse
import cv2
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
#import json
from enum import IntEnum
try:
    import pretty_errors
    pretty_errors.configure(
        display_locals      = True
    )
except ImportError:
    pass
import warnings
#warnings.filterwarnings("ignore",)
from rich import print,console
console = console.Console()
from model import build_model
from dataset import prepare_data
from importlib import machinery
_cur=Path(__file__).absolute().parent
machinery.SourceFileLoader('general',str(_cur/'../libs/general.py')).load_module()
machinery.SourceFileLoader('m_utils',str(_cur/'../libs/model_utils.py')).load_module()
from general import clock,create_log,seed_everything
from m_utils import Meter, Model_Saver,History, get_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Multi-Label network')
    parser.add_argument('--gpu', dest='gpu',help='gpu ID',default='0', type=str)
    parser.add_argument('--epoch', dest='epoch',default=41, type=int)
    parser.add_argument('--epoch_freeze', dest='epoch_freeze',default=20, type=int)
    parser.add_argument('--bs', dest='bs',help='batch size',default=32, type=int)
    parser.add_argument('--cp', dest='check',help='chechpoint,or pool/pretrained.pt ',default=None)
    parser.add_argument('--lr', dest='lr',help='learning rate, 0 means set by this script',default=0, type=float)
    parser.add_argument('--tm', dest='tm',help='train mode',default='part', type=str)
    parser.add_argument('--pretrained', dest='pretrained',default=1, type=int)
    parser.add_argument('--catch', dest='catch',help='send save model request',default=1, type=int)
    parser.add_argument('--shape', dest='shape',help='img resize',nargs='+',type=int )
    parser.add_argument('--mname', dest='mname',help='model name',type=str)
    parser.add_argument('--print_freq', dest='freq',type=int)
    parser.add_argument('--thre', dest='thre',help='threshold',type=float)
    args = parser.parse_args()
    return args 
        

def train_epoch(loader,epoch,args):
    model.train()
    meter=Meter(device)
    scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (images, target) in enumerate(loader):       
        images=images.to(device).float();target=target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images)  
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        pred=torch.sigmoid(output)>args.thre
        meter.update(pred,target)
        meter.collect(loss.item(),target.shape[0])
    acc=meter.acc
    loss=meter.average
    msg=f'Train Epoch{epoch}, Loss:{loss:.2f}, Acc:{acc:.2f}'
    meter.message(msg,epoch,args.freq)
    return loss,acc

def val_epoch(loader,epoch, args):
    meter=Meter(device)
    model.eval()
    with torch.inference_mode():
        for batch_idx, (images, target) in enumerate(loader):
            images=images.to(device).float();target=target.to(device)
            output = model(images)
            loss = criterion(output, target)
            pred=torch.sigmoid(output)>args.thre
            meter.update(pred,target)
            meter.collect(loss.item(),target.shape[0])
    acc=meter.acc
    loss=meter.average
    msg=f'Val Epoch{epoch}, Loss:{loss:.2f}, Acc:{acc:.2f}'
    meter.message(msg,epoch,args.freq)
    return loss,acc

@clock
def fit(history,ms,args,train_loader,val_loader,epoch_s,epoch_e):
    for epoch in range(epoch_s, epoch_e):
        tloss,tacc=train_epoch(train_loader,epoch,args)
        vloss,vacc=val_epoch(val_loader,epoch,args)
        history.collect(tloss,tacc,'train')
        history.collect(vloss,vacc,'val')   
        if args.catch:
            ms.save_model_request(model,epoch,vacc)
    
def hunt(args):
    global model,  optimizer, criterion
    criterion=nn.BCEWithLogitsLoss(weight=None)
    mname=args.mname
    ms=Model_Saver(mname,timestamp)
    checkpoint=args.check
    model = build_model(mname,num_classes,args.pretrained,checkpoint)
    torch.backends.cudnn.benchmark = True
    model = model.to(device)
    if args.lr==0:
        lr=1e-3
        if checkpoint.lower() != 'none':
            lr=lr*0.01
    else:
        lr=args.lr
    train_loss,train_acc, val_loss, val_acc=[],[],[],[]
    print('='*10+mname+'='*10)

    epoch_freeze=args.epoch_freeze
    history=History(mname)
    if epoch_freeze>0:
        print('"freezing epoch"')
        for p in model.named_parameters():
            if 'backbone'  in p[0]:
                p[1].requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        fit(history,ms,args,train_loader,val_loader,0,epoch_freeze)
        
    if args.epoch>epoch_freeze:
        print('"unfreezing epoch"')
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.01, betas=(0.9, 0.99))
        fit(history,ms,args,train_loader,val_loader,epoch_freeze,args.epoch)
                  
    print(ms)
    history.plot(f'{_cur}/output/{timestamp}/result.jpg')  

if __name__ == '__main__':
    classLabels=get_labels('labels.csv')
    num_classes=len(classLabels)
    seed_everything(42)
    timestamp=create_log(level=logging.INFO)
    args = parse_args()  
    logging.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device='cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)
    train_loader,val_loader=prepare_data(args,classLabels)    
    hunt(args)
    

