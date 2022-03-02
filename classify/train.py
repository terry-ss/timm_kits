'''
dirs:{
:data: for image and label
  ./train, ./val:in part train mode, train in ./train in full train mode, train in both.

:output: saved model and others
:pool: storing possible pretrained model

'''

import torch
import torch.nn as nn
import numpy as np
import os,sys
from pathlib import Path
#import hydra
import cv2
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
try:
    import pretty_errors
    pretty_errors.configure(display_locals= True)
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
from general import clock,create_log,seed_everything,get_args
from m_utils import Meter, Model_Saver,History, get_labels
   

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
        pred=torch.max(output,1)[1]
        meter.update(pred,target)
        bs=target.shape[0]
        meter.collect(loss.item()*bs,bs)
    acc=meter.acc
    loss=meter.average
    #print(vars(meter))
    msg=f'Train Epoch{epoch}, Loss:{loss:.2f}, Acc:{acc:.2f}'
    meter.message(msg,epoch,args.print_freq)
    return loss,acc

def train_boost_epoch(loader,epoch,args):
    
    meter=Meter(device)
    scaler = torch.cuda.amp.GradScaler()
    update=False;loss,acc=0,0
    for batch_idx, (images, target) in enumerate(loader): 
        model.eval()
        images=images.to(device).float();target=target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images)  
            pred=torch.max(output,1)[1]
            wrong_num=torch.sum(pred!=target)
            if wrong_num.item()==0:
                continue
            else:
                update=True
            target_=target[pred != target]
            output_=output[pred != target]
            loss = criterion(output_, target_)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()      
        meter.update(pred,target)
        bs=target.shape[0]
        meter.collect(loss.item()*bs,bs)
    acc=meter.acc
    loss=meter.average
    if update:
        msg=f'Train Boost Epoch{epoch}, Loss:{loss:.2f}, Acc:{acc:.2f}'
        meter.message(msg,epoch,args.print_freq)
    return loss,acc,update

def val_epoch(loader,epoch, args):
    meter=Meter(device)
    model.eval()
    with torch.inference_mode():
        for batch_idx, (images, target) in enumerate(loader):
            images=images.to(device).float();target=target.to(device)
            output = model(images)
            loss = criterion(output, target)
            pred=torch.max(output,1)[1]
            meter.update(pred,target)
            bs=target.shape[0]
            meter.collect(loss.item()*bs,bs)
    acc=meter.acc
    loss=meter.average
    msg=f'Val Epoch{epoch}, Loss:{loss:.2f}, Acc:{acc:.2f}'
    meter.message(msg,epoch,args.print_freq)
    return loss,acc

def fit(history,ms,args,train_loader,val_loader,epoch_s,epoch_e):
    for epoch in range(epoch_s, epoch_e):
        tloss,tacc=train_epoch(train_loader,epoch,args)
        if args.if_boost:
            bloss,bacc,update=train_boost_epoch(train_loader,epoch,args)
            if not update:
                bloss,bacc=tloss,tacc
            vloss,vacc=val_epoch(val_loader,epoch,args)
        history.collect(tloss,tacc,'train')
        history.collect(vloss,vacc,'val')   
        if args.catch:
            ms.save_model_request(model,epoch,vacc)   

def hunt(args):
    train_loader,val_loader=prepare_data(args,classLabels)    
    global model,  optimizer, criterion
    #criterion=nn.functional.binary_cross_entropy_with_logits
    criterion=nn.CrossEntropyLoss(reduction='mean')
    mname=args.mname
    save_info=args.save_info
    ms=Model_Saver(mname,timestamp,save_info=save_info)
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
    args=get_args('config.yaml',sys.argv)
    device='cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)
    timestamp=create_log(level=logging.INFO)
    classLabels=get_labels('labels.csv')
    logging.info(classLabels.__members__)
    logging.info(args)
    num_classes=len(classLabels)
    seed_everything(42)
    hunt(args)
    

