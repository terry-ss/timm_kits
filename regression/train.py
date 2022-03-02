
import os,sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
#import timm
from pathlib import Path
import yaml
import gc
try:
    import pretty_errors
    pretty_errors.configure(display_locals= True)
except ImportError:
    pass
import warnings
#warnings.filterwarnings("ignore",)
from rich import print,console
console = console.Console()
from importlib import machinery
_cur=Path(__file__).absolute().parent
machinery.SourceFileLoader('general',str(_cur/'../libs/general.py')).load_module()
machinery.SourceFileLoader('m_utils',str(_cur/'../libs/model_utils.py')).load_module()
from general import get_args,set_dir
from m_utils import get_labels
from dataset import prepare_data
from model import build_model

class LitModel(LightningModule):
    def __init__(self,args):
        super().__init__()
        self.save_hyperparameters()
        in_chans=3 #RGB
        in_features=1000
        self.lr=args.lr
        self.model=build_model(args.mname,args.num_classes,args.pretrained)
        if args.freeze:
            for p in model.named_parameters():
                if 'backbone'  in p[0]:
                    p[1].requires_grad = False
        self.criterion=nn.SmoothL1Loss()
        self.measure=nn.L1Loss()
        
    def forward(self,x):
        y=self.model(x)
        return y 
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        out = self(image.float())
        loss = self.criterion(out,label)
        self.log("train_loss", loss)
        return {'loss':loss}
        
    def validation_step(self, batch, batch_idx):

        image, label = batch
        out = self(image.float())
        loss = self.criterion(out,label)
        score = self.c(out,label)
        self.log("val_loss", loss)
        self.log('val_score',score)
        return {'loss':loss,'val_score':score}
    
    @staticmethod
    def c(gt,pred):
        return torch.mean(torch.abs(gt-pred))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer':optimizer}
    

 
def hunt(args,if_catch=False):
    checkpoint=args.check
    train_loader,val_loader=prepare_data(args,'data/all_labels.yaml')       
    torch.backends.cudnn.benchmark = True
    encoder=LitModel(args)
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="min",
        monitor="val_loss",dirpath='outputs',save_last=False,save_top_k=1)
    if args.check.lower()=='none':
        check_path=None
    else:
        check_path=args.check
    trainer=pl.Trainer(gpus=args.gpu_num,strategy='dp',
        max_epochs=args.epoch,
        auto_lr_find=False,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            #RichProgressBar(text_color='green'),
        ],  
        resume_from_checkpoint=check_path,
        log_every_n_steps=10,
        precision=16)
    trainer.fit(encoder,train_loader,val_loader)

    # del train_loader, val_loader
    # gc.collect()
    #trainer.predict()


if __name__ == '__main__':
    args=get_args('config.yaml',sys.argv)
    device='cuda' if torch.cuda.is_available() else 'cpu' 
    print(f'{device}')
    classLabels=get_labels('data/labels_name.csv')
    args['num_classes']=len(classLabels)
    pl.seed_everything(42)
    set_dir('outputs')
    print(args)
    hunt(args)
