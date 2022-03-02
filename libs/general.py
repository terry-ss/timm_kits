import shutil
import time
from pathlib import Path
from functools import wraps
import logging
import random,os
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict  
import warnings
from pprint import pprint
from collections import Counter

def clock(fn):
    @wraps(fn)
    def wrapper(*args,**kwargs):
        '''ths is decorator'''
        #print(fn.__name__)
        start = time.time()
        res=fn(*args)
        end = time.time()
        cost=(end - start)
        if cost>60:
            hour=cost//3600
            minute=cost//60 - hour*60
            second=cost%60
            print(' time cost: %dh %dm %ds'%(hour, minute,second))
        else:
            print(f' time cost: {cost:.2f}s')
        return res
    return wrapper

def set_dir(filepath,mode='whole'):
    if mode=='whole':
        p=Path(filepath)
        p.mkdir(exist_ok=True,parents=True)
        shutil.rmtree(p)
        p.mkdir(exist_ok=True)
    elif mode=='file_only':
        p=Path(filepath)
        p.mkdir(exist_ok=True,parents=True)
        for file in p.rglob('*'):
            if file.is_file():
                file.unlink()
            
def create_log(level=logging.INFO):
    timestamp=time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    Path(f'output/{timestamp}').mkdir(exist_ok=True,parents=True) 
    logging.basicConfig(level=level,
                        filename=f'output/{timestamp}/train_logs.log',
                        filemode='w+',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    return timestamp

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_args(yaml_path,console=None):
    '''
    hydra-style argumentations
    '''
    with open(yaml_path,'r') as f:
        config=yaml.load(f,Loader=yaml.SafeLoader)
        #config=edict(config)
    #pprint(config)
    for key in ['env','common']:
        for k,v in config[key].items():
            config['common'][k]=_input(v)
    if console:
        invalid=[]
        for x in console[1:]:
            if x.startswith('env.'):
                x=x.removeprefix('env.')
                x0,x1=split_into_two_parts(x)
                config['env'][x0]=_input(x1)
            elif x.startswith('common.'):
                x=x.removeprefix('common.')
                x0,x1=split_into_two_parts(x)
                config['common'][x0]=_input(x1)
            else:
                invalid.append(x)
        if invalid:
            msg=f'unsupported args: {invalid}'
            warnings.warn(msg)
    args=edict(config['common']|config['env'])
    return args



def split_into_two_parts(x:str,split:str='='):
    list_=x.split(split)
    x0=list_[0]
    x1=split.join(list_[1:])
    return x0,x1
    
def _input(x):
    if not isinstance(x,str):
        return x
    if x.startswith('-'):
        x=x[1:]
        sign=-1
    else:
        sign=1
    if x.isdigit():
        x=int(x)
    elif x.replace('.','').isdigit():
        x=float(x)
    elif Counter(x)['e']==1:
        y=x.split('e')
        if len(y)==2:
            if y[1].startswith('-'):
                y[1]=y[1][1:]
            if y[0].replace('.','').isdigit() and y[1].isdigit():
                x=float(x)
    if isinstance(x,(float,int)):
        x=x*sign
    
    return x
    
if __name__ == '__main__':
    pass
    #set_dir('test','file_only')
    
