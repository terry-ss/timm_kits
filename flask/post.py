import requests
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np
from imutils.paths import list_images
import pandas as pd

def test_img(project,port):
    url=f'http://127.0.0.1:{port}/{project}'
    print(url)
    img_dir='img/'
    pbar=tqdm(list(list_images(img_dir)),colour='green')
    recored={'filename':[],'pred':[]}
    for img_file in pbar:
        data={'filename':img_file,'mask':False}
        res=requests.post(url=url,data=data)
        if res.status_code==200:
            result=eval(res.text)['predict']       
            real=convert(result)
            recored['filename'].append(img_file)
            recored['pred'].append(real)
        else:
            print(res.status_code)
            break
    df=pd.DataFrame(recored)
    df.to_csv('result.csv')
    print(df.head())

def convert(res):
    y=np.array(res).squeeze()
    return y
  
if __name__=='__main__':
    port='1111'
    project='meter'
    test_img(project,port)
           
