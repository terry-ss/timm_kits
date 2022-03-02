from pathlib import Path
import cv2
from imutils import paths
import sys,shutil
import numpy as np
from tqdm import tqdm
import hashlib
import time

##
_md=hashlib.md5()

##

def img_hash(img):
    img=cv2.resize(img,(10,10))
    hash_=hash(str(img))
    return hash_

def img_md5(img):
    _md.update(img)
    res=_md.hexdigest()
    return res

def clear_same_img(path):
    imgs=list(paths.list_images(path))
    imgs=sorted(imgs,key=lambda i:len(i),reverse=False)
    hashs=[]
    for img_file in tqdm(imgs):
        img=cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        hashs.append(img_hash(img))
    print('\r HASH has been calculated')
    n=len(imgs)
    for i in range(n) :
        file1=imgs[i]
        hash1=hashs[i]
        for j in range(i+1,n):
            file2=imgs[j]
            hash2=hashs[j]
            if hash1==hash2:
                if Path(file1).exists():
                    Path(file1).unlink()
                
def all_files_number(path):
    p=Path(path)
    f=[y for y in p.rglob(f'*')]
    dir_count = 0
    file_count = 0
    for f in p.rglob(f'*'):
        if f.is_dir():
            dir_count+=1
        else:
            file_count+=1
    n=dir_count+file_count
    print(f'total number of "{path}" is \n {n}')
    return dir_count,file_count

def clear_redundant_xml(path):
    p=Path(path)
    xmls=[y for y in p.glob(f'*.xml')]
    imgs=list(paths.list_images(p))
    img_names=[Path(y).stem for y in imgs]
    for xml in xmls:
        name=Path(xml).stem
        if not name in img_names:
            Path(xml).unlink()
            
def clear_same_names_between_paths(path_keep,path_remove):
    files1=Path(path_keep).glob('*')
    names_ref=[Path(x).name for x in files1]
    files2=Path(path_remove).glob('*')
    for file in files2:
        p=Path(file)
        if p.name in names_ref:
            p.unlink()
            
def clear_same_imgs_between_paths(path_keep,path_remove):    
    files1=paths.list_images(path)
    hashs=[]
    for img_file in files1:
        img=cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        hashs.append(img_hash(img))
    files2=paths.list_images(path)
    for img_file in files2:
        img=cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        if img_hash(img) in hashs:
            Path(img_file).unlink()  
            
def eliminate_space_in_name(path):
    p=Path(path)
    for file in p.rglob('*'):
        old_name=Path(file).name
        if ' ' in old_name:
            new_name=''.join(old_name.split())
            replace=Path(file).parent/new_name
            Path(file).rename(replace)
    
def get_files(path,extensions):
    #extension should start with '.'
    all_files = []
    for ext in extensions:
        all_files.extend(list(Path(path).rglob('*'+ext)))
    return all_files

if __name__ == '__main__':
    path=' '
    all_files_number(path)
    time.sleep(0.2)
    clear_same_img(path)
    all_files_number(path)
    