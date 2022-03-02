import torch
from torch.utils.data import Dataset, DataLoader
from imutils.paths import list_images
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from pathlib import Path
from collections import namedtuple
import yaml
from easydict import EasyDict

class R_Dataset(Dataset):       
    def __init__(self,images_data,yaml_path, transform=None):
        self.imgs=list(list_images(images_data))
        self.transform=transform
        with open(yaml_path,'r') as f:
            self.labels=yaml.load(f,Loader=yaml.SafeLoader)
        self.temple=namedtuple('Item',('image','label'))
        #print(self.labels)
           
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self,idx):
        img_name=Path(self.imgs[idx]).name
        image=cv2.imread(self.imgs[idx])
        label=torch.tensor(self.labels[img_name])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        #return self.temple(*(image,label))
        #return {'image':image,'label':label}
        return image,label
       
def create_transform(args):
    trfm_train = A.Compose([
        A.Resize(args.shape[0],args.shape[1]),
        A.HorizontalFlip(p=0.5),
        A.SomeOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                       saturation=0.1, hue=0.1, always_apply=False, p=0.3),
            A.ShiftScaleRotate(),   

        ], n=2,p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            A.ShiftScaleRotate(),
            ], p=1),
        A.OneOf([
            A.RandomSnow(),
            A.RandomSunFlare(),
            A.RandomFog(),
            A.RandomShadow(),
            ], p=1),
        ToTensorV2()
    ])
    trfm_val=A.Compose([
        A.Resize(args.shape[0],args.shape[1]),
        ToTensorV2()
        ])
    return trfm_train,trfm_val

def prepare_data(args,labels_path):
    trfm_train,trfm_val=create_transform(args)
    
    #train_mode=args.tm
    train_dataset = R_Dataset('data/train', labels_path,
        transform=trfm_train, )
    val_dataset = R_Dataset('data/val', labels_path,
        transform=trfm_val,)
    batch_size = args.bs
    train_loader = DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           num_workers=4, 
                                           pin_memory=False, 
                                           shuffle=True,
                                          )
    val_loader = DataLoader(val_dataset,
                                           batch_size=batch_size, 
                                           num_workers=4, 
                                           pin_memory=False, 
                                           shuffle=False,
                                          )
    return train_loader,val_loader

