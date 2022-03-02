import torch
from torch.utils.data import Dataset, DataLoader
from imutils.paths import list_images
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from pathlib import Path

def filter_out(images_filepaths,classLabels):
    screen=[]
    for path in images_filepaths:
        label=Path(path).parts[-2]
        if label in classLabels.__members__:
            screen.append(path)
    return screen
    

class C_Dataset(Dataset):
    def __init__(self, images_filepaths, classes,transform=None):
        self.classLabels=classes
        if isinstance(images_filepaths,str):
            self.images_filepaths = list(list_images(images_filepaths))
        elif isinstance(images_filepaths,list):
            self.images_filepaths=[]
            for fp in images_filepaths:
                self.images_filepaths.extend(list(list_images(fp)))
        self.transform = transform
        self.images_filepaths=filter_out(self.images_filepaths,self.classLabels)
       
    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        label=Path(image_filepath).parts[-2]
        label=int(self.classLabels[label])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

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

def prepare_data(args,classLabels):
    trfm_train,trfm_val=create_transform(args)
    
    train_mode=args.tm
    if train_mode=='part':
        train_dataset = C_Dataset('data/train', classLabels,
            transform=trfm_train, )
        val_dataset = C_Dataset('data/val', classLabels,
        transform=trfm_val,)
    elif train_mode=='full':
        data_list=['data/train','data/val']
        train_dataset = C_Dataset(data_list, classLabels,
            transform=trfm_train,yaml_path=yaml_path
            )
        val_dataset = C_Dataset('data/val',classLabels, 
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
                                           num_workers=2, 
                                           pin_memory=False, 
                                           shuffle=False,
                                          )
    return train_loader,val_loader

