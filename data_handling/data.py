import os

import cv2
import random

from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader

from .augmentations import closestNdivisibleByM, RandomCrop



def get_data_loaders(config, num_workers=2):
    batch_size = config['batch_size']
    train_img_path = config["train_img_path"]
    train_mask_path = config["train_mask_path"]
    val_img_path = config["val_img_path"]
    val_mask_path = config["val_mask_path"]
    
    train_dataset = SegmentationDataset(train_img_path, train_mask_path, split='train', 
                                        norm_means=config['norm_means'], norm_stds=config['norm_stds'], 
                                        img_shape=config['img_shape'])
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_dataset = SegmentationDataset(val_img_path, val_mask_path, split='val',
                                      norm_means=config['norm_means'], norm_stds=config['norm_stds'], 
                                      img_shape=config['img_shape'])
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return loader_train, loader_val



class SegmentationDataset(Dataset):
    def __init__(self, img_path, mask_path=None, split='train',
                 norm_means=None, norm_stds=None, img_shape=None, divisibility=None, image_channels=3):
        super(SegmentationDataset, self).__init__()
        # split
        self.split=split
        
        # files
        self.img_files = os.listdir(img_path)
        self.img_files.sort()
        self.img_files = [os.path.join(img_path, fname) for fname in self.img_files]
        
        self.mask_path = mask_path
        if self.mask_path is not None:
            self.mask_files = os.listdir(mask_path)
            self.mask_files.sort()
            self.mask_files = [os.path.join(mask_path, fname) for fname in self.mask_files]
        
        # normalize
        if norm_means is None:
            self.means = [0.485, 0.456, 0.406]
        else:
            self.means = norm_means
        if norm_stds is None:
            self.stds = [0.229, 0.224, 0.225]
        else:
            self.stds = norm_stds
        
        # image shape to return
        if img_shape is None:
            img_shape = (1000, 1000)
        self.img_shape = img_shape

        self.divisibility = divisibility
        if self.divisibility is not None:
            self.img_shape = (closestNdivisibleByM(self.img_shape[0], divisibility), 
                              closestNdivisibleByM(self.img_shape[1], divisibility))
        
        self.img_channels = image_channels
        
        if split == 'train':
            self.transform = transforms.Compose([
                #transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.stds),
            ])
            self.crop = RandomCrop(img_shape)
            self.flips = transforms.Compose([
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        ])
            
        else: # i.e. split == 'val' or split == 'test'
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.means, std=self.stds),
            ])
        
    def __len__(self) -> int:
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img = Image.fromarray(cv2.imread(self.img_files[idx]))
        resize_shape = self.img_shape
            
        if self.split == 'train':
            mask = torch.from_numpy(cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE))
            img = self.transform(img)
            
            img_mask_cat = torch.cat([img, mask.unsqueeze(0)], dim=0)
            #if torch.rand(1)[0] < 0.25:
                #img_mask_cat = self.crop(img_mask_cat) 
            img_mask_cat = self.flips(img_mask_cat)

            img = img_mask_cat[:3]
            mask = img_mask_cat[-1].long()

            img = transforms.functional.resize(img, resize_shape, antialias=True)
            mask = cv2.resize(mask.numpy(), resize_shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long()

        else:
            if self.mask_path is None:
                mask = None
            else:
                mask = torch.from_numpy(cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE))
                mask = cv2.resize(mask.numpy(), resize_shape[::-1], interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask).long()
            img = self.transform(img)
            img = transforms.functional.resize(img, resize_shape, antialias=True)
            
        return img, mask





