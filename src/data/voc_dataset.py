"voc_dataset: utility to access Pascal VOC 2012 data"

from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize, RandomCrop


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__()

        self.transform = transform
        self.root_dir = root_dir
        self.train = train
        self.files_path = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', 'train.txt' if train else 'val.txt')
        with open(self.files_path, 'r') as f:
            self.files = f.read().split()
        for i, fname in enumerate(self.files):
            img = Image.open(os.path.join(self.root_dir, 'JPEGImages' if train else 'SegmentationClass', fname + '.jpg' if train else fname + '.png'))
            if img.size[0] < 250 or img.size[1] < 250:
                self.files.pop(i)
        #ToTensor converts image (HxWxC) in the range [0,255] into torch.FloatTensor (CxHxW) in the range [0.0, 1.0]
        self.images_transform = transforms.Compose(
            [ToTensor(), Normalize((0.5,), (0.5,))])
        
    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        fname = self.files[index]
        img = Image.open(os.path.join(self.root_dir, 'JPEGImages', fname + '.jpg'))
        lbel = Image.open(os.path.join(self.root_dir, 'SegmentationClass', fname + '.png'))
        
        crop_h, crop_w = 250, 250
        #print(img.size)
        if self.train:
            crop_list = []
            for _ in range(4):

                try:
                    i, j, h, w = transforms.RandomCrop.get_params(
                        img, output_size=(crop_h, crop_w) )
                    image = transforms.functional.crop(img, i, j, h, w)
                    mask = transforms.functional.crop(lbel, i, j, h, w)
                    r = random.random()
                    if r > 0.5:
                        image = transforms.functional.hflip(image)
                        mask = transforms.functional.hflip(mask)

                    mask = np.array(mask, dtype=np.int64)
                    mask[np.where(mask==255)] = 0
                    
                    crop_list.append( (self.images_transform(image), torch.from_numpy(mask)) )
            
                except:
                    break
            img = transforms.functional.resize(img, (250,250))
            lbel = transforms.functional.resize(lbel, (250,250))
            lbel = np.array(lbel, dtype=np.int64)
            lbel[np.where(lbel==255)] = 0
            crop_list.append( (self.images_transform(img), torch.from_numpy(lbel)) )
            return crop_list
        else:
            img = transforms.functional.resize(img, (250,250))
            lbel = transforms.functional.resize(lbel, (250,250))
            lbel = np.array(lbel, dtype=np.int64)
            lbel[np.where(lbel==255)] = 0
            return self.images_transform(img), torch.from_numpy(lbel)