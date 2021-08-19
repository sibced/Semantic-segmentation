"dataset: utility to access the data"

from os import listdir
from os.path import join, splitext

from typing import Tuple

import numpy as np
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize

from . import INPUT_IMAGES_FOLDER, LABEL_IMAGES_FOLDER, TRAIN_CSV, TEST_CSV, VALIDATION_CSV

#converts image (HxWxC) in the range [0,255] into torch.FloatTensor (CxHxW) in the range [0.0, 1.0]
TO_TENSOR = ToTensor()

BASE_TRANSFORM = transforms.Compose([RandomHorizontalFlip(
), RandomVerticalFlip(), ToTensor(), Normalize((0.5,), (0.5,))])
def DROP_TRANSFORM(_): return torch.zeros((0,))


class RawDataset(torch.utils.data.Dataset):
    "RawDataset: dataset for the whole dataset"

    def __init__(self, transform=None, label_only=False):
        """Initializes a dataset containing images and labels."""
        super().__init__()

        self.transform = transform
        if transform is None:
            self.transform = lambda e: e
        self.label_only = label_only

        self.dataframe = pd.DataFrame(
            data=[(f,
                   join(INPUT_IMAGES_FOLDER, f),
                   join(LABEL_IMAGES_FOLDER, splitext(f)[0]+'.png')
                   ) for f in listdir(INPUT_IMAGES_FOLDER) if f.endswith("jpg")],
            columns=(
                'path',
                'input',
                'label'))

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """Returns the index-th data item of the dataset."""
        path = self.dataframe.loc[index, 'path']
        lbel = Image.open(self.dataframe.loc[index, 'label'])

        if self.label_only:
            return torch.zeros((0,)), self.transform(lbel), path

        inpt = Image.open(self.dataframe.loc[index, 'input'])
        return self.transform(inpt), self.transform(lbel), path


class CSVDataset(torch.utils.data.Dataset):
    "dataset for a subset of the raw dataset given by a csv file"

    def __init__(self, source, transform=None):
        "Initializes a dataset from a csv file created by split"
        super().__init__()

        self.transform = transform
        if transform is None:
            self.transform = lambda e: e

        # get pd.Series containing all filename
        self.imgs = pd.read_csv(source, names=('img',)).img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        "returns the index-th data item of the dataset."
        fname = self.imgs[index]
        inpt = Image.open(join(INPUT_IMAGES_FOLDER, fname))
        lbel = Image.open(join(LABEL_IMAGES_FOLDER, splitext(fname)[0]+'.png'))

        return self.transform(inpt), self.transform(lbel), fname


class SegmentationDataset(torch.utils.data.Dataset):
    "dataset holding images and their segmentation masks (greyscale)"

    def __init__(self, source: str, size: Tuple[int, int]):
        "source: file path, size: (width, height)"
        super().__init__()
        self.images_transform = transforms.Compose(
            [Resize(size), ToTensor(), Normalize((0.5,), (0.5,))])
        self.label_transform = Resize(size, Image.NEAREST)

        # get pd.Series containing all filename
        self.imgs = pd.read_csv(source, names=('img',)).img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):

        fname = self.imgs[index]
        inpt = Image.open(join(INPUT_IMAGES_FOLDER, fname))
        lbel = Image.open(join(LABEL_IMAGES_FOLDER, splitext(fname)[0]+'.png'))

        mask = torch.randint(0, 2, (2,))
        if mask[0] == 1:  # HFlip
            inpt = inpt.transpose(Image.FLIP_LEFT_RIGHT)
            lbel = lbel.transpose(Image.FLIP_LEFT_RIGHT)

        if mask[1] == 1:  # VFlip
            inpt = inpt.transpose(Image.FLIP_TOP_BOTTOM)
            lbel = lbel.transpose(Image.FLIP_TOP_BOTTOM)

        # apply transform
        inpt: torch.Tensor = self.images_transform(inpt)  # C, H, W
        lbel: Image.Image = self.label_transform(lbel)  # H, W, C

        return inpt, np.array(lbel, dtype=np.int64), fname


def trainset(size: Tuple[int, int] = (6000, 4000)) -> CSVDataset:
    "trainset: returns the training set"
    return SegmentationDataset(TRAIN_CSV, size)


def testset(size: Tuple[int, int] = (6000, 4000)) -> CSVDataset:
    "trainset: returns the testing set"
    return SegmentationDataset(TEST_CSV, size)


def validationset(size: Tuple[int, int] = (6000, 4000)) -> CSVDataset:
    "trainset: returns the validation set"
    return SegmentationDataset(VALIDATION_CSV, size)


def trainset_old(transform=BASE_TRANSFORM) -> CSVDataset:
    "trainset: returns the training set"
    return CSVDataset(TRAIN_CSV, transform)


def testset_old(transform=BASE_TRANSFORM) -> CSVDataset:
    "trainset: returns the testing set"
    return CSVDataset(TEST_CSV, transform)


def validationset_old(transform=BASE_TRANSFORM) -> CSVDataset:
    "trainset: returns the validation set"
    return CSVDataset(VALIDATION_CSV, transform)


def split(test=0.1, validation=0.1):
    df = pd.DataFrame(data=[f for f in listdir(
        INPUT_IMAGES_FOLDER) if f.endswith("jpg")])
    df = df.sample(frac=1, random_state=42)

    test_size = int(test * len(df))
    validation_size = int(validation * len(df))
    train_size = len(df) - test_size - validation_size

    test_end = train_size + test_size
    valid_end = test_end + validation_size

    df.iloc[0:train_size].to_csv(TRAIN_CSV, index=False, header=False)
    df.iloc[train_size:test_end].to_csv(TEST_CSV, index=False, header=False)
    df.iloc[test_end:valid_end].to_csv(
        VALIDATION_CSV, index=False, header=False)


if __name__ == "__main__":
    split()
