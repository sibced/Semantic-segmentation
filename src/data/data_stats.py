"data_stats: information on the dataset"

import torch
import numpy as np
from matplotlib import pyplot as plt

from .dataset import *


def show_images(img):
    img = img 
    npimg = img.numpy() * .5 + .5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def stats():
    # get as H*W*C uint8
    ds = RawDataset(np.array, label_only=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    MAX_SIZE = len(dl)
    count_per_image = np.zeros((MAX_SIZE), dtype=int)
    colors_dict = {}

    for i, (_, lbel, _) in enumerate(dl):
        if i == MAX_SIZE:
            break
        colors = np.unique(lbel.reshape(-1, 3), axis = 0)
        count_per_image[i] = len(colors)
        for c in colors:
            key = c.tobytes()
            if key in colors_dict:
                colors_dict[key] += 1
            else:
                colors_dict[key] = 1
    
    print(count_per_image)
    print(colors_dict)

"""results of stats on the whole dataset: 

[11 11  4 13 12 14 14  8  5 12 10  7 15  8 13  8 10 12  7  9  4 13  5 15
 14 13 17 13 12 13 13 13 15 11  7 12  7 15 14 12 16  9  2  9 13 13 13  6
 11 16 12  8 12 14  9 17 11 19  8 16 12 12 16 14  7 12 14 14 14  6 10 11
 13  7 11  5 14 14 12 13 16 13 14 19  9 12 22 13 15  9 18  8 10 12  7  7
 11 22 14  8  7 15 14 13 13 11  8 12 15 13 14 16 15 16 11  9 17 18 14 16
 11 11 15 15 11 10 15 15 13 16 16 12 16 12 14  8 14 17 13 14 15 12 17 11
 10  9  6 16 13 10 18  9 14  5 16  5  9 15  7 16 16 11  6 14  8  8 16 16
 15 16  9  5 10 17 12 14 11 10 17  9 17 10 13 17  9 15 14 14 14  9  9 16
 15 13 17 13 14 12 12 13 14 12 15  6 15 14 16 10  5 10 13 10 14 16 14 12
 12 10 13  8  5  8 16 16 16 14 14 12 13 12 12 12 11 14 12  9  5 15 14 10
  9 15  3 15 17 13 11 12 13 15 13  6 13 15 14 13 12 14  4 15 18 17 12 15
 16 10 10 12 14 15 12 16 10 18 10 15 15  5 10 11 14 14 12 12 13  9 15 13
 16 12  9 11 17 14 11 16 14 10 10  8 14 16 12 15 15 12 13 11 12 15 15 11
  8 11 14  6 10 14 16 15 16 15 17 10 14 12  5 14 14 13  6  5 17 15  7 12
 10 18 11 10  7 12 14 16 13 12 12 12 19 10 19 13 14  8 13 14 13 15  8 10
 19 16 12 13 12 14 13 10 10 10  7 10 16 16 16  7 12 11 13 15 10  9 15  7
  9 17 14 15 14 13  9 12 15 13  8 12 17 15 13 10]
{b'\x00\x00\x00': 398, b'\x02\x87s': 389, b'\t\x8f\x96': 64, b'k\x8e#': 359, b'pgW': 330, b'w\x0b ': 154, b'\x80@\x80': 380, b'\x82L\x00': 332, b'\x99\x99\x99': 167, b'\xbe\x99\x99': 212, b'\xff\x16`': 367, b'\x00f\x00': 273, b'33\x00': 186, b'ff\x9c': 292, b'\xfe\xe4\x0c': 151, b'FFF': 199, b'p\x96\x92': 165, b'\xbe\xfa\xbe': 83, b'\xfe\x94\x0c': 37, b'\x1c*\xa8': 111, b'0)\x1e': 210, b'\x002Y': 40, b'f3\x00': 21}

decoded:
[0 0 0] : 398
[  2 135 115] : 389
[  9 143 150] : 64
[107 142  35] : 359
[112 103  87] : 330
[119  11  32] : 154
[128  64 128] : 380
[130  76   0] : 332
[153 153 153] : 167
[190 153 153] : 212
[255  22  96] : 367
[  0 102   0] : 273
[51 51  0] : 186
[102 102 156] : 292
[254 228  12] : 151
[70 70 70] : 199
[112 150 146] : 165
[190 250 190] : 83
[254 148  12] : 37
[ 28  42 168] : 111
[48 41 30] : 210
[ 0 50 89] : 40
[102  51   0] : 21
"""

if __name__ == "__main__":
    stats()