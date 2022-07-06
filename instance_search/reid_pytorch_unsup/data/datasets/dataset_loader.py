# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
#cv2.setNumThreads(0)
#cv2.ocl.setUseOpenCL(False)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_image_cv2(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, size=None):
        self.dataset = dataset
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, setid, trkid, raw_image_name, bbox = self.dataset[index]
        img = read_image_cv2(img_path)
        img = cv2.resize(img,(self.size[1],self.size[0]))
        img = img[:,:,::-1]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path, setid, trkid, raw_image_name, bbox
