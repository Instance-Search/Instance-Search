# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
from modeling import build_model
from config import cfg
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

class ReID_Inference:

    def __init__(self,model_dir,gpus):
    
        cfg.MODEL.DEVICE_ID = gpus

        # useful setting
        cfg.MODEL.LAST_STRIDE = 1
        cfg.MODEL.NECK = 512
        cfg.MODEL.NAME = 'resnet50'
        cfg.INPUT.PIXEL_MEAN = [123.675,116.280,103.530]
        cfg.INPUT.PIXEL_STD = [57.0,57.0,57.0]
        cfg.TEST.WEIGHT = model_dir
        self.w = 128
        self.h = 384

        # useless setting
        cfg.MODEL.PRETRAIN_PATH = ''
        cfg.MODEL.DROPOUT = 0.
        cfg.MODEL.PRETRAIN_CHOICE = 'self'
        cfg.freeze()

        # cuda
        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True

        # model
        self.model = build_model(cfg, 750)
        self.model.load_param(cfg.TEST.WEIGHT,'self')
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to('cuda')
        self.model.eval()

        # save onnx
        save_onnx = False
        if save_onnx:
            dummy_input = torch.randn(1, 3, self.h, self.w, device='cuda')
            torch.onnx.export(self.model, dummy_input, "./reid.onnx", verbose=False)

        # preprocessing
        mean = cfg.INPUT.PIXEL_MEAN
        std = cfg.INPUT.PIXEL_STD
        mean = [k/255. for k in cfg.INPUT.PIXEL_MEAN]
        std = [k/255. for k in cfg.INPUT.PIXEL_STD]
        trans_list = [T.ToTensor(),T.Normalize(mean=mean, std=std)]
        self.transform = T.Compose(trans_list)

    def extract(self,img):
        img = cv2.resize(img,(self.w,self.h))
        img = img[:,:,::-1]
        data = Image.fromarray(img)
        data = self.transform(data)
        data = torch.unsqueeze(data,0)
    
        with torch.no_grad():
            data = data.to('cuda')
            feat = self.model(data)

        return feat.cpu().numpy()

    def distance(self,feat1,feat2):
        sim=np.linalg.norm(feat1-feat2,ord=2)
        return sim

def main():

    model_dir = ''
    gpus = '0'
    reid = ReID_Inference(model_dir,gpus)

    imgpath = '0108_03_0000.png'
    img = cv2.imread(imgpath)
    feat = reid.extract(img)

    save_feat = False
    if save_feat:
        f = open('./feat.txt','w')
        f.write(','.join(list(feat.flatten().astype(str))))
        f.close()

if __name__ == '__main__':
    main()

