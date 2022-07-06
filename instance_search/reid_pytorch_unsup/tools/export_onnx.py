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

import onnx
from onnx import optimizer

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


def onnx_apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    remove_initializer_from_input(model)
    onnx.save(model, outfile)
    return model



class ReID_Inference:

    def __init__(self,model_dir,gpus):

        cfg.MODEL.DEVICE_ID = gpus

        # useful setting
        cfg.MODEL.LAST_STRIDE = 1
        cfg.MODEL.NECK = 512
        # cfg.MODEL.NAME = 'resnet50_ibn_a'
        cfg.MODEL.NAME = 'resnet50'
        # cfg.MODEL.NAME = 'detnet_small'
        cfg.INPUT.PIXEL_MEAN = [123.675,116.280,103.530]
        cfg.INPUT.PIXEL_STD = [57.0,57.0,57.0]
        cfg.TEST.WEIGHT = model_dir
        # cfg.MODEL.GEM = True
        self.w = 224
        self.h = 224

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
        self.model = build_model(cfg, [750, 750, 750])
        self.model.load_param(cfg.TEST.WEIGHT,'self')
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to('cuda')
        self.model.eval()

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

    def export_onnx(self, prefix):

        # fp32
        onnxfile = prefix+'_fp32.onnx'
        dummy_input = torch.randn(1, 3, self.h, self.w, device='cuda')
        torch.onnx.export(self.model, dummy_input, onnxfile, verbose=True,
                          keep_initializers_as_inputs=True)
        onnx_model = onnx.load(onnxfile)
        passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, onnxfile)
        model = onnx_apply(change_input_dim, onnxfile, onnxfile)

        # fp16
        onnxfile = prefix+'_fp16.onnx'
        dummy_input = torch.randn(1, 3, self.h, self.w, device='cuda')
        dummy_input = dummy_input.half()
        self.model = self.model.half()
        # Ref" https://github.com/onnx/onnx/issues/1385
        torch.onnx.export(self.model, dummy_input, onnxfile, verbose=True,
                          keep_initializers_as_inputs=True)
        onnx_model = onnx.load(onnxfile)
        passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, onnxfile)
        model = onnx_apply(change_input_dim, onnxfile, onnxfile)

def main():

    # model_dir = '/home/gongyou.zyq/reid_pytorch/log/sot/resnet50_ibn_b_model_30.pth'
    # model_dir = 'log/baseline/resnet50_ibn_a_model_30.pth'
    # model_dir = 'log/baseline/resnet50_model_5.pth'
    # model_dir = '/home/gongyou.zyq/video_object_retrieval/instance_search/reid_pytorch_unsup/log/INS_Cityflow/self_train_reid/query_seed_naive/topk20_noisy/resnet50_model_30.pth'
    model_dir = '/home/gongyou.zyq/video_object_retrieval/instance_search/reid_pytorch_unsup/log/gldv2/resnet50_model_24.pth'
    gpus = '0'
    reid = ReID_Inference(model_dir,gpus)
    # reid.export_onnx('/home/gongyou.zyq/video_object_retrieval/instance_search/reid_pytorch_unsup/log/INS_Cityflow/self_train_reid/query_seed_naive/topk20_noisy/resnet50')
    reid.export_onnx('/home/gongyou.zyq/video_object_retrieval/instance_search/reid_pytorch_unsup/log/gldv2/resnet50')

if __name__ == '__main__':
    main()
