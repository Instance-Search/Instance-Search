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

sys.path.append('.')
from config import cfg
from data import make_val_data_loader
from engine.tester import tester
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(torch.cuda.device_count()))
    logger.info("Running with config:\n{}".format(cfg))

    val_data_loader, num_query = make_val_data_loader(cfg)
    model = build_model(cfg, [1,1,1])
    model.load_param(cfg.TEST.WEIGHT,'self')

    tester(cfg, model, val_data_loader, num_query)

if __name__ == '__main__':
    main()
