# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import time
import numpy as np

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_val_data_loader
from data.datasets import init_dataset
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger

import torch.distributed as dist
import torch.multiprocessing as mp

def train(cur_gpu,ngpus_per_node,cfg):

    cur_rank = cfg.DIST.NODE_RANK * ngpus_per_node + cur_gpu
    world_size = cfg.DIST.NODE_WORLDSIZE * ngpus_per_node
    if cfg.DIST.NODE_WORLDSIZE > 0:
        dist.init_process_group(backend=cfg.DIST.BACKEND, init_method=cfg.DIST.URL,
                                world_size=world_size, rank=cur_rank)
        distribute = True
    else:
        distribute = False

    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR, cur_gpu)
    logger.info("nodes:{} gpus_on_current_node:{}".format(cfg.DIST.NODE_WORLDSIZE,torch.cuda.device_count()))
    logger.info("Running with config:\n{}".format(cfg))

    # prepare dataset
    val_data_loader, num_query = make_val_data_loader(cfg)
    num_classes = np.zeros(len(cfg.DATALOADER.SAMPLER_PROB)).astype(int) - 1
    source_dataset = init_dataset(cfg.SRC_DATA.NAMES, root_train=cfg.SRC_DATA.TRAIN_DIR)
    num_classes[0] = source_dataset.num_train_pids
    num_classes[2] = cfg.TGT_UNSUPDATA.CLUSTER_TOPK

    # prepare model
    model = build_model(cfg, num_classes)

    optimizer,fixed_lr_idxs = make_optimizer(cfg, model)
    loss_fn = make_loss(cfg, num_classes)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Start epoch:%d' %start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        logger.info('Path to the checkpoint of optimizer:%s' %path_to_optimizer)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu'))
        optimizer.load_state_dict(torch.load(path_to_optimizer))
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch, fixed_lr_idxs)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'self' or cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        model.load_param(cfg.MODEL.PRETRAIN_PATH,cfg.MODEL.PRETRAIN_CHOICE)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, -1, fixed_lr_idxs)
    else:
        logger.info('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    do_train(cfg,
            model,
            val_data_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_fn,
            num_query,
            start_epoch,     # add for using self trained model
            cur_gpu,
            distribute
            )

def main():

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DIST.NODE_WORLDSIZE = int(os.getenv('WORLD_SIZE', cfg.DIST.NODE_WORLDSIZE))
    cfg.DIST.NODE_RANK = int(os.getenv('RANK', cfg.DIST.NODE_RANK))
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    ngpus_per_node = torch.cuda.device_count()
    if cfg.DIST.NODE_WORLDSIZE > 0:
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node,cfg))
    else:
        train(0,ngpus_per_node,cfg)

if __name__ == '__main__':
    main()
