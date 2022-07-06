# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, add_psolabels, ImageDataset
from .samplers import Sampler_All
from .transforms import build_transforms
import numpy as np

def make_val_data_loader(cfg):
    val_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.VAL_DATA.NAMES, root_val=cfg.VAL_DATA.TRAIN_DIR)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.INPUT.SIZE_TEST)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn
    )

    return val_loader, len(dataset.query)

def make_target_unsupdata_loader(cfg):
    val_transforms = build_transforms(cfg, is_train=False)

    dataset = init_dataset(cfg.TGT_UNSUPDATA.NAMES, root_train=cfg.TGT_UNSUPDATA.TRAIN_DIR)

    target_unsupdata_set = ImageDataset(dataset.train, val_transforms, cfg.INPUT.SIZE_TEST)
    target_unsupdata_loader = DataLoader(
        target_unsupdata_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn
    )
    return target_unsupdata_loader

def make_alltrain_data_loader(cfg,psolabels):
    train_transforms = build_transforms(cfg, is_train=True)

    infos = [cfg.SRC_DATA,cfg.TGT_SUPDATA,cfg.TGT_UNSUPDATA]

    datasets = [(init_dataset(infos[i].NAMES, root_train=infos[i].TRAIN_DIR, setid=i)).train for i in range(len(infos))]
    if 'cluster' in cfg.TGT_UNSUPDATA.UNSUP_MODE:
        assert psolabels is not None
        datasets[2] = add_psolabels(datasets[2],psolabels)

    merged_dataset = []
    for i in datasets:
        merged_dataset.extend(i)
    train_set = ImageDataset(merged_dataset, train_transforms, cfg.INPUT.SIZE_TRAIN)

    train_loader = DataLoader(
        train_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH,
        sampler=Sampler_All(datasets,infos,cfg.DATALOADER.SAMPLER_PROB,cfg.DATALOADER.IMS_PER_BATCH, cfg),
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn, drop_last=True,
    )
    return train_loader
