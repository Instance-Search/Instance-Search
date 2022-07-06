# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .triplet_loss_v2 import TripletLossV2
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .simclr_loss import SupConLoss


import logging
import copy

def make_loss(cfg, num_classes):

    logger = logging.getLogger("reid_baseline.check")

    # whether semihnm for triplet
    if cfg.LOSS.TRP_HNM == 'yes':
        logger.info('semi-hnm is used for triplet loss.')
        semi = True
    else:
        logger.info('semi-hnm is not used for triplet loss.')
        semi = False

    # whether l2norm for triplet
    if cfg.LOSS.TRP_L2 == 'yes':
        logger.info('l2 normal is used for triplet loss.')
        use_l2 = True
    else:
        logger.info('l2 normal is not used for triplet loss.')
        use_l2 = False

    infos = [cfg.SRC_DATA,cfg.TGT_SUPDATA,cfg.TGT_UNSUPDATA]

    # src loss
    i = 0
    src_func = []
    if 'trp' in cfg.LOSS.LOSS_TYPE[i]:
        if 'v2' in cfg.LOSS.LOSS_TYPE[i]:
            src_triplet = TripletLossV2(cfg.LOSS.TRP_MARGIN,infos[i].NUM_INSTANCE,infos[i].NUM_POS_INSTANCE)
        else:
            src_triplet = TripletLoss(cfg.LOSS.TRP_MARGIN)
        def src_trploss_func(score, feat, target):
            return src_triplet(feat, target, use_l2, semi)[0]
        src_func.append(src_trploss_func)
    if 'cls' in cfg.LOSS.LOSS_TYPE[i]:
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            src_xent = CrossEntropyLabelSmooth(num_classes=num_classes[i])
            def src_clsloss_func(score, feat, target):
                return src_xent(score, target)
        else:
            def src_clsloss_func(score, feat, target):
                return F.cross_entropy(score, target)
        src_func.append(src_clsloss_func)

    # tgt sup loss
    i = 1
    tgt_sup_func = []
    if 'trp' in cfg.LOSS.LOSS_TYPE[i]:
        if 'v2' in cfg.LOSS.LOSS_TYPE[i]:
            tgt_sup_triplet = TripletLossV2(cfg.LOSS.TRP_MARGIN,infos[i].NUM_INSTANCE,infos[i].NUM_POS_INSTANCE)
        else:
            tgt_sup_triplet = TripletLoss(cfg.LOSS.TRP_MARGIN)
        def tgt_sup_trploss_func(score, feat, target):
            return tgt_sup_triplet(feat, target, use_l2, semi)[0]
        tgt_sup_func.append(tgt_sup_trploss_func)
    if 'cls' in cfg.LOSS.LOSS_TYPE[i]:
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            tgt_sup_xent = CrossEntropyLabelSmooth(num_classes=num_classes[i])
            def tgt_sup_clsloss_func(score, feat, target):
                return tgt_sup_xent(score, target)
        else:
            def tgt_sup_clsloss_func(score, feat, target):
                return F.cross_entropy(score, target)
        tgt_sup_func.append(tgt_sup_clsloss_func)

    # tgt unsup loss
    i = 2
    tgt_unsup_func = []
    if 'trp' in cfg.LOSS.LOSS_TYPE[i]:
        if 'v2' in cfg.LOSS.LOSS_TYPE[i]:
            tgt_unsup_triplet = TripletLossV2(cfg.LOSS.TRP_MARGIN,infos[i].NUM_INSTANCE,infos[i].NUM_POS_INSTANCE)
        else:
            tgt_unsup_triplet = TripletLoss(cfg.LOSS.TRP_MARGIN)
        def tgt_unsup_trploss_func(score, feat, target):
            return tgt_unsup_triplet(feat, target, use_l2, semi)[0]
        tgt_unsup_func.append(tgt_unsup_trploss_func)
    if 'cls' in cfg.LOSS.LOSS_TYPE[i]:
        if cfg.LOSS.IF_LABELSMOOTH == 'on':
            tgt_unsup_xent = CrossEntropyLabelSmooth(num_classes=num_classes[i])
            def tgt_unsup_clsloss_func(score, feat, target):
                return tgt_unsup_xent(score, target)
        else:
            def tgt_unsup_clsloss_func(score, feat, target):
                return F.cross_entropy(score, target)
        tgt_unsup_func.append(tgt_unsup_clsloss_func)
    if 'simclr' in cfg.LOSS.LOSS_TYPE[i]:
        temperature = cfg.TGT_UNSUPDATA.TEMPERATURE
        criterion = SupConLoss(temperature=temperature, base_temperature=temperature,
                              num_views=cfg.TGT_UNSUPDATA.NUM_INSTANCE)
        # criterion = SupConLoss()
        def tgt_unsup_simclr_func(score, feat, target):
            return criterion(feat) * cfg.LOSS.SIMCLR_WEIGHT
        tgt_unsup_func.append(tgt_unsup_simclr_func)

    loss_funcs = [src_func,tgt_sup_func,tgt_unsup_func]

    return loss_funcs

