# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import numpy as np

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def _get_anchor_positive_triplet_mask(anchor_pos_num,neg_num):
    indices_equal = torch.eye(anchor_pos_num).cuda()
    indices_not_equal = 1 - indices_equal
    mask1 = indices_not_equal

    mask2 = torch.zeros([anchor_pos_num,neg_num]).cuda()
    mask = torch.cat([mask1,mask2],1)
    return mask

def _get_anchor_negative_triplet_mask(anchor_pos_num,neg_num):
    mask1 = torch.zeros([anchor_pos_num,anchor_pos_num]).cuda()
    mask2 = torch.ones([anchor_pos_num,neg_num]).cuda()

    mask = torch.cat([mask1,mask2],1)
    return mask

class TripletLossV2(object):

    def __init__(self, margin=None, num=12, posnum=4):
        self.margin = margin
        self.posnum = posnum
        self.num = num
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False, use_semi=True):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        embeddings_list = torch.split(global_feat,self.num,0)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(self.posnum,self.num-self.posnum)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(self.posnum,self.num-self.posnum)

        loss = torch.tensor(0.).cuda()
        for i in range(0,len(embeddings_list)):
            pairwise_dist = euclidean_dist(embeddings_list[i],embeddings_list[i])
            pairwise_dist = pairwise_dist[0:self.posnum]

            anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)
            hardest_positive_dist,_ = torch.max(anchor_positive_dist, 1, keepdim=True)

            max_anchor_negative_dist,_ = torch.max(pairwise_dist, 1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
            hardest_negative_dist,_ = torch.min(anchor_negative_dist, 1, keepdim=True)

            if not use_semi:
                hardest_positive_dist,_ = torch.max(hardest_positive_dist)
                hardest_negative_dist,_ = torch.min(hardest_negative_dist)

            dist_ap = hardest_positive_dist.flatten()
            dist_an = hardest_negative_dist.flatten()

            y = dist_an.new().resize_as_(dist_an).fill_(1)
            if self.margin is not None:
                loss += self.ranking_loss(dist_an, dist_ap, y)
            else:
                loss += self.ranking_loss(dist_an - dist_ap, y)
        loss = loss/len(embeddings_list)
        return loss, dist_ap, dist_an
