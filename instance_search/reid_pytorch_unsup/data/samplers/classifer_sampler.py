# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import math
import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler_ImgUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item[1]
            if pid < 0 :
                continue
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.count = 0
        self.imglist = []

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.imglist = final_idxs
        self.count = 0

        return iter(final_idxs)

    def __next__(self):
        if self.count+self.batch_size >= self.length or self.imglist == []:
            iter(self)
        batchimgs = self.imglist[self.count:self.count+self.batch_size]
        self.count += self.batch_size
        return batchimgs 

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_IdUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            if pid < 0 :
                continue
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.length = self.num_identities * self.num_instances
        self.batch_size = batch_size
        self.count = 0
        self.imglist = []

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        
        self.imglist = ret
        self.count = 0

        return iter(ret)

    def __next__(self):
        if self.count+self.batch_size >= self.length or self.imglist == []:
            iter(self)
        batchimgs = self.imglist[self.count:self.count+self.batch_size]
        self.count += self.batch_size
        return batchimgs 

    def __len__(self):
        return self.num_identities * self.num_instances

# New add by cwh
class RandomIdentitySampler_IdUniform_DiffCam(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.cam_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            if pid < 0 :
                continue
            self.index_dic[pid].append(index)
            cam = item[2]
            self.cam_dic[pid].append(cam)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.length = self.num_identities * self.num_instances
        self.batch_size = batch_size
        self.count = 0
        self.imglist = []

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            idxs = np.asarray(self.index_dic[pid])
            cams = np.asarray(self.cam_dic[pid])
            camlist = list(set(self.cam_dic[pid]))
            num = math.ceil(self.num_instances/float(len(camlist)))
            t = []
            for k in range(0,len(camlist)):
                tmp_t = idxs[cams==camlist[k]]
                replace = False if len(tmp_t) >= num else True
                tmp_t = np.random.choice(tmp_t, size=num, replace=replace)
                t.extend(tmp_t)
            t = np.random.choice(t, size=self.num_instances, replace=False)
            ret.extend(t)
        
        self.imglist = ret
        self.count = 0

        return iter(ret)

    def __next__(self):
        if self.count+self.batch_size >= self.length or self.imglist == []:
            iter(self)
        batchimgs = self.imglist[self.count:self.count+self.batch_size]
        self.count += self.batch_size
        return batchimgs 

    def __len__(self):
        return self.num_identities * self.num_instances

class IDFixedIter_Sampler(RandomIdentitySampler_IdUniform):
    """
    For SimCLR, we have to modify __len__ implementation
    """
    def __init__(self, data_source, batch_size, num_instances, fixed_iter):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        cam_list = []
        for index, item in enumerate(data_source):
            pid = item[1]
            if pid < 0 :
                continue
            cam = item[2]
            cam_list.append(cam)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # self.cluster_num_identities = max(cluster_num_identities, 128)
        # self.cluster_num_identities = len(np.unique(cam_list))
        self.fixed_iter = fixed_iter
        # self.length = self.cluster_num_identities * self.num_instances
        # self.length = fixed_iter * self.num_instances # old code, fixed id
        self.length = batch_size * fixed_iter
        self.batch_size = batch_size
        self.count = 0
        self.imglist = []

    def __len__(self):
        return self.length
