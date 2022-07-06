# encoding: utf-8

import copy
import random
import torch
from collections import defaultdict

import math
import numpy as np
from torch.utils.data.sampler import Sampler

def build_pos_set(path):
    items = []
    f = open(path,'r')
    while True:
        line = f.readline().replace('\n','')
        if not line:
            break
        line = line.split(':')
        if line[1]=='':
            continue
        query_id = int(line[0])
        ranklist = np.asarray(line[1].split(',')).astype(int)
        items.append(set([query_id]+list(ranklist)))
    f.close()

    idsets = []
    for item in items:
        found = 0
        for i in range(len(idsets)):
             if len(item & idsets[i])>0:
                 idsets[i] = idsets[i] | item
                 found = 1
                 break
        if found == 0:
            idsets.append(item)
    return idsets

def build_neg_dict(path):
    dic = {}
    f = open(path,'r')
    while True:
        line = f.readline().replace('\n','')
        if not line:
            break
        line = line.split(':')
        query_id = line[0]
        if line[1]=='':
            continue
        ranklist = np.asarray(line[1].split(',')).astype(int)
        dic[query_id] = set(list(ranklist))
    f.close()
    return dic

def negdict_to_set(pos_sets,neg_dict):
    neg_sets = []
    neg_dict_keys = neg_dict.keys()
    for pos_set in pos_sets:
        neg_set = set() 
        for pos_id in pos_set:
            if str(pos_id) in neg_dict_keys:
                neg_set = neg_set | neg_dict[str(pos_id)]
        neg_sets.append(neg_set)
    assert(len(pos_sets)==len(neg_sets))
    return neg_sets

def merge_set(pos_sets,neg_sets):
    sets = []
    for i in range(len(pos_sets)):
        if len(pos_sets[i])>0 and len(neg_sets[i])>0:
            sets.append([pos_sets[i],neg_sets[i]])
    return sets

# New add by cwh
class TripletSampler_FromTxt(Sampler):
    """
    Randomly sample N items.
    randomly sample K instances, in which there are K1 pos instances
    therefore batch size is N*K.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per item.
        num_pos_instances (int): number of pos instances per item.
    """
    def __init__(self, data_source, batch_size, num_instances, num_pos_instances, itemnum = 500):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_pos_instances = num_pos_instances
        self.ids = []
        for item in data_source:
            path = item[0]
            id = item[4]
            self.ids.append(id)
            if path == '':
                vvv
        self.ids = np.hstack(self.ids)

        # read pos connects
        #pos_path = '/root/kugang.cwh/projects/selftraining/sup_data/gt_face/face_v2/pos_ranklist.txt'
        pos_path = '/root/kugang.cwh/projects/selftraining/sup_data/gt_callback/pos_ranklist.txt'
        #pos_path = '/root/kugang.cwh/projects/selftraining/sup_data/gt_callback/pos_ranklist_market.txt'
        pos_sets = build_pos_set(pos_path)
        # read neg ranklist
        neg_path = '/root/kugang.cwh/projects/selftraining/sup_data/gt_timespace/neg_ranklist_1km_60s.txt'
        #neg_path = '/root/kugang.cwh/projects/selftraining/sup_data/gt_callback/neg_ranklist_market.txt'
        neg_dict = build_neg_dict(neg_path)
        neg_sets = negdict_to_set(pos_sets,neg_dict)
        self.sets = merge_set(pos_sets,neg_sets)

        self.sets = self.sets[:1000]

        # print
        #print('itemnum:',len(self.sets))
        posnum = 0
        negnum = 0
        for s in self.sets:
            posnum += len(s[0])
            negnum += len(s[1])
        #print('posimgs:',posnum)
        #print('negimgs:',negnum)

        self.itemnum = len(self.sets)
        
        self.length = self.itemnum * self.num_instances
        self.batch_size = batch_size
        self.count = 0
        self.imglist = []

    def __iter__(self):
        random.shuffle(self.sets)

        ret = []
        for s in self.sets[:self.itemnum]:
            pos_set = s[0]
            neg_set = s[1]

            # get pos
            replace = False
            if len(pos_set) < self.num_pos_instances:
                replace = True
            pos_ids = np.random.choice(list(pos_set), size=self.num_pos_instances, replace=replace)
            pos_ids = list(pos_ids)

            # get neg
            replace = False
            if len(neg_set) < self.num_instances - self.num_pos_instances:
                replace = True
            neg_ids = np.random.choice(list(neg_set), size=self.num_instances-self.num_pos_instances, replace=replace)
            neg_ids = list(neg_ids)

            ids = pos_ids + neg_ids
            idxs = []
            for id in ids:
                tmpidxs = np.where(self.ids==id)[0]
                tmpidx = np.random.choice(tmpidxs, size=1)
                idxs.append(tmpidx[0])
            ret.extend(idxs)

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
        return self.itemnum * self.num_instances

