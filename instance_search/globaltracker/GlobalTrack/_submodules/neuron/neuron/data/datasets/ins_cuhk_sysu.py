import pickle
import os
import os.path as osp
import numpy as np
import cv2

import neuron.ops as ops
from neuron.config import registry
from .dataset import SeqDataset


__all__ = ['INS_CUHK_SYSU']


@registry.register_module
class INS_CUHK_SYSU(SeqDataset):
    r"""`INS_CUHK_SYSU
    """

    def __init__(self, root_dir=None, subset='test', list_file=None):
        root_dir = '/home/gongyou.zyq/datasets/instance_search/INS_CUHK_SYSU/'
        super(INS_CUHK_SYSU, self).__init__(
            name=f'INS_CUHK_SYSU_{subset}',
            root_dir=None,
            subset=subset,
            list_file=None)

    def _construct_seq_dict(self, root_dir, subset, list_file):

        _, _, _ = root_dir, subset, list_file
        return {}


    def _check_obj(self, obj):
        _, _, w, h = obj['bbox']
        ignore = obj.get('ignore', False)
        if obj['iscrowd'] or ignore or w < 1 or h < 1:
            return False
        return True