# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os

import os.path as osp
import numpy as np

from .bases import BaseImageDataset


class UNSUP(BaseImageDataset):

    def __init__(self, root_train='', root_val='', setid=0, verbose=True, **kwargs):
        super(UNSUP, self).__init__()
        self.train_dir = osp.join(root_train, 'trainval')
        self.query_dir = osp.join(root_val, 'test_probe')
        self.gallery_dir = osp.join(root_val, 'test_gallery')
        self.setid = setid

        #self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            #print("=> ourapi loaded from: {} and {}".format(root_train,root_val))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if dir_path == '':
            return []

        img_paths = glob.glob(osp.join(dir_path, '*.png')) + glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        camid = 0
        for i, img_path in enumerate(img_paths):
            # for unsup, each image has an unique ID
            # to save the original ID info, we put it into camid
            # image_name = os.path.basename(img_path)
            # assert 'bbox' in image_name
            # raw_image_name, bbox_string = image_name.split('_name_')[1].split('_bbox_')
            # bbox = np.array(bbox_string.split('.')[0].split('_')).astype('int')
            raw_image_name, bbox = None, None
            pid = i
            camid = int(os.path.basename(img_path).split('_')[0])
            dataset.append((img_path, pid, camid, self.setid, i, raw_image_name, bbox))

        return dataset
