# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class OURAPI(BaseImageDataset):

    def __init__(self, root_train='', root_val='', setid=0, verbose=True, **kwargs):
        super(OURAPI, self).__init__()
        self.train_dir = osp.join(root_train, 'trainval')
        self.query_dir = osp.join(root_val, 'test_probe')
        self.gallery_dir = osp.join(root_val, 'test_gallery')
        self.setid = setid

        #self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            # print("=> ourapi loaded from: {} and {}".format(root_train,root_val))
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
        pattern = re.compile(r'([\d]+)_([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            pid, camid, pidx = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            #camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            raw_image_name, bbox = None, None
            dataset.append((img_path, pid, camid, self.setid, i, raw_image_name, bbox))

        return dataset
