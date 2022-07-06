"""Make data for self train loc and reid.
Author: gongyou.zyq
Date: 2020.12.24
"""

import argparse
import pickle
from glob import glob
import os
import numpy as np
import cv2

from instance_search.config import cfg
from instance_search.utils.io_utils import split_group2pred


def save_pickle(_cfg, seq_dict, mode):
    """Save pickle label as cache for GlobalTrack."""

    pkl_dir = f'./instance_search/globaltracker/GlobalTrack/cache/'\
              f'{_cfg.SELF_TRAIN.MODE}/topk{_cfg.SELF_TRAIN.TOPK_LOC}_noisy'
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    dataset_name = _cfg.EVAL.DATASET_NAME
    pkl_name = os.path.join(pkl_dir, f'{dataset_name}_{mode}.pkl')
    pickle.dump(seq_dict, open(pkl_name, 'wb'), pickle.HIGHEST_PROTOCOL)


class SelfTrainDataMaker:
    """Convert ranklist into self train data for loc and reid."""

    def __init__(self, _cfg):
        self._cfg = _cfg
        self.all_image_dic = {}
        for mode in ['trainval', 'test_gallery', 'test_probe']:
            mode_dir = os.path.join(self._cfg.PATH.SEARCH_DIR, mode)
            if not os.path.exists(mode_dir):
                continue
            for device in os.listdir(mode_dir):
                device_dir = os.path.join(mode_dir, device)
                image_files = glob(device_dir+'/*.jpg')
                for item in image_files:
                    self.all_image_dic[item] = cv2.imread(item)
        print('all image data cached')

    def set_config(self, _cfg):
        """Set config."""

        self._cfg = _cfg

    def make_test_loc(self, label_dic):
        """Convert instance search label ID for test."""

        gal_dir = os.path.join(self._cfg.PATH.SEARCH_DIR, 'test_gallery')
        seq_dict = {}
        for object_id, query_bbox_dic in label_dic.items():
            img_files = []
            bboxes = []
            meta = {}
            probe_dir = gal_dir.replace('test_gallery', 'test_probe')
            query_img_file = os.path.join(probe_dir,
                                          query_bbox_dic['image_name'])
            assert os.path.isfile(query_img_file)
            img_files.append(query_img_file)
            bboxes.append(query_bbox_dic['bbox'].astype('float32'))
            # query + gallery as label
            gallery_list = query_bbox_dic['pos_gallery_list']
            for gal_data in gallery_list:
                gallery_img_file = os.path.join(gal_dir, gal_data['image_name'])
                assert os.path.isfile(gallery_img_file)
                img_files.append(gallery_img_file)
                bboxes.append(gal_data['bbox'].astype('float32'))
            img = self.all_image_dic[query_img_file]
            meta.update({
                'width': img.shape[1],
                'height': img.shape[0],
                'frame_num': len(bboxes),
                'target_num': 1,
                'total_instances': len(bboxes)})

            # update seq_dict
            if len(img_files) < 2:
                print('short sequence found')
            seq_dict[object_id] = {
                'img_files': img_files,
                'target': {
                    'anno': np.array(bboxes),
                    'meta': meta}}
        save_pickle(self._cfg, seq_dict, 'test')

    def make_sup_loc(self, label_dic):
        """Convert instance search label ID for sup train."""

        _cfg = self._cfg
        train_dir = os.path.join(_cfg.PATH.SEARCH_DIR, 'trainval')
        seq_dict = {}
        for object_id, object_data_list in label_dic.items():
            img_files = []
            bboxes = []
            meta = {}
            for sample_data in object_data_list:
                img_file = os.path.join(train_dir,
                                        sample_data['image_name'])
                assert os.path.isfile(img_file)
                img_files.append(img_file)
                bboxes.append(sample_data['bbox'].astype('float32'))
            img = self.all_image_dic[img_file]
            meta.update({
                'width': img.shape[1],
                'height': img.shape[0],
                'frame_num': len(bboxes),
                'target_num': 1,
                'total_instances': len(bboxes)})

            # update seq_dict
            seq_dict[object_id] = {
                'img_files': img_files,
                'target': {
                    'anno': np.array(bboxes),
                    'meta': meta}}
        save_pickle(_cfg, seq_dict, 'train')

    def make_one_seq_reid(self, query_index, pred_dic, query_label):
        """Make self_train reid data for one object id."""

        unsup_reid_dir = f'{self._cfg.PATH.SEARCH_DIR}../'\
                         f'{self._cfg.SELF_TRAIN.MODE}/'\
                         f'topk{self._cfg.SELF_TRAIN.TOPK_IDE}_noisy/trainval'
        if not os.path.exists(unsup_reid_dir):
            os.makedirs(unsup_reid_dir)
        pred_bboxes, image_ids = pred_dic['bbox'], pred_dic['image_name']
        if self._cfg.SELF_TRAIN.IDE_MODE == 'topk':
            orders = np.argsort(-pred_dic['sim'])
            indexes = orders[:self._cfg.SELF_TRAIN.TOPK_IDE]
        if self._cfg.SELF_TRAIN.IDE_MODE == 'thr':
            indexes = pred_dic['sim'] > self._cfg.SELF_TRAIN.THR_IDE
        image_ids = image_ids[indexes]
        pred_bboxes = pred_bboxes[indexes]
        # NOTE: do not include query itself as it will be retrieved 1st.
        # self_train_data = [[os.path.join(self._cfg.PATH.SEARCH_DIR,
        #                                  self._cfg.EVAL.QUERY_DATA_SOURCE,
        #                                  query_label['image_name']),
        #                     query_label['bbox'].astype('int')]]
        self_train_data = []
        for k, bbox in enumerate(pred_bboxes):
            img_file = os.path.join(self._cfg.PATH.SEARCH_DIR,
                                    self._cfg.EVAL.GALLERY_DATA_SOURCE,
                                    image_ids[k])
            bbox = bbox.astype('int')
            self_train_data.append([img_file, bbox])
        for k, [img_file, bbox] in enumerate(self_train_data):
            short_name = '_'.join(img_file.split('/')[-2:])
            short_name = short_name.split('.')[0]
            new_name = f'{query_index:04d}_00_{k:04d}_name_{short_name}_bbox_'\
                       f'{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg'
            new_name = f'{self._cfg.PATH.SEARCH_DIR}../{self._cfg.SELF_TRAIN.MODE}'\
                       f'/topk{self._cfg.SELF_TRAIN.TOPK_IDE}_noisy/trainval/{new_name}'
            self.crop_image(img_file, new_name, bbox)

    def crop_image(self, img_file, new_name, bbox):
        """Crop image."""

        img = self.all_image_dic[img_file]
        height, width = img.shape[:2]
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::3] = np.clip(bbox[1::3], 0, height - 1)
        cv2.imwrite(new_name, img[bbox[1]: bbox[3], bbox[0]: bbox[2], :])

    def make_selftrain_noisy_reid(self, query_instance_dic):
        """Make self train noisy data for reid."""

        _cfg = self._cfg
        pkl_file = _cfg.SELF_TRAIN.QUERY_SEED_BASELINE_FEAT
        all_pred_dic = pickle.load(open(pkl_file, 'rb'))
        print(f'make self train noisy reid data from: {pkl_file}')

        selected_test_id = sorted(query_instance_dic.keys())
        for query_index, query_instance_id in enumerate(selected_test_id):
            pred_dic = all_pred_dic[query_instance_id]
            pred_dic = split_group2pred(pred_dic)
            query_label = query_instance_dic[query_instance_id]
            self.make_one_seq_reid(query_index, pred_dic, query_label)

    def make_one_seq_loc(self, pred_dic, query_label):
        """Make self_train loc data for one object id."""

        pred_dic = split_group2pred(pred_dic)
        pred_bboxes, image_ids = pred_dic['bbox'], pred_dic['image_name']
        img_files = []
        bboxes = []
        sim_list = []
        meta = {}

        if self._cfg.SELF_TRAIN.LOC_MODE in ['topk', 'rss']:
            indexes = np.argsort(-pred_dic['sim'])[:self._cfg.SELF_TRAIN.TOPK_LOC]
        if self._cfg.SELF_TRAIN.LOC_MODE in ['thr']:
            indexes = pred_dic['sim'] > self._cfg.SELF_TRAIN.THR_LOC
        image_ids = image_ids[indexes]
        pred_bboxes = pred_bboxes[indexes].astype('float32')
        sims = pred_dic['sim'][indexes]
        if len(pred_bboxes) < 1:
            return [], [], [], []


        img_files.append(os.path.join(self._cfg.PATH.SEARCH_DIR,
                                      self._cfg.EVAL.QUERY_DATA_SOURCE,
                                      query_label['image_name']))
        bboxes.append(query_label['bbox'])
        sim_list.append(1.0)

        for k, bbox in enumerate(pred_bboxes):
            img_file = os.path.join(self._cfg.PATH.SEARCH_DIR,
                                    self._cfg.EVAL.GALLERY_DATA_SOURCE,
                                    image_ids[k])
            img_files.append(img_file)
            bboxes.append(bbox)
            sim_list.append(sims[k])

        img = self.all_image_dic[img_file]
        meta.update({
            'width': img.shape[1],
            'height': img.shape[0],
            'frame_num': len(bboxes),
            'target_num': 1,
            'total_instances': len(bboxes)})
        # save raw rss value
        sim_list = np.array(sim_list)
        return img_files, bboxes, meta, sim_list

    def make_selftrain_noisy_loc(self, query_instance_dic):
        """Make self train noisy data for loc."""

        pkl_file = self._cfg.SELF_TRAIN.QUERY_SEED_BASELINE_FEAT
        all_pred_dic = pickle.load(open(pkl_file, 'rb'))
        print(f'make self train noisy loc data from: {pkl_file}')

        seq_dict = {}
        # selected_test_id = sorted(query_instance_dic.keys(), key=int)
        selected_test_id = sorted(query_instance_dic.keys())
        # selected_test_id = selected_test_id[:self._cfg.EVAL.TEST_QUERY_NUM]
        for query_instance_id in selected_test_id:
            pred_dic = all_pred_dic[query_instance_id]
            img_files, bboxes, meta, sim_list = self.make_one_seq_loc(pred_dic,\
                                         query_instance_dic[query_instance_id])
            if len(img_files) == 0:
                continue

            # update seq_dict
            seq_dict[query_instance_id] = {
                'img_files': img_files,
                'target': {
                    'sim_list': sim_list,
                    'anno': np.array(bboxes),
                    'meta': meta}}
        print(f'{len(seq_dict)} seq pairs for loc')
        pkl_dir = f'./instance_search/globaltracker/GlobalTrack/'\
                  f'cache/{self._cfg.SELF_TRAIN.MODE}/'\
                  f'topk{self._cfg.SELF_TRAIN.TOPK_LOC}_noisy'
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        data_mode = self._cfg.EVAL.DATASET_NAME
        pkl_name = os.path.join(pkl_dir, f'{data_mode}_train.pkl')
        pickle.dump(seq_dict, open(pkl_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def main():
    """Main method"""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="Cityflow.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    selftrain_maker = SelfTrainDataMaker(cfg)

    self_flag = 'self' in cfg.SELF_TRAIN.MODE
    sup_flag = 'sup' in cfg.SELF_TRAIN.MODE
    loc_flag = 'loc' in cfg.SELF_TRAIN.MODE
    reid_flag = 'reid' in cfg.SELF_TRAIN.MODE

    if sup_flag and loc_flag:
        print('make sup loc')
        label_file = cfg.EVAL.LABEL_FILE[:-4] + '_train.pkl'
        train_dic = pickle.load(open(label_file, 'rb'))
        selftrain_maker.make_sup_loc(train_dic)
    if self_flag and reid_flag:
        print('Make selftrain noisy reid')
        query_instance_dic = pickle.load(open(cfg.SELF_TRAIN.QUERY_SEED_FILE, 'rb'))
        # for topk in [2, 5, 10, 20]:    # no use for many choices
        selftrain_maker.set_config(cfg)
        selftrain_maker.make_selftrain_noisy_reid(query_instance_dic)
    if self_flag and loc_flag:
        print('Make selftrain noisy loc')
        query_instance_dic = pickle.load(open(cfg.SELF_TRAIN.QUERY_SEED_FILE, 'rb'))
        query_instance_dic_test = pickle.load(open(cfg.EVAL.LABEL_FILE, 'rb'))
        # for topk in [2, 5, 10, 20]:    # no use for many choices
        selftrain_maker.set_config(cfg)
        selftrain_maker.make_selftrain_noisy_loc(query_instance_dic)
        selftrain_maker.make_test_loc(query_instance_dic_test)


if __name__ == "__main__":
    main()
