import unittest
import os
import numpy as np
import cv2

import _init_paths
import neuron.ops as ops
from datasets import *
from mmcv.parallel import DataContainer as DC


class TestDatasets(unittest.TestCase):

    def setUp(self):
        # self.visualize = False
        self.visualize = True

    def test_pair_wrapper(self):
        # dataset = PairWrapper(base_transforms='extra_partial')
        # dataset = PairWrapper(base_transforms='extra_partial', base_dataset='coco_train', sampling_prob=1)
        # dataset = PairWrapper(base_transforms='extra_partial', base_dataset='Cityflow_train', sampling_prob=1)
        # dataset = PairWrapper(base_transforms='extra_partial', base_dataset='cityflow_test', sampling_prob=1)
        # dataset = PairWrapper(base_transforms='extra_partial', base_dataset='got10k_train', sampling_prob=1)
        # dataset = PairWrapper(base_transforms='extra_partial', base_dataset='CUHK_SYSU_train', sampling_prob=1)
        dataset = PairWrapper(base_transforms='extra_partial', base_dataset='INS_PRW_train', sampling_prob=1)
        indices = np.random.choice(len(dataset), 10)
        print(len(dataset))
        for i in indices:
            item = dataset[i]

            # check keys
            keys = [
                'img_z',
                'img_x',
                'img_meta_z',
                'img_meta_x',
                'gt_bboxes_z',
                'gt_bboxes_x']
            self.assertTrue(all([k in item for k in keys]))

            # check data types
            for _, v in item.items():
                self.assertTrue(isinstance(v, DC))
            
            # check sizes
            self.assertEqual(
                len(item['gt_bboxes_z'].data),
                len(item['gt_bboxes_x'].data))
            if 'gt_labels' in item:
                self.assertEqual(
                    len(item['gt_bboxes_x'].data),
                    len(item['gt_labels'].data))
            
            # visualize pair
            if self.visualize:
                ops.sys_print('Item index:', i)
                self._show_image(
                    item['img_z'].data, item['gt_bboxes_z'].data,
                    fig=0, delay=i)
                self._show_image(
                    item['img_x'].data, item['gt_bboxes_x'].data,
                    fig=1, delay=i)

    def _show_image(self, img, bboxes, fig, delay):
        img = 255. * (img - img.min()) / (img.max() - img.min())
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        bboxes = bboxes.cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        debug_dir = './debug_loc_pair/'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        for bbox in bboxes:
            bbox = bbox.astype('int')
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f'index_{delay}_{fig}.jpg'), img)
        # ops.show_image(img, bboxes, fig=fig, delay=delay)


if __name__ == '__main__':
    unittest.main()
