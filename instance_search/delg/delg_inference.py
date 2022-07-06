"""Extract delg features and global-local matching.
Author: gongyou.zyq
Date: 2020.11.26
Ref: https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html
"""

import os
from multiprocessing import Pool
import numpy as np
from google.protobuf import text_format
from scipy import spatial
from skimage import measure
from skimage import transform
import cv2

import tensorflow as tf
from delf.python.utils import RgbLoader
from delf import extractor, delf_config_pb2
from instance_search.base_matcher import BaseMatcher


class DelgMatcher(BaseMatcher):
    """Delg global local inference."""

    def __init__(self, gpus, _cfg):

        BaseMatcher.__init__(self, gpus, _cfg)
        if gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
            config = delf_config_pb2.DelfConfig()
            # pylint: disable=line-too-long
            # R50 + GLDv1 model
            with tf.io.gfile.GFile('/home/gongyou.zyq/models/research/delf/delf/python/delg/r50delg_gld_config.pbtxt', 'r') as f_config:
                text_format.Parse(f_config.read(), config)
            config.model_path = '/home/gongyou.zyq/models/research/delf/delf/python/delg/parameters/r50delg_gld_20200814'
            # R101 + GLDv2 naive model
            # with tf.io.gfile.GFile('/home/gongyou.zyq/models/research/delf/delf/python/delg/r101delg_gldv2clean_config.pbtxt', 'r') as f_config:
            #     text_format.Parse(f_config.read(), config)
            # config.model_path = '/home/gongyou.zyq/models/research/delf/delf/python/delg/parameters/r101delg_gldv2clean_20200914'
            config.use_local_features = _cfg.DELG.USE_LOCAL_FEATURE
            # config.delf_local_config.max_feature_num = 100000
            # config.delf_local_config.score_threshold = 1.0
            self.extractor_fn = extractor.MakeExtractor(config)
            self.delg_config = config

    def extract(self, large_path, bbox):
        """Extract feature. See base class"""

        resize_factor = 1.0
        pil_im = RgbLoader(large_path)
        # very important for query preprocessing...
        if self._cfg.DELG.CROP_QUERY and bbox is not None:
            # Crop query image according to bounding box.
            original_image_size = max(pil_im.size)
            bbox = [int(round(b)) for b in bbox]
            pil_im = pil_im.crop(bbox)
            width, height = pil_im.size
            cropped_image_size = max(pil_im.size)
            resize_factor = cropped_image_size / original_image_size
            bbox_info = {'w': width, 'h': height}
        else:
            width, height = pil_im.size
            bbox_info = {'w': width, 'h': height}

        img = np.array(pil_im)
        # Extract and save features.
        extracted_features = self.extractor_fn(img, resize_factor)
        global_feat = extracted_features['global_descriptor']
        if self._cfg.DELG.USE_LOCAL_FEATURE:
            local_feat = extracted_features['local_features']
        else:
            local_feat = []
        # print(np.max(local_descriptor['locations'][:, 0]))
        # print(np.max(local_descriptor['locations'][:, 1]))
        res = {'global_feat': global_feat, 'local_feat': local_feat,
               'bbox_info': bbox_info, 'bbox': bbox}
        return res

    @staticmethod
    def get_paired_pts(sample_query_local_feat, sample_index_local_feat):
        """Gets paired pts in the original image space by delf.

        Args:
            sample_query_local_feat: A dict for query local feat.
            sample_index_local_feat: A dict for index local feat.

        Returns:
            A tuple of two elements:
                query_locations_to_use: source points.
                index_image_locations_to_use: target points.
        """

        query_locations = sample_query_local_feat['locations']
        query_descriptors = sample_query_local_feat['descriptors']
        if len(query_descriptors) == 0:    # for empty query local feat
            return np.array([]), np.array([])
        index_image_locations = sample_index_local_feat['locations']
        index_image_descriptors = sample_index_local_feat['descriptors']

        index_image_tree = spatial.cKDTree(index_image_descriptors)
        descriptor_matching_threshold = 1.0
        _, indices = index_image_tree.query(query_descriptors,\
                 distance_upper_bound=descriptor_matching_threshold, n_jobs=-1)
        num_features_query = query_locations.shape[0]
        num_features_index_image = index_image_locations.shape[0]
        # Select feature locations for putative matches.
        query_locations_to_use = np.array([
            query_locations[i,]
            for i in range(num_features_query)
            if indices[i] != num_features_index_image
        ])
        index_image_locations_to_use = np.array([
            index_image_locations[indices[i],]
            for i in range(num_features_query)
            if indices[i] != num_features_index_image
        ])
        # print(query_locations_to_use.shape[0])
        return query_locations_to_use, index_image_locations_to_use

    def local_matching(self, query_res, index_res):
        """DELG local matching. See base class"""

        query_bbox_info = query_res['bbox_info']
        (query_locations_to_use,
         index_image_locations_to_use) = self.get_paired_pts(
                 query_res['local_feat'], index_res['local_feat'])
        if (query_locations_to_use.shape[0] <=
                self._cfg.DELG.MIN_RANSAC_SAMPLES):
            return np.array([[0, 0, 10000, 10000, 0]])
        # Perform geometric verification using RANSAC.
        model_robust, inliers = measure.ransac(
            (index_image_locations_to_use, query_locations_to_use),
            transform.AffineTransform,
            min_samples=self._cfg.DELG.MIN_RANSAC_SAMPLES,
            residual_threshold=20.0,
            max_trials=1000,
            random_state=0)
        score = np.sum(inliers)
        if score is None:
            score = 0
        if score > 0:
            height, width = query_bbox_info['h'], query_bbox_info['w']
            pts = np.float32([[0, 0], [0, height-1], [width-1, height-1],
                              [width-1, 0]])
            pts = np.reshape(pts, (-1, 1, 2))
            matrix = model_robust.params.astype('float32')
            try:
                # matrix might be Singular matrix
                matrix = np.linalg.inv(matrix)
                dst = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
                det = np.array([[dst[:, 1].min(), dst[:, 0].min(),
                                 dst[:, 1].max(), dst[:, 0].max(), score]])
            except np.linalg.LinAlgError:
                # print('bad M', score)
                det = np.array([[0, 0, 10000, 10000, 0]])
        else:
            det = np.array([[0, 0, 10000, 10000, 0]])
        return det

    def local_matching_batch(self, all_index_res, index_name_list):
        """DELG local matching. See base class"""

        pool = Pool()
        pool_result = []
        for rank, index_name in enumerate(index_name_list):
            if self._cfg.DELG.MULTI_PROCESS:
                dets = pool.apply_async(self.local_matching,
                                        args=(self.query_res,
                                              all_index_res[index_name],))
            else:
                dets = self.local_matching(self.query_res,
                                           all_index_res[index_name])
            if rank % 10 == 0:
                print(f'Re-ranking: i = {rank} out of {self.rerank_num}')
            pool_result.append(dets)
        pool.close()
        pool.join()
        local_rank_dict = {}
        for index_name, res in zip(index_name_list, pool_result):
            if self._cfg.DELG.MULTI_PROCESS:
                local_rank_dict[index_name] = res.get()
            else:
                local_rank_dict[index_name] = res
        return local_rank_dict
