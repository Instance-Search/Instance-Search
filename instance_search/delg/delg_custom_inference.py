"""Extract delg features and global-local matching.
Author: gongyou.zyq
Date: 2020.11.26
Ref: https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html
"""

import os
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

class DelgCustomMatcher(BaseMatcher):
    """Delg global local inference."""

    def __init__(self, gpus, _cfg):

        BaseMatcher.__init__(self, gpus, _cfg)
        if gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
            config = delf_config_pb2.DelfConfig()
            # pylint: disable=line-too-long
            with tf.io.gfile.GFile('/home/gongyou.zyq/models/research/delf/delf/python/delg/r50delg_gld_config.pbtxt', 'r') as f_config:
                text_format.Parse(f_config.read(), config)
            config.model_path = '/home/gongyou.zyq/models/research/delf/delf/python/delg/parameters/r50delg_gld_20200814'
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
            # [x1, y1, x2, y2]
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
        # print(np.max(local_descriptor['locations'][:, 0]))
        # print(np.max(local_descriptor['locations'][:, 1]))
        local_feat = {}
        if self._cfg.DELG.CROP_QUERY and bbox is not None:
            local_feat = {'locations': global_feat,
                          'descriptors': global_feat}
        else:
            offset_x = int(width/4.0)
            offset_y = int(height/4.0)
            bbox_list = [[0, 0, 2*offset_x, 2*offset_y],
                         [2*offset_x, 0, width-1, 2*offset_y],
                         [0, 2*offset_y, 2*offset_x, height-1],
                         [2*offset_x, 2*offset_y, width-1, height-1],
                         [offset_x, offset_y, 3*offset_x, 3*offset_y]]
            local_feat_list = []
            for gal_bbox in bbox_list:
                original_image_size = max(pil_im.size)
                cropped_im = pil_im.crop(gal_bbox)
                width, height = cropped_im.size
                cropped_image_size = max(cropped_im.size)
                resize_factor = cropped_image_size / original_image_size
                img = np.array(cropped_im)
                # Extract and save features.
                extracted_features = self.extractor_fn(img, resize_factor)
                patch_global_feat = extracted_features['global_descriptor']
                local_feat_list.append(patch_global_feat)
            local_feat_list = np.array(local_feat_list)
            local_feat = {'locations': local_feat_list,
                          'descriptors': local_feat_list}
        return global_feat, local_feat, bbox_info

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

        sample_query_local_feat = query_res['local_feat']
        sample_index_local_feat = index_res['local_feat']
        query_feat = sample_query_local_feat['descriptors'][:, np.newaxis]
        gallery_feat = sample_index_local_feat['descriptors']
        sims = np.dot(gallery_feat, query_feat)
        score = np.max(sims)
        det = np.array([[0, 0, 10000, 10000, score]])
        return det
