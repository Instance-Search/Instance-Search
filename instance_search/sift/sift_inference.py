"""Sift keypoint extraction and local matching.
Author: gongyou.zyq
Date: 2020.11.26
Ref: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
"""

import cv2
import numpy as np

from instance_search.base_matcher import BaseMatcher

def cvpoint2numpy(kp_cv):
    """Convert cvpoint to numpy.
    # pylint: disable=line-too-long
    Ref: https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object
    """

    kp_numpy = np.array([kp_cv[idx].pt for idx in range(0, len(kp_cv))])
    kp_numpy = kp_numpy.reshape(-1, 2).astype('float32')
    return kp_numpy

def numpy2cvpoint(kp_numpy):
    """Convert numpy to cvpoint."""

    result = []
    for sample_point in kp_numpy:
        pt_cv = cv2.KeyPoint(x=sample_point[0], y=sample_point[1], _size=2)
        result.append(pt_cv)
    return result

class SiftMatcher(BaseMatcher):
    """Sift extractor and matcher."""

    def __init__(self, gpus, _cfg):

        BaseMatcher.__init__(self, gpus, _cfg)
        self.sift = cv2.SIFT_create()

    def extract(self, large_path, bbox):
        """Extract. See base class."""

        large_bgr = cv2.imread(large_path)
        if bbox is not None:
            bbox = bbox.astype('int')
            img = large_bgr[bbox[1]: bbox[3], bbox[0]:bbox[2], :]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bbox_info = {'w': bbox[2]-bbox[0], 'h': bbox[3]-bbox[1]}
        else:
            gray = cv2.cvtColor(large_bgr, cv2.COLOR_BGR2GRAY)
            bbox_info = {'w': gray.shape[1], 'h': gray.shape[0]}
        kp_cv, des = self.sift.detectAndCompute(gray, None)
        kp_numpy = cvpoint2numpy(kp_cv)
        global_feat = np.array([])
        local_feat = {'locations': kp_numpy, 'descriptors': des}
        return global_feat, local_feat, bbox_info

    @staticmethod
    def _get_paired_pts(sample_query_local_feat, sample_index_local_feat):
        """Gets paired pts in the original image space by opencv.

        Args:
            sample_query_local_feat: A dict for query local feat.
            sample_index_local_feat: A dict for index local feat.

        Returns:
            A tuple of two elements:
                src_pts: source points.
                dst_pts: target points.
        """

        kp1 = numpy2cvpoint(sample_query_local_feat['locations'])
        kp2 = numpy2cvpoint(sample_index_local_feat['locations'])
        des1 = sample_query_local_feat['descriptors']
        des2 = sample_index_local_feat['descriptors']

        index_params = dict(algorithm=1, trees=5)    # FLANN_INDEX_KDTREE=1
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=1)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for match in matches:
            good.append(match[0])

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        src_pts = np.reshape(src_pts, (-1, 1, 2))
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        dst_pts = np.reshape(dst_pts, (-1, 1, 2))

        return src_pts, dst_pts

    def local_matching(self, query_res, index_res):
        """Opencv sift matching. See base class."""

        query_bbox_info = query_res['bbox_info']
        src_pts, dst_pts = self._get_paired_pts(query_res['local_feat'],
                                                index_res['local_feat'])
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # pylint: disable=line-too-long
        # Ref: https://stackoverflow.com/questions/24456788/opencv-how-to-get-inlier-points-using-findhomography-findfundamental-and-ra
        if matrix is not None:
            score = np.sum(mask)
            height, width = query_bbox_info['h'], query_bbox_info['w']
            pts = np.float32([[0, 0], [0, height-1], [width-1, height-1],
                              [width-1, 0]])
            pts = np.reshape(pts, (-1, 1, 2))
            dst = cv2.perspectiveTransform(pts, matrix)
            det = np.array([[dst[:, :, 0].min(), dst[:, :, 1].min(),
                             dst[:, :, 0].max(), dst[:, :, 1].max(), score]])
        else:
            det = np.array([[0, 0, 10000, 10000, 0]])
        return det
