"""Test local matching as localizer for an image pair.
Author: gongyou.zyq
Date: 2020.11.26
"""
import os
import pickle

from instance_search.config import cfg
from instance_search.global_local_matching import global_local_factory
from instance_search.utils.io_utils import get_query_info, get_gallery_info,\
                                           parse_args, select_good_id


def load_sample_data(_cfg):
    """Load query and gallery info."""

    query_instance_dic = pickle.load(open(cfg.EVAL.LABEL_FILE, 'rb'))
    selected_test_id = select_good_id(_cfg, query_instance_dic)
    query_instance_id = selected_test_id[0]
    query_bbox_dic = query_instance_dic[query_instance_id]
    query_large_path, query_bbox = get_query_info(_cfg, query_bbox_dic)
    query_bbox = query_bbox.astype('int')
    object_id = query_bbox_dic['object_id']
    index_large_path, _ = get_gallery_info(
            _cfg, query_bbox_dic['pos_gallery_list'], sample_index=0)
    print(f'Query instance id: {query_instance_id}, object id: {object_id}, '
          f'query_large_path: {query_large_path},  bbox: {query_bbox}')
    print(f'Index large path: {index_large_path}')
    return query_large_path, query_bbox, index_large_path


def test_local_matching():
    """Test local matching."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    global_local_matcher = global_local_factory(0, cfg)

    query_large_path, query_bbox, index_large_path = load_sample_data(cfg)
    # query_large_path = '/home/gongyou.zyq/datasets/instance_search/GLDv2/reid_images/test_gallery/0000_0008_0008.jpg'
    # query_bbox = []
    query_res = global_local_matcher.extract(query_large_path, query_bbox)
    query_global_feat = query_res['global_feat']
    query_local_feat = query_res['local_feat']
    query_bbox_info = query_res['bbox_info']
    print('query_global_feat shape, query_local_locations shape, '
          'query_local_descriptor shape, query_bbox_info')
    print(query_global_feat.shape, query_local_feat['locations'].shape,
          query_local_feat['descriptors'].shape, query_bbox_info)
    index_res = global_local_matcher.extract(index_large_path, None)
    index_global_feat = index_res['global_feat']
    index_local_feat = index_res['local_feat']
    index_bbox_info = index_res['bbox_info']
    print('index_global_feat shape, index_local_locations shape, '
          'index_local_descriptor shape, index_bbox_info')
    print(index_global_feat.shape, index_local_feat['locations'].shape,
          index_local_feat['descriptors'].shape, index_bbox_info)

    det_result = global_local_matcher.local_matching(query_res, index_res)
    save_name = f'./tests/images/debug_{cfg.EVAL.SIM_MODE}.jpg'
    print(f'save results in {save_name}')
    global_local_matcher.draw([query_large_path, query_bbox],
                              [index_large_path, det_result],
                              save_name)


def main():
    """Main function."""

    test_local_matching()


if __name__ == "__main__":
    main()
