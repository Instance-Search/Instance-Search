"""Read global local features and perform global ranking and local matching.
Author: gongyou.zyq
Date: 2020.11.27
"""

import pickle
import os
import time

import numpy as np

from instance_search.global_local_matching import global_local_factory
from instance_search.config import cfg
from instance_search.utils.io_utils import parse_args, save_pickle


def perform_retrieval(_cfg, all_query_res, all_index_res):
    """Perform global/local retrieval. Save pred results in standard format.

    Args:
        query_res: A dict of query feat
        index_res: A dict of index feat
    """

    global_local_matcher = global_local_factory(None, _cfg)
    query_name_list = sorted(all_query_res.keys())

    # some strange image (too thin or too fat) may have no feat
    index_global_feat = []
    for index_name, index_res in all_index_res.items():
        if len(index_res['global_feat']) >1:
            index_global_feat.append(index_res['global_feat'])
        else:
            rand_feat = np.ones((2048,), dtype=np.float32)
            index_global_feat.append(rand_feat)
            print(f'{index_name} has no feat, use rand feat instead')
    index_global_feat = np.array(index_global_feat)
    query_global_feat = np.array([all_query_res[query_name]['global_feat']
                                  for query_name in query_name_list])
    all_sims = np.dot(query_global_feat, index_global_feat.T)
    print(all_sims.shape)
    all_rank_dict = {}
    for query_index, query_name in enumerate(
            query_name_list[:_cfg.EVAL.TEST_QUERY_NUM]):
        # NOTE: might be wrong for other dataset
        query_instance_id = query_name.split('/')[0]
        query_res = all_query_res[query_name]
        query_res['image_name'] = query_name
        global_local_matcher.set_query_feat(query_res, query_instance_id)
        print(f'{query_index}/{len(query_name_list)}')
        rank_dict = global_local_matcher.get_global_rank(all_sims[query_index],
                                                         all_index_res)
        if _cfg.DELG.USE_LOCAL_FEATURE:
            local_rank_dict = global_local_matcher.rerank_by_local(
                    rank_dict, all_index_res)
            rank_dict = global_local_matcher.merge_ranks(local_rank_dict,
                                                         rank_dict)
        all_rank_dict[query_instance_id] = convert_rawmatch2dict(rank_dict)
    start_time = time.time()
    save_pickle(all_rank_dict, _cfg.EVAL.SIM_MODE, _cfg.EVAL.DATA_MODE)
    print(f'{time.time()-start_time} seconds to save featrue')


def convert_rawmatch2dict(raw_rank_dict):
    """Convert query-gallery raw match result into dict format."""

    pred_dict = {}
    for gal_name, matched_bbox in raw_rank_dict.items():
        pair_res = {'sim': matched_bbox[:, -1],
                    'bbox': matched_bbox[:, :4]}
        pred_dict[gal_name] = pair_res
    return pred_dict


def main():
    """Main function."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    sim_dir = f'./tests/features/{cfg.EVAL.DATA_MODE}/{cfg.EVAL.SIM_MODE}/'
    query_res_pure = pickle.load(open(f'{sim_dir}/query_pure.pkl', 'rb'))
    query_res_crop = pickle.load(open(f'{sim_dir}/query_cropresize.pkl', 'rb'))
    all_index_res = pickle.load(open(f'{sim_dir}/index.pkl', 'rb'))
    print('finish load data')
    if not cfg.DELG.CROP_QUERY:
        all_query_res = query_res_pure
    else:
        # this is common case
        all_query_res = query_res_crop
    perform_retrieval(cfg, all_query_res, all_index_res)


if __name__ == "__main__":
    main()
