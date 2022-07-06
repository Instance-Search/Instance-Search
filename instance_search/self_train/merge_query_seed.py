"""Test localizer with a pair of images.
Author: gongyou.zyq
Date: 2020.11.18
"""

import pickle
import os
import argparse
import random
import numpy as np

from instance_search.utils.io_utils import split_group2pred
from instance_search.config import cfg

KEEP_NUM = 1
np.random.seed(1)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="Cityflow.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def merge_query_seed(_cfg):
    """Expand neighbour."""

    query_seed_list = _cfg.SELF_TRAIN.QUERY_SEED_TYPE_LIST
    sample_per_type = int(_cfg.SELF_TRAIN.QUERY_SEED_NUM/len(query_seed_list))
    global_count = 0
    query_instance_dic = {}
    for query_seed_type in _cfg.SELF_TRAIN.QUERY_SEED_TYPE_LIST:
        query_seed_file_proposal = f'./tests/features/{_cfg.EVAL.DATA_MODE}/'\
                                   f'{query_seed_type}.pkl'
        query_instance_dic_proposal = pickle.load(
                open(query_seed_file_proposal, 'rb'))
        query_keys = list(query_instance_dic_proposal.keys())
        if len(query_keys) > sample_per_type:
            sampled_query = random.sample(query_keys, sample_per_type)
        else:
            sampled_query = query_keys
        for old_query_instance_id in sampled_query:
            new_query_instance_id = f'query_seed_{global_count:08d}'
            query_instance_dic[new_query_instance_id] = \
                    query_instance_dic_proposal[old_query_instance_id]
            global_count += 1

    pickle.dump(query_instance_dic,
                open(_cfg.SELF_TRAIN.QUERY_SEED_FILE, 'wb'))


def main():
    """Main function."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    merge_query_seed(cfg)


if __name__ == "__main__":
    main()
