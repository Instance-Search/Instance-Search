"""Extract global local feature.
Author: gongyou.zyq
Date: 2020.11.28
"""

import os
import pickle
import time
from itertools import cycle
from multiprocessing import Pool, Queue

import numpy as np
import tqdm

from instance_search.global_local_matching import global_local_factory
from instance_search.utils.io_utils import load_all_data, parse_args
from instance_search.config import cfg


def init_worker(_cfg, gpu_id):
    """
    This gets called to initialize the worker process.
    Here we load the ML model into GPU memory.
    Each worker process will pull an GPU ID from a queue of available IDs
    (e.g. [0, 1, 2, 3]) to ensure that multiple. GPUs are consumed evenly.
    """

    # pylint: disable=global-variable-undefined, invalid-name
    global global_local_extractor, worker_cfg
    gpu_id = gpu_id.get()
    global_local_extractor = global_local_factory(gpu_id, _cfg)
    worker_cfg = _cfg


def process_by_worker(data_info):
    """Process input by worker process."""

    image_path, bbox = data_info['image_path'], data_info['bbox']
    image_name = data_info['image_name']
    large_path = os.path.join(worker_cfg.PATH.SEARCH_DIR, image_path)

    global_local_feat = global_local_extractor.extract(large_path, bbox)
    return {image_name: global_local_feat}


class GlobalLocalManager:
    """GlobalLocal manager to manage multiple process."""

    def __init__(self, _cfg):

        self._cfg = _cfg
        num_gpu = _cfg.DISTRIBUTE.NUM_GPU
        worker_per_gpu = _cfg.DISTRIBUTE.WORKER_PER_GPU
        self.num_process = worker_per_gpu * num_gpu
        gpu_ids = Queue()
        gpu_id_cycle_iterator = cycle(range(0, num_gpu))
        for _ in range(self.num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))
        # pylint: disable=consider-using-with
        self.process_pool = Pool(processes=self.num_process,
                                 initializer=init_worker,
                                 initargs=(_cfg, gpu_ids, ))
        print('worker init done')

    @staticmethod
    def split_data_list(query_dic, full_name_list):
        """Split data into query and index.

        For Oxford5k and Paris6k, we remove query from all data in advance.

        Args:
            query_dic: A dict containing query info. For example,
                       {'object_1': {image_name, object_id}, 'object_2': ...}
            full_name_list: A list containing all candidate images.

        Returns: A 4-element tuple containing:
                    query_name_list: A list of query names.
                    query_bbox_list: A list of query gt bboxes.
                    index_name_list: A list of index names.
        """

        query_name_list = []
        query_bbox_list = []
        for object_data in query_dic.values():
            query_name_list.append(object_data['image_name'])
            query_bbox_list.append(object_data['bbox'])

        index_name_list = np.array(full_name_list)
        return (query_name_list, query_bbox_list, index_name_list)

    def save_pickle(self, pool_output, query_flag):
        """Save pickle for global local features.
        We keep the order of query_pid_list for code debug with origin delg

        Args:
            pool_output: A list of multi processing results. Each element is
                         global_local_feat which is 3 tuples
                         (global_feat, local_feat, bbox_info)
            query_flag: A flag to indicate query.
        """

        all_feature_dic = {}
        for feature_dic in pool_output:
            all_feature_dic.update(feature_dic)
        if self._cfg.DELG.CROP_QUERY:
            postfix = 'cropresize'
        else:
            postfix = 'pure'
        feature_dir = ('./tests/features/'
                       f'{self._cfg.EVAL.DATA_MODE}/{self._cfg.EVAL.SIM_MODE}')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        if query_flag:
            pkl_name = os.path.join(feature_dir, f'query_{postfix}.pkl')
            pickle.dump(all_feature_dic, open(pkl_name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)
        else:
            pkl_name = os.path.join(feature_dir, 'index.pkl')
            pickle.dump(all_feature_dic, open(pkl_name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)

    def extract_images(self, query_dic, full_name_list, skip_index=False):
        """Extract global local feature for images.

        Args:
            query_dic: A dict containing query images. For example,
                       {'object_1': [aaa.jpg, bbb.jpg], 'object_2': [ccc.jpg]}
            full_name_list: A list containing all candidate images.
        """

        (query_name_list, query_bbox_list,
         index_name_list) = self.split_data_list(query_dic, full_name_list)
        query_data_list = []
        for k, query_name in enumerate(query_name_list):
            image_path = os.path.join(self._cfg.EVAL.QUERY_DATA_SOURCE,
                                      query_name)
            query_data_list.append({'image_path': image_path,
                                    'bbox': query_bbox_list[k],
                                    'image_name': query_name_list[k]})
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
            process_by_worker, query_data_list), total=len(query_data_list)))
        self.save_pickle(pool_output, query_flag=True)

        if skip_index:
            return
        index_data_list = []
        for k, index_name in enumerate(index_name_list):
            image_path = os.path.join(self._cfg.EVAL.GALLERY_DATA_SOURCE,
                                      index_name)
            index_data_list.append({'image_path': image_path,
                                    'bbox': None,
                                    'image_name': index_name_list[k]})
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
            process_by_worker, index_data_list), total=len(index_data_list)))
        self.save_pickle(pool_output, query_flag=False)

    def finish_worker(self):
        """Finish worker."""

        self.process_pool.close()
        self.process_pool.join()


def main():
    """Main method."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.DELG.USE_LOCAL_FEATURE = 1

    query_instance_dic = pickle.load(open(cfg.EVAL.LABEL_FILE, 'rb'))
    start_time = time.time()
    all_data = load_all_data(cfg)
    full_name_list = [item['image_name'] for item in all_data]
    load_time = time.time() - start_time
    print('%.4f s for load images' % load_time)

    cfg.DELG.CROP_QUERY = False
    global_local_manager = GlobalLocalManager(cfg)
    global_local_manager.extract_images(query_instance_dic, full_name_list,
                                        skip_index=False)
    global_local_manager.finish_worker()

    cfg.DELG.CROP_QUERY = True
    global_local_manager = GlobalLocalManager(cfg)
    global_local_manager.extract_images(query_instance_dic, full_name_list,
                                        skip_index=True)
    global_local_manager.finish_worker()


if __name__ == "__main__":
    main()
