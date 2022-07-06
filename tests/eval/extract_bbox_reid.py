"""Rerank the gallery-feature sim by the model.
We read one large-gallery image once and compute features of all its boxes.
Author: gongyou.zyq
Date: 2020.11.28
"""

import os
import pickle
import time
from itertools import cycle
from multiprocessing import Pool, Queue

import cv2
import numpy as np
import tqdm

from instance_search.instance_search import InstanceSearcher
from instance_search.utils.io_utils import convert_rawdet2dict, \
                                           get_query_info, load_all_data, \
                                           parse_args, save_pickle, \
                                           select_good_id
from instance_search.config import cfg


def add_new_gallery(old_dict, gallery_pred_dict, query_instance_id):
    """Add new gallery pred."""

    gallery_pred = gallery_pred_dict[query_instance_id]
    if len(gallery_pred) > 0:
        old_dict[query_instance_id].update(gallery_pred)
    return old_dict


def init_worker(_cfg, gpu_id, loc_info):
    """
    This gets called to initialize the worker process.
    Here we load the ML model into GPU memory.
    Each worker process will pull an GPU ID from a queue of available IDs
    (e.g. [0, 1, 2, 3]) to ensure that multiple. GPUs are consumed evenly.
    """

    # pylint: disable=global-variable-undefined, invalid-name
    global instance_searcher, worker_cfg, gallery_bbox_dic
    global gq_activate_dic
    gpu_id = gpu_id.get()
    instance_searcher = InstanceSearcher(gpu_id, _cfg)
    worker_cfg = _cfg
    gallery_bbox_dic = loc_info
    gq_activate_dic = {}

    for query_instance_id, q_pred_dic in loc_info.items():
        for gallery_name in q_pred_dic:
            if gallery_name not in gq_activate_dic:
                gq_activate_dic[gallery_name] = [query_instance_id]
            else:
                gq_activate_dic[gallery_name].append(query_instance_id)


def set_query_worker(query_meta_list):
    """Set query for each worker."""

    instance_searcher.reset()
    for query_meta in query_meta_list:
        query_large_path, query_bbox, query_instance_id = query_meta[:3]
        query_img_bgr = cv2.imread(query_large_path)
        query_bbox = query_bbox.astype('int')
        instance_searcher.refiner.set_query(query_img_bgr, query_bbox,
                                            query_instance_id)
    time.sleep(10)


def process_by_worker(data_info):
    """Process_input_by_worker_process."""

    large_img_list = []
    gallery_large_path = data_info['image_path']
    gallery_name = data_info['image_name']
    large_img_list.append(cv2.imread(gallery_large_path))

    try:
        activated_query_list = gq_activate_dic[gallery_name]
    except KeyError:
        # no query is responsible for this gallery
        activated_query_list = []
    instance_searcher.set_activated_query(activated_query_list)
    rough_det_queries = {}
    for query_instance_id in activated_query_list:
        pair_res = gallery_bbox_dic[query_instance_id][gallery_name]
        concat_bbox = np.concatenate((pair_res['bbox'],
                                      pair_res['sim'][:, np.newaxis]), axis=1)
        rough_det_queries[query_instance_id] = [concat_bbox]
    refined_det_queries = instance_searcher.reid_rerank(rough_det_queries,
                                                        large_img_list)

    query_refine_dict = {}
    for query_instance_id in activated_query_list:
        # rough_det, (N, 5)
        refine_det = refined_det_queries[query_instance_id][0]
        refine_dict = convert_rawdet2dict(refine_det)
        query_refine_dict[query_instance_id] = {gallery_name: refine_dict}
    return query_refine_dict


class RefineManager:
    """Refine manager to manage multiple process."""

    def __init__(self, _cfg):
        self._cfg = _cfg
        num_gpu = _cfg.DISTRIBUTE.NUM_GPU
        worker_per_gpu = _cfg.DISTRIBUTE.WORKER_PER_GPU
        self.num_process = worker_per_gpu * num_gpu
        gpu_ids = Queue()
        gpu_id_cycle_iterator = cycle(range(0, num_gpu))
        for _ in range(self.num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))
        pkl_file = f'./tests/features/{_cfg.EVAL.DATA_MODE}/'\
                   f'{_cfg.EVAL.ROUGH_LOCALIZER}.pkl'
        self.all_loc_info = pickle.load(open(pkl_file, 'rb'))
        self.all_query_instance_list = sorted(self.all_loc_info.keys())
        print('worker init done')
        # pylint: disable=consider-using-with
        self.process_pool = Pool(processes=self.num_process,
                                 initializer=init_worker,
                                 initargs=(_cfg, gpu_ids, self.all_loc_info, ))

    def get_process_num(self):
        """Get process number."""

        return self.num_process

    def set_query(self, query_meta_list):
        """Set query."""

        self.process_pool.map(set_query_worker,
                              [query_meta_list] * self.get_process_num())

    def recompute_sim(self, data_list):
        """Rerank feature to measure speed."""

        if self._cfg.EVAL.OUTPUT_SPEED:    # warmup
            pool_output = self.process_pool.map(process_by_worker,
                                                data_list[:1000])
            start_time = time.time()
            num_loops = len(data_list)

        # pool_output = self.process_pool.map(process_by_worker, data_list)
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
                process_by_worker, data_list), total=len(data_list)))
        if self._cfg.EVAL.OUTPUT_SPEED:    # warmup
            duration = (time.time()-start_time)*1000.0/num_loops
            fps = 1000.0 / duration
            print(f'time for one image: {duration:4f} ms, fps: {int(fps)}')

        start_time = time.time()
        refine_dict = {item: {'image_name': [], 'sim': [], 'bbox': []}
                       for item in self.all_query_instance_list}

        for sample_dic in pool_output:
            query_refine_dict = sample_dic
            for query_instance_id in query_refine_dict:
                refine_dict = add_new_gallery(refine_dict, query_refine_dict,
                                              query_instance_id)

        save_pickle(refine_dict, self._cfg.EVAL.REFINER,
                    self._cfg.EVAL.DATA_MODE)
        print(f'save feature time {time.time()-start_time}')

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
    cfg.freeze()

    query_instance_dic = pickle.load(open(cfg.EVAL.LABEL_FILE, 'rb'))
    selected_test_id = select_good_id(cfg, query_instance_dic)
    refine_manager = RefineManager(cfg)
    data_list = load_all_data(cfg)
    start_time = time.time()
    query_meta_list = []
    for query_instance_id in selected_test_id[:cfg.EVAL.TEST_QUERY_NUM]:
        query_bbox_dic = query_instance_dic[query_instance_id]
        query_large_path, query_bbox = get_query_info(cfg, query_bbox_dic)
        query_meta_list.append([query_large_path, query_bbox,
                                query_instance_id, query_bbox_dic, data_list])
        if cfg.EVAL.VERBOSE:
            print(f'Query instance id: {query_instance_id}, '
                  f'query_large_path: {query_large_path}, '
                  f'bbox: {query_bbox}')
    refine_manager.set_query(query_meta_list)
    print('All queries init done!')
    print('%.4f s for set query' % (time.time() - start_time))
    refine_manager.recompute_sim(data_list)


if __name__ == "__main__":
    main()
