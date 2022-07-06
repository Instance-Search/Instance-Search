"""Extract end2end instance search results with multi processing.
Author: gongyou.zyq
Date: 2020.11.28
"""

import os
import pickle
import time
from itertools import cycle
from multiprocessing import Pool, Queue

import cv2
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


def init_worker(_cfg, gpu_id):
    """
    This gets called to initialize the worker process.
    Here we load the ML model into GPU memory.
    Each worker process will pull an GPU ID from a queue of available IDs
    (e.g. [0, 1, 2, 3]) to ensure that multiple. GPUs are consumed evenly.
    """

    # pylint: disable=global-variable-undefined, invalid-name
    global instance_searcher, worker_cfg
    gpu_id = gpu_id.get()
    instance_searcher = InstanceSearcher(gpu_id, _cfg)
    worker_cfg = _cfg


def get_gallery_list(_cfg, query_bbox_dic, full_gallery_list):
    """Get gallery list for a query instance bbox."""

    if _cfg.EVAL.GALLERY_RANGE == 'global':
        gallery_list = [item['image_name'] for item in full_gallery_list]
    elif _cfg.EVAL.GALLERY_RANGE == 'local':
        gallery_list = [item['image_name']
                        for item in query_bbox_dic['gallery_list']]
    elif _cfg.EVAL.GALLERY_RANGE == 'pos':
        gallery_list = [item['image_name']
                        for item in query_bbox_dic['pos_gallery_list']]
    else:
        print('Unknown GALLERY_RANGE')
        gallery_list = None
    return gallery_list


def process_by_worker(work_info):
    """Process input by worker process."""

    instance_searcher.reset()
    [query_large_path, query_bbox, query_instance_id, gallery_name] = work_info
    query_img_bgr = cv2.imread(query_large_path)
    instance_searcher.set_query(query_img_bgr, query_bbox, query_instance_id)

    gallery_large_path = os.path.join(worker_cfg.PATH.SEARCH_DIR,
                                      worker_cfg.EVAL.GALLERY_DATA_SOURCE,
                                      gallery_name)

    instance_searcher.set_activated_query([query_instance_id])
    rough_det_queries, refined_det_queries = instance_searcher.detect(
            [cv2.imread(gallery_large_path)])

    rough_det = rough_det_queries[query_instance_id][0]
    refine_det = refined_det_queries[query_instance_id][0]
    rough_dict = convert_rawdet2dict(rough_det)
    refine_dict = convert_rawdet2dict(refine_det)
    query_rough_dict, query_refine_dict = {}, {}
    query_rough_dict[query_instance_id] = {gallery_name: rough_dict}
    query_refine_dict[query_instance_id] = {gallery_name: refine_dict}

    return [query_rough_dict, query_refine_dict]


class End2EndManager:
    """End2End manager to manage multiple process."""

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
        self.all_query_instance_list = []
        print('worker init done')

    def instance_search_images(self, work_list):
        """Instance search images."""

        if self._cfg.EVAL.OUTPUT_SPEED:    # warmup
            pool_output = self.process_pool.map(process_by_worker,
                                                work_list[:100])
            start_time = time.time()
            num_loops = len(work_list)

        start_time = time.time()
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
            process_by_worker, work_list), total=len(work_list)))
        print(f'process time {time.time()-start_time} for '
              f'{len(self.all_query_instance_list)} query instances')

        if self._cfg.EVAL.OUTPUT_SPEED:
            duration = (time.time()-start_time)*1000.0/num_loops
            fps = 1000.0 / duration
            print(f'time for instance search one image: {duration:4f} ms, '
                  f'fps: {int(fps)}')

        start_time = time.time()
        rough_dict = {item: {} for item in self.all_query_instance_list}
        refine_dict = {item: {} for item in self.all_query_instance_list}
        for sample_dic in pool_output:
            [query_rough_dict, query_refine_dict] = sample_dic
            for query_instance_id in query_rough_dict:
                rough_dict = add_new_gallery(rough_dict, query_rough_dict,
                                             query_instance_id)
                refine_dict = add_new_gallery(refine_dict, query_refine_dict,
                                              query_instance_id)

        save_pickle(rough_dict, self._cfg.EVAL.ROUGH_LOCALIZER,
                    self._cfg.EVAL.DATA_MODE)
        if self._cfg.EVAL.REFINER != 'null':
            save_pickle(refine_dict, self._cfg.EVAL.REFINER,
                        self._cfg.EVAL.DATA_MODE)
        print(f'save feature time {time.time()-start_time}')

    def finish_worker(self):
        """Finish worker."""

        self.process_pool.close()
        self.process_pool.join()
        self.all_query_instance_list = []

    def add_query_instance(self, query_instance_id):
        """Add query instance."""

        self.all_query_instance_list.append(query_instance_id)


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
    end2end_manager = End2EndManager(cfg)
    data_list = load_all_data(cfg)
    work_list = []
    for query_instance_id in selected_test_id[:cfg.EVAL.TEST_QUERY_NUM]:
        end2end_manager.add_query_instance(query_instance_id)
        query_bbox_dic = query_instance_dic[query_instance_id]
        gallery_list = get_gallery_list(cfg, query_bbox_dic, data_list)
        query_large_path, query_bbox = get_query_info(cfg, query_bbox_dic)
        for item in gallery_list:
            work_list.append([query_large_path, query_bbox,
                              query_instance_id, item])
        object_id = query_bbox_dic['object_id']
        if cfg.EVAL.VERBOSE:
            print(f'Query instance id: {query_instance_id}, '
                  f'object id: {object_id}, bbox: {query_bbox}'
                  f'query_large_path: {query_large_path}')
    print(f'{len(work_list)} pairs to forward')

    end2end_manager.instance_search_images(work_list)
    end2end_manager.finish_worker()


if __name__ == "__main__":
    main()
