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
import numpy as np
import tqdm

from instance_search.instance_search import InstanceSearcher
from instance_search.utils.io_utils import convert_rawdet2dict, \
                                           get_query_info, load_all_data, \
                                           parse_args, save_pickle, \
                                           select_good_id
from instance_search.config import cfg
from tests.eval.eval_ins import INSEvaluator


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
    global instance_searcher, worker_cfg, worker_gq_activate_dic
    gpu_id = gpu_id.get()
    instance_searcher = InstanceSearcher(gpu_id, _cfg)
    worker_cfg = _cfg
    worker_gq_activate_dic = {}


def set_query_worker(query_feat_list):
    """Set query for each worker."""

    instance_searcher.reset()
    [all_query_loc_feat, all_query_reid_feat, gq_activate_dic] = query_feat_list
    for k, v in gq_activate_dic.items():
        worker_gq_activate_dic[k] = v
    # worker_gq_activate_dic = gq_activate_dic.copy()
    instance_searcher.set_query_feat(all_query_loc_feat, all_query_reid_feat)
    time.sleep(60)


def extract_query_worker(query_meta):
    """Extract one query feature."""

    query_large_path, query_bbox, query_instance_id = query_meta[:3]
    query_img_bgr = cv2.imread(query_large_path)
    query_bbox = query_bbox.astype('int')
    [query_loc_feat_dic, query_reid_feat_dic] = instance_searcher.set_query(
            query_img_bgr, query_bbox, query_instance_id)
    return [query_loc_feat_dic, query_reid_feat_dic]


def process_by_worker(data_info):
    """Process input by worker process."""

    large_img_list = []
    gallery_large_path = data_info['image_path']
    gallery_name = data_info['image_name']
    large_img_list.append(cv2.imread(gallery_large_path))

    try:
        activated_query_list = worker_gq_activate_dic[gallery_name]
    except KeyError:
        # no query is responsible for this gallery
        activated_query_list = []
    instance_searcher.set_activated_query(activated_query_list)
    rough_det_queries, refined_det_queries = instance_searcher.detect(
            large_img_list)

    query_rough_dict = {}
    query_refine_dict = {}
    for query_instance_id in activated_query_list:
        # rough_det, (N, 5)
        rough_det = rough_det_queries[query_instance_id][0]
        refine_det = refined_det_queries[query_instance_id][0]
        rough_dict = convert_rawdet2dict(rough_det)
        refine_dict = convert_rawdet2dict(refine_det)
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

    def get_gq_activate(self, query_meta_list):
        """Get gq_activate_dic for global or local."""

        if self._cfg.EVAL.GALLERY_RANGE == 'global':
            gq_activate_dic = self.get_gq_activate_global(query_meta_list)
        else:
            gq_activate_dic = self.get_gq_activate_local(query_meta_list)
        return gq_activate_dic

    @staticmethod
    def get_gq_activate_global(query_meta_list):
        """Get gq_activate_dic with global range."""

        gallery_list = [item['image_name'] for item in query_meta_list[0][-1]]
        query_id_list = []
        gq_activate_dic = {}
        for query_meta in query_meta_list:
            query_instance_id = query_meta[2]
            query_id_list.append(query_instance_id)
        for activate_gal_name in gallery_list:
            gq_activate_dic[activate_gal_name] = query_id_list
        return gq_activate_dic

    def get_gq_activate_local(self, query_meta_list):
        """Get gq_activate_dic with local range."""

        gq_activate_dic = {}
        for query_meta in query_meta_list:
            query_instance_id = query_meta[2]
            query_bbox_dic = query_meta[3]
            if self._cfg.EVAL.GALLERY_RANGE == 'local':
                gallery_list = [item['image_name']
                                for item in query_bbox_dic['gallery_list']]
            elif self._cfg.EVAL.GALLERY_RANGE == 'pos':
                gallery_list = [item['image_name']
                                for item in query_bbox_dic['pos_gallery_list']]
            else:
                print('Unknown GALLERY_RANGE')
                return {}

            for activate_gal_name in gallery_list:
                if activate_gal_name not in gq_activate_dic:
                    gq_activate_dic[activate_gal_name] = [query_instance_id]
                else:
                    gq_activate_dic[activate_gal_name].append(query_instance_id)
            # print('set query ok', current_process())
        # instance_searcher.localizer.get_mean_template()
        return gq_activate_dic

    def get_process_num(self):
        """Get process number."""

        return self.num_process

    def set_query(self, query_meta_list):
        """Set query."""

        query_meta_list = query_meta_list
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
                extract_query_worker, query_meta_list), total=len(query_meta_list)))
        all_query_loc_feat = {}
        all_query_reid_feat = {}
        for sample_dic in pool_output:
            [query_loc_feat_dic, query_reid_feat_dic] = sample_dic
            all_query_loc_feat.update(query_loc_feat_dic)
            all_query_reid_feat.update(query_reid_feat_dic)
        # convert to cpu numpy
        # for query_instance_id in list(all_query_loc_feat.keys()):
        #     all_query_loc_feat[query_instance_id] = \
        #             all_query_loc_feat[query_instance_id].cpu().numpy()
        gq_activate_dic = self.get_gq_activate(query_meta_list)

        query_feat_list = [all_query_loc_feat, all_query_reid_feat,
                           gq_activate_dic]
        self.process_pool.map(set_query_worker,
                              [query_feat_list] * self.get_process_num())

    def instance_search_images(self, data_list):
        """Instance search images."""

        if self._cfg.EVAL.OUTPUT_SPEED:    # warmup
            pool_output = self.process_pool.map(process_by_worker,
                                                data_list[:100])
            start_time = time.time()
            num_loops = len(data_list)

        start_time = time.time()
        # pool_output = self.process_pool.map(process_by_worker, data_list)
        pool_output = list(tqdm.tqdm(self.process_pool.imap_unordered(
                process_by_worker, data_list), total=len(data_list)))
        print(f'Process time {time.time()-start_time:.4f} s for '
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
        print(f'Postprocess time {time.time()-start_time:.4f} s')

        if self._cfg.EVAL.CACHE_FEATURE:
            save_pickle(rough_dict, self._cfg.EVAL.ROUGH_LOCALIZER,
                        self._cfg.EVAL.DATA_MODE)
            if self._cfg.EVAL.REFINER != 'null':
                save_pickle(refine_dict, self._cfg.EVAL.REFINER,
                            self._cfg.EVAL.DATA_MODE)
                print(f'save feature time {time.time()-start_time:.4f} s')
        else:
            ins_evaluator = INSEvaluator(self._cfg)
            ins_evaluator.eval_data(rough_dict, refine_dict)
            ins_evaluator.output_final_result()

    def finish_worker(self):
        """Finish worker."""

        print('Finish worker')
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
    start_time = time.time()
    query_meta_list = []
    for query_instance_id in selected_test_id[:cfg.EVAL.TEST_QUERY_NUM]:
        end2end_manager.add_query_instance(query_instance_id)
        query_bbox_dic = query_instance_dic[query_instance_id]
        query_large_path, query_bbox = get_query_info(cfg, query_bbox_dic)
        object_id = query_bbox_dic['object_id']

        query_meta_list.append([query_large_path, query_bbox,
                                query_instance_id, query_bbox_dic, data_list])
        if cfg.EVAL.VERBOSE:
            print(f'Query instance id: {query_instance_id}, '
                  f'object id: {object_id}, bbox: {query_bbox}'
                  f'query_large_path: {query_large_path}')
    end2end_manager.set_query(query_meta_list)
    print('All queries init done!')
    print('%.4f s for set query' % (time.time() - start_time))

    end2end_manager.instance_search_images(data_list)
    end2end_manager.finish_worker()


def test_save_speed():
    """Test save speed."""

    query_num = 50
    gallery_num = 700000
    topk_per_image = 5
    all_query_instance_list = [f'query_{i}' for i in range(query_num)]
    rough_dict = {item: {} for item in all_query_instance_list}
    refine_dict = {item: {} for item in all_query_instance_list}

    for j in range(gallery_num):
        if j % 100 == 0:
            print(j)
        dummy_bbox = {'sim': np.zeros((topk_per_image)),
                      'bbox': np.zeros((topk_per_image, 4))}
        dummy_gallery = {f'gallery_{j}': dummy_bbox}
        query_dict = {item: dummy_gallery for item in all_query_instance_list}
        for query_instance_id in query_dict:
            rough_dict = add_new_gallery(rough_dict, query_dict,
                                         query_instance_id)
            refine_dict = add_new_gallery(refine_dict, query_dict,
                                          query_instance_id)
    start_time = time.time()
    save_pickle(rough_dict, 'debug', 'debug')
    save_pickle(refine_dict, 'debug', 'debug')
    print(f'save feature time {time.time()-start_time}')


if __name__ == "__main__":
    main()
    # test_save_speed()
