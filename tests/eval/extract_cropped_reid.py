# pylint: disable=line-too-long, global-variable-undefined, invalid-name
"""
Inspired by reid annotation multi processing, we could use multi processing
to speed up.
Author: Yuqi Zhang
Date: 2019.10.25
"""

import os
from glob import glob
import time
from itertools import cycle
from multiprocessing import Pool, Queue
import cv2

from instance_search.config import cfg
from instance_search.reid.reid_inference import ReIDInference
# from sift_inference import SiftInference
# from delg_inference import DelgInference

def init_worker(_cfg, gpu_id):
    """
    This gets called to initialize the worker process. Here we load the ML model into GPU memory.
    Each worker process will pull an GPU ID from a queue of available IDs (e.g. [0, 1, 2, 3]) to ensure that multiple
    GPUs are consumed evenly.
    """

    global model
    model = ReIDInference(gpu_id.get(), _cfg)
    # model = SiftInference(_cfg, gpu_id.get())
    # model = DelgInference(_cfg, gpu_id.get())

def process_input_by_worker_process(image_path):
    """Process_input_by_worker_process."""

    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    reid_feat = model.extract(img)
    reid_feat = reid_feat.astype('float32')
    sample_mode = image_path.split('/')[-2]
    return [{image_name: reid_feat}, sample_mode]


def load_all_data(_cfg):
    """Load all mode data."""

    valset_dir = _cfg.PATH.REID_DIR
    image_list = []
    for mode in ['test_gallery', 'test_probe']:
        mode_list = glob(valset_dir + mode + '/*.*')
        print(f'{len(mode_list)} images for {mode}')
        image_list += mode_list

    return image_list


def gather_feature(_cfg, pool_output):
    """Gather feature."""

    gallery_dic = {}
    probe_dic = {}
    for sample_result in pool_output:
        [sample_dic, sample_mode] = sample_result
        if sample_mode in ['test_gallery', 'test_distract']:
            gallery_dic.update(sample_dic)
        else:
            probe_dic.update(sample_dic)
    return probe_dic, gallery_dic


def extract_feature(_cfg):
    """Extract reid feat for each image, using multiprocessing."""

    image_list = load_all_data(_cfg)

    num_process = 16
    gpu_ids = Queue()
    gpu_id_cycle_iterator = cycle(range(0, 8))
    for _ in range(num_process):
        gpu_ids.put(next(gpu_id_cycle_iterator))
    process_pool = Pool(processes=num_process, initializer=init_worker,
                        initargs=(_cfg, gpu_ids, ))
    start_time = time.time()
    pool_output = process_pool.map(process_input_by_worker_process, image_list)
    process_pool.close()
    process_pool.join()
    print('%.4f s' % (time.time() - start_time))

    probe_dic, gallery_dic = gather_feature(_cfg, pool_output)
    return probe_dic, gallery_dic


def main():
    """Main method."""

    cfg.merge_from_file('./config/market.yml')
    cfg.freeze()
    extract_feature(cfg)


if __name__ == "__main__":
    main()
