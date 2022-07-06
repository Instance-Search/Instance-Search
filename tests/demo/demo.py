"""Eval end-to-end."""

from glob import glob
import os
import time
import pickle
import numpy as np
import cv2

# from instance_search.utils.io_utils import load_all_data
from tests.eval.extract_ins import End2EndManager
from instance_search.instance_search import InstanceSearcher

from instance_search.config import cfg
from instance_search.utils.io_utils import split_group2pred

# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements
def draw_bbox(image_path, bbox, color):
    """Draw bbox."""

    img = cv2.imread(image_path)
    [x1, y1, x2, y2] = bbox    # pylint:disable=invalid-name
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return img

def vis_retrieval(_cfg, query_instance_id, query_large_path, query_bbox, all_pred_dic):
    """Vis retrieval."""

    pred_info = all_pred_dic[query_instance_id]
    pred_info = split_group2pred(pred_info)
    save_dir = f'./tests/images/vis_pred/{_cfg.EVAL.DATA_MODE}/' \
               f'reid_result/{query_instance_id}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = draw_bbox(query_large_path, query_bbox, (255, 0, 0))
    cv2.putText(img, 'query',
                (int(query_bbox[0]), int(query_bbox[1])),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(save_dir, 'query.jpg'), img)

    orders = np.argsort(-pred_info['sim'])

    bbox_list = pred_info['bbox'][orders]
    sim = pred_info['sim'][orders]
    image_ids = pred_info['image_name'][orders]
    for index, image_name in enumerate(image_ids[:_cfg.EVAL.VIS_TOPK]):
        sim_str = '%.4f' % sim[index]
        gal_bbox = bbox_list[index].astype('int')
        save_name = sim_str + '_' + image_name[:-4]+\
        f'_bbox_{gal_bbox[0]}_{gal_bbox[1]}_{gal_bbox[2]}_{gal_bbox[3]}.jpg'
        prefix = image_name.split('_')[0]
        old_path = os.path.join(_cfg.PATH.SEARCH_DIR, 'train', prefix, image_name)
        if not os.path.exists(old_path):
            old_path = os.path.join(_cfg.PATH.SEARCH_DIR, 'test', prefix, image_name)
        img = draw_bbox(old_path, gal_bbox, (0, 255, 0))
        cv2.putText(img, sim_str, (int(gal_bbox[0]), int(gal_bbox[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, save_name), img)


def eval_offline(_cfg):
    """Eval offline."""

    # _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/frames_all/"
    # _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/PRW-v16.04.20/frames/"
    # _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/jiasheng_test/folder/"

    MARKET_FLAG = False
    if MARKET_FLAG:
        _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/instance_search/xianzhe/L2G/Market-1501/"
        xianzhe_anno = pickle.load(open('/home/gongyou.zyq/datasets/instance_search/xianzhe/L2G/Market_query_gallery.pkl', 'rb'))
        _cfg.EVAL.DATASET_NAME = "aa_market"
    else:
        _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/instance_search/xianzhe/L2G/MSMT/"
        xianzhe_anno = pickle.load(open('/home/gongyou.zyq/datasets/instance_search/xianzhe/L2G/MSMT_query_gallery.pkl', 'rb'))
        _cfg.EVAL.DATASET_NAME = "aa_msmt"
    QUERY_NUM = 50
    GALLERY_NUM = 100000

    # _cfg.EVAL.ROUGH_LOCALIZER = 'full_bbox'
    # _cfg.EVAL.REFINER = "reid"
    _cfg.EVAL.ROUGH_LOCALIZER = 'globaltrack'
    _cfg.EVAL.REFINER = "null"

    _cfg.EVAL.VIS_TOPK = 50
    _cfg.EVAL.DATA_MODE = "demo/baseline"
    _cfg.DISTRIBUTE.WORKER_PER_GPU = 3

    # NOTE: tricky but fast config, for local search only
    _cfg.EVAL.MULTI_QUERY = False
    _cfg.INFERENCE.REID_BATCH_SIZE = 1
    _cfg.INFERENCE.LOCALIZER_TOPK = 1
    _cfg.EVAL.MIN_HEIGHT = 0
    _cfg.EVAL.REFINE_MIN_THR = -1.0    # NOTE: very important for speed
    _cfg.EVAL.CACHE_FEATURE = True

    # # NOTE: common setting, almost impossible
    # _cfg.EVAL.MULTI_QUERY = False
    # _cfg.INFERENCE.REID_BATCH_SIZE = 1
    # _cfg.INFERENCE.LOCALIZER_TOPK = 1
    # _cfg.EVAL.MIN_HEIGHT = 5
    # _cfg.EVAL.REFINE_MIN_THR = 0.4    # NOTE: very important for speed

    _cfg.INFERENCE.GLOBALTRACK_SCALE = (128, 256)    # NOTE: for xianzhe
    print(_cfg)
    end2end_manager = End2EndManager(_cfg)

    # self_defined = [['./demo/test_1.jpg', [1090, 550, 1160, 680]]]
    # self_defined = [['./demo/test_2.jpg', [440, 480, 520, 560]]]
    # self_defined = [['./demo/test_1.jpg', [500, 290, 720, 483]],
    #                 ['./demo/test_2.jpg', [520, 450, 800, 690]],
    #                 ['./demo/test_2.jpg', [157, 40, 240, 220]]]

    # self_defined = [['./demo/c2s2_120577.jpg', [1100, 552, 1210, 608]],
    #                 ['./demo/c3s2_033412.jpg', [1500, 355, 1747, 487]],
    #                 ['./demo/c4s2_000642.jpg', [1110, 486, 1220, 609]],
    #                 ['./demo/c6s2_000343.jpg', [30, 209, 76, 245]]]
    # self_defined = [['/home/gongyou.zyq/datasets/jiasheng_test/_crop_face/face_1f919d8a73f4bde653f3537e320d1b98_344480.jpg', [44, 38, 109, 141]],
    #                 ['/home/gongyou.zyq/datasets/jiasheng_test/_crop_face/face_2b6bb3dfb6e522b8e7892407b2e50ed2_375737.jpg', [27, 28, 84, 109]],
    #                 ['/home/gongyou.zyq/datasets/jiasheng_test/_crop_face/face_ac8c021f61c37e1dbfb6eef051ff0ac5_163173.jpg', [42, 33, 106, 122]]]


    # query_name_list = [x for x,y in xianzhe_anno['query']]
    # print(len(query_name_list), len(set(query_name_list)))
    # print(query_name_list[:10])
    self_defined = []
    for query_image_name, query_bbox in xianzhe_anno['query']:
        self_defined.append([os.path.join(_cfg.PATH.SEARCH_DIR, query_image_name), query_bbox])
    self_defined = self_defined[:QUERY_NUM]
    print(f'{len(self_defined)} queries')
    image_list = []
    for gallery_image_name in xianzhe_anno['gallery']:
        image_list.append(os.path.join(_cfg.PATH.SEARCH_DIR, gallery_image_name))
    data_list = [{'image_path': item, 'image_name': os.path.basename(item)} for item in image_list]
    data_list = data_list[:GALLERY_NUM]
    # print(f'{len(data_list)} gallery data')
    query_meta_list = []
    start_time = time.time()
    for sample_index, sample_info in enumerate(self_defined):
        [query_large_path, query_bbox] = sample_info
        query_bbox = np.array(query_bbox)
        query_instance_id = f'demo_{sample_index:08d}'
        # query_instance_id = query_large_path.split(_cfg.PATH.SEARCH_DIR)[1]
        end2end_manager.add_query_instance(query_instance_id)
        query_bbox_dic = {query_instance_id: None}
        query_meta_list.append([query_large_path, query_bbox, query_instance_id, query_bbox_dic, data_list])
    end2end_manager.set_query(query_meta_list)
    print(f'{time.time()-start_time} for set query')
    end2end_manager.instance_search_images(data_list)
    end2end_manager.finish_worker()

    all_pred_dic = pickle.load(open(f'./tests/features/{_cfg.EVAL.DATA_MODE}/globaltrack.pkl', 'rb'))
    # all_pred_dic = pickle.load(open(f'./tests/features/{_cfg.EVAL.DATA_MODE}/reid.pkl', 'rb'))
    save_dic = {}
    save_dic['query'] = [x for x,y in xianzhe_anno['query'][:QUERY_NUM]]
    save_dic['gallery'] = [x for x in xianzhe_anno['gallery'][:GALLERY_NUM]]
    sim_matrix = np.zeros((QUERY_NUM, len(save_dic['gallery'])), dtype=np.float16)
    for query_index in range(len(all_pred_dic)):
        query_instnace_id = f'demo_{query_index:08d}'
        query_res = all_pred_dic[query_instnace_id]
        for gallery_index, item in enumerate(data_list):
            gallery_name = item['image_name']
            res = query_res[os.path.basename(gallery_name)]
            res['sim'] = res['sim'].astype('float16')
            try:
                sim = res['sim'][0]
            except:
                sim = 0.0
            sim_matrix[query_index, gallery_index] = sim
    save_dic['sim'] = sim_matrix
    pickle.dump(save_dic, open(f'{_cfg.EVAL.DATASET_NAME}_{_cfg.EVAL.ROUGH_LOCALIZER}.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    dfa

    # all_pred_dic = pickle.load(open(f'./tests/features/{_cfg.EVAL.DATA_MODE}/reid.pkl', 'rb'))
    # pred_num = []
    # for query_result in all_pred_dic.values():
    #     pred_num.append(len(query_result['image_name']))
    # print(f'mean pred num: {np.mean(pred_num)}')
    for query_meta in query_meta_list:
        query_large_path, query_bbox, query_instance_id = query_meta[:3]
        vis_retrieval(cfg, query_instance_id, query_large_path, query_bbox, all_pred_dic)

def test_search_speed(_cfg):
    """Eval offline."""

    _cfg.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/xianzhe/AA-Market/patch_person/"
    _cfg.EVAL.REFINER = "null"
    _cfg.EVAL.VIS_TOPK = 50
    _cfg.EVAL.DATASET_NAME = "demo"
    _cfg.EVAL.DATA_MODE = "demo/baseline"

    _cfg.EVAL.MULTI_QUERY = False
    _cfg.INFERENCE.REID_BATCH_SIZE = 1
    _cfg.INFERENCE.LOCALIZER_TOPK = 1
    _cfg.EVAL.MIN_HEIGHT = 0
    _cfg.EVAL.REFINE_MIN_THR = 0.0    # NOTE: very important for speed
    _cfg.INFERENCE.GLOBALTRACK_SCALE = (128, 256)    # NOTE: for xianzhe
    print(_cfg)

    instance_searcher = InstanceSearcher(0, _cfg)
    xianzhe_anno = pickle.load(open('/home/gongyou.zyq/datasets/xianzhe/AA-Market/patchLabel.pkl', 'rb'))
    query_num = 1
    gallery_num = 1
    activated_query_list = []
    for query_image_name, query_bbox in sorted(list(xianzhe_anno.items())):
        if 'test_probe' not in query_image_name:
            continue
        if len(activated_query_list) >= query_num:
            continue
        query_img_bgr = cv2.imread(_cfg.PATH.SEARCH_DIR + query_image_name)
        instance_searcher.set_query(query_img_bgr, query_bbox, query_image_name)
        activated_query_list.append(query_image_name)
    print(f'{len(activated_query_list)} queries')
    instance_searcher.localizer.get_mean_template()
    instance_searcher.set_activated_query(activated_query_list)
    gallery_image_list = glob(_cfg.PATH.SEARCH_DIR+'test_gallery/*.jpg')
    for gallery_path in gallery_image_list[:gallery_num]:
        img_bgr = cv2.imread(gallery_path)
        start_time = time.time()
        for _ in range(100):
            instance_searcher.reset()
            instance_searcher.set_query(query_img_bgr, query_bbox, query_image_name)
            activated_query_list = [query_image_name]
            instance_searcher.set_activated_query(activated_query_list)
            det = instance_searcher.detect([img_bgr])
        print(time.time()-start_time)

def main():
    """Main method."""

    cfg.merge_from_file('./instance_search/config/PRW.yml')
    eval_offline(cfg)
    # test_search_speed(cfg)

if __name__ == "__main__":
    main()
