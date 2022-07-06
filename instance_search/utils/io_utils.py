"""
IO utils for instance search.
label example: a dict, key is object_id, value is a list of sub-dict.
sub-dict example: {'device_id': device_id,
                   'raw_image_name': raw image name,
                   'image_name': customed short image name,
                   'bbox': bbox with [x1, y1, x2, y2],
                   'object_id': object_id,
                   'ignore': 0/1}
See `features/bag_search_label.pkl` as an example.
pred example: a dict, {'image_name': image_name_list(short), 'bbox': bbox_list,
                       'sim': np.array(sim_list)}.
See `features/bag/6047/globaltrack_sim.pkl` as an example.
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import cv2
import numpy as np

from instance_search.config import cfg


# pylint: disable=line-too-long, too-many-locals, invalid-name, no-else-return
def get_query_info(_cfg, query_bbox_dic):
    """Convenience use for only image_full_path and bbox."""

    query_dir = os.path.join(_cfg.PATH.SEARCH_DIR, _cfg.EVAL.QUERY_DATA_SOURCE)
    query_image_name = query_bbox_dic['image_name']
    query_large_path = os.path.join(query_dir, query_image_name)
    query_bbox = query_bbox_dic['bbox']
    # query_object_id = query_bbox_dic['object_id']
    return query_large_path, query_bbox


def get_gallery_info(_cfg, gallery_bbox_dic_list, sample_index):
    """Convenience use for only image_full_path and bbox."""

    gallery_bbox_dic = gallery_bbox_dic_list[sample_index]
    gallery_image_name = gallery_bbox_dic['image_name']
    gallery_large_path = os.path.join(_cfg.PATH.SEARCH_DIR,
                                      _cfg.EVAL.GALLERY_DATA_SOURCE,
                                      gallery_image_name)
    gallery_bbox = gallery_bbox_dic['bbox']
    return gallery_large_path, gallery_bbox


def select_good_id(_cfg, query_instance_dic):
    """Select ids with more samples for better evaluation."""

    selected_test_id = []
    for object_id in query_instance_dic:
        pos_gallery_list = query_instance_dic[object_id]['pos_gallery_list']
        if len(pos_gallery_list) >= _cfg.EVAL.MIN_SAMPLE_PER_ID - 1:
            selected_test_id.append(object_id)
    print(f'{len(query_instance_dic.keys())} query instance ids, '
          f'{len(selected_test_id)} valid query instance ids')
    selected_test_id = sorted(selected_test_id)
    return selected_test_id


def refine_bboxes(rough_det, _cfg):
    """
    Refine bboxes by their size and score.
    In: rough_det, (N, 5)
    Out: good mask
    """

    diff_height = rough_det[:, 3] - rough_det[:, 1]
    diff_width = rough_det[:, 2] - rough_det[:, 0]
    size_condition_mask = (diff_height > _cfg.EVAL.MIN_HEIGHT) * \
                          (diff_width > 1.0)
    if _cfg.EVAL.REFINE_MIN_THR != 0.0:
        score_condition_mask = rough_det[:, -1] > _cfg.EVAL.REFINE_MIN_THR
        good_data_mask = score_condition_mask * size_condition_mask
    else:
        good_data_mask = size_condition_mask
    return good_data_mask


def split_group2pred(result_dic):
    """
    Split group format {image_name: [all its bbox]}
    into pred format {image_name: [], sim: [], bbox: []}, for output results.
    """

    merged_bboxes = []
    merged_sims = []
    unique_image_ids = []
    repeat_times = []
    for image_name, pred_for_large in result_dic.items():
        if len(pred_for_large) == 0:
            continue
        merged_bboxes.append(pred_for_large['bbox'])
        merged_sims.append(pred_for_large['sim'])
        repeat_times.append(len(pred_for_large['bbox']))
        unique_image_ids.append(image_name)

    merged_bboxes = np.concatenate(merged_bboxes)
    merged_sims = np.concatenate(merged_sims)
    image_ids = np.repeat(unique_image_ids, repeat_times)
    return {'sim': merged_sims,
            'bbox': merged_bboxes,
            'image_name': image_ids}


def merge_pred2group(pred_dic):
    """
    Merge pred format {image_name: [], sim: [], bbox: []}
    into group format {image_name: [all its bbox]} for reid feature purpose.
    """

    bboxes = np.array(pred_dic['bbox'])
    image_ids = np.array(pred_dic['image_name'])
    sims = np.array(pred_dic['sim'])
    full_bboxes = np.concatenate([bboxes, sims[:, np.newaxis]], axis=1)

    """
    # naive implementation
    unique_image_name = np.unique(image_ids)
    # we also keep the name_dic for emtpy bbox
    gallery_bbox_dict = {image_name: [] for image_name in unique_image_name}
    # we only use the four value for latter reid feature extraction
    for index, full_bbox in enumerate(full_bboxes):
        image_name = image_ids[index]
        gallery_bbox_dict[image_name].append(full_bbox)
    for image_name in gallery_bbox_dict:
        gallery_bbox_dict[image_name] = np.array(gallery_bbox_dict[image_name])
    return gallery_bbox_dict
    """

    # Ref: https://stackoverflow.com/questions/23268605/grouping-indices-of-unique-elements-in-numpy
    a = image_ids
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    gallery_bbox_dict = {}
    for index, image_name in enumerate(unq_items):
        gallery_bbox_dict[image_name] = full_bboxes[unq_idx[index]]
    return gallery_bbox_dict


def load_image(image_path):
    """Load image."""

    img = cv2.imread(image_path)
    return [image_path, img]


def load_image_string(image_path):
    """Load image string."""

    img = cv2.imread(image_path)
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    return [image_path, img_str]


def load_all_data(_cfg):
    """Load all data into memory."""

    search_dir = os.path.join(_cfg.PATH.SEARCH_DIR,
                              _cfg.EVAL.GALLERY_DATA_SOURCE)
    image_list = []
    for device_id in os.listdir(search_dir):
        device_dir = os.path.join(search_dir, device_id)
        for image_name in os.listdir(device_dir):
            image_path = os.path.join(device_dir, image_name)
            data_info = {'image_path': image_path, 'image_name': f'{device_id}/{image_name}'}
            image_list.append(data_info)
    print(f'{len(image_list)} large images')

    if _cfg.DATA.LOAD_FORMAT == 'disk':
        data_list = image_list
    elif _cfg.DATA.LOAD_FORMAT == 'numpy':
        load_pool = Pool()
        data_list = load_pool.map(load_image, image_list)
    elif _cfg.DATA.LOAD_FORMAT == 'string':
        load_pool = Pool()
        data_list = load_pool.map(load_image_string, image_list)
    else:
        print('invalid data type, exit')
        data_list = None
    return data_list


def save_pickle(all_pred_dic, sim_mode, data_mode):
    """Save pickle for pool_output. One file for one sim mode.

    Args:
        all_pred_dic: A dict with key query_instance_id and value a long list
        sim_mode: A string to indicate sim format
        data_mode: A string to indicate data mode
    """

    instance_sim_dir = f'./tests/features/{data_mode}/'
    if not os.path.exists(instance_sim_dir):
        os.makedirs(instance_sim_dir)
    pkl_name = instance_sim_dir + f'/{sim_mode}.pkl'
    pickle.dump(all_pred_dic, open(pkl_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def test_load_query(_cfg):
    """Get an object_id and draw bboxes on query and gallery
    to debug annotation.
    """

    label_file = _cfg.EVAL.LABEL_FILE
    query_dic, gallery_dic = pickle.load(open(label_file, 'rb'))
    selected_test_id = select_good_id(_cfg, gallery_dic)
    object_id = selected_test_id[0]
    large_path, query_name, bbox, _ = get_object_full_info(_cfg, query_dic[object_id], sample_index=0)
    print(f'Query: object_id: {object_id}, large_path: {large_path}, bbox: {bbox} device_id: {bbox}')
    debug_dir = f'./tests/images/debug_annotation/{object_id}'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    img = cv2.imread(large_path)
    [x1, y1, x2, y2] = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    save_name = query_name.replace('/', '_')
    cv2.imwrite(os.path.join(debug_dir, f'query_{save_name}'), img)
    # shutil.copy(large_path, debug_dir)
    for i in range(len(gallery_dic[object_id])):
        large_path, gallery_name, bbox, _ = get_object_full_info(_cfg, gallery_dic[object_id], sample_index=i)
        if not gallery_dic[object_id][i]['ignore']:
            # shutil.copy(large_path, debug_dir)
            img = cv2.imread(large_path)
            [x1, y1, x2, y2] = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            save_name = gallery_name.replace('/', '_')
            cv2.imwrite(os.path.join(debug_dir, f'gt_{save_name}'), img)


def refine_pred2large(_cfg, rough_pred_dic, debug=False):
    """
    Refine rough det with some rules.
    In: pred_dic, individual preds.
    Out: refined large merge dic. Used for reid extraction later.
    gallery_bbox_dic:  One large image for multple bboxes.
    """

    rough_bboxes, rough_image_ids, rough_sim = rough_pred_dic['bbox'], rough_pred_dic['image_name'], rough_pred_dic['sim']
    gallery_bbox_dict = merge_pred2group(rough_pred_dic)
    if debug:
        assert len(gallery_bbox_dict) == len(rough_bboxes)
        print(f'For {_cfg.EVAL.SIM_MODE}, rough large image num: {len(gallery_bbox_dict)}, raw bboxes num {len(rough_bboxes)}')

    rough_det = np.concatenate([rough_bboxes, rough_sim[:, np.newaxis]], axis=1)
    good_data_mask = refine_bboxes(rough_det, _cfg)
    refined_bboxes, refined_image_ids, refined_sim = rough_bboxes[good_data_mask], rough_image_ids[good_data_mask], rough_sim[good_data_mask]
    refined_pred_dic = {'bbox': refined_bboxes, 'image_name': refined_image_ids, 'sim': refined_sim}
    gallery_bbox_dict = merge_pred2group(refined_pred_dic)

    bboxes_per_large = float(len(refined_bboxes)) / len(gallery_bbox_dict)
    if debug:
        print(f'refined large image num: {len(gallery_bbox_dict)}, refined bboxes num: {len(refined_bboxes)}, refined bboxes_per_large: {bboxes_per_large}')
        print(f'gallery_bbox_dict keys: {list(gallery_bbox_dict.keys())[:10]}')
    return gallery_bbox_dict


def test_merge_split(_cfg):
    """
    Pred file is not grouped. Use refine_pred2large to merge individual pred into large dict.
    Save pickle then split thse merged dict into individual pred for eval purpose.
    """

    label_file = _cfg.EVAL.LABEL_FILE
    _, gallery_dic = pickle.load(open(label_file, 'rb'))
    selected_test_id = select_good_id(_cfg, gallery_dic)
    object_id = selected_test_id[0]

    sim_mode = _cfg.EVAL.SIM_MODE
    rough_pred_dic = load_sim_dic(_cfg, object_id, sim_mode)
    gallery_bbox_dict = refine_pred2large(_cfg, rough_pred_dic, True)    # refined results

    object_id = '123456789'
    sim_mode = 'debug'
    data_mode = 'debug'
    save_pickle(object_id, gallery_bbox_dict, sim_mode, data_mode)


def test_load_data(_cfg):
    """Test load data."""

    start_time = time.time()
    data_list = load_all_data(_cfg)
    load_time = time.time() - start_time
    print('%.4f s for load %d images' % (load_time, len(data_list)))


def compare_pred(_cfg):
    """Compare pred."""

    label_file = _cfg.EVAL.LABEL_FILE
    gt_dic = pickle.load(open(label_file, 'rb'))
    selected_test_id = select_good_id(_cfg, gt_dic)
    object_id = selected_test_id[0]

    sim_mode = 'rerank'
    pred_dic_a = load_sim_dic(_cfg, object_id, sim_mode)
    gallery_name_dic_a = merge_pred2group(pred_dic_a)

    sim_mode = 'rerank_back'
    pred_dic_b = load_sim_dic(_cfg, object_id, sim_mode)
    gallery_name_dic_b = merge_pred2group(pred_dic_b)
    for gallery_name in gallery_name_dic_a:
        len_a = len(gallery_name_dic_a[gallery_name])
        len_b = len(gallery_name_dic_b[gallery_name])
        if len_a != len_b:
            print(gallery_name, len_a, len_b)
            # print(gallery_name_dic_a[gallery_name])
            # print(gallery_name_dic_b[gallery_name])


def refine_gt_dic(_cfg, query_instance_dic):
    """Remove query instance id that is too small."""

    refined_query_instance_dic = {}
    min_height = _cfg.EVAL.MIN_HEIGHT
    for pid in query_instance_dic:
        query_bbox = query_instance_dic[pid]
        bbox = query_bbox['bbox']
        if bbox is not None:
            if bbox[3] - bbox[1] < min_height:
                continue
        refined_query_instance_dic[pid] = query_instance_dic[pid]
    return refined_query_instance_dic


def convert_rawdet2dict(raw_det):
    """Convert query-gallery raw detection into dict format."""

    pred_dict = {}
    if len(raw_det) == 0:
        pred_dict = {'bbox': np.array([]), 'sim': np.array([])}
    else:
        pred_dict = {'bbox': raw_det[:, :4],
                     'sim': raw_det[:, -1]}
    return pred_dict


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="Cityflow.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    """Main method."""

    cfg.merge_from_file('./instance_search/config/PRW_PERSON.yml')
    cfg.freeze()

    test_load_query(cfg)
    # test_merge_split(cfg)
    # test_load_data(cfg)
    # compare_pred(cfg)


if __name__ == '__main__':
    main()
