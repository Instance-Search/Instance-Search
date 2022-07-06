"""Convert data."""

import os
import json
from multiprocessing import Pool
from glob import glob

import numpy as np
import cv2

# pylint: disable=invalid-name, too-many-locals, no-member
def toBBox(image, shape):
    """
    Raw bbox to template size. image: 511 size, shape is bbox"""

    imh, imw = image.shape[:2]
    if len(shape) == 4:
        w, h = shape[2]-shape[0], shape[3]-shape[1]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = 127  # 127
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w, h = w * scale_z, h * scale_z
    cx, cy = imw//2, imh//2
    bbox = [cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5]
    return bbox

def crop_image(frame_info, track_id, reid_dir):
    """Crop image."""

    image_path = frame_info['image_path']
    bbox = frame_info['bbox']
    frame_index = frame_info['frame_index']
    image = cv2.imread(image_path)
    new_bbox = toBBox(image, bbox)
    [x1, y1, x2, y2] = new_bbox
    patch = image[int(y1): int(y2), int(x1): int(x2), :]
    cv2.imwrite(os.path.join(reid_dir, '%08d_00_0000_%08d.jpg' % (track_id, frame_index)), patch)

def process_json(json_file, sub_data_dir):
    """Load json into track list. image path and raw bbox"""

    with open(json_file) as fin:
        meta_data = json.load(fin)

    track_list = []
    for video, tracks in meta_data.items():
        # if video.split('/')[0] != 'train':
        #     continue
        for trk, frames in tracks.items():
            seq_list = []
            for frm, info in frames.items():
                if not isinstance(info, dict):
                    continue
                if 'bbox' in info:
                    bbox = info['bbox']
                    valid = info['valid']
                else:
                    continue
                if not valid:
                    continue
                image_path = os.path.join(sub_data_dir, video, '{}.{}.{}.jpg'.format(frm, trk, 'x'))
                if not os.path.exists(image_path):
                    print('null file %s' % image_path)
                    continue
                frame_info = {'image_path': image_path, 'bbox': bbox, 'frame_index': int(frm)}
                seq_list.append(frame_info)
            track_list.append(seq_list)
    print(f'{len(track_list)} tracks for {json_file}')
    return track_list

def convert_data():
    """Convert data."""

    name_lsit = ['GOT-10k_crop511', 'VID_crop511', 'y2b']
    pid_offset = 0
    reid_dir = '/home/gongyou.zyq/datasets/Corrected_sot_data/trainval/'
    # name_lsit = ['TAO_crop511']
    # pid_offset = 321340
    # reid_dir = '/home/gongyou.zyq/datasets/Corrected_TAO_data/trainval/'

    data_dir = '/home/gongyou.zyq/datasets/SOT_data/'
    if not os.path.exists(reid_dir):
        os.makedirs(reid_dir)
    all_track_list = []
    for dataset_name in name_lsit:
        json_files = glob(os.path.join(data_dir, dataset_name) + '/*.json')
        for json_file in json_files:
            sub_data_dir = os.path.join(data_dir, dataset_name)
            sub_track_list = process_json(json_file, sub_data_dir)
            all_track_list += sub_track_list
    print(f'{len(all_track_list)} tracks for all data')

    pool = Pool()
    for track_id, seq_list in enumerate(all_track_list):
        for frame_info in seq_list:
            # crop_image(frame_info, track_id+pid_offset, reid_dir)
            pool.apply_async(crop_image, args=(frame_info, track_id+pid_offset, reid_dir, ))
    pool.close()
    pool.join()

convert_data()
