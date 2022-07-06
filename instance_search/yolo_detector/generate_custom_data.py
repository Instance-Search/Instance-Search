"""Generate custom bbox annotation."""

import os
import pickle
from glob import glob
import cv2

from utils.get_labels import get_object_info, select_good_id
from config import cfg

# pylint: disable=invalid-name, line-too-long, too-many-locals
def generate_custom_data(large_path, bbox):
    """Generate custom data."""

    detector_dir = '/home/gongyou.zyq/video_object_retrieval/detector/'
    obj_dir = detector_dir + 'data/obj/'
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    [x1, y1, x2, y2] = bbox
    large = cv2.imread(large_path)
    fixed_height, fixed_width = large.shape[:2]
    image_height = y2 - y1
    image_width = x2 -x1
    image_height_ratio = image_height / float(fixed_height)
    image_width_ratio = image_width / float(fixed_width)
    center_x = (x1+x2)/2
    center_x_ratio = center_x / float(fixed_width)
    center_y = (y1+y2)/2
    center_y_ratio = center_y / float(fixed_height)
    index = 0
    cv2.imwrite(obj_dir + '%06d.jpg' % index, large)
    f = open(obj_dir + '%06d.txt' % index, 'w')
    f.write('0 %.6f %.6f %.6f %.6f' % (center_x_ratio, center_y_ratio, image_width_ratio, image_height_ratio))
    f.close()

    image_list = glob(obj_dir + '*.jpg')
    f_train = open(detector_dir + 'data/train.txt', 'w')
    for item in sorted(image_list):
        f_train.write(item + '\n')
    f_train.close()

def debug_annotation():
    """Debug annotation."""

    detector_dir = '/home/gongyou.zyq/video_object_retrieval/detector/'
    obj_dir = detector_dir + 'data/obj/'
    rand_data = glob(obj_dir+'*.jpg')[0]
    img = cv2.imread(rand_data)
    image_height, image_width = img.shape[:2]
    f_txt = open(rand_data.replace('.jpg', '.txt'), 'r')
    annotation = f_txt.readlines()[0].strip()
    [_, center_ratio_x, center_ratio_y, width_ratio, height_ratio] = annotation.split(' ')
    center_ratio_x = float(center_ratio_x)
    center_ratio_y = float(center_ratio_y)
    width_ratio = float(width_ratio)
    height_ratio = float(height_ratio)
    x1 = int(center_ratio_x*image_width) - int(width_ratio*image_width/2.0)
    x2 = int(center_ratio_x*image_width) + int(width_ratio*image_width/2.0)
    y1 = int(center_ratio_y*image_height) - int(height_ratio*image_height/2.0)
    y2 = int(center_ratio_y*image_height) + int(height_ratio*image_height/2.0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('./test_images/debug_annotation.jpg', img)

def main():
    """Main method."""

    cfg.merge_from_file('./config/bag.yml')
    cfg.freeze()

    data_mode = 'reid'
    gt_dic = pickle.load(open(f'./features/{data_mode}_search_label.pkl', 'rb'))
    selected_test_id = select_good_id(gt_dic)
    object_id = selected_test_id[0]
    large_path, _, bbox, _ = get_object_info(cfg, gt_dic[object_id], 0)
    print(f'object_id: {object_id}, large_path: {large_path}, bbox: {bbox}')

    generate_custom_data(large_path, bbox)
    debug_annotation()

if __name__ == "__main__":
    main()
