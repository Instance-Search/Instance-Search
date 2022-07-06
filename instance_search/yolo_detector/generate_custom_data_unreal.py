"""Generate custom bbox annotation."""

import os
import pickle
import random
from glob import glob
from multiprocessing import Pool
import cv2

# pylint: disable=invalid-name, line-too-long, too-many-locals
BLANK_MODE = 'real'    # pure, replicate, onbg
IMAGE_NUM = 1
FIXED_HEIGHT = 1080
FIXED_WIDTH = 1920
LOWER_RATIO = 0.2
UPPER_RATIO = 0.8

def get_image(raw_img, obj_dir, index):
    """Get image."""

    if BLANK_MODE == 'real':
        # large_path = '/home/gongyou.zyq/datasets/huanan_data/frames/44030659001320113060/00020600.jpg'
        # x1 = 1671
        # x2 = 1792
        # y1 = 633
        # y2 = 754

        # large_path = '/home/gongyou.zyq/datasets/huanan_data/frames/44030659001320010007/00044675.jpg'
        # tizi
        # x1 = 166
        # y1 = 832
        # x2 = 500
        # y2 = 1066

        # large_path = '/home/gongyou.zyq/datasets/huanan_data/frames/44030659001320010007/00044675.jpg'
        # small bag
        # x1 = 215
        # y1 = 430
        # x2 = 253
        # y2 = 460

        # large_path = './test_images/00024825.jpg'
        # x1 = 310
        # y1 = 215
        # x2 = 360
        # y2 = 350

        large_path = '/home/gongyou.zyq/datasets/huanan_data/frames/44030659001320010066/00031250.jpg'
        # large bagage
        x1 = 620
        y1 = 950
        x2 = 750
        y2 = 1080

        large = cv2.imread(large_path)
        fixed_height, fixed_width = large.shape[:2]
        image_height = y2 - y1
        image_width = x2 -x1
    else:
        raw_height = raw_img.shape[0]
        raw_width = raw_img.shape[1]
        h_w = raw_height / float(raw_width)
        random_h_w = h_w * random.uniform(0.5, 2.0)
        scale_y = random.uniform(int(LOWER_RATIO*FIXED_HEIGHT)/float(raw_height), int(UPPER_RATIO*FIXED_HEIGHT)/float(raw_height))
        scale_x = scale_y / random_h_w
        img = cv2.resize(raw_img.copy(), None, fx=scale_x, fy=scale_y)
        image_height, image_width = img.shape[:2]
        x1 = int((FIXED_WIDTH-image_width)/2)
        y1 = int((FIXED_HEIGHT-image_height)/2)
        x2 = x1 + image_width
        y2 = y1 + image_height
        if x1 < 0 or y1 < 0 or x2 > FIXED_WIDTH or y2 > FIXED_HEIGHT:
            return 0
        # if image_width < int(LOWER_RATIO*FIXED_WIDTH) or image_height < int(LOWER_RATIO*FIXED_HEIGHT) or image_width > int(UPPER_RATIO*FIXED_WIDTH) or image_height > int(UPPER_RATIO*FIXED_HEIGHT):
        #     return 0
        fixed_height = FIXED_HEIGHT
        fixed_width = FIXED_WIDTH

        if BLANK_MODE == 'pure':
            rand_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            large = cv2.copyMakeBorder(img, y1, FIXED_HEIGHT-y2, x1, FIXED_WIDTH-x2, cv2.BORDER_CONSTANT, value=rand_color)
        elif BLANK_MODE == 'replicate':
            large = cv2.copyMakeBorder(img, y1, FIXED_HEIGHT-y2, x1, FIXED_WIDTH-x2, cv2.BORDER_REPLICATE)
        elif BLANK_MODE == 'onbg':
            frame_dir = '/home/gongyou.zyq/datasets/huanan_data/frames/'
            device_id = random.choice(os.listdir(frame_dir))
            device_dir = os.path.join(frame_dir, device_id)
            rand_background_image = random.choice(os.listdir(device_dir))
            large = cv2.imread(os.path.join(device_dir, rand_background_image))
            large[y1:y2, x1:x2, :] = img

    image_height_ratio = image_height / float(fixed_height)
    image_width_ratio = image_width / float(fixed_width)
    center_x = (x1+x2)/2
    center_x_ratio = center_x / float(fixed_width)
    center_y = (y1+y2)/2
    center_y_ratio = center_y / float(fixed_height)

    cv2.imwrite(obj_dir + '%06d.jpg' % index, large)
    f = open(obj_dir + '%06d.txt' % index, 'w')
    f.write('0 %.6f %.6f %.6f %.6f' % (center_x_ratio, center_y_ratio, image_width_ratio, image_height_ratio))
    f.close()
    return 0

def generate_data(query_image):
    """Generate data."""

    detector_dir = '/home/gongyou.zyq/video_object_retrieval/detector/'
    obj_dir = detector_dir + 'data/obj/'
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    raw_img = cv2.imread(query_image)
    pool = Pool()
    for i in range(IMAGE_NUM):
        # get_image(raw_img, obj_dir, i)
        pool.apply_async(get_image, args=(raw_img, obj_dir, i, ))
    pool.close()
    pool.join()

    image_list = glob(obj_dir + '*.jpg')
    f_train = open(detector_dir + 'data/train.txt', 'w')
    for item in sorted(image_list):
        f_train.write(item + '\n')
    f_train.close()

def get_query_info(_person_id, _mode):
    """Get query info."""

    gt_dic = pickle.load(open(f'./features/{_mode}_search_label.pkl', 'rb'))
    person_data = gt_dic[_person_id]
    print(person_data)
    fds

    gt_large_list = gt_dic['large_name']
    gt_bbox_list = gt_dic['bbox']
    gt_personid_list = gt_dic['person_id']
    for k, gt_personid in enumerate(gt_personid_list):
        large_name = gt_large_list[k]
        if gt_personid == PERSON_ID:
            bbox = gt_bbox_list[k]
            device_id, frame_index = large_name.split('_')
            large_path = f'/home/gongyou.zyq/datasets/huanan_data/frames/{device_id}/{frame_index}'
            # we choose the first matched gt_personid image as query.
            break
    return large_path, bbox

def generate_real_data(_person_id, _mode):
    """Generate real data."""

    detector_dir = '/home/gongyou.zyq/video_object_retrieval/detector/'
    obj_dir = detector_dir + 'data/obj/'
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    large_path, bbox = get_query_info(_person_id, _mode)
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

    # generate_data('./test_images/test_3.png')
    generate_real_data('46096', 'reid')
    debug_annotation()

if __name__ == "__main__":
    main()
