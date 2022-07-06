from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# DATA_PATH
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.VIDEO_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/raw_video/"
_C.PATH.PB_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/pb_files/"
_C.PATH.VCS_TXT_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/bag_log/"
_C.PATH.FRAME_ALL_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/frame_all/"
_C.PATH.FRAME_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/frame_bag/"
_C.PATH.REID_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/{data_mode}_data/test_gallery/"
_C.PATH.SEARCH_DIR = "/home/gongyou.zyq/datasets/test_video/shannon_data/first_annotation/frame_bag/"


# -----------------------------------------------------------------------------
# EVAL PARAM
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.DATASET_NAME = "bag"
_C.EVAL.DATA_MODE = "bag"
_C.EVAL.ROUGH_LOCALIZER = 'siamrpn'
_C.EVAL.REFINER = 'reid'    # 'reid', 'k_reciprocal', 'sift', 'delg', 'null'
_C.EVAL.SIM_MODE = "siamrpn"    # 'ROUGH_LOCALIZER + REFINER
_C.EVAL.SIM_MODE = "globaltrack"
_C.EVAL.MIN_SAMPLE_PER_ID = 2
_C.EVAL.LABEL_FILE = "./features/bag_search_label.pkl"
_C.EVAL.TEST_QUERY_NUM = 1
_C.EVAL.EVAL_MODE = "eval_only"    # 'rough', 'refine', 'rough_refine', 'end2end', 'eval_only', not used anymore
_C.EVAL.IOU_THR = 0.5
_C.EVAL.NMS = 0.3
_C.EVAL.RANK_GLOBAL_TOPK = [1, 5, 10, 100]
_C.EVAL.MIN_HEIGHT = 70
_C.EVAL.REFINE_MIN_THR = 0.001
_C.EVAL.SEPARATE_CAM = False
_C.EVAL.VIS_FLAG = False
_C.EVAL.VIS_TOPK = 100
_C.EVAL.EVAL_LEVEL = 'image'    # 'image', 'tracklet'
_C.EVAL.K_RECIPROCAL_TOPK = 200
_C.EVAL.LOCALIZER_GLOBAL_TOPK = 200
_C.EVAL.QUERY_FORMAT = 'large_bbox'    # large_bbox, small_pure, small_pad, small_pad_context
_C.EVAL.MULTI_QUERY = False
_C.EVAL.OUTPUT_SPEED = False    # output inference speed
_C.EVAL.PROPOSAL_MODE = 'local'    # How we select proposal, 'local', 'global', 'thr'
_C.EVAL.PROPOSAL_PER_LARGE = -1    # at proposal stage, we only consider top-k proposal per large image
_C.EVAL.VERBOSE = False    # draw results for every query
_C.EVAL.GALLERY_RANGE = 'global'    # gallery range for eval, pos, local, global
_C.EVAL.QUERY_DATA_SOURCE = 'test_probe'    # test_probe trainval
_C.EVAL.GALLERY_DATA_SOURCE = 'test_gallery'    # test_gallery trainval
_C.EVAL.MAP_METRIC = 'delg'    # 'voc', 'retrieval', 'delg'
_C.EVAL.CACHE_FEATURE = False

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.LOC_HALF_FLAG = True
_C.INFERENCE.LOCALIZER_MODEL = './instance_search/globaltracker/GlobalTrack/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
_C.INFERENCE.REID_HALF_FLAG = True
_C.INFERENCE.REID_MODEL = './reid/reid.onnx'
_C.INFERENCE.REID_IMAGE_HEIGHT = 384
_C.INFERENCE.REID_IMAGE_WIDTH = 128
_C.INFERENCE.REID_BATCH_SIZE = 8
_C.INFERENCE.REID_FLIP = False
_C.INFERENCE.GLOBALTRACK_SCALE = (640, 480)    # (640, 480), (1333, 800)
_C.INFERENCE.LOCALIZER_TOPK = 5

# -----------------------------------------------------------------------------
# DATA PARAM
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.MIN_HEIGHT_RATIO = 0.07
_C.DATA.SKIP_FRAMES = 25
_C.DATA.LOAD_FORMAT = 'disk'    # disk, numpy, string

# -----------------------------------------------------------------------------
# DISTRIBUTE
# -----------------------------------------------------------------------------
_C.DISTRIBUTE = CN()
_C.DISTRIBUTE.NUM_GPU = 8
_C.DISTRIBUTE.WORKER_PER_GPU = 3
_C.DISTRIBUTE.GALLERY_BATCH_SIZE = 1
_C.DISTRIBUTE.QUERY_BATCH_SIZE = 100

# -----------------------------------------------------------------------------
# DELG
# -----------------------------------------------------------------------------
_C.DELG = CN()
_C.DELG.DELG_FLAG = False    # Whether to use delg
_C.DELG.CROP_QUERY = True    # Crop and resize query to a fixed size, True for DELG
_C.DELG.USE_LOCAL_FEATURE = 0
_C.DELG.NUM_TO_RERANK = 100
_C.DELG.MIN_RANSAC_SAMPLES = 3
_C.DELG.MULTI_PROCESS = True    # Use multi cores for speed-up

# -----------------------------------------------------------------------------
# SELF TRAIN
# -----------------------------------------------------------------------------
_C.SELF_TRAIN = CN()
_C.SELF_TRAIN.MODE = 'self_train/current'
_C.SELF_TRAIN.LOC_MODE = 'topk'    # how to select nosiy data for localizer 'topk', 'thr', 'rss'
_C.SELF_TRAIN.TOPK_LOC = 10
_C.SELF_TRAIN.THR_LOC = 0.001
_C.SELF_TRAIN.IDE_MODE = 'topk'    # how to select nosiy data for reid 'topk', 'thr'
_C.SELF_TRAIN.UNSUP_TYPE = 'cluster'    # how to self-train for reid 'cluster', 'simclr', 'combined'
_C.SELF_TRAIN.TOPK_IDE = 10
_C.SELF_TRAIN.THR_IDE = 0.001
_C.SELF_TRAIN.SOURCE = 'naive'    # naive test_gallery trainval
_C.SELF_TRAIN.QUERY_SEED_TYPE = 'general_detection'
_C.SELF_TRAIN.QUERY_SEED_TYPE_LIST = ('general_detection', 'selective_search')
_C.SELF_TRAIN.QUERY_SEED_NUM = 1000
_C.SELF_TRAIN.QUERY_SEED_NEIGHBOUR = 2
_C.SELF_TRAIN.QUERY_SEED_FILE = "tests/features/CUHK_SYSU/query_seed_trainval/query_seed.pkl"
_C.SELF_TRAIN.QUERY_SEED_BASELINE_FEAT = "tests/features/CUHK_SYSU/query_seed_trainval/baseline/reid.pkl"
_C.SELF_TRAIN.ONLY_PERSON = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
