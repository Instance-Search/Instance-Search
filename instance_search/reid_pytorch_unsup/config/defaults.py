from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0,1,2,3,4,5,6,7'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 512
# whether use dropout
_C.MODEL.DROPOUT = 0.7
# whether use GEM for pooling
_C.MODEL.GEM = False

# -----------------------------------------------------------------------------
# DISTRIBUTE
# -----------------------------------------------------------------------------
_C.DIST = CN()
# number of nodes for distributed training 
_C.DIST.NODE_WORLDSIZE = 1
# node rank for distributed training
_C.DIST.NODE_RANK = 0
# url used to set up distributed training
_C.DIST.URL = 'tcp://127.0.0.1:12345'
# distributed backend
_C.DIST.BACKEND = 'nccl'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 1.0
# Whether use the mean value for random erasing
_C.INPUT.RE_USING_MEAN = 'no'
# Parameter sh for random erasing
_C.INPUT.RE_SH = 0.2
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [123.675,116.280,103.530]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [57.0,57.0,57.0]
# Whether normalize image into [0,1]
_C.INPUT.PIXEL_NORM = 'yes'
# Whether the resize is used in the first and at the last in tranforms
_C.INPUT.RESIZE_ORDER = 'first'
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Unsupervise data for Domain Adaption
# -----------------------------------------------------------------------------
_C.TGT_UNSUPDATA = CN()
# number of top k cluster
# _C.TGT_UNSUPDATA.CLUSTER_SOURCE = ['visual', 'iou']
_C.TGT_UNSUPDATA.IOU_WEIGHT = 0.0
_C.TGT_UNSUPDATA.UNSUP_MODE = 'cluster'
_C.TGT_UNSUPDATA.CLUSTER_TOPK = 500
_C.TGT_UNSUPDATA.CLUSTER_DIST_THRD = 0.5
# epoch number of labeling
_C.TGT_UNSUPDATA.PSOLABEL_PERIOD = 6
# List of the dataset names for training
_C.TGT_UNSUPDATA.NAMES = ('market1501')
# Root directory where datasets should be used
_C.TGT_UNSUPDATA.TRAIN_DIR = ('./data')
# Sampler for data loading
_C.TGT_UNSUPDATA.SAMPLER_UNIFORM = 'id'
# imgs per id
_C.TGT_UNSUPDATA.NUM_INSTANCE = 4
# pos imgs per id
_C.TGT_UNSUPDATA.NUM_POS_INSTANCE = 0
# learning rate for the classifier
_C.TGT_UNSUPDATA.CLS_LR = 0.01
# min samples for clustering
_C.TGT_UNSUPDATA.MIN_SAMPLES = 4
# SimCLR temperature
_C.TGT_UNSUPDATA.TEMPERATURE = 0.05

# -----------------------------------------------------------------------------
# Supervise data for Domain Adaption
# -----------------------------------------------------------------------------
_C.TGT_SUPDATA = CN()
# List of the dataset names for training
_C.TGT_SUPDATA.NAMES = ('market1501')
# Root directory where datasets should be used
_C.TGT_SUPDATA.TRAIN_DIR = ('./data')
# Sampler for data loading
_C.TGT_SUPDATA.SAMPLER_UNIFORM = 'trp'
# imgs per id
_C.TGT_SUPDATA.NUM_INSTANCE = 12
# pos imgs per id
_C.TGT_SUPDATA.NUM_POS_INSTANCE = 4
 
# -----------------------------------------------------------------------------
# Source Dataset
# -----------------------------------------------------------------------------
_C.SRC_DATA = CN()
# List of the dataset names for training
_C.SRC_DATA.NAMES = ('market1501')
# Root directory where datasets should be used
_C.SRC_DATA.TRAIN_DIR = ('./data')
# Sampler for data loading
_C.SRC_DATA.SAMPLER_UNIFORM = 'id'
# Number of instance for one batch
_C.SRC_DATA.NUM_INSTANCE = 4
# pos imgs per id of source domain
_C.SRC_DATA.NUM_POS_INSTANCE = 0

# -----------------------------------------------------------------------------
# Val Dataset
# -----------------------------------------------------------------------------
_C.VAL_DATA = CN()
# List of the dataset names for val
_C.VAL_DATA.NAMES = ('market1501')
# Root directory where val datasets should be used
_C.VAL_DATA.TRAIN_DIR = ('./data')

# -----------------------------------------------------------------------------
# Dataloader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Probablity of sampling from multi-datasets: source, supdata, unsupdata
_C.DATALOADER.SAMPLER_PROB = [1.0,0.0,0.0]
# Number of images per batch
_C.DATALOADER.IMS_PER_BATCH = 84

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# The loss type of metric loss
_C.LOSS.LOSS_TYPE = ['trp_cls','trpv2','trp_cls']
# If train with label smooth, options: 'on', 'off'
_C.LOSS.IF_LABELSMOOTH = 'off'
# loss weights
_C.LOSS.LOSS_WEIGHTS = [1.0,0.001,1.0]
# Margin of triplet loss
_C.LOSS.TRP_MARGIN = 0.3
_C.LOSS.TRP_L2 = 'yes'
_C.LOSS.TRP_HNM = 'yes'
_C.LOSS.SIMCLR_WEIGHT = 0.0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.009
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of cluster loss
_C.SOLVER.CLUSTER_MARGIN = 0.3

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 80)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 3.5e-6
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 0
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 40
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 40
# fixed iter per epoch
_C.SOLVER.FIXED_ITER = 1000

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 64
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
