# sup training for person search
set -e
DATASET_NAME="CUHK_SYSU"    # PRW CUHK_SYSU
PREPARE_DATA=0
TRAIN_LOC=0
TRAIN_REID=0
EXTRACT_TEST_FEAT=0
EVAL_FLAG=1
# Following code should be fixed

ROOT_DIR="/home/gongyou.zyq/video_object_retrieval"
LOC_DIR="${ROOT_DIR}/instance_search/globaltracker/GlobalTrack"
LOC_MODEL="${LOC_DIR}/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth"
UNSUP_DIR="${ROOT_DIR}/instance_search/reid_pytorch_unsup"
ST_MODE="sup_train_loc_reid/clean_gt"
SEARCH_DIR="/home/gongyou.zyq/datasets/instance_search/INS_${DATASET_NAME}/valid_images/"
SUP_IMAGE_DIR="/home/gongyou.zyq/datasets/instance_search/INS_${DATASET_NAME}/reid_images/"

if [ ${PREPARE_DATA} -eq 1 ]; then
    echo "Prepare sup train data"
    cd ${ROOT_DIR} && python instance_search/self_train/make_selftrain_data.py --config_file="${DATASET_NAME}.yml" SELF_TRAIN.MODE ${ST_MODE} PATH.SEARCH_DIR ${SEARCH_DIR}
    cd ${LOC_DIR} && cp "./cache/${ST_MODE}/${DATASET_NAME}_train.pkl" "./cache/${DATASET_NAME}_train.pkl" && cp "./cache/${ST_MODE}/${DATASET_NAME}_test.pkl" "./cache/${DATASET_NAME}_test.pkl"
fi

if [ $TRAIN_LOC -eq 1 ]; then
    echo "Train localizer"
    cd ${LOC_DIR} && python -m torch.distributed.launch --nproc_per_node=8 tools/train_qg_rcnn.py --launcher pytorch --load_from ${LOC_MODEL} --base_dataset "${DATASET_NAME}_train" --sampling_prob "1" --gpus 8 --work_dir "log/${DATASET_NAME}/${ST_MODE}" --config "configs/custom_train.py" --validate --dataset_name=${DATASET_NAME}
fi

if [ $TRAIN_REID -eq 1 ]; then
    echo "Train ReID"
    cd ${UNSUP_DIR} && python tools/train.py --config_file="configs/sup_train/${DATASET_NAME}.yml" MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR "./log/${DATASET_NAME}/${ST_MODE}" SRC_DATA.TRAIN_DIR ${SUP_IMAGE_DIR} TGT_SUPDATA.TRAIN_DIR ${SUP_IMAGE_DIR} TGT_UNSUPDATA.TRAIN_DIR ${SUP_IMAGE_DIR} VAL_DATA.TRAIN_DIR ${SUP_IMAGE_DIR}
fi

if [ $EXTRACT_TEST_FEAT -eq 1 ]; then
    echo "Extract test feat"
    LOC_MODEL="${LOC_DIR}/log/${DATASET_NAME}/${ST_MODE}/latest.pth"
    IDE_ONNX_MODEL="${UNSUP_DIR}/log/${DATASET_NAME}/${ST_MODE}/reid_resnet50_30.onnx"
    cd ${ROOT_DIR} && python tests/eval/extract_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} INFERENCE.REID_MODEL ${IDE_ONNX_MODEL} INFERENCE.LOCALIZER_MODEL ${LOC_MODEL} INFERENCE.REID_IMAGE_HEIGHT 384 INFERENCE.REID_IMAGE_WIDTH 128 EVAL.MULTI_QUERY False INFERENCE.REID_BATCH_SIZE 5 INFERENCE.LOCALIZER_TOPK 5
fi

if [ ${EVAL_FLAG} -eq 1 ]; then
    echo "Strat eval"
    cd ${ROOT_DIR} && python tests/eval/eval_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} EVAL.PROPOSAL_PER_LARGE 1 EVAL.PROPOSAL_MODE "local" EVAL.SIM_MODE "globaltrack"
    python tests/eval/eval_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE}
fi
