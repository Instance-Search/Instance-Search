set -e
DATASET_NAME="INS_CUHK_SYSU"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "Instance160" "Instance335"

PREPARE_DATA=0
TRAIN_LOC=0
TRAIN_REID=1
EXTRACT_TEST_FEAT=1
SELF_TRAIN_SOURCE="trainval" 

################################## Following code should be fixed ####################################

ROOT_DIR="./"
LOC_DIR="${ROOT_DIR}/instance_search/globaltracker/GlobalTrack"
LOC_MODEL="${LOC_DIR}/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth"
UNSUP_DIR="${ROOT_DIR}/instance_search/reid_pytorch_unsup"
IMAGE_DIR="/home/gongyou.zyq/datasets/instance_search/${DATASET_NAME}"
IDE_PTH_MODEL="${UNSUP_DIR}/log/baseline/resnet50_model_5.pth"
IDE_ONNX_MODEL="${UNSUP_DIR}/log/baseline/sot_r50_fp16.onnx"
QUERY_SEED_DIR="${DATASET_NAME}/query_seed/query_seed_${SELF_TRAIN_SOURCE}"
QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed.pkl"

QUERY_DATA_SOURCE=${SELF_TRAIN_SOURCE}
GALLERY_DATA_SOURCE=${SELF_TRAIN_SOURCE}

SelfTrain(){
    ST_MODE=$1
    EVAL_DIR=$2
    CUR_INDEX=$3

    if [ ${PREPARE_DATA} -eq 1 ]; then
        echo "Extract feat from query seed"
        cd ${ROOT_DIR} && python tests/eval/extract_ins_pair.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} INFERENCE.REID_MODEL ${IDE_ONNX_MODEL} INFERENCE.LOCALIZER_MODEL ${LOC_MODEL} EVAL.LABEL_FILE ${QUERY_SEED_FILE} EVAL.GALLERY_RANGE "local" EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE} EVAL.TEST_QUERY_NUM 1000000
        echo "Prepare self train data"
        QUERY_SEED_FEAT="tests/features/${DATASET_NAME}/${ST_MODE}/reid.pkl"
        cd ${ROOT_DIR} && python instance_search/self_train/make_selftrain_data.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} SELF_TRAIN.MODE ${ST_MODE} SELF_TRAIN.QUERY_SEED_FILE ${QUERY_SEED_FILE} SELF_TRAIN.QUERY_SEED_BASELINE_FEAT ${QUERY_SEED_FEAT} EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE}
        cd ${LOC_DIR} && cp ./cache/${ST_MODE}/topk1_noisy/${DATASET_NAME}_train.pkl ./cache/${DATASET_NAME}_train.pkl && cp ./cache/${ST_MODE}/topk1_noisy/${DATASET_NAME}_test.pkl ./cache/${DATASET_NAME}_test.pkl
    fi

    if [ ${TRAIN_LOC} -eq 1 ]; then
        echo "Train localizer"
        cd ${LOC_DIR} && python -m torch.distributed.launch --nproc_per_node=8 tools/train_qg_rcnn.py --launcher pytorch --load_from $LOC_MODEL --base_dataset "${DATASET_NAME}_train" --sampling_prob "1" --gpus 8 --work_dir log/${DATASET_NAME}/${ST_MODE} --config configs/custom_train.py --dataset_name=${DATASET_NAME} --validate
        LOC_MODEL="${LOC_DIR}/log/${DATASET_NAME}/${ST_MODE}/latest.pth"
    fi

    if [ ${TRAIN_REID} -eq 1 ]; then
        echo "Train ReID"
        cd ${UNSUP_DIR} && python tools/train.py --config_file=configs/self_train_cluster/${DATASET_NAME}.yml MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${IMAGE_DIR}/${ST_MODE}/topk1_noisy VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE} MODEL.PRETRAIN_PATH $IDE_PTH_MODEL DIST.URL 'tcp://127.0.0.17:11113'
        IDE_ONNX_MODEL="${UNSUP_DIR}/log/${DATASET_NAME}/${ST_MODE}/reid_resnet50_3.onnx"
        IDE_PTH_MODEL="${UNSUP_DIR}/log/${DATASET_NAME}/${ST_MODE}/resnet50_model_3.pth"
    fi

    if [ ${EXTRACT_TEST_FEAT} -eq 1 ]; then
        cd ${ROOT_DIR} && python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} INFERENCE.REID_MODEL $IDE_ONNX_MODEL INFERENCE.LOCALIZER_MODEL $LOC_MODEL DISTRIBUTE.WORKER_PER_GPU 1
    fi

    echo "..................................................."
}

# Only for one round
for i in $(seq 0 1); 
do
    echo "self train for round $((i+1))"
    ST_MODE="self_train_loc_reid/query_seed_${SELF_TRAIN_SOURCE}/round/round_$((i+1))_noisy"
    EVAL_DIR="self_train_loc_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/round_${i}_noisy"
    SelfTrain $ST_MODE $EVAL_DIR $((i+1))
done
