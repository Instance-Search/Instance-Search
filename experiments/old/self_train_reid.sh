set -e
DATASET_NAME="Instance160"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "Instance160" "Instance335"

TRAIN_REID=1
EXTRACT_TEST_FEAT=1
SELF_TRAIN_SOURCE="trainval"    # trainval test_gallery naive

################################## Following code should be fixed ####################################

ROOT_DIR="/home/gongyou.zyq/video_object_retrieval"
UNSUP_DIR="${ROOT_DIR}/instance_search/reid_pytorch_unsup"
IMAGE_DIR="/home/gongyou.zyq/datasets/instance_search/${DATASET_NAME}"
QUERY_SEED_DIR="${DATASET_NAME}/query_seed/query_seed_${SELF_TRAIN_SOURCE}"
QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed.pkl"

IDESelfTrain(){
    ST_MODE=$1
    EVAL_DIR=$2
    TOPK_IDE=$3
    CMD_STRING=$4

    if [ ${TRAIN_REID} -eq 1 ]; then
        echo "Train reid"
        cd ${UNSUP_DIR} && eval $CMD_STRING
    fi

    if [ ${EXTRACT_TEST_FEAT} -eq 1 ]; then
        echo "Extract test feat and eval"
        IDE_ONNX_MODEL="${UNSUP_DIR}/log/${DATASET_NAME}/${ST_MODE}/reid_resnet50_3.onnx"
        cd ${ROOT_DIR} && python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} INFERENCE.REID_MODEL ${IDE_ONNX_MODEL}
    fi

    echo "..................................................."
}

ReIDSearchClusterTopk(){
    # cluster-based
    # for TOPK_IDE in 2 5 20; do
    for TOPK_IDE in 20; do
        ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/grid_search/topk${TOPK_IDE}_noisy"
        EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/grid_search/topk${TOPK_IDE}_noisy"
        TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
        CMD_STRING="python tools/train.py --config_file=configs/self_train_cluster/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA} VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE}"
        echo ${DATASET_NAME} ${ST_MODE}
        IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
    done
}

ReIDSearchClusterIou(){
    TOPK_IDE=5
    for IOU_WEIGHT in 1.0 0.0; do
        ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/grid_search/iouweight${IOU_WEIGHT}_noisy"
        EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/grid_search/iouweight${IOU_WEIGHT}_noisy"
        TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
        CMD_STRING="python tools/train.py --config_file=configs/self_train_cluster/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA} VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE} TGT_UNSUPDATA.IOU_WEIGHT ${IOU_WEIGHT}"
        echo ${DATASET_NAME} ${ST_MODE}
        IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
    done
}

ReIDSearchSimCLRTemperature(){
    TOPK_IDE=5
    for TEMPERATURE in 0.5 0.05; do
        ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/grid_search/temperature${TEMPERATURE}_noisy"
        EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/grid_search/temperature${TEMPERATURE}_noisy"
        TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
        CMD_STRING="python tools/train.py --config_file=configs/self_train_simclr/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA} VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE} TGT_UNSUPDATA.TEMPERATURE ${TEMPERATURE} TGT_UNSUPDATA.UNSUP_MODE simclr"
        echo ${DATASET_NAME} ${ST_MODE}
        IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
    done
}

ReIDSearchSimCLRSampler(){
    TOPK_IDE=5
    for UNSUP_MODE in "cluster_simclr" "simclr"; do
        ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/grid_search/${UNSUP_MODE}_noisy"
        EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/grid_search/${UNSUP_MODE}_noisy"
        TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
        CMD_STRING="python tools/train.py --config_file=configs/self_train_simclr/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA}  VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE} TGT_UNSUPDATA.UNSUP_MODE ${UNSUP_MODE}"
        echo ${DATASET_NAME} ${ST_MODE}
        IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
    done
}

ReIDSearchCombine(){
    TOPK_IDE=5
    # for SIMCLR_WEIGHT in 0.0 0.5 1.0; do
    for SIMCLR_WEIGHT in 0.1; do
        ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/grid_search/simclrweight${SIMCLR_WEIGHT}_noisy"
        EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/grid_search/simclrweight${SIMCLR_WEIGHT}_noisy"
        TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
        CMD_STRING="python tools/train.py --config_file=configs/self_train_combine/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA}  VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE} TGT_UNSUPDATA.UNSUP_MODE cluster_simclr LOSS.SIMCLR_WEIGHT ${SIMCLR_WEIGHT}"
        echo ${DATASET_NAME} ${ST_MODE}
        IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
    done
}

ReIDBest(){
    TOPK_IDE=1
    ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/reid_noisy"
    EVAL_DIR="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/reid_noisy"
    TGT_UNSUPDATA="${IMAGE_DIR}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_IDE}_noisy"
    CMD_STRING="python tools/train.py --config_file=configs/self_train_cluster/${DATASET_NAME}.yml SRC_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_SUPDATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ TGT_UNSUPDATA.TRAIN_DIR ${TGT_UNSUPDATA} VAL_DATA.TRAIN_DIR ${IMAGE_DIR}/reid_images/ OUTPUT_DIR ./log/${DATASET_NAME}/${ST_MODE}"
    echo ${DATASET_NAME} ${ST_MODE}
    IDESelfTrain ${ST_MODE} ${EVAL_DIR} ${TOPK_IDE} "${CMD_STRING}"
}

# ReIDSearchClusterTopk
# ReIDSearchClusterIou
# ReIDSearchSimCLRTemperature
# ReIDSearchSimCLRSampler
# ReIDSearchCombine
ReIDBest
