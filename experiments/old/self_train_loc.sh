set -e
DATASET_NAME="Instance160"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "Instance160" "Instance335"

TRAIN_LOC=1
EXTRACT_TEST_FEAT=0
EVAL_FLAG=0
SELF_TRAIN_SOURCE="trainval"    # trainval test_gallery naive

################################## Following code should be fixed ####################################

ROOT_DIR="/home/gongyou.zyq/video_object_retrieval"
LOC_DIR="${ROOT_DIR}/instance_search/globaltracker/GlobalTrack"
INIT_LOC_MODEL="${LOC_DIR}/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth"
QUERY_SEED_DIR="${DATASET_NAME}/query_seed/query_seed_${SELF_TRAIN_SOURCE}"
QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed.pkl"

LocSelfTrain(){
    LOC_MODE=$1
    EVAL_DIR=$2
    TOPK_LOC=$3
    ST_MODE=$4

    if [ ${TRAIN_LOC} -eq 1 ]; then
        echo "Train localizer"
        export PYTHONPATH=${LOC_DIR}:$PYTHONPATH
        CACHE_DIR="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/topk${TOPK_LOC}_noisy"
        cd ${LOC_DIR} && cp ./cache/${CACHE_DIR}/${DATASET_NAME}_train.pkl ./cache/${DATASET_NAME}_train.pkl && cp ./cache/${CACHE_DIR}/${DATASET_NAME}_test.pkl ./cache/${DATASET_NAME}_test.pkl
        cd ${LOC_DIR} && python -m torch.distributed.launch --nproc_per_node=8 tools/train_qg_rcnn.py --launcher pytorch --load_from ${INIT_LOC_MODEL} --base_dataset "${DATASET_NAME}_train" --sampling_prob "1" --gpus 8 --work_dir "log/${DATASET_NAME}/${ST_MODE}" --config "configs/custom_train.py" --dataset_name=${DATASET_NAME} --sample_strategy=${LOC_MODE} --validate
    fi

    if [ ${EXTRACT_TEST_FEAT} -eq 1 ]; then
        echo "Extract test feat"
        LOC_MODEL="${LOC_DIR}/log/${DATASET_NAME}/${ST_MODE}/latest.pth"
        cd ${ROOT_DIR} && python tests/eval/extract_ins_pair.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} EVAL.REFINER "null" EVAL.GALLERY_RANGE "pos" INFERENCE.LOCALIZER_MODEL ${LOC_MODEL} INFERENCE.LOCALIZER_TOPK 20
    fi

    if [ ${EVAL_FLAG} -eq 1 ]; then
        echo "Strat eval"
        cd ${ROOT_DIR} && python tests/eval/eval_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} EVAL.ROUGH_LOCALIZER "globaltrack" EVAL.SIM_MODE "globaltrack" EVAL.PROPOSAL_PER_LARGE 20 EVAL.PROPOSAL_MODE "local"
        # python tests/eval/eval_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR}
    fi
    echo "..................................................."
}

LocGridSearch(){
    LOC_MODE="topk"
    for TOPK_LOC in 2 5 20; do
        ST_MODE="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/grid_search/${LOC_MODE}${TOPK_LOC}_noisy"
        EVAL_DIR="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/eval/${LOC_MODE}${TOPK_LOC}_noisy"
        echo "${DATASET_NAME} ${ST_MODE}"
        LocSelfTrain ${LOC_MODE} ${EVAL_DIR} ${TOPK_LOC} ${ST_MODE}
    done
}

RSSSearch(){
    LOC_MODE="topk"
    TOPK_LOC=5
    ST_MODE="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/grid_search/${LOC_MODE}${TOPK_LOC}_noisy"
    EVAL_DIR="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/eval/${LOC_MODE}${TOPK_LOC}_noisy"
    echo "${DATASET_NAME} ${ST_MODE}"
    LocSelfTrain ${LOC_MODE} ${EVAL_DIR} ${TOPK_LOC} ${ST_MODE}
}

LocBest(){
    TOPK_LOC=1
    LOC_MODE="topk"
    ST_MODE="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/loc_noisy"
    EVAL_DIR="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/eval/loc_noisy"
    echo "${DATASET_NAME} ${ST_MODE}"
    LocSelfTrain ${LOC_MODE} ${EVAL_DIR} ${TOPK_LOC} ${ST_MODE}
}

# LocGridSearch
# RSSSearch
LocBest
