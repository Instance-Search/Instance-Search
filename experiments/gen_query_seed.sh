# generate query seed for self train
set -e
DATASET_NAME="INS_CUHK_SYSU"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "Instance160" "Instance335"
SELF_TRAIN_SOURCE="trainval"

################################## Following code should be fixed ####################################

ROOT_DIR="./"
SEARCH_DIR="/home/gongyou.zyq/datasets/instance_search/${DATASET_NAME}/valid_images/"
QUERY_SEED_DIR="${DATASET_NAME}/query_seed/query_seed_${SELF_TRAIN_SOURCE}"

QUERY_DATA_SOURCE=${SELF_TRAIN_SOURCE}
GALLERY_DATA_SOURCE=${SELF_TRAIN_SOURCE}

CreateQuerySeed(){
    ROUGH_LOCALIZER=$1

    echo "get query seed from ${ROUGH_LOCALIZER}"
    echo "Extract bbox proposal"
    python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${QUERY_SEED_DIR}/bbox_proposal" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "null" DISTRIBUTE.WORKER_PER_GPU 1 PATH.SEARCH_DIR ${SEARCH_DIR} EVAL.GALLERY_RANGE "global" EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE} INFERENCE.LOCALIZER_TOPK 5 EVAL.TEST_QUERY_NUM 1 EVAL.CACHE_FEATURE True

    echo "Generate query seed proposal from bbox proposal"
    QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed_proposal/${ROUGH_LOCALIZER}.pkl"
    python instance_search/self_train/generate_query_seed.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${QUERY_SEED_DIR}/bbox_proposal/" PATH.SEARCH_DIR ${SEARCH_DIR} SELF_TRAIN.QUERY_SEED_FILE ${QUERY_SEED_FILE} SELF_TRAIN.SOURCE ${SELF_TRAIN_SOURCE} SELF_TRAIN.QUERY_SEED_TYPE ${ROUGH_LOCALIZER}
}


MergeQuerySeed(){
    echo "Merge query proposals and sample enough query seed."
    QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed.pkl"
    python instance_search/self_train/merge_query_seed.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${QUERY_SEED_DIR}/query_seed_proposal/" PATH.SEARCH_DIR ${SEARCH_DIR} SELF_TRAIN.QUERY_SEED_FILE ${QUERY_SEED_FILE}

    echo "Extract baseline feat for query seed"
    python tests/eval/extract_ins_pair.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${QUERY_SEED_DIR}/baseline" EVAL.LABEL_FILE ${QUERY_SEED_FILE} EVAL.GALLERY_RANGE "local" EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE} EVAL.TEST_QUERY_NUM 1000000 EVAL.CACHE_FEATURE True
    # python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${QUERY_SEED_DIR}/baseline" EVAL.LABEL_FILE ${QUERY_SEED_FILE} EVAL.GALLERY_RANGE "global" EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE} EVAL.TEST_QUERY_NUM 1000000 EVAL.CACHE_FEATURE True
}

MakeSelfTrainData(){
    echo "Make self train data"
    QUERY_SEED_FILE="tests/features/${QUERY_SEED_DIR}/query_seed.pkl"
    QUERY_SEED_BASELINE_FEAT="tests/features/${QUERY_SEED_DIR}/baseline/reid.pkl"
    ST_MODE="self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/"
    python instance_search/self_train/make_selftrain_data.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} SELF_TRAIN.MODE ${ST_MODE} SELF_TRAIN.QUERY_SEED_FILE ${QUERY_SEED_FILE} SELF_TRAIN.QUERY_SEED_BASELINE_FEAT ${QUERY_SEED_BASELINE_FEAT} EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE}
    ST_MODE="self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/"
    python instance_search/self_train/make_selftrain_data.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${ST_MODE} SELF_TRAIN.MODE ${ST_MODE} SELF_TRAIN.QUERY_SEED_FILE ${QUERY_SEED_FILE} SELF_TRAIN.QUERY_SEED_BASELINE_FEAT ${QUERY_SEED_BASELINE_FEAT} EVAL.QUERY_DATA_SOURCE ${QUERY_DATA_SOURCE} EVAL.GALLERY_DATA_SOURCE ${GALLERY_DATA_SOURCE}
}


cd ${ROOT_DIR}
for ROUGH_LOCALIZER in "general_detection"; do
    CreateQuerySeed ${ROUGH_LOCALIZER}
done

MergeQuerySeed
MakeSelfTrainData
