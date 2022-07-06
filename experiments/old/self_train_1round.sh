set -e
DATASET_NAME="INSTRE"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "INS_Cityflow"
EXTRACT_FLAG=1
SELF_TRAIN_SOURCE="trainval"    # trainval test_gallery naive

################################## Following code should be fixed ####################################

ROOT_DIR="/home/gongyou.zyq/video_object_retrieval"
UNSUP_DIR="${ROOT_DIR}/instance_search/reid_pytorch_unsup"
IDE_ONNX_MODEL="${UNSUP_DIR}/log/${DATASET_NAME}/self_train_reid/query_seed_${SELF_TRAIN_SOURCE}/reid_noisy/reid_resnet50_3.onnx"
LOC_DIR="${ROOT_DIR}/instance_search/globaltracker/GlobalTrack"
LOC_MODEL="${LOC_DIR}/log/${DATASET_NAME}/self_train_loc/query_seed_${SELF_TRAIN_SOURCE}/loc_noisy/latest.pth"
EVAL_DIR="self_train_loc_reid/query_seed_${SELF_TRAIN_SOURCE}/eval/one_round"


TEST_QUERY_NUM=10000
echo "${DATASET_NAME} for self train one round."
if [ ${EXTRACT_FLAG} -eq 1 ]; then
    CMD_STRING="--config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} INFERENCE.LOCALIZER_MODEL ${LOC_MODEL} INFERENCE.REID_MODEL ${IDE_ONNX_MODEL} EVAL.TEST_QUERY_NUM ${TEST_QUERY_NUM}"
    python tests/eval/extract_ins.py ${CMD_STRING}
fi

python tests/eval/eval_ins.py --config_file="${DATASET_NAME}.yml" EVAL.DATA_MODE ${DATASET_NAME}/${EVAL_DIR} EVAL.TEST_QUERY_NUM ${TEST_QUERY_NUM} # EVAL.VERBOSE True # EVAL.VIS_FLAG True
