set -e
DATASET_NAME="INS_CUHK_SYSU"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "Instance160" "Instance335"
WORKER_PER_GPU=3    # siamrpn may out of memory
WITH_DUKE_REID=0
declare -a LOC_LIST=("globaltrack")    # edge_box general_detection siamrpn globaltrack

################################## Following code should be fixed ####################################

TwoStageBaseline(){
    for ROUGH_LOCALIZER in "${LOC_LIST[@]}"; do
        echo "Two stage baseline for ${DATASET_NAME}, rough localizer: ${ROUGH_LOCALIZER} + reid" 
        if [ ${WITH_DUKE_REID} == 0 ]; then
            python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/${ROUGH_LOCALIZER}" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "reid" DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU}
        else
            python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/${ROUGH_LOCALIZER}" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "reid" DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU} INFERENCE.REID_MODEL "instance_search/reid/duke_strongbaseline.onnx" INFERENCE.REID_HALF_FLAG False INFERENCE.REID_IMAGE_HEIGHT 256 INFERENCE.REID_IMAGE_WIDTH 128 INFERENCE.REID_BATCH_SIZE 1
        fi
    done
}

OneStageBaseline(){
    for ROUGH_LOCALIZER in "${LOC_LIST[@]}"; do
        echo "One stage baseline for ${DATASET_NAME}, rough localizer: ${ROUGH_LOCALIZER}"
        python tests/eval/extract_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/${ROUGH_LOCALIZER}_one" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "null" DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU}
    done
}

BboxReidFeature(){
    for ROUGH_LOCALIZER in "${LOC_LIST[@]}"; do
        echo "Bbox reid feature for ${DATASET_NAME}, rough localizer: ${ROUGH_LOCALIZER}"
        python tests/eval/extract_bbox_reid.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/${ROUGH_LOCALIZER}" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "reid" DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU}
    done
}

################################## Command line ####################################

# OneStageBaseline
# BboxReidFeature
TwoStageBaseline
