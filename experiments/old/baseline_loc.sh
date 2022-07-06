set -e
DATASET_NAME="INSTRE"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "INS_Cityflow"
EXTRACT_FLAG=1
WORKER_PER_GPU=1    # siamrpn may out of memory
# Following code should be fixed

# for ROUGH_LOCALIZER in sliding_window general_detection edge_box siamrpn siamrpn_full globaltrack; do
# for ROUGH_LOCALIZER in general_detection siamrpn globaltrack; do
for ROUGH_LOCALIZER in globaltrack; do
    echo "Rough localizer type: ${ROUGH_LOCALIZER}"
 
    if [ ${EXTRACT_FLAG} -eq 1 ]; then
        python tests/eval/extract_ins_pair.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_loc" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.REFINER "null" EVAL.GALLERY_RANGE "pos" INFERENCE.LOCALIZER_TOPK 20 DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU}
    fi

    python tests/eval/eval_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_loc" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} EVAL.SIM_MODE ${ROUGH_LOCALIZER} EVAL.PROPOSAL_PER_LARGE 20 EVAL.PROPOSAL_MODE "local"
    echo "..................................................."
done
