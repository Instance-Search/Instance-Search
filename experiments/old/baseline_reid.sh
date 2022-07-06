set -e
DATASET_NAME="INS_Cityflow"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "INS_Cityflow"
# Following code should be fixed

for REFINER in sot_r50_fp32.onnx imagenet.onnx vehicle.onnx; do
    echo "Ranker type: ${REFINER}"
 
    python tests/eval/eval_reid_image.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_pos" INFERENCE.REID_MODEL "instance_search/reid/${REFINER}" INFERENCE.REID_HALF_FLAG False
    echo "..................................................."
done
