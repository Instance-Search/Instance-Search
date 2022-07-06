set -e

for DATASET_NAME in "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "INS_Cityflow"; do
    for ROUGH_LOCALIZER in "sliding_window" "general_detection" "selective_search" "edge_box" "siamrpn" "siamrpn_full" "globaltrack"; do
        echo "Dataset: ${DATASET_NAME}, rough localizer type: ${ROUGH_LOCALIZER}"
        python tests/localizer/test_localizer.py --config_file="${DATASET_NAME}.yml" EVAL.ROUGH_LOCALIZER ${ROUGH_LOCALIZER} 
        echo "..................................................."
    done
done
