# num of localizer boxes for LOC/INS: 3, 5, 10, 15, 20
set -e
DATASET_NAME="INS_Cityflow"    # "INS_PRW" "INS_CUHK_SYSU" "PRW" "CUHK_SYSU" "INS_Cityflow"
PROPOSAL_MODE="thr"    # local global thr
# Following code should be fixed

if [ "$PROPOSAL_MODE" = "local" ]; then
    for PROPOSAL_PER_LARGE in 1 5 20; do
        echo "proposal num per large: ${PROPOSAL_PER_LARGE}"
        python tests/eval/eval_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/globaltrack" EVAL.PROPOSAL_PER_LARGE ${PROPOSAL_PER_LARGE} EVAL.PROPOSAL_MODE "local"
        echo "..................................................."
    done
fi

if [ "$PROPOSAL_MODE" = "thr" ]; then
    for LOC_THR in 0.1 0.01 0.001; do
        echo "localizer thr: ${LOC_THR}"
        python tests/eval/eval_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/globaltrack" EVAL.REFINE_MIN_THR ${LOC_THR} EVAL.PROPOSAL_MODE "thr"
        echo "..................................................."
    done
fi

if [ "$PROPOSAL_MODE" = "global" ]; then
    for LOCALIZER_GLOBAL_TOPK in 1000 2500 5000 7500 10000 15000 30000; do
        echo "localizer global top-k: ${LOCALIZER_GLOBAL_TOPK}"
        python tests/eval/eval_ins.py --config_file=${DATASET_NAME}.yml EVAL.DATA_MODE "${DATASET_NAME}/baseline_ins/globaltrack" EVAL.LOCALIZER_GLOBAL_TOPK ${LOCALIZER_GLOBAL_TOPK} EVAL.PROPOSAL_MODE "global"
        echo "..................................................."
    done
fi
