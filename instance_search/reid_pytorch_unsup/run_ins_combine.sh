set -e
DATASET_NAME=PRW    # Cityflow or PRW

# -----------simclr unsup, grid search best param--------------
# for temperature in 0.05 0.1 0.5; do
for temperature in 0.05; do
    for re in 0.0 1.0; do
        python tools/train.py --config_file=configs/ins_${DATASET_NAME}_combine.yml MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR ./log/${DATASET_NAME}/grid_search_combine/temp${temperature}_re${re} TGT_UNSUPDATA.TEMPERATURE ${temperature} INPUT.RE_PROB ${re}
    done
done 
