# cluster based unsup baseline
set -e
DATASET_NAME=Cityflow    # Cityflow or PRW
GRID_MIN_SAMPLES=0
GRID_BATCH=0
GRID_LR=1

for DATASET_NAME in Cityflow PRW; do
    # -----------grid search min_samples--------------
    if [ $GRID_MIN_SAMPLES -eq 1 ]; then
        for min_samples in 1 2 4; do
            python tools/train.py --config_file=configs/ins_${DATASET_NAME}_cluster.yml MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR ./log/${DATASET_NAME}/grid_search_cluster/lr0.009_6x8_min${min_samples}sample TGT_UNSUPDATA.CLS_LR 0.009 SOLVER.BASE_LR 0.009 DATALOADER.IMS_PER_BATCH 48 TGT_UNSUPDATA.NUM_INSTANCE 8 TGT_UNSUPDATA.MIN_SAMPLES ${min_samples} DIST.URL 'tcp://127.0.0.17:11113'
        done
        # min_samples=2 could be good for clean data. However for noisy data with few instnaces, we had better set min_samples=1
    fi

    # -----------grid search batch settings--------------
    if [ $GRID_BATCH -eq 1 ]; then
        BATCH_SETTING_LIST=("6 8 48" "12 8 96" "12 4 48")
        for element in "${BATCH_SETTING_LIST[@]}"; do
            arrIN=(${element// / })
            ids_per_batch=${arrIN[0]}
            imgs_per_id=${arrIN[1]}
            batch_size=${arrIN[2]}
            python tools/train.py --config_file=configs/ins_${DATASET_NAME}_cluster.yml MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR ./log/${DATASET_NAME}/grid_search_cluster/lr0.0009_${ids_per_batch}x${imgs_per_id}_min2sample TGT_UNSUPDATA.CLS_LR 0.0009 SOLVER.BASE_LR 0.0009 DATALOADER.IMS_PER_BATCH ${batch_size} TGT_UNSUPDATA.NUM_INSTANCE ${imgs_per_id} TGT_UNSUPDATA.MIN_SAMPLES 2 DIST.URL 'tcp://127.0.0.17:11113'
        done
        # 6x8 could be good
    fi

    # -----------grid search lr--------------
    if [ $GRID_LR -eq 1 ]; then
        for lr in 0.009; do    # 0.009 0.0009
            python tools/train.py --config_file=configs/ins_${DATASET_NAME}_cluster.yml MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR ./log/${DATASET_NAME}/grid_search_cluster/lr${lr}_6x8_min1sample TGT_UNSUPDATA.CLS_LR ${lr} SOLVER.BASE_LR ${lr} DATALOADER.IMS_PER_BATCH 48 TGT_UNSUPDATA.NUM_INSTANCE 8 TGT_UNSUPDATA.MIN_SAMPLES 1 DIST.URL 'tcp://127.0.0.17:11112'
        done
    fi
done

# -----------self train on noisy data--------------
# in experimetns folder
