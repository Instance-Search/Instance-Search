# pair-wise localizer self training
set -e

TOPK_LOC=10
# ST_MODE="self_train_loc/${TOPK_LOC}_per_query"    # self training for noisy reid data
ST_MODE="self_train_loc/clean_gt"
ROOT_DIR="/home/gongyou.zyq/video_object_retrieval"
LOC_DIR="${ROOT_DIR}/instance_search/globaltracker/GlobalTrack"
LOC_MODEL=${LOC_DIR}/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth

for DATASET_NAME in Cityflow PRW; do
    cd ${ROOT_DIR}
    python instance_search/self_train/make_selftrain_data.py --config_file=${DATASET_NAME}.yml SELF_TRAIN.TOPK_LOC ${TOPK_LOC} SELF_TRAIN.MODE ${ST_MODE}
    cd ${LOC_DIR}
    cp ./cache/${ST_MODE}/${DATASET_NAME}_train.pkl ./cache/${DATASET_NAME}_train.pkl
    cp ./cache/${ST_MODE}/${DATASET_NAME}_test.pkl ./cache/${DATASET_NAME}_test.pkl

    sampler=RandomSampler
    for lr in 0.01; do    # for lr in 0.01 0.001
        for batch_size in 1; do    # for batch_size in 1 2
            for sampler in RandomSampler; do    # RandomSampler OHEMSampler
                python -m torch.distributed.launch --nproc_per_node=8 --master_port=1111 tools/train_qg_rcnn.py --launcher pytorch --load_from $LOC_MODEL --base_dataset "${DATASET_NAME}_train" --sampling_prob "1" --gpus 8 --work_dir log/${DATASET_NAME}/grid_search/lr${lr}_sampler${sampler}_bs${batch_size} --config configs/self_train.py --validate --lr=${lr} --sampler=${sampler} --dataset_name=${DATASET_NAME} --imgs_per_gpu=${batch_size}
            done
        done
    done
done
