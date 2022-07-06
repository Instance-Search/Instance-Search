set -e
DATASET_NAME=Oxford5k    # Oxford5k, Paris6k, GLDv2
GLOBAL_LOCAL_TYPE=delg_reid   # delg, delg_custom, delg_vit, sift, delg_reid, delg_competition_baseline, delg_3rd, delg_torch
LOCAL_FLAG=0
WORKER_PER_GPU=1    # 1 for delg tf model

################################## Following code should be fixed ####################################

# basic mathcing runtest
# python tests/global_local/test_global_local.py --config_file=${DATASET_NAME}.yml EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE}

# Oxford5k sample from DELG code
python tests/eval/extract_global_local.py --config_file=${DATASET_NAME}.yml EVAL.REFINER ${GLOBAL_LOCAL_TYPE} EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE} DISTRIBUTE.WORKER_PER_GPU ${WORKER_PER_GPU}
python tests/eval/perform_global_local_retrieval.py --config_file=${DATASET_NAME}.yml EVAL.REFINER ${GLOBAL_LOCAL_TYPE} EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE} DELG.USE_LOCAL_FEATURE ${LOCAL_FLAG}
python tests/eval/eval_cbir.py --config_file=${DATASET_NAME}.yml EVAL.ROUGH_LOCALIZER ${GLOBAL_LOCAL_TYPE} EVAL.REFINER ${GLOBAL_LOCAL_TYPE} EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE} EVAL.VERBOSE True

# python tests/eval/perform_global_local_retrieval.py --config_file=${DATASET_NAME}.yml EVAL.REFINER ${GLOBAL_LOCAL_TYPE} EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE} DELG.MULTI_PROCESS False DELG.NUM_TO_RERANK 30
# python tests/eval/eval_cbir.py --config_file=${DATASET_NAME}.yml EVAL.ROUGH_LOCALIZER ${GLOBAL_LOCAL_TYPE} EVAL.REFINER ${GLOBAL_LOCAL_TYPE} EVAL.SIM_MODE ${GLOBAL_LOCAL_TYPE}


# extract patch features by DELG model, similar to jubusou
# python tests/eval/extract_global_local.py --config_file=Oxford5k.yml EVAL.REFINER delg_custom EVAL.SIM_MODE delg_custom DISTRIBUTE.WORKER_PER_GPU 1
# python tests/eval/perform_global_local_retrieval.py --config_file=Oxford5k.yml EVAL.REFINER delg_custom EVAL.SIM_MODE delg_custom
# python tests/eval/eval_cbir.py --config_file=Oxford5k.yml EVAL.ROUGH_LOCALIZER delg_custom EVAL.REFINER delg_custom EVAL.SIM_MODE delg_custom

# python tests/eval/extract_global_local.py --config_file=Cityflow.yml EVAL.SIM_MODE delg EVAL.ROUGH_LOCALIZER delg EVAL.REFINER delg
# python tests/eval/perform_global_local_retrieval.py --config_file=Cityflow.yml EVAL.SIM_MODE delg EVAL.ROUGH_LOCALIZER delg EVAL.REFINER delg DELG.USE_LOCAL_FEATURE 0 
# python tests/eval/perform_global_local_retrieval.py --config_file=Cityflow.yml EVAL.SIM_MODE delg DELG.USE_LOCAL_FEATURE 1
# python tests/eval/eval_ins.py --config_file=Cityflow.yml EVAL.PROPOSAL_PER_LARGE 1 EVAL.SIM_MODE delg
# python tests/eval/eval_cbir.py --config_file=Cityflow.yml EVAL.PROPOSAL_PER_LARGE 1 EVAL.SIM_MODE delg


# python tests/eval/extract_global_local.py --config_file=Cityflow.yml EVAL.SIM_MODE sift EVAL.ROUGH_LOCALIZER sift EVAL.REFINER sift
# python tests/eval/perform_global_local_retrieval.py --config_file=Cityflow.yml EVAL.SIM_MODE sift EVAL.ROUGH_LOCALIZER sift EVAL.REFINER sift DELG.USE_LOCAL_FEATURE 0 
# python tests/eval/perform_global_local_retrieval.py --config_file=Cityflow.yml EVAL.SIM_MODE sift DELG.USE_LOCAL_FEATURE 1
# python tests/eval/eval_cbir.py --config_file=Cityflow.yml EVAL.PROPOSAL_PER_LARGE 1 EVAL.SIM_MODE sift
