set -e
DATASET_NAME="INS_CUHK_SYSU"
python tests/instance_searcher/test_instance_searcher.py --config_file="${DATASET_NAME}.yml"
