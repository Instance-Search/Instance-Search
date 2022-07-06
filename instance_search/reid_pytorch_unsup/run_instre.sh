python tools/train.py --config_file='configs/instre_sup.yml' MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR "('log/instre_sup/')" DIST.NODE_WORLDSIZE "1" DIST.NODE_RANK "0" DIST.URL 'tcp://127.0.0.17:12347'

python tools/train.py --config_file='configs/instre.yml' MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" OUTPUT_DIR "('log/instre_unsup/')" DIST.NODE_WORLDSIZE "1" DIST.NODE_RANK "0" DIST.URL 'tcp://127.0.0.17:12347'
