# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .ourapi import OURAPI
from .unsup import UNSUP
from .bbox_unsup import BBOX_UNSUP
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'ourapi': OURAPI,
    'unsup': UNSUP,
    'bbox_unsup': BBOX_UNSUP,
}


def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

def add_psolabels(dataset,psolabels):
    assert(len(dataset)==len(psolabels))
    outset = []
    for i in range(len(dataset)):
        img_path, pid, camid, setid, trkid, raw_image_name, bbox = dataset[i]
        outset.append((img_path,psolabels[i],camid,setid,trkid, raw_image_name, bbox))
    return outset
