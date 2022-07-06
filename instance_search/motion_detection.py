"""Moving object detection
Author: gongyou.zyq
Date: 2021.02.07
"""

from instance_search.background_subtraction.bs_inference import \
        BackgroundSubtraction
from instance_search.optical_flow.optical_flow_inference import OpticalFlow

def motion_detection_factory(gpu_id, _cfg):
    """Selects motion detector for multiple kinds.

    We supoort background subtraction and optical flow currently.

    Args:
        gpu_id: A int indicating which gpu to use
        _cfg: Instance search config.

    Returns:
        Motion detector object.
    """

    if _cfg.EVAL.ROUGH_LOCALIZER == 'background_subtraction':
        _motion_detector = BackgroundSubtraction(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'optical_flow':
        _motion_detector = OpticalFlow(gpu_id, _cfg)
    else:
        print('unknown sim_mode.')
        _motion_detector = None
    return _motion_detector
