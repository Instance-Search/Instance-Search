"""Global local inference.
Author: gongyou.zyq
Date: 2020.11.27
"""

from instance_search.sift.sift_inference import SiftMatcher
from instance_search.delg.delg_inference import DelgMatcher
from instance_search.delg.delg_custom_inference import DelgCustomMatcher
from instance_search.delg.delg_vit_inference import DelgVitMatcher
from instance_search.delg.delg_reid_inference import DelgReIDMatcher
from instance_search.delg.delg_competition import DelgCompetitionMatcher
from instance_search.delg.delg_3rd import DelgCompetition3rdMatcher
# from instance_search.delg.delg_torch_inference import DelgTorchMatcher

def global_local_factory(gpu_id, _cfg):
    """Selects global local matcher for multiple kinds.

    We supoort sift and delg currently.

    Args:
        gpu_id: A int indicating which gpu to use
        _cfg: Instance search config.

    Returns:
        Global local matcher object.
    """

    if _cfg.EVAL.SIM_MODE== 'sift':
        _matcher = SiftMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg':
        _matcher = DelgMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_custom':
        _matcher = DelgCustomMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_vit':
        _matcher = DelgVitMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_reid':
        _matcher = DelgReIDMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_torch':
        _matcher = DelgTorchMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_competition_baseline':
        _matcher = DelgCompetitionMatcher(gpu_id, _cfg)
    elif _cfg.EVAL.SIM_MODE == 'delg_3rd':
        _matcher = DelgCompetition3rdMatcher(gpu_id, _cfg)
    else:
        print('unknown sim_mode.')
        _matcher = None
    return _matcher
