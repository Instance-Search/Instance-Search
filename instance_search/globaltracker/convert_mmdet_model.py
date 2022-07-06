import torch

old_model_path = '/home/gongyou.zyq/video_object_retrieval/instance_search/globaltracker/GlobalTrack/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot_origin.pth'
new_model_path = '/home/gongyou.zyq/video_object_retrieval/instance_search/globaltracker/GlobalTrack/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
old_model = torch.load(old_model_path)
old_list = ['bbox_head.fc_cls.weight', 'bbox_head.fc_cls.bias', 'bbox_head.fc_reg.weight', 'bbox_head.fc_reg.bias', 'bbox_head.shared_fcs.0.weight', 'bbox_head.shared_fcs.0.bias', 'bbox_head.shared_fcs.1.weight', 'bbox_head.shared_fcs.1.bias']
reg_list = ['bbox_head.fc_reg.weight', 'bbox_head.fc_reg.bias']
new_model = old_model.copy()
new_model['state_dict'] = old_model['state_dict'].copy()
for name, param in old_model['state_dict'].items():
    if name in old_list:
        new_name = 'roi_head.' + name
        new_model['state_dict'][new_name] = param
    if name in reg_list:
        new_model['state_dict'][new_name] = param[:4]
torch.save(new_model, new_model_path)
new_model = torch.load(new_model_path)
for name, param in new_model['state_dict'].items():
    print(name)
