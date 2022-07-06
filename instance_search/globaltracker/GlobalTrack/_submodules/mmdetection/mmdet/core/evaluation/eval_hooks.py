import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet import datasets
from .coco_utils import fast_eval_recall, results2json
from .mean_ap import eval_map

# DEBUG_LIST = [1, 50, 100, 150, 200, 250]
DEBUG_LIST = []
def debug_input(idx, data, result):
    """Debug input."""

    import cv2
    img_z = data['img_z'].data.cpu().numpy()
    img_x = data['img_x'].data.cpu().numpy()
    gt_bboxes_z = data['gt_bboxes_z'].data.cpu().numpy().astype('int')
    gt_bboxes_x = data['gt_bboxes_x'].data.cpu().numpy()
    img_z = np.transpose(img_z, (1, 2, 0))
    img_x = np.transpose(img_x, (1, 2, 0))
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_z = ((img_z*std) + mean)[:, :, ::-1].astype('uint8')
    img_x = ((img_x*std) + mean)[:, :, ::-1].astype('uint8')
    input_img_x = img_x.copy()
    [x1, y1, x2, y2] = gt_bboxes_z[0]
    cv2.rectangle(img_z, (x1, y1), (x2, y2), (0, 0, 255), 2)
    [x1, y1, x2, y2] = gt_bboxes_x[0]
    cv2.rectangle(img_x, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('./debug_globaltrack/debug_img_z.jpg', img_z)
    cv2.imwrite('./debug_globaltrack/debug_img_x.jpg', img_x)
    # print(gt_bboxes_z[0], gt_bboxes_x[0])
    # print(result[0].shape, result[1].shape)
    best_index = np.argmax(result[1][:, -1])
    best_pred = result[1][best_index]
    # print(result[1])
    print(idx, best_pred)
    img_x = ((img_x*std) + mean)[:, :, ::-1].astype('uint8')
    [x1, y1, x2, y2, conf] = best_pred.astype('int')
    cv2.rectangle(img_x, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('./debug_globaltrack/pred.jpg', img_x)

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        # we only test and save the last epoch
        if (runner.epoch + 1) % runner._max_epochs != 0:
            return
        self.custom_eval(runner)

    def before_train_epoch(self, runner):
        if runner.epoch != 0:
            return
        self.custom_eval(runner)

    def custom_eval(self, runner):
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        # Some node are fast and some are slow, some may lose some node
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            # if idx in DEBUG_LIST:
            #     debug_input(idx, data, result)
            orig_gt_label = {'bboxes': data['img_meta_x'].data['ori_gt_bboxes_x'],
                             'labels': data['gt_labels'].data.cpu().numpy()}
            pred = result[1]
            results[idx] = [orig_gt_label, pred]

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()
        dist.barrier()
        print('test inference finished')
        if runner.rank == 0:
            print('Start to sync all gpu\n')
            dist.barrier()
            print('load cached feature file on other node')
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                print(f'load tmp_results on node {i}')
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            print(f'save tmp_file in {tmp_file}')
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        new_results = []
        for i in range(len(self.dataset)):
            # ann = self.dataset.get_ann_info(i)
            [ann, pred] = results[i]
            new_results.append([pred])
            bboxes = ann['bboxes']
            labels = ann['labels']
            if 'bboxes_ignore' in ann:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
            if i in DEBUG_LIST:
                print(i, bboxes)
        results = new_results
        if not gt_ignore:
            gt_ignore = None
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        num_gts = eval_results[0]['num_gts']
        num_dets = eval_results[0]['num_dets']
        recall = eval_results[0]['recall'][-1]
        topk = int(num_dets/num_gts)
        runner.logger.info(f"Epoch [{runner.epoch + 1}] Evaluation Result: mAP: {mean_ap:.4f}, Recall@{topk}: {recall:.4f} num_gts: {num_gts}, num_dets: {num_dets}")
        # runner.logger.info(eval_results)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'
                     ] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            try:
                cocoDt = cocoGt.loadRes(result_files[res_type])
            except IndexError:
                print('No prediction found.')
                break
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])
