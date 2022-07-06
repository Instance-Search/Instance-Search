# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from layers import make_loss
from data import make_target_unsupdata_loader, make_alltrain_data_loader
from utils.reid_metric import R1_mAP
from utils.reid_metric import Cluster

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
import onnx
from onnx import optimizer as onnx_optimizer
from tools.export_onnx import onnx_apply, change_input_dim
from modeling.baseline import weights_init_classifier

def create_supervised_trainer(model, optimizer, loss_fn, loss_weight,
                              device=None, device_id=-1, distribute=False):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if distribute:
            torch.cuda.set_device(device_id)
            model.cuda(device_id)
            # TODO: where is the layer without grad?
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model.to(device)
        else:
            model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, setid = batch
        # print(target.max())
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feats = model(img)

        losses = []
        total_loss = torch.tensor(0.).cuda() # 第一句
        for i in range(len(loss_fn)):
            loss = torch.tensor(0.).cuda()
            if i == setid[0]:
                for j in range(len(loss_fn[i])):
                    loss += loss_fn[i][j](feats[i], feats[-1], target)
            else:
                loss += 0. * torch.sum(feats[i])
            # loss += 0. * sum(p.sum() for p in model.parameters()) #第二句没用注释
            total_loss += loss * loss_weight[i] # 第三句
            losses.append(loss)

        total_loss.backward() # 第四句
        optimizer.step()
        # compute acc
        #acc = (feats[0].max(1)[1] == target).float().mean()
        return {'src':losses[0].item(),'tgt_sup':losses[1].item(),'tgt_unsup':losses[2].item()}

    return Engine(_update)

def create_supervised_evaluator(model, metrics,
                                device=None, device_id=-1, distribute=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    '''
    if device:
        if distribute:
            torch.cuda.set_device(device_id)
            model.cuda(device_id)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model.to(device)
        else:
            model.to(device)
    '''

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, _, _, _ = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_psolabel_producer(model, metrics,
                                device=None, device_id=-1, distribute=False):

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, trkids, raw_image_name, bbox = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids, trkids, raw_image_name, bbox

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_train(
        cfg,
        model,
        val_data_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        device_id,
        distribute
):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    psolabel_period = cfg.TGT_UNSUPDATA.PSOLABEL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, cfg.LOSS.LOSS_WEIGHTS, device=device, device_id=device_id, distribute=distribute)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, device_id=device_id, distribute=distribute)
    psolabel_producer = create_psolabel_producer(model, metrics={'cluster': Cluster(config=cfg, topk=cfg.TGT_UNSUPDATA.CLUSTER_TOPK,dist_thrd=cfg.TGT_UNSUPDATA.CLUSTER_DIST_THRD, min_samples=cfg.TGT_UNSUPDATA.MIN_SAMPLES)}, device=device, device_id=device_id, distribute=distribute)
    if device_id == 0:
        checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['src']).attach(trainer, 'src_loss')
    RunningAverage(output_transform=lambda x: x['tgt_sup']).attach(trainer, 'tgt_sup_loss')
    RunningAverage(output_transform=lambda x: x['tgt_unsup']).attach(trainer, 'tgt_unsup_loss')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        #scheduler.step()
        pass

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        alltrain_data_loader = engine.state.dataloader
        iter = (engine.state.iteration - 1) % len(alltrain_data_loader) + 1
        if iter % log_period == 0:
            if cfg.DATALOADER.SAMPLER_PROB[0] != 0:
                src_loss = engine.state.metrics['src_loss']/cfg.DATALOADER.SAMPLER_PROB[0]
            else:
                src_loss = 0.
            if cfg.DATALOADER.SAMPLER_PROB[1] != 0:
                tgt_sup_loss = engine.state.metrics['tgt_sup_loss']/cfg.DATALOADER.SAMPLER_PROB[1]
            else:
                tgt_sup_loss = 0.
            if cfg.DATALOADER.SAMPLER_PROB[2] != 0:
                tgt_unsup_loss = engine.state.metrics['tgt_unsup_loss']/cfg.DATALOADER.SAMPLER_PROB[2]
            else:
                tgt_unsup_loss = 0.
            logger.info("Epoch[{}] Iter[{}/{}] src: {:.3f}, sup: {:.3f}, unsup: {:.3f}, lr: {:.2e}/{:.2e}"
                        .format(engine.state.epoch, iter, len(alltrain_data_loader),
                                src_loss, tgt_sup_loss, tgt_unsup_loss,
                                scheduler.get_lr()[0],scheduler.get_lr()[-1]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_psolabels(engine):
        if engine.state.epoch % psolabel_period == 0 and 'cluster' in cfg.TGT_UNSUPDATA.UNSUP_MODE:
            target_unsupdata_loader = make_target_unsupdata_loader(cfg)
            psolabel_producer.run(target_unsupdata_loader)
            psolabels,cluster_acc,num,mean_feat = psolabel_producer.state.metrics['cluster']
            num_classes = max(set(psolabels))+1
            logger.info("Cluster Acc: {:.3f}, classes: {} imgnum: {}".format(cluster_acc,num_classes,num))
            alltrain_data_loader = make_alltrain_data_loader(cfg,psolabels)
            # Ref: https://github.com/pytorch/ignite/commit/fb63728405266e20c578180e1a32290b27671e6f
            engine.state.dataloader = alltrain_data_loader
            engine._dataloader_iter = iter(engine.state.dataloader)
            reset_fc(num_classes, mean_feat)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        alltrain_data_loader = engine.state.dataloader
        scheduler.step()
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            alltrain_data_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    def reset_fc(num_classes, mean_feat):
        """Reset fc for cluster-based unsup."""

        logger.info(f"Reset last fc with class num: {num_classes}")
        device_id = next(model.parameters()).device
        if mean_feat is not None:
            # model.to('cpu')
            # model.tgt_unsup_classifier = nn.Linear(model.in_planes, num_classes,
            #                                        bias=False)
            # torch.manual_seed(0)
            # torch.cuda.manual_seed(0)
            # torch.cuda.manual_seed_all(0)
            model.tgt_unsup_classifier.apply(weights_init_classifier)
            # model.tgt_unsup_classifier.weight = nn.Parameter(mean_feat)
        # print(model.tgt_unsup_classifier.state_dict())
        # Ref: https://github.com/pytorch/pytorch/issues/7460
        model.to(device_id)

    def custom_eval(engine):
        evaluator.run(val_data_loader)
        cmc, mAP = evaluator.state.metrics['r1_mAP']
        logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        # output onnx
        # if device_id == 0 and engine.state.epoch == epochs - 1:    # only keep the last model onnx
        if device_id == 0 and engine.state.epoch == epochs:    # only keep the last model onnx
            # h,w = cfg.INPUT.SIZE_TRAIN
            # dummy_input = torch.randn(1, 3, h, w, device='cuda')
            # torch.onnx.export(model, dummy_input, "%s/reid_%s_%d.onnx" %(output_dir,cfg.MODEL.NAME,engine.state.epoch), verbose=False)

            onnxfile = '%s/reid_%s_%d.onnx' % (output_dir,cfg.MODEL.NAME,engine.state.epoch)
            h,w = cfg.INPUT.SIZE_TRAIN
            dummy_input = torch.randn(1, 3, h, w, device='cuda')
            dummy_input = dummy_input.half()
            half_model = copy.deepcopy(model).eval().half()
            torch.onnx.export(half_model, dummy_input, onnxfile, verbose=True,
                              keep_initializers_as_inputs=True)
            onnx_model = onnx.load(onnxfile)
            passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
            optimized_model = onnx_optimizer.optimize(onnx_model, passes)
            onnx.save(optimized_model, onnxfile)
            onnx_apply(change_input_dim, onnxfile, onnxfile)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            custom_eval(engine)

    @trainer.on(Events.EPOCH_STARTED)
    def log_validation_results_init(engine):
        if engine.state.epoch == 1:
            pass
            # custom_eval(engine)

    if 'cluster' in cfg.TGT_UNSUPDATA.UNSUP_MODE:
        target_unsupdata_loader = make_target_unsupdata_loader(cfg)
        psolabel_producer.run(target_unsupdata_loader)
        psolabels, cluster_acc, num, mean_feat = psolabel_producer.state.metrics['cluster']
        # logger.info("Cluster Acc: {:.3f}, classes: {} imgnum: {}".format(cluster_acc, len(set(psolabels))-1, num))
        num_classes = max(set(psolabels))+1
        logger.info("Cluster Acc: {:.3f}, classes: {} imgnum: {}".format(cluster_acc, num_classes, num))
        alltrain_data_loader = make_alltrain_data_loader(cfg, psolabels)
        reset_fc(num_classes, None)
    else:
        alltrain_data_loader = make_alltrain_data_loader(cfg, None)
    trainer.run(alltrain_data_loader, max_epochs=epochs)
