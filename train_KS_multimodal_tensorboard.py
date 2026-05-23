#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
import json
import numpy as np
import argparse
import random

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

from utils.min_norm_solvers import MinNormSolver
from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import GSPlugin, weight_init

warnings.filterwarnings("ignore")


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


def get_label_indices(y):
    if y.dim() > 1:
        return torch.argmax(y, dim=1)
    return y.long()


def compute_mean_gt_confidence(logits, label_idx):
    """
    计算整体模态置信度：
    对所有样本在真实类别上的 softmax 概率取平均
    """
    probs = F.softmax(logits, dim=1)
    idx = torch.arange(label_idx.shape[0], device=label_idx.device)
    gt_conf = probs[idx, label_idx]
    return gt_conf.mean().item(), gt_conf.detach().cpu().numpy()


def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    tl_f = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    all_audio_conf = []
    all_video_conf = []

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        label_idx = get_label_indices(y)

        optimizer.zero_grad()
        result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

        loss_a = criterion(result_a, y).mean()
        loss_v = criterion(result_v, y).mean()
        loss_fusion = criterion(logits_ratio * result_a + logits_ratio * result_v, y).mean()
        loss = loss_a + loss_v + loss_fusion

        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_a.add(loss_a.item())
        tl_v.add(loss_v.item())
        tl_f.add(loss_fusion.item())

        _, audio_conf_arr = compute_mean_gt_confidence(result_a, label_idx)
        _, video_conf_arr = compute_mean_gt_confidence(result_v, label_idx)
        all_audio_conf.extend(audio_conf_arr.tolist())
        all_video_conf.extend(video_conf_arr.tolist())

    loss_ave = tl.item()
    loss_audio = tl_a.item()
    loss_video = tl_v.item()
    loss_fusion = tl_f.item()

    train_conf_audio = float(np.mean(all_audio_conf)) if len(all_audio_conf) > 0 else 0.0
    train_conf_video = float(np.mean(all_video_conf)) if len(all_video_conf) > 0 else 0.0

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, '
         'Average loss_audio:{loss_audio:.3f}, Average loss_video:{loss_video:.3f}, '
         'Average loss_fusion:{loss_fusion:.3f}, '
         'Train conf_audio:{conf_a:.4f}, Train conf_video:{conf_v:.4f}')
        .format(
            epoch=epoch,
            loss_ave=loss_ave,
            loss_audio=loss_audio,
            loss_video=loss_video,
            loss_fusion=loss_fusion,
            conf_a=train_conf_audio,
            conf_v=train_conf_video,
        )
    )

    train_stats = {
        'loss_total': loss_ave,
        'loss_audio': loss_audio,
        'loss_video': loss_video,
        'loss_fusion': loss_fusion,
        'conf_audio': train_conf_audio,
        'conf_video': train_conf_video,
    }
    return model, train_stats


def val(epoch, val_loader, model, logger):
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []

    all_audio_conf = []
    all_video_conf = []

    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_idx = get_label_indices(y)
            label_list = label_list + label_idx.tolist()
            one_hot_label = one_hot_label + y.tolist()

            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            label_idx = label_idx.cuda()

            result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(result_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(result_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(0.5 * result_a + 0.5 * result_v, dim=1)).tolist()

            pred = (F.softmax(0.5 * result_a + 0.5 * result_v, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(result_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(result_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

            _, audio_conf_arr = compute_mean_gt_confidence(result_a, label_idx)
            _, video_conf_arr = compute_mean_gt_confidence(result_v, label_idx)
            all_audio_conf.extend(audio_conf_arr.tolist())
            all_video_conf.extend(video_conf_arr.tolist())

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')

        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)

        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)

        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    conf_audio = float(np.mean(all_audio_conf)) if len(all_audio_conf) > 0 else 0.0
    conf_video = float(np.mean(all_video_conf)) if len(all_video_conf) > 0 else 0.0

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: '
         'f1:{f1:.4f}, acc:{acc:.4f}, mAP:{mAP:.4f}, '
         'f1_a:{f1_a:.4f}, acc_a:{acc_a:.4f}, mAP_a:{mAP_a:.4f}, '
         'f1_v:{f1_v:.4f}, acc_v:{acc_v:.4f}, mAP_v:{mAP_v:.4f}, '
         'conf_audio:{conf_a:.4f}, conf_video:{conf_v:.4f}')
        .format(
            epoch=epoch,
            f1=f1, acc=acc, mAP=mAP,
            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v,
            conf_a=conf_audio, conf_v=conf_video
        )
    )

    val_stats = {
        'acc': acc,
        'acc_audio': acc_a,
        'acc_video': acc_v,
        'f1': f1,
        'f1_audio': f1_a,
        'f1_video': f1_v,
        'mAP': mAP,
        'mAP_audio': mAP_a,
        'mAP_video': mAP_v,
        'conf_audio': conf_audio,
        'conf_video': conf_video,
    }
    return val_stats


def log_stats_to_tensorboard(writer, epoch, train_stats, val_stats, lr):
    writer.add_scalar('lr', lr, epoch)

    writer.add_scalar('train/loss_total', train_stats['loss_total'], epoch)
    writer.add_scalar('train/loss_audio', train_stats['loss_audio'], epoch)
    writer.add_scalar('train/loss_video', train_stats['loss_video'], epoch)
    writer.add_scalar('train/loss_fusion', train_stats['loss_fusion'], epoch)

    writer.add_scalar('train/conf_audio', train_stats['conf_audio'], epoch)
    writer.add_scalar('train/conf_video', train_stats['conf_video'], epoch)

    writer.add_scalar('val/acc', val_stats['acc'], epoch)
    writer.add_scalar('val/f1', val_stats['f1'], epoch)
    writer.add_scalar('val/mAP', val_stats['mAP'], epoch)

    writer.add_scalar('val/acc_audio', val_stats['acc_audio'], epoch)
    writer.add_scalar('val/f1_audio', val_stats['f1_audio'], epoch)
    writer.add_scalar('val/mAP_audio', val_stats['mAP_audio'], epoch)

    writer.add_scalar('val/acc_video', val_stats['acc_video'], epoch)
    writer.add_scalar('val/f1_video', val_stats['f1_video'], epoch)
    writer.add_scalar('val/mAP_video', val_stats['mAP_video'], epoch)

    writer.add_scalar('val/conf_audio', val_stats['conf_audio'], epoch)
    writer.add_scalar('val/conf_video', val_stats['conf_video'], epoch)


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--tb_log_dir', type=str, default='/data/zyh/NeurIPS24-LFM/_tensorboard_runs',
                        help='TensorBoard log root directory.')

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logits_ratio = cfg['train']['logits_ratio']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    tb_dir = os.path.join(args.tb_log_dir, f'ks_{exp_id}')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f'TensorBoard log dir: {tb_dir}')

    # ----- SET DATALOADER -----
    train_dataset = VADataset(config, mode='train')
    test_dataset = VADataset(config, mode='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True
    )

    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr_adjust,
        momentum=config['train']['optimizer']['momentum'],
        weight_decay=config['train']['optimizer']['wc']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        config['train']['lr_scheduler']['patience'],
        0.1
    )

    best_acc = 0
    cls_k = []

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        model, train_stats = train_audio_video(
            epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio
        )

        val_stats = val(epoch, test_loader, model, logger)

        log_stats_to_tensorboard(
            writer=writer,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            lr=current_lr
        )
        writer.flush()

        acc = val_stats['acc']
        acc_a = val_stats['acc_audio']
        acc_v = val_stats['acc_video']

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )

    writer.close()