#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import json
import time
import random
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import weight_init


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


class LearningProgressRecorder:
    """Record unimodal performance and plot normalized modality learning progress."""

    def __init__(self, save_root: str, smooth_window: int = 5):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_root, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.audio_acc = []
        self.video_acc = []
        self.multi_acc = []
        self.smooth_window = smooth_window

    def update(self, acc: float, acc_a: float, acc_v: float) -> None:
        self.multi_acc.append(float(acc))
        self.audio_acc.append(float(acc_a))
        self.video_acc.append(float(acc_v))

    @staticmethod
    def _normalize_progress(x, eps: float = 1e-8):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return x
        x0 = x[0]
        xT = x[-1]
        return (x - x0) / (xT - x0 + eps)

    @staticmethod
    def _moving_average(x, w: int = 5):
        x = np.asarray(x, dtype=np.float64)
        if len(x) < w or w <= 1:
            return x
        pad = w // 2
        x_pad = np.pad(x, (pad, pad), mode='edge')
        kernel = np.ones(w, dtype=np.float64) / float(w)
        return np.convolve(x_pad, kernel, mode='valid')

    def plot(self, lr_decay_epoch: int | None = None):
        epochs = np.arange(1, len(self.audio_acc) + 1)

        prog_a = self._normalize_progress(self.audio_acc)
        prog_v = self._normalize_progress(self.video_acc)

        prog_a_s = self._moving_average(prog_a, self.smooth_window)
        prog_v_s = self._moving_average(prog_v, self.smooth_window)

        plt.figure(figsize=(7.4, 5.3))

        # raw points
        plt.plot(epochs, prog_a, 'o', alpha=0.22, markersize=4, color='#1f77b4')
        plt.plot(epochs, prog_v, 'o', alpha=0.22, markersize=4, color='#ff7f0e')

        # smoothed curves
        plt.plot(epochs, prog_a_s, linewidth=2.8, color='#1f77b4', label='Audio progress')
        plt.plot(epochs, prog_v_s, linewidth=2.8, color='#ff7f0e', label='Video progress')

        # learning gap
        plt.fill_between(
            epochs,
            np.minimum(prog_a_s, prog_v_s),
            np.maximum(prog_a_s, prog_v_s),
            color='gray',
            alpha=0.12,
            label='Learning gap'
        )

        if lr_decay_epoch is not None and lr_decay_epoch >= 0:
            plt.axvline(x=lr_decay_epoch, linestyle='--', color='gray', alpha=0.7)
            ymax = max(float(np.max(prog_a_s)), float(np.max(prog_v_s)), 0.1)
            plt.text(lr_decay_epoch + 0.5, max(0.05, 0.08 * ymax), 'LR decay', fontsize=10, color='gray')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Normalized Learning Progress', fontsize=12)
        plt.title('Normalized Modality Learning Progress', fontsize=13)
        plt.ylim(-0.02, 1.05)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        png_path = os.path.join(self.save_dir, 'normalized_modality_learning_progress.png')
        pdf_path = os.path.join(self.save_dir, 'normalized_modality_learning_progress.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        return png_path, pdf_path


def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        spectrogram = spectrogram.unsqueeze(1).float().cuda(non_blocking=True)

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

    loss_ave = tl.item()
    loss_audio = tl_a.item()
    loss_video = tl_v.item()

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , '
         'Average loss_audio : {loss_audio:.3f},Average loss_video : {loss_video:.3f}')
        .format(epoch=epoch, loss_ave=loss_ave, loss_audio=loss_audio, loss_video=loss_video)
    )

    return model


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

    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_list.extend(torch.argmax(y, dim=1).tolist())
            one_hot_label.extend(y.tolist())
            image = image.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            spectrogram = spectrogram.unsqueeze(1).float().cuda(non_blocking=True)

            result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

            soft_pred_a.extend(F.softmax(result_a, dim=1).tolist())
            soft_pred_v.extend(F.softmax(result_v, dim=1).tolist())
            soft_pred.extend(F.softmax(0.5 * result_a + 0.5 * result_v, dim=1).tolist())
            pred = F.softmax(0.5 * result_a + 0.5 * result_v, dim=1).argmax(dim=1)
            pred_a = F.softmax(result_a, dim=1).argmax(dim=1)
            pred_v = F.softmax(result_v, dim=1).argmax(dim=1)

            pred_list.extend(pred.tolist())
            pred_list_a.extend(pred_a.tolist())
            pred_list_v.extend(pred_v.tolist())

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

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},'
         'f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},'
         'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}')
        .format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v)
    )
    return acc, acc_a, acc_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--progress_save_root', type=str,
                        default='/data/zyh/NeurIPS24-LFM/_figure/acc_progress')
    parser.add_argument('--progress_smooth_window', type=int, default=5)
    parser.add_argument('--mark_lr_decay', action='store_true',
                        help='Mark LR decay epoch on the progress curve.')
    args = parser.parse_args()

    cfg = config
    with open(args.config, 'r') as f:
        exp_params = json.load(f)
    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logits_ratio = cfg['train']['logits_ratio']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = VADataset(cfg, mode='train')
    test_dataset = VADataset(cfg, mode='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True,
    )

    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = cfg['train']['optimizer']['lr']
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr_adjust,
        momentum=cfg['train']['optimizer']['momentum'],
        weight_decay=cfg['train']['optimizer']['wc']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        cfg['train']['lr_scheduler']['patience'],
        0.1
    )

    best_acc = 0
    cls_k = []

    progress_recorder = LearningProgressRecorder(
        save_root=args.progress_save_root,
        smooth_window=args.progress_smooth_window,
    )

    logger.info(f'Progress figure will be saved to: {progress_recorder.save_dir}')

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        progress_recorder.update(acc, acc_a, acc_v)
        lr_decay_epoch = cfg['train']['lr_scheduler']['patience'] if args.mark_lr_decay else None
        png_path, pdf_path = progress_recorder.plot(lr_decay_epoch=lr_decay_epoch)
        logger.info(f'Normalized learning progress figure saved to: {png_path}')

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/'
                f'multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )
