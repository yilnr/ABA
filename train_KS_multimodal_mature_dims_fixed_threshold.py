#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
import json
import random
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import create_logger, Averager, deep_update_dict
from utils.tools import weight_init

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


def to_label_index(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.long()
    return torch.argmax(y, dim=1).long()


def compute_feature_maturity_scores(features: np.ndarray, labels: np.ndarray, eps: float = 1e-8):
    """
    Fisher-style maturity score for each feature dimension.
    A larger value means stronger class separability for that dimension.
    """
    classes = np.unique(labels)
    global_mean = features.mean(axis=0)

    between = np.zeros(features.shape[1], dtype=np.float64)
    within = np.zeros(features.shape[1], dtype=np.float64)

    for c in classes:
        feat_c = features[labels == c]
        if len(feat_c) == 0:
            continue
        mean_c = feat_c.mean(axis=0)
        var_c = feat_c.var(axis=0)
        n_c = len(feat_c)

        between += n_c * (mean_c - global_mean) ** 2
        within += n_c * var_c

    return between / (within + eps)


@torch.no_grad()
def collect_epoch_features(loader, model, max_batches=None):
    model.eval()
    audio_feats = []
    video_feats = []
    labels_all = []

    for step, (spectrogram, image, y) in enumerate(tqdm(loader, leave=False, desc="CollectFeat")):
        if max_batches is not None and step >= max_batches:
            break

        image = image.float().cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        spectrogram = spectrogram.unsqueeze(1).float().cuda(non_blocking=True)

        _, _, _, f_a, f_v = model(spectrogram, image)

        audio_feats.append(f_a.detach().cpu().numpy())
        video_feats.append(f_v.detach().cpu().numpy())
        labels_all.append(to_label_index(y).cpu().numpy())

    audio_feats = np.concatenate(audio_feats, axis=0)
    video_feats = np.concatenate(video_feats, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return audio_feats, video_feats, labels_all


class MatureDimRecorder:
    def __init__(
        self,
        save_root,
        threshold_ratio=0.3,
        smooth_window=1,
        warmup_epochs=10,
        fixed_threshold_mode="warmup_max",
        fixed_threshold_quantile=0.8,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_root, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.threshold_ratio = threshold_ratio
        self.smooth_window = max(1, int(smooth_window))
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.fixed_threshold_mode = fixed_threshold_mode
        self.fixed_threshold_quantile = float(fixed_threshold_quantile)

        self.audio_scores = []
        self.video_scores = []
        self.audio_mature_counts = []
        self.video_mature_counts = []

        self.audio_threshold = None
        self.video_threshold = None
        self.threshold_frozen = False
        self.audio_warmup_scores = []
        self.video_warmup_scores = []

    @staticmethod
    def _moving_average(x, window):
        if window <= 1 or len(x) < window:
            return np.array(x, dtype=np.float64)
        x = np.array(x, dtype=np.float64)
        out = np.convolve(x, np.ones(window) / window, mode='valid')
        pad = [out[0]] * (window - 1)
        return np.array(pad + out.tolist(), dtype=np.float64)

    def _freeze_thresholds_if_needed(self):
        if self.threshold_frozen:
            return
        if len(self.audio_scores) < self.warmup_epochs:
            return

        audio_pool = np.concatenate(self.audio_warmup_scores, axis=0)
        video_pool = np.concatenate(self.video_warmup_scores, axis=0)

        if self.fixed_threshold_mode == "warmup_quantile":
            self.audio_threshold = float(np.quantile(audio_pool, self.fixed_threshold_quantile))
            self.video_threshold = float(np.quantile(video_pool, self.fixed_threshold_quantile))
        else:
            self.audio_threshold = self.threshold_ratio * float(audio_pool.max())
            self.video_threshold = self.threshold_ratio * float(video_pool.max())

        self.threshold_frozen = True

    def update(self, audio_feat: np.ndarray, video_feat: np.ndarray, labels: np.ndarray):
        score_a = compute_feature_maturity_scores(audio_feat, labels)
        score_v = compute_feature_maturity_scores(video_feat, labels)

        self.audio_scores.append(score_a)
        self.video_scores.append(score_v)

        if not self.threshold_frozen:
            self.audio_warmup_scores.append(score_a)
            self.video_warmup_scores.append(score_v)
            self._freeze_thresholds_if_needed()

        if self.threshold_frozen:
            th_a = self.audio_threshold
            th_v = self.video_threshold
        else:
            # warmup阶段先沿用原始动态阈值，等积累完 warmup 再冻结
            th_a = self.threshold_ratio * float(score_a.max())
            th_v = self.threshold_ratio * float(score_v.max())

        self.audio_mature_counts.append(int((score_a > th_a).sum()))
        self.video_mature_counts.append(int((score_v > th_v).sum()))

    def plot_curve(self):
        epochs = np.arange(1, len(self.audio_mature_counts) + 1)
        audio_curve = self._moving_average(self.audio_mature_counts, self.smooth_window)
        video_curve = self._moving_average(self.video_mature_counts, self.smooth_window)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, audio_curve, marker='o', linewidth=2.2, markersize=4.5, label='Audio')
        plt.plot(epochs, video_curve, marker='o', linewidth=2.2, markersize=4.5, label='Video')
        plt.scatter(epochs, self.audio_mature_counts, s=18, alpha=0.30)
        plt.scatter(epochs, self.video_mature_counts, s=18, alpha=0.30)

        if self.warmup_epochs < len(epochs):
            plt.axvline(self.warmup_epochs, linestyle='--', color='gray', alpha=0.7)
            plt.text(self.warmup_epochs + 0.5, max(max(audio_curve), max(video_curve)) * 0.08,
                     'threshold frozen', fontsize=9, color='gray')

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Number of Mature Dimensions", fontsize=12)
        plt.title("Number of Mature Dimensions vs Epoch", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        png_path = os.path.join(self.save_dir, "mature_dimensions_vs_epoch.png")
        pdf_path = os.path.join(self.save_dir, "mature_dimensions_vs_epoch.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()

        meta = {
            "threshold_frozen": self.threshold_frozen,
            "audio_threshold": self.audio_threshold,
            "video_threshold": self.video_threshold,
            "warmup_epochs": self.warmup_epochs,
            "fixed_threshold_mode": self.fixed_threshold_mode,
            "fixed_threshold_quantile": self.fixed_threshold_quantile,
            "threshold_ratio": self.threshold_ratio,
        }
        with open(os.path.join(self.save_dir, "mature_threshold_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return png_path, pdf_path


def train_audio_video(epoch, train_loader, model, optimizer, logger, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc=f"Train {epoch}")):
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

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , '
         'Average loss_audio : {loss_audio:.3f}, Average loss_video : {loss_video:.3f}').format(
            epoch=epoch,
            loss_ave=tl.item(),
            loss_audio=tl_a.item(),
            loss_video=tl_v.item(),
        )
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
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader, desc=f"Val {epoch}")):
            label_list += torch.argmax(y, dim=1).tolist()
            one_hot_label += y.tolist()
            image = image.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            spectrogram = spectrogram.unsqueeze(1).float().cuda(non_blocking=True)

            result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

            soft_pred_a += F.softmax(result_a, dim=1).tolist()
            soft_pred_v += F.softmax(result_v, dim=1).tolist()
            soft_pred += F.softmax(0.5 * result_a + 0.5 * result_v, dim=1).tolist()
            pred = F.softmax(0.5 * result_a + 0.5 * result_v, dim=1).argmax(dim=1)
            pred_a = F.softmax(result_a, dim=1).argmax(dim=1)
            pred_v = F.softmax(result_v, dim=1).argmax(dim=1)

            pred_list += pred.tolist()
            pred_list_a += pred_a.tolist()
            pred_list_v += pred_v.tolist()

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
         'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(
            epoch=epoch,
            f1=f1, acc=acc, mAP=mAP,
            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v,
        )
    )
    return acc, acc_a, acc_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--mature_threshold_ratio', type=float, default=0.3,
                        help='Used only when fixed_threshold_mode=warmup_max')
    parser.add_argument('--mature_smooth_window', type=int, default=3)
    parser.add_argument('--mature_max_batches', type=int, default=0,
                        help='0 means use all batches from the evaluation loader')
    parser.add_argument('--mature_save_root', type=str, default='/data/zyh/NeurIPS24-LFM/_figure/mature_fixed')
    parser.add_argument('--mature_warmup_epochs', type=int, default=10)
    parser.add_argument('--mature_fixed_threshold_mode', type=str, default='warmup_max',
                        choices=['warmup_max', 'warmup_quantile'])
    parser.add_argument('--mature_fixed_threshold_quantile', type=float, default=0.8)
    args = parser.parse_args()

    cfg = config
    with open(args.config, 'r') as f:
        exp_params = json.load(f)
    cfg = deep_update_dict(exp_params, cfg)

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

    local_rank = cfg['train']['local_rank']
    logits_ratio = cfg['train']['logits_ratio']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

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

    model = AVClassifier(config=cfg).cuda()
    model.apply(weight_init)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        momentum=cfg['train']['optimizer']['momentum'],
        weight_decay=cfg['train']['optimizer']['wc'],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        cfg['train']['lr_scheduler']['patience'],
        0.1,
    )

    best_acc = 0.0
    mature_recorder = MatureDimRecorder(
        save_root=args.mature_save_root,
        threshold_ratio=args.mature_threshold_ratio,
        smooth_window=args.mature_smooth_window,
        warmup_epochs=args.mature_warmup_epochs,
        fixed_threshold_mode=args.mature_fixed_threshold_mode,
        fixed_threshold_quantile=args.mature_fixed_threshold_quantile,
    )
    logger.info(f'Mature-dimension plots will be saved to: {mature_recorder.save_dir}')
    logger.info(
        f'Fixed-threshold mode: {args.mature_fixed_threshold_mode}, '
        f'warmup_epochs={args.mature_warmup_epochs}, '
        f'threshold_ratio={args.mature_threshold_ratio}, '
        f'quantile={args.mature_fixed_threshold_quantile}'
    )

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, logits_ratio)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        max_batches = None if args.mature_max_batches <= 0 else args.mature_max_batches
        audio_feat, video_feat, labels = collect_epoch_features(test_loader, model, max_batches=max_batches)
        mature_recorder.update(audio_feat, video_feat, labels)
        png_path, pdf_path = mature_recorder.plot_curve()
        logger.info(f'Mature-dimension curve saved to: {png_path}')

        if acc > best_acc:
            best_acc = acc

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks'
                f'multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )
