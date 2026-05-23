#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import json
import random
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.tools import weight_init
from utils.utils import create_logger, Averager, deep_update_dict

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


class ClassWiseStatProjector(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.9, eps=1e-5, loss_type="smooth_l1", cosine_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.eps = eps
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight

        self.register_buffer("audio_mean", torch.zeros(num_classes, feat_dim))
        self.register_buffer("video_mean", torch.zeros(num_classes, feat_dim))
        self.register_buffer("audio_var", torch.ones(num_classes, feat_dim))
        self.register_buffer("video_var", torch.ones(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @staticmethod
    def _get_label_index(y):
        if y.ndim == 1:
            return y.long()
        return torch.argmax(y, dim=1).long()

    @torch.no_grad()
    def update_stats(self, feat_a, feat_v, y):
        labels = self._get_label_index(y)
        unique_labels = labels.unique(sorted=False)
        for cls in unique_labels.tolist():
            mask = labels == cls
            if mask.sum() == 0:
                continue
            cls_a = feat_a[mask]
            cls_v = feat_v[mask]
            mean_a = cls_a.mean(dim=0)
            mean_v = cls_v.mean(dim=0)
            var_a = torch.clamp(cls_a.var(dim=0, unbiased=False), min=self.eps)
            var_v = torch.clamp(cls_v.var(dim=0, unbiased=False), min=self.eps)

            if not self.initialized[cls]:
                self.audio_mean[cls] = mean_a
                self.video_mean[cls] = mean_v
                self.audio_var[cls] = var_a
                self.video_var[cls] = var_v
                self.initialized[cls] = True
            else:
                m = self.momentum
                self.audio_mean[cls] = m * self.audio_mean[cls] + (1.0 - m) * mean_a
                self.video_mean[cls] = m * self.video_mean[cls] + (1.0 - m) * mean_v
                self.audio_var[cls] = m * self.audio_var[cls] + (1.0 - m) * var_a
                self.video_var[cls] = m * self.video_var[cls] + (1.0 - m) * var_v

    def project_audio_to_video(self, feat_a, y):
        labels = self._get_label_index(y)
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_a - mu_a) / std_a * std_v + mu_v

    def project_video_to_audio(self, feat_v, y):
        labels = self._get_label_index(y)
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_v - mu_v) / std_v * std_a + mu_a

    def _pair_loss(self, x, y):
        if self.loss_type == "mse":
            base = F.mse_loss(x, y)
        else:
            base = F.smooth_l1_loss(x, y)
        if self.cosine_weight > 0:
            cos = 1.0 - F.cosine_similarity(x, y, dim=1).mean()
            base = base + self.cosine_weight * cos
        return base

    def forward(self, feat_a, feat_v, y):
        self.update_stats(feat_a.detach(), feat_v.detach(), y)
        proj_a2v = self.project_audio_to_video(feat_a, y)
        proj_v2a = self.project_video_to_audio(feat_v, y)
        loss_a2v = self._pair_loss(proj_a2v, feat_v)
        loss_v2a = self._pair_loss(proj_v2a, feat_a)
        loss = 0.5 * (loss_a2v + loss_v2a)
        stats = {
            "proj_loss": float(loss.detach().item()),
            "proj_a2v": float(loss_a2v.detach().item()),
            "proj_v2a": float(loss_v2a.detach().item()),
        }
        return loss, proj_a2v, proj_v2a, stats


@torch.no_grad()
def get_label_index(y):
    if y.ndim == 1:
        return y.long()
    return torch.argmax(y, dim=1).long()


@torch.no_grad()
def select_two_classes(loader, logger=None):
    counts = defaultdict(int)
    for spectrogram, image, y in loader:
        labels = get_label_index(y)
        for lab in labels.tolist():
            counts[int(lab)] += 1
    # choose the two most frequent classes for stable visualization
    selected = [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]]
    if len(selected) < 2:
        raise RuntimeError("Could not find two classes from the KS dataset for visualization.")
    if logger is not None:
        logger.info(f"Selected classes for visualization: {selected} with counts {[counts[c] for c in selected]}")
    return selected


@torch.no_grad()
def collect_epoch_projection_features(loader, model, projector, selected_classes, max_points_per_class=120):
    model.eval()
    projector.eval()

    audio_feats = []
    video_feats = []
    proj_a2v_feats = []
    proj_v2a_feats = []
    labels_kept = []
    per_class_count = defaultdict(int)

    selected_classes = set(int(x) for x in selected_classes)

    for spectrogram, image, y in loader:
        labels = get_label_index(y)
        keep_mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls in selected_classes:
            keep_mask |= (labels == cls)
        if keep_mask.sum() == 0:
            continue

        image = image.float().cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        spectrogram = spectrogram.unsqueeze(1).float().cuda(non_blocking=True)
        labels_gpu = get_label_index(y)

        _, _, _, f_a, f_v = model(spectrogram, image)
        proj_a2v = projector.project_audio_to_video(f_a, y)
        proj_v2a = projector.project_video_to_audio(f_v, y)

        for idx in range(labels_gpu.shape[0]):
            cls = int(labels_gpu[idx].item())
            if cls not in selected_classes:
                continue
            if per_class_count[cls] >= max_points_per_class:
                continue

            audio_feats.append(f_a[idx].detach().cpu().numpy())
            video_feats.append(f_v[idx].detach().cpu().numpy())
            proj_a2v_feats.append(proj_a2v[idx].detach().cpu().numpy())
            proj_v2a_feats.append(proj_v2a[idx].detach().cpu().numpy())
            labels_kept.append(cls)
            per_class_count[cls] += 1

        done = all(per_class_count[c] >= max_points_per_class for c in selected_classes)
        if done:
            break

    if len(labels_kept) == 0:
        return None

    return {
        "audio": np.asarray(audio_feats, dtype=np.float32),
        "video": np.asarray(video_feats, dtype=np.float32),
        "proj_a2v": np.asarray(proj_a2v_feats, dtype=np.float32),
        "proj_v2a": np.asarray(proj_v2a_feats, dtype=np.float32),
        "labels": np.asarray(labels_kept, dtype=np.int64),
    }

def lighten_color(color, factor=0.55):
    """
    把颜色往白色方向提亮。
    factor 越大越浅。
    """
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(rgb * (1 - factor) + white * factor)


def draw_pretty_panel(ax,
                      X_base,
                      labels_base,
                      class_ids,
                      title,
                      class_colors,
                      X_proj=None,
                      labels_proj=None,
                      proj_name=None):
    """
    美化后的单个子图：
    - 原始样本：半透明柔和色实心圆
    - 投影样本：更浅颜色、稍大点、深色边
    """
    ax.set_title(title, fontsize=15, weight='bold', pad=10)

    # 原始样本
    for cls in class_ids:
        mask = labels_base == cls
        if not np.any(mask):
            continue
        ax.scatter(
            X_base[mask, 0], X_base[mask, 1],
            s=34,
            c=[class_colors[cls]],
            alpha=0.48,
            edgecolors='white',
            linewidths=0.55,
            zorder=2,
        )

    # 投影样本
    if X_proj is not None and labels_proj is not None:
        for cls in class_ids:
            mask = labels_proj == cls
            if not np.any(mask):
                continue
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                s=46,
                c=[lighten_color(class_colors[cls], factor=0.35)],
                alpha=0.78,
                edgecolors=class_colors[cls],
                linewidths=0.9,
                zorder=3,
            )

        if proj_name is not None:
            ax.text(
                0.03, 0.97, proj_name,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=10.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#B8B8B8', lw=0.8, alpha=0.92),
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FBFBFB')

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#C8C8C8')

def lighten_color(color, factor=0.45):
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(rgb * (1 - factor) + white * factor)


def draw_target_only_panel(ax, X, labels, class_ids, title, class_colors):
    ax.set_title(title, fontsize=15, weight='bold', pad=10)

    for cls in class_ids:
        mask = labels == cls
        if not np.any(mask):
            continue
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=42,
            c=[class_colors[cls]],
            alpha=0.42,
            edgecolors='white',
            linewidths=0.55,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FBFBFB')
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#CFCFCF')


def draw_before_after_panel(ax,
                            X_target,
                            labels_target,
                            X_before,
                            labels_before,
                            X_after,
                            labels_after,
                            class_ids,
                            title,
                            class_colors,
                            before_name,
                            after_name):
    ax.set_title(title, fontsize=15, weight='bold', pad=10)

    # 目标模态簇
    for cls in class_ids:
        mask = labels_target == cls
        if not np.any(mask):
            continue
        ax.scatter(
            X_target[mask, 0], X_target[mask, 1],
            s=46,
            c=[class_colors[cls]],
            alpha=0.46,
            edgecolors='white',
            linewidths=0.55,
            zorder=2,
        )

    # before：未投影源模态
    for cls in class_ids:
        mask = labels_before == cls
        if not np.any(mask):
            continue
        ax.scatter(
            X_before[mask, 0], X_before[mask, 1],
            s=24,
            c=[lighten_color(class_colors[cls], factor=0.62)],
            alpha=0.28,
            edgecolors='none',
            zorder=1,
        )

    # after：投影后的样本
    for cls in class_ids:
        mask = labels_after == cls
        if not np.any(mask):
            continue
        ax.scatter(
            X_after[mask, 0], X_after[mask, 1],
            s=50,
            c=[lighten_color(class_colors[cls], factor=0.20)],
            alpha=0.85,
            edgecolors=class_colors[cls],
            linewidths=0.95,
            zorder=3,
        )

    ax.text(
        0.03, 0.97,
        f'Before: {before_name}\nAfter: {after_name}',
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=10.2, weight='bold',
        bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#BDBDBD', lw=0.8, alpha=0.95),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FBFBFB')
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#CFCFCF')

@torch.no_grad()
def save_epoch_projection_figure(epoch, feat_dict, class_ids, save_dir, logger=None, perplexity=30, seed=42):
    os.makedirs(save_dir, exist_ok=True)

    labels = feat_dict["labels"]
    audio = feat_dict["audio"]
    video = feat_dict["video"]
    proj_a2v = feat_dict["proj_a2v"]
    proj_v2a = feat_dict["proj_v2a"]

    # -------- Audio target space --------
    # joint t-SNE on [audio, raw video(before), projected video->audio(after)]
    audio_joint = np.concatenate([audio, video, proj_v2a], axis=0)
    tsne_audio = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(audio_joint) - 1)),
        learning_rate='auto',
        init='pca',
        random_state=seed,
    )
    audio_joint_2d = tsne_audio.fit_transform(audio_joint)
    n = len(audio)
    audio_2d = audio_joint_2d[:n]
    video_in_audio_2d = audio_joint_2d[n:2*n]
    proj_v2a_2d = audio_joint_2d[2*n:]

    # -------- Video target space --------
    # joint t-SNE on [video, raw audio(before), projected audio->video(after)]
    video_joint = np.concatenate([video, audio, proj_a2v], axis=0)
    tsne_video = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(video_joint) - 1)),
        learning_rate='auto',
        init='pca',
        random_state=seed,
    )
    video_joint_2d = tsne_video.fit_transform(video_joint)
    video_2d = video_joint_2d[:n]
    audio_in_video_2d = video_joint_2d[n:2*n]
    proj_a2v_2d = video_joint_2d[2*n:]

    # 柔和配色，接近你给的参考图风格
    palette = [
        '#F28E8C',  # 柔和红
        '#97D5C9',  # 柔和青绿
        '#B6A6E3',  # 柔和紫
        '#C9E49C',  # 柔和浅绿
        '#9ED0F6',  # 柔和蓝
        '#F3C97A',  # 柔和黄
    ]
    class_colors = {
        cls: palette[i % len(palette)] for i, cls in enumerate(class_ids)
    }

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.9))

    # 1) 原始 audio
    draw_target_only_panel(
        axes[0],
        X=audio_2d,
        labels=labels,
        class_ids=class_ids,
        title='Audio Feature Space',
        class_colors=class_colors,
    )

    # 2) 原始 video
    draw_target_only_panel(
        axes[1],
        X=video_2d,
        labels=labels,
        class_ids=class_ids,
        title='Video Feature Space',
        class_colors=class_colors,
    )

    # 3) Audio target space: before/after
    draw_before_after_panel(
        axes[2],
        X_target=audio_2d,
        labels_target=labels,
        X_before=video_in_audio_2d,
        labels_before=labels,
        X_after=proj_v2a_2d,
        labels_after=labels,
        class_ids=class_ids,
        title='Audio Space: Before / After',
        class_colors=class_colors,
        before_name='raw Video',
        after_name='Video→Audio',
    )

    # 4) Video target space: before/after
    draw_before_after_panel(
        axes[3],
        X_target=video_2d,
        labels_target=labels,
        X_before=audio_in_video_2d,
        labels_before=labels,
        X_after=proj_a2v_2d,
        labels_after=labels,
        class_ids=class_ids,
        title='Video Space: Before / After',
        class_colors=class_colors,
        before_name='raw Audio',
        after_name='Audio→Video',
    )

    # 图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=class_colors[class_ids[0]],
                   markeredgecolor='white', markeredgewidth=0.8,
                   markersize=8, alpha=0.9, label=f'Class {class_ids[0]}'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=class_colors[class_ids[1]],
                   markeredgecolor='white', markeredgewidth=0.8,
                   markersize=8, alpha=0.9, label=f'Class {class_ids[1]}'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#E6E6E6',
                   markeredgecolor='none',
                   markersize=7, alpha=0.5, label='Before'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#F4D4D3',
                   markeredgecolor='#A55A5A',
                   markeredgewidth=1.0,
                   markersize=8, alpha=0.95, label='After (projected)'),
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.03)
    )

    fig.suptitle(f'Epoch {epoch}', fontsize=17, weight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    png_path = os.path.join(save_dir, f'epoch_{epoch:03d}_projection_before_after.png')
    pdf_path = os.path.join(save_dir, f'epoch_{epoch:03d}_projection_before_after.pdf')
    plt.savefig(png_path, dpi=320, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=320, bbox_inches='tight')
    plt.close(fig)

    if logger is not None:
        logger.info(f'Saved projection visualization for epoch {epoch}: {png_path}')
        logger.info(f'Saved projection visualization for epoch {epoch}: {pdf_path}')


def train_audio_video(epoch, train_loader, model, projector, optimizer, logger, logits_ratio, proj_weight):
    model.train()
    projector.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    tl_f = Averager()
    tl_proj = Averager()
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

        loss_proj, _, _, _ = projector(f_a, f_v, y)

        loss = loss_a + loss_v + loss_fusion + proj_weight * loss_proj
        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_a.add(loss_a.item())
        tl_v.add(loss_v.item())
        tl_f.add(loss_fusion.item())
        tl_proj.add(loss_proj.item())

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        (
            'Epoch {epoch:d}: '
            'Average Training Loss:{loss_ave:.4f}, '
            'loss_audio:{loss_audio:.4f}, '
            'loss_video:{loss_video:.4f}, '
            'loss_fusion:{loss_fusion:.4f}, '
            'loss_proj:{loss_proj:.4f}'
        ).format(
            epoch=epoch,
            loss_ave=tl.item(),
            loss_audio=tl_a.item(),
            loss_video=tl_v.item(),
            loss_fusion=tl_f.item(),
            loss_proj=tl_proj.item(),
        )
    )

    return model, projector


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
            label_idx = torch.argmax(y, dim=1)
            label_list.extend(label_idx.tolist())
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
        (
            'Epoch {epoch:d}: '
            'f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},'
            'f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},'
            'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}'
        ).format(
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
    parser.add_argument('--proj_weight', type=float, default=0.2)
    parser.add_argument('--proj_momentum', type=float, default=0.9)
    parser.add_argument('--proj_eps', type=float, default=1e-5)
    parser.add_argument('--proj_loss', type=str, default='smooth_l1', choices=['smooth_l1', 'mse'])
    parser.add_argument('--proj_cosine_weight', type=float, default=0.1)
    parser.add_argument('--vis_every', type=int, default=1)
    parser.add_argument('--vis_max_points_per_class', type=int, default=100)
    parser.add_argument('--vis_perplexity', type=int, default=30)
    parser.add_argument('--vis_root', type=str, default='/data/zyh/NeurIPS24-LFM/_tsne/projection_BeforeAfter')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = deep_update_dict(json.load(f), config)

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

    selected_classes = select_two_classes(test_loader, logger)
    time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_save_dir = os.path.join(args.vis_root, time_tag)
    os.makedirs(vis_save_dir, exist_ok=True)
    logger.info(f'Projection visualization directory: {vis_save_dir}')

    model = AVClassifier(config=cfg).cuda()
    model.apply(weight_init)

    projector = ClassWiseStatProjector(
        num_classes=cfg['setting']['num_class'],
        feat_dim=model.hidden_dim,
        momentum=args.proj_momentum,
        eps=args.proj_eps,
        loss_type=args.proj_loss,
        cosine_weight=args.proj_cosine_weight,
    ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        momentum=cfg['train']['optimizer']['momentum'],
        weight_decay=cfg['train']['optimizer']['wc'],
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg['train']['lr_scheduler']['patience'], 0.1)

    best_acc = 0.0
    save_dir = '/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks'
    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f'Projection alignment enabled: proj_weight={args.proj_weight}, '
        f'momentum={args.proj_momentum}, eps={args.proj_eps}, '
        f'loss={args.proj_loss}, cosine_weight={args.proj_cosine_weight}'
    )

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        scheduler.step()

        model, projector = train_audio_video(
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            projector=projector,
            optimizer=optimizer,
            logger=logger,
            logits_ratio=logits_ratio,
            proj_weight=args.proj_weight,
        )

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        if epoch % args.vis_every == 0:
            feat_dict = collect_epoch_projection_features(
                loader=test_loader,
                model=model,
                projector=projector,
                selected_classes=selected_classes,
                max_points_per_class=args.vis_max_points_per_class,
            )
            if feat_dict is not None:
                save_epoch_projection_figure(
                    epoch=epoch,
                    feat_dict=feat_dict,
                    class_ids=selected_classes,
                    save_dir=vis_save_dir,
                    logger=logger,
                    perplexity=args.vis_perplexity,
                    seed=cfg['seed'],
                )

        if acc > best_acc:
            best_acc = acc
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'best_acc': best_acc,
                'cfg': cfg,
                'proj_args': vars(args),
            }
            torch.save(ckpt, os.path.join(save_dir, 'multi_KS_best_model_with_projection.pth'))
            logger.info(f'New best checkpoint saved at epoch {epoch}, acc={acc:.4f}')

        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'acc': acc,
                'acc_a': acc_a,
                'acc_v': acc_v,
                'cfg': cfg,
                'proj_args': vars(args),
            }
            torch.save(
                ckpt,
                os.path.join(save_dir, f'multi_KS_proj_epoch_{epoch}_acc_{acc:.4f}_{acc_a:.4f}_{acc_v:.4f}.pth')
            )
