#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from utils.min_norm_solvers import MinNormSolver
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from dataset.VGGSoundDataset import VGGSound,SemiVGGSound
import random
import re
from collections import defaultdict
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)

from utils.tools import GSPlugin, weight_init

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def flatten_epoch_metrics(epoch, train_metrics, val_metrics):
    """Merge train/validation metrics into a flat dict for CSV logging."""
    row = {'epoch': int(epoch)}
    for key, value in train_metrics.items():
        row[f'train_{key}'] = value
    for key, value in val_metrics.items():
        row[f'val_{key}'] = value
    return row


def init_epoch_info_file(info_dir, run_name, args, cfg):
    ensure_dir(info_dir)
    info_path = os.path.join(info_dir, f'{run_name}.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write('# KS ABA training information. One epoch per row.\n')
        f.write(f'# run_name={run_name}\n')
        f.write(f'# config={args.config}\n')
        f.write(f'# tensorboard_root={args.tensorboard_root}\n')
        f.write(f'# seed={cfg.get("seed", "NA")}\n')
        f.write(f'# gpu_id={cfg.get("gpu_id", "NA")}\n')
        f.write('# columns will be written after the first epoch.\n')
    return info_path


def append_epoch_info(info_path, row):
    """Append one CSV-style line per epoch for later plotting."""
    file_exists = os.path.exists(info_path)
    with open(info_path, 'r', encoding='utf-8') as f:
        content = f.read()
    has_header = 'epoch,' in content

    keys = list(row.keys())
    with open(info_path, 'a', encoding='utf-8') as f:
        if not has_header:
            f.write(','.join(keys) + '\n')
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                values.append(f'{value:.8f}')
            else:
                values.append(str(value))
        f.write(','.join(values) + '\n')


def write_tensorboard_scalars(writer, epoch, train_metrics, val_metrics):
    """Write all losses, accuracies, mAP, F1, and ABA statistics to TensorBoard."""
    for key, value in train_metrics.items():
        tag_prefix = 'ABA' if key in [
            'mass_a2v', 'mass_v2a', 'lambda_a2v', 'lambda_v2a',
            'reliability_audio', 'reliability_video', 'compat_a2v', 'compat_v2a',
            'valid_classes_per_batch', 'beta'
        ] else 'Train'
        writer.add_scalar(f'{tag_prefix}/{key}', value, epoch)

    for key, value in val_metrics.items():
        writer.add_scalar(f'Val/{key}', value, epoch)

class AdaptiveBidirectionalAligner(nn.Module):
    """
    Practical implementation of the paper idea for audio-video training:
      1) class-conditional statistical projection with diagonal Gaussian statistics;
      2) bidirectional feature-component partial OT alignment;
      3) adaptive directional weighting from unimodal reliability;
      4) transferability-aware transported mass based on source reliability,
         target deficiency, and projection compatibility.

    Notation:
      a -> v: project audio features into video space and align with video features.
      v -> a: project video features into audio space and align with audio features.
    """
    def __init__(self,
                 num_classes,
                 feat_dim,
                 momentum=0.9,
                 eps=1e-5,
                 rho=0.5,
                 tau_m=0.5,
                 tau_lambda=0.5,
                 tau_c=1.0,
                 ot_temp=0.07,
                 ot_iters=8,
                 min_samples_per_class=2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.rho = float(rho)
        # tau_m is kept for backward compatibility with old scripts.
        # The new transported mass is controlled by tau_c through projection compatibility.
        self.tau_m = float(tau_m)
        self.tau_lambda = float(tau_lambda)
        self.tau_c = float(tau_c)
        self.ot_temp = float(ot_temp)
        self.ot_iters = int(ot_iters)
        self.min_samples_per_class = int(min_samples_per_class)

        self.register_buffer('audio_mean', torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer('video_mean', torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer('audio_var', torch.ones(self.num_classes, self.feat_dim))
        self.register_buffer('video_var', torch.ones(self.num_classes, self.feat_dim))
        self.register_buffer('initialized', torch.zeros(self.num_classes, dtype=torch.bool))

    @staticmethod
    def get_label_index(y):
        if y.ndim == 1:
            return y.long()
        return torch.argmax(y, dim=1).long()

    @torch.no_grad()
    def update_stats(self, feat_a, feat_v, y):
        labels = self.get_label_index(y)
        for cls in labels.unique(sorted=False).tolist():
            cls = int(cls)
            mask = labels == cls
            if mask.sum() == 0:
                continue

            cur_a = feat_a[mask]
            cur_v = feat_v[mask]
            mean_a = cur_a.mean(dim=0)
            mean_v = cur_v.mean(dim=0)
            var_a = torch.clamp(cur_a.var(dim=0, unbiased=False), min=self.eps)
            var_v = torch.clamp(cur_v.var(dim=0, unbiased=False), min=self.eps)

            if not bool(self.initialized[cls]):
                self.audio_mean[cls].copy_(mean_a)
                self.video_mean[cls].copy_(mean_v)
                self.audio_var[cls].copy_(var_a)
                self.video_var[cls].copy_(var_v)
                self.initialized[cls] = True
            else:
                m = self.momentum
                self.audio_mean[cls].mul_(m).add_(mean_a, alpha=1.0 - m)
                self.video_mean[cls].mul_(m).add_(mean_v, alpha=1.0 - m)
                self.audio_var[cls].mul_(m).add_(var_a, alpha=1.0 - m)
                self.video_var[cls].mul_(m).add_(var_v, alpha=1.0 - m)

    def project_audio_to_video(self, feat_a, labels):
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_a - mu_a) / std_a * std_v + mu_v

    def project_video_to_audio(self, feat_v, labels):
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_v - mu_v) / std_v * std_a + mu_a

    def _component_cost(self, source_components, target_components):
        """
        source_components / target_components: [n_k, D]
        Return feature-component cost matrix C in [D, D].
        Each feature dimension is treated as one component over samples of the same class.
        """
        src = source_components - source_components.mean(dim=0, keepdim=True)
        tgt = target_components - target_components.mean(dim=0, keepdim=True)
        src_comp = F.normalize(src.transpose(0, 1), dim=1, eps=self.eps)  # [D, n_k]
        tgt_comp = F.normalize(tgt.transpose(0, 1), dim=1, eps=self.eps)  # [D, n_k]
        cost = 1.0 - torch.matmul(src_comp, tgt_comp.transpose(0, 1))
        return torch.clamp(cost, min=0.0, max=2.0)

    def _approx_partial_ot_loss(self, cost, mass):
        """
        Lightweight entropy-regularized partial transport.
        The plan is softly balanced with row/column upper bounds 1/D and total mass m.
        This is efficient and stable for feature-level alignment in minibatch training.
        """
        D = cost.shape[0]
        m = torch.clamp(mass, min=self.eps, max=1.0)
        row_cap = 1.0 / float(D)
        col_cap = 1.0 / float(D)

        plan = torch.exp(-cost / max(self.ot_temp, self.eps)) + self.eps
        plan = plan / (plan.sum() + self.eps) * m

        for _ in range(self.ot_iters):
            row_sum = plan.sum(dim=1, keepdim=True)
            plan = plan * torch.clamp(row_cap / (row_sum + self.eps), max=1.0)
            col_sum = plan.sum(dim=0, keepdim=True)
            plan = plan * torch.clamp(col_cap / (col_sum + self.eps), max=1.0)
            total = plan.sum()
            if total > self.eps:
                plan = plan / (total + self.eps) * m

        return (plan * cost).sum() / (m + self.eps)

    def forward(self, feat_a, feat_v, logits_a, logits_v, y):
        labels = self.get_label_index(y)
        self.update_stats(feat_a.detach(), feat_v.detach(), labels)

        prob_a = F.softmax(logits_a.detach(), dim=1)
        prob_v = F.softmax(logits_v.detach(), dim=1)

        class_losses = []
        mass_a2v_vals, mass_v2a_vals = [], []
        lambda_a2v_vals, lambda_v2a_vals = [], []
        reliability_a_vals, reliability_v_vals = [], []
        compat_a2v_vals, compat_v2a_vals = [], []

        for cls in labels.unique(sorted=False).tolist():
            cls = int(cls)
            mask = labels == cls
            if mask.sum().item() < self.min_samples_per_class:
                continue
            if not bool(self.initialized[cls]):
                continue

            fa = feat_a[mask]
            fv = feat_v[mask]
            cls_labels = labels[mask]

            proj_a2v = self.project_audio_to_video(fa, cls_labels)
            proj_v2a = self.project_video_to_audio(fv, cls_labels)

            # Class-wise modality reliability, following the paper's learning-status signal.
            s_a = prob_a[mask, cls].mean()
            s_v = prob_v[mask, cls].mean()

            lambda_logits = torch.stack([s_a / max(self.tau_lambda, self.eps),
                                         s_v / max(self.tau_lambda, self.eps)])
            lambda_weights = F.softmax(lambda_logits, dim=0)
            lambda_a2v = lambda_weights[0]
            lambda_v2a = lambda_weights[1]

            cost_a2v = self._component_cost(proj_a2v, fv)
            cost_v2a = self._component_cost(proj_v2a, fa)

            # Transferability-aware transported mass:
            #   m_{j->k} = rho * source_reliability * target_deficiency * projection_compatibility.
            # The compatibility is detached for stability, so it controls the transport budget
            # without encouraging the model to reduce cost only by manipulating the mass term.
            compat_a2v = torch.exp(-cost_a2v.detach().mean() / max(self.tau_c, self.eps))
            compat_v2a = torch.exp(-cost_v2a.detach().mean() / max(self.tau_c, self.eps))
            m_a2v = self.rho * s_a * (1.0 - s_v) * compat_a2v
            m_v2a = self.rho * s_v * (1.0 - s_a) * compat_v2a
            m_a2v = torch.clamp(m_a2v, min=self.eps, max=1.0)
            m_v2a = torch.clamp(m_v2a, min=self.eps, max=1.0)

            loss_a2v = self._approx_partial_ot_loss(cost_a2v, m_a2v)
            loss_v2a = self._approx_partial_ot_loss(cost_v2a, m_v2a)
            class_loss = lambda_a2v * loss_a2v + lambda_v2a * loss_v2a
            class_losses.append(class_loss)

            mass_a2v_vals.append(float(m_a2v.detach().cpu()))
            mass_v2a_vals.append(float(m_v2a.detach().cpu()))
            lambda_a2v_vals.append(float(lambda_a2v.detach().cpu()))
            lambda_v2a_vals.append(float(lambda_v2a.detach().cpu()))
            reliability_a_vals.append(float(s_a.detach().cpu()))
            reliability_v_vals.append(float(s_v.detach().cpu()))
            compat_a2v_vals.append(float(compat_a2v.detach().cpu()))
            compat_v2a_vals.append(float(compat_v2a.detach().cpu()))

        if len(class_losses) == 0:
            zero = (feat_a.sum() + feat_v.sum()) * 0.0
            stats = {
                'align_loss': 0.0,
                'mass_a2v': 0.0,
                'mass_v2a': 0.0,
                'lambda_a2v': 0.0,
                'lambda_v2a': 0.0,
                'reliability_audio': 0.0,
                'reliability_video': 0.0,
                'compat_a2v': 0.0,
                'compat_v2a': 0.0,
                'valid_classes': 0,
            }
            return zero, stats

        align_loss = torch.stack(class_losses).mean()
        stats = {
            'align_loss': float(align_loss.detach().cpu()),
            'mass_a2v': float(np.mean(mass_a2v_vals)),
            'mass_v2a': float(np.mean(mass_v2a_vals)),
            'lambda_a2v': float(np.mean(lambda_a2v_vals)),
            'lambda_v2a': float(np.mean(lambda_v2a_vals)),
            'reliability_audio': float(np.mean(reliability_a_vals)),
            'reliability_video': float(np.mean(reliability_v_vals)),
            'compat_a2v': float(np.mean(compat_a2v_vals)),
            'compat_v2a': float(np.mean(compat_v2a_vals)),
            'valid_classes': len(class_losses),
        }
        return align_loss, stats


def train_audio_video(epoch, train_loader, model, aligner, optimizer, logger, cls_k, logits_ratio, aba_beta=0.1, aba_warmup=0):
    model.train()
    aligner.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    tl_f = Averager()
    tl_align = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    if aba_warmup is not None and aba_warmup > 0:
        beta_scale = min(1.0, float(epoch + 1) / float(aba_warmup))
    else:
        beta_scale = 1.0
    cur_aba_beta = aba_beta * beta_scale

    stat_mass_a2v = Averager()
    stat_mass_v2a = Averager()
    stat_lambda_a2v = Averager()
    stat_lambda_v2a = Averager()
    stat_reliability_audio = Averager()
    stat_reliability_video = Averager()
    stat_compat_a2v = Averager()
    stat_compat_v2a = Averager()
    stat_valid_cls = Averager()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

        loss_a = criterion(result_a, y).mean()
        loss_v = criterion(result_v, y).mean()
        loss_fusion = criterion(logits_ratio * result_a + logits_ratio * result_v, y).mean()

        loss_align, align_stats = aligner(
            feat_a=f_a,
            feat_v=f_v,
            logits_a=result_a,
            logits_v=result_v,
            y=y,
        )

        loss = loss_a + loss_v + loss_fusion + cur_aba_beta * loss_align

        loss.backward()
        optimizer.step()
        tl.add(loss.item())
        tl_a.add(loss_a.item())
        tl_v.add(loss_v.item())
        tl_f.add(loss_fusion.item())
        tl_align.add(loss_align.item() if torch.is_tensor(loss_align) else float(loss_align))

        stat_mass_a2v.add(align_stats['mass_a2v'])
        stat_mass_v2a.add(align_stats['mass_v2a'])
        stat_lambda_a2v.add(align_stats['lambda_a2v'])
        stat_lambda_v2a.add(align_stats['lambda_v2a'])
        stat_reliability_audio.add(align_stats['reliability_audio'])
        stat_reliability_video.add(align_stats['reliability_video'])
        stat_compat_a2v.add(align_stats['compat_a2v'])
        stat_compat_v2a.add(align_stats['compat_v2a'])
        stat_valid_cls.add(align_stats['valid_classes'])

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        (
            'Epoch {epoch:d}: Average Training Loss:{loss_ave:.4f}, '
            'loss_audio:{loss_audio:.4f}, loss_video:{loss_video:.4f}, '
            'loss_fusion:{loss_fusion:.4f}, loss_align:{loss_align:.4f}, beta:{beta:.4f}'
        ).format(
            epoch=epoch,
            loss_ave=tl.item(),
            loss_audio=tl_a.item(),
            loss_video=tl_v.item(),
            loss_fusion=tl_f.item(),
            loss_align=tl_align.item(),
            beta=cur_aba_beta,
        )
    )
    logger.info(
        (
            'ABA stats: mass_a2v:{mass_a2v:.4f}, mass_v2a:{mass_v2a:.4f}, '
            'lambda_a2v:{lambda_a2v:.4f}, lambda_v2a:{lambda_v2a:.4f}, '
            'reliability_audio:{rel_a:.4f}, reliability_video:{rel_v:.4f}, '
            'compat_a2v:{compat_a2v:.4f}, compat_v2a:{compat_v2a:.4f}, '
            'valid_classes_per_batch:{valid_cls:.2f}'
        ).format(
            mass_a2v=stat_mass_a2v.item(),
            mass_v2a=stat_mass_v2a.item(),
            lambda_a2v=stat_lambda_a2v.item(),
            lambda_v2a=stat_lambda_v2a.item(),
            rel_a=stat_reliability_audio.item(),
            rel_v=stat_reliability_video.item(),
            compat_a2v=stat_compat_a2v.item(),
            compat_v2a=stat_compat_v2a.item(),
            valid_cls=stat_valid_cls.item(),
        )
    )

    train_metrics = {
        'loss_total': float(tl.item()),
        'loss_audio': float(tl_a.item()),
        'loss_video': float(tl_v.item()),
        'loss_fusion': float(tl_f.item()),
        'loss_align': float(tl_align.item()),
        'beta': float(cur_aba_beta),
        'mass_a2v': float(stat_mass_a2v.item()),
        'mass_v2a': float(stat_mass_v2a.item()),
        'lambda_a2v': float(stat_lambda_a2v.item()),
        'lambda_v2a': float(stat_lambda_v2a.item()),
        'reliability_audio': float(stat_reliability_audio.item()),
        'reliability_video': float(stat_reliability_video.item()),
        'compat_a2v': float(stat_compat_a2v.item()),
        'compat_v2a': float(stat_compat_v2a.item()),
        'valid_classes_per_batch': float(stat_valid_cls.item()),
    }

    return model, aligner, train_metrics


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
    score_a = 0.0
    score_v = 0.0
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            
            result_b, result_a, result_v, f_a , f_v   = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(result_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(result_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(0.5 * result_a +  0.5 * result_v, dim=1)).tolist()
            pred = (F.softmax(0.5 * result_a +  0.5 * result_v, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(result_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(result_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

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
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                              f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    val_metrics = {
        'f1_multi': float(f1),
        'acc_multi': float(acc),
        'mAP_multi': float(mAP),
        'f1_audio': float(f1_a),
        'acc_audio': float(acc_a),
        'mAP_audio': float(mAP_a),
        'f1_video': float(f1_v),
        'acc_video': float(acc_v),
        'mAP_video': float(mAP_v),
    }
    return val_metrics
    

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--aba_beta', type=float, default=0.1, help='Weight beta for adaptive bidirectional alignment loss.')
    parser.add_argument('--aba_warmup', type=int, default=0, help='Warmup epochs for gradually enabling ABA loss. 0 means no warmup.')
    parser.add_argument('--aba_rho', type=float, default=0.5, help='Total transported mass budget for partial OT.')
    parser.add_argument('--aba_tau_m', type=float, default=0.5, help='Temperature for adaptive transported mass.')
    parser.add_argument('--aba_tau_lambda', type=float, default=0.5, help='Temperature for adaptive directional weighting.')
    parser.add_argument('--aba_tau_c', type=float, default=1.0, help='Temperature for projection compatibility in transferability-aware transported mass.')
    parser.add_argument('--aba_momentum', type=float, default=0.9, help='Momentum for class-wise statistical feature estimation.')
    parser.add_argument('--aba_eps', type=float, default=1e-5, help='Numerical stability epsilon.')
    parser.add_argument('--aba_ot_temp', type=float, default=0.07, help='Entropy temperature for approximate partial OT.')
    parser.add_argument('--aba_ot_iters', type=int, default=8, help='Number of balancing iterations for approximate partial OT.')
    parser.add_argument('--aba_min_samples_per_class', type=int, default=2, help='Minimum samples per class in a batch to compute feature-level OT.')
    parser.add_argument('--tensorboard_root', type=str, default='/data/zyh/NeurIPS24-LFM/_tensorboard_runs', help='Root directory for TensorBoard runs.')
    parser.add_argument('--epoch_info_dir', type=str, default='/data/zyh/NeurIPS24-LFM/_logs/ks_information', help='Directory for per-epoch text logs.')

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

    run_name = f"ks_ABA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{cfg['seed']}"
    tensorboard_dir = os.path.join(args.tensorboard_root, run_name)
    ensure_dir(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    info_path = init_epoch_info_file(args.epoch_info_dir, run_name, args, cfg)
    logger.info(f'TensorBoard directory: {tensorboard_dir}')
    logger.info(f'Epoch information file: {info_path}')

    # ----- SET DATALOADER -----
    train_dataset = VADataset(cfg, mode='train')
    test_dataset = VADataset(cfg, mode='test')


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    val_batch = next(iter(train_loader))


    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    feat_dim = getattr(model, 'hidden_dim', cfg.get('visual', {}).get('hidden_dim', 512))
    aligner = AdaptiveBidirectionalAligner(
        num_classes=cfg['setting']['num_class'],
        feat_dim=feat_dim,
        momentum=args.aba_momentum,
        eps=args.aba_eps,
        rho=args.aba_rho,
        tau_m=args.aba_tau_m,
        tau_lambda=args.aba_tau_lambda,
        tau_c=args.aba_tau_c,
        ot_temp=args.aba_ot_temp,
        ot_iters=args.aba_ot_iters,
        min_samples_per_class=args.aba_min_samples_per_class,
    ).cuda()
    logger.info(
        f'ABA enabled: beta={args.aba_beta}, warmup={args.aba_warmup}, rho={args.aba_rho}, '
        f'tau_m={args.aba_tau_m}, tau_lambda={args.aba_tau_lambda}, tau_c={args.aba_tau_c}, '
        f'ot_temp={args.aba_ot_temp}, ot_iters={args.aba_ot_iters}, feat_dim={feat_dim}'
    )

    lr_adjust = cfg['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=cfg['train']['optimizer']['momentum'],
                          weight_decay=cfg['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0
    cls_k = []
    
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model, aligner, train_metrics = train_audio_video(
            epoch,
            train_loader,
            model,
            aligner,
            optimizer,
            logger,
            cls_k,
            logits_ratio,
            aba_beta=args.aba_beta,
            aba_warmup=args.aba_warmup,
        )

        val_metrics = val(epoch, test_loader, model, logger)
        acc = val_metrics['acc_multi']
        acc_a = val_metrics['acc_audio']
        acc_v = val_metrics['acc_video']

        write_tensorboard_scalars(writer, epoch, train_metrics, val_metrics)
        epoch_row = flatten_epoch_metrics(epoch, train_metrics, val_metrics)
        append_epoch_info(info_path, epoch_row)
        writer.flush()
        logger.info(f'Epoch {epoch:d} information appended to: {info_path}')
        # if acc > best_acc:
            # best_acc = acc
            # print('Find a better model and save it!')
            # logger.info('Find a better model and save it!')
        m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
        
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth')
        

        ### TODO:before
        # if acc > best_acc:
        #     best_acc = acc
        #     print('Find a better model and save it!')
        #     logger.info('Find a better model and save it!')
        #     m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
        #     torch.save(model.state_dict(), '/data/lxe/multimodel/NeurIPS24-LFM-main/KS_model/multi_KS_best_model.pth')
    writer.close()
    logger.info('TensorBoard writer closed.')
