#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)

from utils.tools import weight_init

PROJECT_ROOT = Path(__file__).resolve().parent

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
    
    for key, value in train_metrics.items():
        tag_prefix = 'ABA' if key in ['mass_a2v', 'mass_v2a', 'lambda_a2v', 'lambda_v2a', 'compat_a2v', 'compat_v2a', 'valid_classes_per_batch', 'beta'] else 'Train'
        writer.add_scalar(f'{tag_prefix}/{key}', value, epoch)

    for key, value in val_metrics.items():
        writer.add_scalar(f'Val/{key}', value, epoch)

class AdaptiveBidirectionalAligner(nn.Module):

    def __init__(self,
                 num_classes,
                 feat_dim,
                 momentum=0.9,
                 eps=1e-5,
                 eps_m=1e-6,
                 rho=0.75,
                 tau_c=0.4,
                 tau_lambda=0.45,
                 cov_shrinkage=0.1,
                 ot_eta=0.07,
                 ot_iters=50,
                 min_samples_per_class=2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.eps_m = float(eps_m)
        self.rho = float(rho)
        self.tau_c = float(tau_c)
        self.tau_lambda = float(tau_lambda)
        self.cov_shrinkage = float(cov_shrinkage)
        self.ot_eta = float(ot_eta)
        self.ot_iters = int(ot_iters)
        self.min_samples_per_class = int(min_samples_per_class)

        eye = torch.eye(self.feat_dim)
        self.register_buffer('audio_mean', torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer('video_mean', torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer('audio_cov', eye.unsqueeze(0).repeat(self.num_classes, 1, 1))
        self.register_buffer('video_cov', eye.unsqueeze(0).repeat(self.num_classes, 1, 1))
        self.register_buffer('initialized', torch.zeros(self.num_classes, dtype=torch.bool))

    @staticmethod
    def get_label_index(y):
        if y.ndim == 1:
            return y.long()
        return torch.argmax(y, dim=1).long()

    def _symmetrize(self, mat):
        return 0.5 * (mat + mat.transpose(-1, -2))

    def _matrix_sqrt(self, mat):
        
        mat = self._symmetrize(mat)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = torch.clamp(eigvals, min=self.eps)
        return (eigvecs * torch.sqrt(eigvals).unsqueeze(-2)) @ eigvecs.transpose(-1, -2)

    def _matrix_invsqrt(self, mat):
        
        mat = self._symmetrize(mat)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = torch.clamp(eigvals, min=self.eps)
        return (eigvecs * torch.rsqrt(eigvals).unsqueeze(-2)) @ eigvecs.transpose(-1, -2)

    def _shrink_cov(self, cov):
        
        cov = self._symmetrize(cov)
        D = cov.shape[-1]
        eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
        trace = torch.trace(cov)
        return ((1.0 - self.cov_shrinkage) * cov
                + self.cov_shrinkage * trace / float(D) * eye
                + self.eps * eye)

    @torch.no_grad()
    def update_stats(self, feat_a, feat_v, y):
        """Eq. (6)-(9): update class-wise Gaussian statistics with EMA."""
        labels = self.get_label_index(y)
        D = feat_a.shape[1]
        eye = torch.eye(D, device=feat_a.device, dtype=feat_a.dtype)

        for cls in labels.unique(sorted=False).tolist():
            cls = int(cls)
            mask = labels == cls
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue

            cur_a = feat_a[mask]
            cur_v = feat_v[mask]
            mean_a = cur_a.mean(dim=0)
            mean_v = cur_v.mean(dim=0)

            if not bool(self.initialized[cls]):
                self.audio_mean[cls].copy_(mean_a)
                self.video_mean[cls].copy_(mean_v)
                # If n_k < 2, keep covariance as identity, as in the paper.
                if n_k >= 2:
                    ca = cur_a - mean_a.unsqueeze(0)
                    cv = cur_v - mean_v.unsqueeze(0)
                    cov_a = (ca.transpose(0, 1) @ ca) / float(n_k)
                    cov_v = (cv.transpose(0, 1) @ cv) / float(n_k)
                    self.audio_cov[cls].copy_(self._symmetrize(cov_a) + self.eps * eye)
                    self.video_cov[cls].copy_(self._symmetrize(cov_v) + self.eps * eye)
                self.initialized[cls] = True
            else:
                m = self.momentum
                self.audio_mean[cls].mul_(m).add_(mean_a, alpha=1.0 - m)
                self.video_mean[cls].mul_(m).add_(mean_v, alpha=1.0 - m)

                # For single-sample classes, update mean only and keep covariance unchanged.
                if n_k >= 2:
                    ca = cur_a - mean_a.unsqueeze(0)
                    cv = cur_v - mean_v.unsqueeze(0)
                    cov_a = (ca.transpose(0, 1) @ ca) / float(n_k)
                    cov_v = (cv.transpose(0, 1) @ cv) / float(n_k)
                    self.audio_cov[cls].mul_(m).add_(self._symmetrize(cov_a) + self.eps * eye, alpha=1.0 - m)
                    self.video_cov[cls].mul_(m).add_(self._symmetrize(cov_v) + self.eps * eye, alpha=1.0 - m)

    def _gaussian_ot_project(self, z, labels, source='audio_to_video'):
        
        outs = []
        for cls in labels.unique(sorted=False).tolist():
            cls = int(cls)
            mask = labels == cls
            z_c = z[mask]

            if source == 'audio_to_video':
                mu_src = self.audio_mean[cls]
                mu_tgt = self.video_mean[cls]
                cov_src = self._shrink_cov(self.audio_cov[cls])
                cov_tgt = self._shrink_cov(self.video_cov[cls])
            elif source == 'video_to_audio':
                mu_src = self.video_mean[cls]
                mu_tgt = self.audio_mean[cls]
                cov_src = self._shrink_cov(self.video_cov[cls])
                cov_tgt = self._shrink_cov(self.audio_cov[cls])
            else:
                raise ValueError(f'Unknown projection direction: {source}')

            cov_src_sqrt = self._matrix_sqrt(cov_src)
            cov_src_invsqrt = self._matrix_invsqrt(cov_src)
            middle = self._matrix_sqrt(cov_src_sqrt @ cov_tgt @ cov_src_sqrt)
            A = cov_src_invsqrt @ middle @ cov_src_invsqrt
            b = mu_tgt - A @ mu_src
            z_proj = z_c @ A.transpose(0, 1) + b.unsqueeze(0)
            outs.append((mask, z_proj))

        z_out = torch.zeros_like(z)
        for mask, z_proj in outs:
            z_out[mask] = z_proj
        return z_out

    def project_audio_to_video(self, feat_a, labels):
        return self._gaussian_ot_project(feat_a, labels, source='audio_to_video')

    def project_video_to_audio(self, feat_v, labels):
        return self._gaussian_ot_project(feat_v, labels, source='video_to_audio')

    def _component_cost(self, source_components, target_components):
        
        src_comp = source_components.transpose(0, 1)  # [D, n_k]
        tgt_comp = target_components.transpose(0, 1)  # [D, n_k]
        numerator = src_comp @ tgt_comp.transpose(0, 1)
        denom = src_comp.norm(p=2, dim=1, keepdim=True) @ tgt_comp.norm(p=2, dim=1, keepdim=True).transpose(0, 1)
        cost = 1.0 - numerator / (denom + self.eps)
        return torch.clamp(cost, min=0.0, max=2.0)

    def _entropic_partial_ot_plan(self, cost, mass):
        
        D = cost.shape[0]
        device, dtype = cost.device, cost.dtype
        m = torch.clamp(mass, min=self.eps_m, max=1.0)

        u = torch.full((D,), 1.0 / float(D), device=device, dtype=dtype)
        v = torch.full((D,), 1.0 / float(D), device=device, dtype=dtype)
        dummy_mass = torch.clamp(1.0 - m, min=0.0)
        a_aug = torch.cat([u, dummy_mass.view(1)], dim=0)
        b_aug = torch.cat([v, dummy_mass.view(1)], dim=0)

        # Augmented cost. Matching real components to dummy has zero cost;
        # dummy-to-dummy is discouraged so that the real-real block carries mass m.
        C_aug = torch.zeros(D + 1, D + 1, device=device, dtype=dtype)
        C_aug[:D, :D] = cost
        C_aug[D, D] = cost.detach().max() + 10.0

        K = torch.exp(-C_aug / max(self.ot_eta, self.eps)) + self.eps
        left = torch.ones_like(a_aug)
        right = torch.ones_like(b_aug)
        for _ in range(self.ot_iters):
            left = a_aug / (K @ right + self.eps)
            right = b_aug / (K.transpose(0, 1) @ left + self.eps)

        plan_aug = left.unsqueeze(1) * K * right.unsqueeze(0)
        plan = plan_aug[:D, :D]

        # Tiny numerical correction so that sum(Pi) is close to m.
        plan_sum = plan.sum()
        if plan_sum > self.eps:
            plan = plan * (m / (plan_sum + self.eps))
        return plan

    def _partial_ot_loss(self, cost, mass):
        
        plan = self._entropic_partial_ot_plan(cost, mass)
        return (plan * cost).sum()

    def forward(self, feat_a, feat_v, logits_a, logits_v, y):
        labels = self.get_label_index(y)
        self.update_stats(feat_a.detach(), feat_v.detach(), labels)

        prob_a = F.softmax(logits_a.detach(), dim=1)
        prob_v = F.softmax(logits_v.detach(), dim=1)

        class_losses = []
        mass_a2v_vals, mass_v2a_vals = [], []
        lambda_a2v_vals, lambda_v2a_vals = [], []
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

            cost_a2v = self._component_cost(proj_a2v, fv)
            cost_v2a = self._component_cost(proj_v2a, fa)
            cbar_a2v = cost_a2v.mean()
            cbar_v2a = cost_v2a.mean()

            
            r_a = prob_a[mask, cls].mean()
            r_v = prob_v[mask, cls].mean()

            
            compat_a2v = torch.exp(-cbar_a2v / max(self.tau_c, self.eps))
            compat_v2a = torch.exp(-cbar_v2a / max(self.tau_c, self.eps))
            m_a2v = self.rho * r_a * (1.0 - r_v) * compat_a2v
            m_v2a = self.rho * r_v * (1.0 - r_a) * compat_v2a
            m_a2v = torch.clamp(m_a2v, min=self.eps_m, max=self.rho)
            m_v2a = torch.clamp(m_v2a, min=self.eps_m, max=self.rho)

            lambda_logits = torch.stack([r_a / max(self.tau_lambda, self.eps),
                                         r_v / max(self.tau_lambda, self.eps)])
            lambda_weights = F.softmax(lambda_logits, dim=0)
            lambda_a2v = lambda_weights[0]
            lambda_v2a = lambda_weights[1]

            loss_a2v = self._partial_ot_loss(cost_a2v, m_a2v)
            loss_v2a = self._partial_ot_loss(cost_v2a, m_v2a)
            class_loss = lambda_a2v * loss_a2v + lambda_v2a * loss_v2a
            class_losses.append(class_loss)

            mass_a2v_vals.append(float(m_a2v.detach().cpu()))
            mass_v2a_vals.append(float(m_v2a.detach().cpu()))
            lambda_a2v_vals.append(float(lambda_a2v.detach().cpu()))
            lambda_v2a_vals.append(float(lambda_v2a.detach().cpu()))
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
            'compat_a2v': float(np.mean(compat_a2v_vals)),
            'compat_v2a': float(np.mean(compat_v2a_vals)),
            'valid_classes': len(class_losses),
        }
        return align_loss, stats


def train_audio_video(epoch, train_loader, model, aligner, optimizer, logger, aba_beta=0.1, aba_warmup=0):
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
    stat_compat_a2v = Averager()
    stat_compat_v2a = Averager()
    stat_valid_cls = Averager()

    for spectrogram, image, y in tqdm(train_loader):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

        loss_a = criterion(result_a, y).mean()
        loss_v = criterion(result_v, y).mean()
        loss_fusion = criterion(result_b, y).mean()

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
        stat_compat_a2v.add(align_stats.get('compat_a2v', 0.0))
        stat_compat_v2a.add(align_stats.get('compat_v2a', 0.0))
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
            'compat_a2v:{compat_a2v:.4f}, compat_v2a:{compat_v2a:.4f}, '
            'valid_classes_per_batch:{valid_cls:.2f}'
        ).format(
            mass_a2v=stat_mass_a2v.item(),
            mass_v2a=stat_mass_v2a.item(),
            lambda_a2v=stat_lambda_a2v.item(),
            lambda_v2a=stat_lambda_v2a.item(),
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
    with torch.no_grad():
        for spectrogram, image, y in tqdm(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            
            result_b, result_a, result_v, f_a , f_v   = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(result_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(result_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(result_b, dim=1)).tolist()
            pred = (F.softmax(result_b, dim=1)).argmax(dim=1)
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
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'data' / 'kinetics_sound.json'))
    parser.add_argument('--aba_beta', type=float, default=0.1, help='Weight beta for adaptive bidirectional alignment loss.')
    parser.add_argument('--aba_warmup', type=int, default=0, help='Warmup epochs for gradually enabling ABA loss. 0 means no warmup.')
    parser.add_argument('--aba_rho', type=float, default=0.75, help='Maximum transport budget rho in Eq. (23).')
    parser.add_argument('--aba_tau_c', type=float, default=0.4, help='Projection compatibility temperature tau_c in Eq. (23).')
    parser.add_argument('--aba_tau_lambda', type=float, default=0.45, help='Directional weighting temperature tau_lambda in Eq. (25).')
    parser.add_argument('--aba_momentum', type=float, default=0.9, help='EMA momentum gamma for class-wise statistical feature estimation.')
    parser.add_argument('--aba_eps', type=float, default=1e-5, help='Numerical stability epsilon.')
    parser.add_argument('--aba_eps_m', type=float, default=1e-6, help='Lower clipping bound for transported mass.')
    parser.add_argument('--aba_cov_shrinkage', type=float, default=0.1, help='Covariance shrinkage coefficient alpha in Eq. (10).')
    parser.add_argument('--aba_ot_eta', type=float, default=0.07, help='Entropy regularization coefficient eta for partial OT.')
    parser.add_argument('--aba_ot_iters', type=int, default=50, help='Number of Sinkhorn iterations for entropic partial OT.')
    parser.add_argument('--aba_min_samples_per_class', type=int, default=2, help='Minimum samples per class in a batch to compute feature-level OT.')
    parser.add_argument('--tensorboard_root', type=str, default=str(PROJECT_ROOT / '_tensorboard_runs'), help='Root directory for TensorBoard runs.')
    parser.add_argument('--epoch_info_dir', type=str, default=str(PROJECT_ROOT / '_logs' / 'ks_information'), help='Directory for per-epoch text logs.')

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
    logger, _, _ = create_logger(cfg, local_rank)

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
        eps_m=args.aba_eps_m,
        rho=args.aba_rho,
        tau_c=args.aba_tau_c,
        tau_lambda=args.aba_tau_lambda,
        cov_shrinkage=args.aba_cov_shrinkage,
        ot_eta=args.aba_ot_eta,
        ot_iters=args.aba_ot_iters,
        min_samples_per_class=args.aba_min_samples_per_class,
    ).cuda()
    logger.info(
        f'ABA enabled: beta={args.aba_beta}, warmup={args.aba_warmup}, rho={args.aba_rho}, '
        f'tau_c={args.aba_tau_c}, tau_lambda={args.aba_tau_lambda}, '
        f'cov_shrinkage={args.aba_cov_shrinkage}, ot_eta={args.aba_ot_eta}, '
        f'ot_iters={args.aba_ot_iters}, feat_dim={feat_dim}'
    )

    lr_adjust = cfg['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=cfg['train']['optimizer']['momentum'],
                          weight_decay=cfg['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg['train']['lr_scheduler']['patience'], 0.1)

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
            aba_beta=args.aba_beta,
            aba_warmup=args.aba_warmup,
        )

        val_metrics = val(epoch, test_loader, model, logger)
        write_tensorboard_scalars(writer, epoch, train_metrics, val_metrics)
        epoch_row = flatten_epoch_metrics(epoch, train_metrics, val_metrics)
        append_epoch_info(info_path, epoch_row)
        writer.flush()
        logger.info(f'Epoch {epoch:d} information appended to: {info_path}')
    writer.close()
    logger.info('TensorBoard writer closed.')
