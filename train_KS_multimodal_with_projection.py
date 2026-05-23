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
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
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


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


class ClassWiseStatProjector(nn.Module):
    """
    Class-wise diagonal Gaussian projection using running mean/variance.

    For class k, project audio -> video via:
        z_a2v = ((z_a - mu_a[k]) / std_a[k]) * std_v[k] + mu_v[k]
    and symmetrically for video -> audio.

    Running statistics are updated with EMA for stability because some mini-batches
    may contain only a few samples per class.
    """
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.9,
        eps: float = 1e-5,
        loss_type: str = "smooth_l1",
        cosine_weight: float = 0.1,
    ):
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
    def _get_label_index(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            return y.long()
        return torch.argmax(y, dim=1).long()

    @torch.no_grad()
    def update_stats(self, feat_a: torch.Tensor, feat_v: torch.Tensor, y: torch.Tensor) -> None:
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
            var_a = cls_a.var(dim=0, unbiased=False)
            var_v = cls_v.var(dim=0, unbiased=False)

            # Avoid zero variance for singleton classes in a batch.
            var_a = torch.clamp(var_a, min=self.eps)
            var_v = torch.clamp(var_v, min=self.eps)

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

    def project_audio_to_video(self, feat_a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        labels = self._get_label_index(y)
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_a - mu_a) / std_a * std_v + mu_v

    def project_video_to_audio(self, feat_v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        labels = self._get_label_index(y)
        mu_a = self.audio_mean[labels]
        mu_v = self.video_mean[labels]
        std_a = torch.sqrt(torch.clamp(self.audio_var[labels], min=self.eps))
        std_v = torch.sqrt(torch.clamp(self.video_var[labels], min=self.eps))
        return (feat_v - mu_v) / std_v * std_a + mu_a

    def _pair_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            base = F.mse_loss(x, y)
        else:
            base = F.smooth_l1_loss(x, y)

        if self.cosine_weight > 0:
            cos = 1.0 - F.cosine_similarity(x, y, dim=1).mean()
            base = base + self.cosine_weight * cos
        return base

    def forward(self, feat_a: torch.Tensor, feat_v: torch.Tensor, y: torch.Tensor):
        # Update the running class-wise statistics using the current batch.
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

        loss_proj, proj_a2v, proj_v2a, proj_stats = projector(f_a, f_v, y)

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
            pred = (F.softmax(0.5 * result_a + 0.5 * result_v, dim=1)).argmax(dim=1)
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
    args = parser.parse_args()

    cfg = deep_update_dict(json.load(open(args.config, 'r')), config)

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

    projector = ClassWiseStatProjector(
        num_classes=cfg['setting']['num_class'],
        feat_dim=model.hidden_dim,
        momentum=args.proj_momentum,
        eps=args.proj_eps,
        loss_type=args.proj_loss,
        cosine_weight=args.proj_cosine_weight,
    ).cuda()

    lr_adjust = cfg['train']['optimizer']['lr']
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr_adjust,
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
