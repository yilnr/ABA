#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
import json
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score

from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import create_logger, Averager, deep_update_dict
from utils.tools import weight_init

warnings.filterwarnings("ignore")


def compute_mAP(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    ap = []
    for i in range(y_true.shape[1]):
        ap.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(ap))


def to_class_indices(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.long()
    return torch.argmax(y, dim=1).long()


def train_audio_video(
    epoch: int,
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    logger,
    logits_ratio: float,
    device: torch.device,
) -> nn.Module:
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    for _, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc=f"Train {epoch}")):
        image = image.float().to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        spectrogram = spectrogram.unsqueeze(1).float().to(device, non_blocking=True)

        optimizer.zero_grad()
        _, result_a, result_v, _, _ = model(spectrogram, image)

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


def val(
    epoch: int,
    val_loader: DataLoader,
    model: nn.Module,
    logger,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    pred_list, pred_list_a, pred_list_v = [], [], []
    label_list, soft_pred, soft_pred_a, soft_pred_v, one_hot_label = [], [], [], [], []

    with torch.no_grad():
        for _, (spectrogram, image, y) in enumerate(tqdm(val_loader, desc=f"Val {epoch}")):
            label_list += torch.argmax(y, dim=1).tolist()
            one_hot_label += y.tolist()

            image = image.float().to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            spectrogram = spectrogram.unsqueeze(1).float().to(device, non_blocking=True)

            _, result_a, result_v, _, _ = model(spectrogram, image)

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
    mAP = compute_mAP(torch.tensor(soft_pred), torch.tensor(one_hot_label))
    mAP_a = compute_mAP(torch.tensor(soft_pred_a), torch.tensor(one_hot_label))
    mAP_v = compute_mAP(torch.tensor(soft_pred_v), torch.tensor(one_hot_label))

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


@torch.no_grad()
def collect_features(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    feats_a: List[torch.Tensor] = []
    feats_v: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for step, (spectrogram, image, y) in enumerate(tqdm(loader, desc="CollectFeat", leave=False)):
        if max_batches is not None and step >= max_batches:
            break

        image = image.float().to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        spectrogram = spectrogram.unsqueeze(1).float().to(device, non_blocking=True)

        _, _, _, f_a, f_v = model(spectrogram, image)
        feats_a.append(f_a.detach().cpu())
        feats_v.append(f_v.detach().cpu())
        labels_all.append(to_class_indices(y).detach().cpu())

    return torch.cat(feats_a, dim=0), torch.cat(feats_v, dim=0), torch.cat(labels_all, dim=0)


def shrink_covariance(cov: torch.Tensor, shrinkage: float, eps: float) -> torch.Tensor:
    d = cov.shape[0]
    trace_term = torch.trace(cov) / d
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    cov = (1.0 - shrinkage) * cov + shrinkage * trace_term * eye
    cov = cov + eps * eye
    return cov


def compute_class_stats(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    shrinkage: float,
    eps: float,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    d = feats.shape[1]
    eye = torch.eye(d, dtype=feats.dtype, device=feats.device)

    for k in range(num_classes):
        idx = (labels == k)
        if idx.sum().item() == 0:
            continue
        x = feats[idx]
        mu = x.mean(dim=0)
        xc = x - mu
        if x.shape[0] > 1:
            cov = (xc.T @ xc) / x.shape[0]
        else:
            cov = eye.clone()
        cov = shrink_covariance(cov, shrinkage=shrinkage, eps=eps)
        stats[k] = (mu, cov)
    return stats


def matrix_sqrt_and_inv_sqrt(cov: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=eps)
    sqrt_e = torch.sqrt(evals)
    inv_sqrt_e = 1.0 / sqrt_e
    cov_sqrt = (evecs * sqrt_e.unsqueeze(0)) @ evecs.T
    cov_inv_sqrt = (evecs * inv_sqrt_e.unsqueeze(0)) @ evecs.T
    return cov_sqrt, cov_inv_sqrt


def build_classwise_ot_map(
    mu_src: torch.Tensor,
    cov_src: torch.Tensor,
    mu_tgt: torch.Tensor,
    cov_tgt: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cov_src_sqrt, cov_src_inv_sqrt = matrix_sqrt_and_inv_sqrt(cov_src, eps)
    middle = cov_src_sqrt @ cov_tgt @ cov_src_sqrt
    middle_sqrt, _ = matrix_sqrt_and_inv_sqrt(middle, eps)
    A = cov_src_inv_sqrt @ middle_sqrt @ cov_src_inv_sqrt
    b = mu_tgt - A @ mu_src
    return A, b


def build_projection_maps(
    stats_a: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    stats_v: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    eps: float,
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    maps_a2v: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    maps_v2a: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    common_classes = sorted(set(stats_a.keys()).intersection(set(stats_v.keys())))
    for k in common_classes:
        mu_a, cov_a = stats_a[k]
        mu_v, cov_v = stats_v[k]
        maps_a2v[k] = build_classwise_ot_map(mu_a, cov_a, mu_v, cov_v, eps)
        maps_v2a[k] = build_classwise_ot_map(mu_v, cov_v, mu_a, cov_a, eps)
    return maps_a2v, maps_v2a


@torch.no_grad()
def evaluate_projection_similarity(
    feats_a: torch.Tensor,
    feats_v: torch.Tensor,
    labels: torch.Tensor,
    maps_a2v: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    maps_v2a: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    raw_a2v_vals = []
    proj_a2v_vals = []
    raw_v2a_vals = []
    proj_v2a_vals = []

    per_class_proj_a2v = defaultdict(list)
    per_class_proj_v2a = defaultdict(list)

    for i in range(feats_a.shape[0]):
        k = int(labels[i].item())
        if k not in maps_a2v or k not in maps_v2a:
            continue

        a = feats_a[i]
        v = feats_v[i]
        raw_sim = F.cosine_similarity(a.unsqueeze(0), v.unsqueeze(0), dim=1).item()
        raw_a2v_vals.append(raw_sim)
        raw_v2a_vals.append(raw_sim)

        A_av, b_av = maps_a2v[k]
        A_va, b_va = maps_v2a[k]

        a_proj = A_av @ a + b_av
        v_proj = A_va @ v + b_va

        sim_a2v = F.cosine_similarity(a_proj.unsqueeze(0), v.unsqueeze(0), dim=1).item()
        sim_v2a = F.cosine_similarity(v_proj.unsqueeze(0), a.unsqueeze(0), dim=1).item()

        proj_a2v_vals.append(sim_a2v)
        proj_v2a_vals.append(sim_v2a)
        per_class_proj_a2v[k].append(sim_a2v)
        per_class_proj_v2a[k].append(sim_v2a)

    metrics = {
        "raw_cosine/a_vs_v": float(np.mean(raw_a2v_vals)) if raw_a2v_vals else 0.0,
        "raw_cosine/v_vs_a": float(np.mean(raw_v2a_vals)) if raw_v2a_vals else 0.0,
        "projected_cosine/a2v_vs_v": float(np.mean(proj_a2v_vals)) if proj_a2v_vals else 0.0,
        "projected_cosine/v2a_vs_a": float(np.mean(proj_v2a_vals)) if proj_v2a_vals else 0.0,
        "delta_cosine/a2v_improvement": float(np.mean(proj_a2v_vals) - np.mean(raw_a2v_vals)) if proj_a2v_vals else 0.0,
        "delta_cosine/v2a_improvement": float(np.mean(proj_v2a_vals) - np.mean(raw_v2a_vals)) if proj_v2a_vals else 0.0,
    }

    for k, vals in per_class_proj_a2v.items():
        metrics[f"per_class_projected_cosine/a2v_vs_v/class_{k}"] = float(np.mean(vals))
    for k, vals in per_class_proj_v2a.items():
        metrics[f"per_class_projected_cosine/v2a_vs_a/class_{k}"] = float(np.mean(vals))
    return metrics


def log_projection_metrics(writer: SummaryWriter, metrics: Dict[str, float], epoch: int):
    for tag, value in metrics.items():
        writer.add_scalar(tag, value, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--tb_root', type=str, default='/data/zyh/NeurIPS24-LFM/_tensorboard')
    parser.add_argument('--projection_shrinkage', type=float, default=0.1,
                        help='Shrinkage coefficient for covariance stabilization.')
    parser.add_argument('--projection_eps', type=float, default=1e-4,
                        help='Diagonal regularization for covariance and eigenvalue clipping.')
    parser.add_argument('--projection_stat_loader', type=str, default='train', choices=['train', 'test'],
                        help='Which split to use for estimating class-wise Gaussian statistics.')
    parser.add_argument('--projection_eval_loader', type=str, default='test', choices=['train', 'test'],
                        help='Which split to use for evaluating projected-feature similarity.')
    parser.add_argument('--projection_max_batches', type=int, default=0,
                        help='Use at most this many batches when collecting projection statistics/evaluation. 0 means all.')
    args = parser.parse_args()

    cfg = config
    with open(args.config, 'r') as f:
        exp_params = json.load(f)
    cfg = deep_update_dict(exp_params, cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = cfg['train']['local_rank']
    logits_ratio = cfg['train']['logits_ratio']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    tb_dir = os.path.join(args.tb_root, 'ks_projection', exp_id)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard directory: {tb_dir}")
    
    logger.info("Start building train dataset...")
    train_dataset = VADataset(cfg, mode='train')

    logger.info("Start building test dataset...")
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
    logger.info("Dataloaders built.")

    logger.info("Start building model...")
    model = AVClassifier(config=cfg).to(device)
    logger.info("Start applying weight_init...")
    model.apply(weight_init)
    logger.info("weight_init done.")
    
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
    num_classes = int(cfg['setting']['num_class'])
    max_batches = None if args.projection_max_batches <= 0 else args.projection_max_batches

    stat_loader = train_loader if args.projection_stat_loader == 'train' else test_loader
    eval_loader = train_loader if args.projection_eval_loader == 'train' else test_loader

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        scheduler.step()

        model = train_audio_video(epoch, train_loader, model, optimizer, logger, logits_ratio, device)
        acc, acc_a, acc_v = val(epoch, test_loader, model, logger, device)

        writer.add_scalar('acc/multimodal', acc, epoch)
        writer.add_scalar('acc/audio', acc_a, epoch)
        writer.add_scalar('acc/video', acc_v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # ===== Class-Conditional Statistical Projection analysis =====
        feats_a_stat, feats_v_stat, labels_stat = collect_features(stat_loader, model, device, max_batches=max_batches)
        feats_a_eval, feats_v_eval, labels_eval = collect_features(eval_loader, model, device, max_batches=max_batches)

        stats_a = compute_class_stats(
            feats_a_stat.float(), labels_stat.long(), num_classes,
            shrinkage=args.projection_shrinkage, eps=args.projection_eps
        )
        stats_v = compute_class_stats(
            feats_v_stat.float(), labels_stat.long(), num_classes,
            shrinkage=args.projection_shrinkage, eps=args.projection_eps
        )
        maps_a2v, maps_v2a = build_projection_maps(stats_a, stats_v, eps=args.projection_eps)

        projection_metrics = evaluate_projection_similarity(
            feats_a_eval.float(), feats_v_eval.float(), labels_eval.long(), maps_a2v, maps_v2a
        )
        log_projection_metrics(writer, projection_metrics, epoch)

        logger.info(
            "Projection similarity | raw(a,v)=%.4f, proj(a->v,v)=%.4f, raw(v,a)=%.4f, proj(v->a,a)=%.4f",
            projection_metrics["raw_cosine/a_vs_v"],
            projection_metrics["projected_cosine/a2v_vs_v"],
            projection_metrics["raw_cosine/v_vs_a"],
            projection_metrics["projected_cosine/v2a_vs_a"],
        )

        if acc > best_acc:
            best_acc = acc

        if epoch % 10 == 0:
            ckpt_dir = '/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f'multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )
            torch.save(model.state_dict(), ckpt_path)

    writer.close()
