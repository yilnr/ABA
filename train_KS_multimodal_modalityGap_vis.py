#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
import json
import numpy as np
import argparse
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import umap

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F

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


def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
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
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , Average loss_audio : {loss_audio:.3f},Average loss_video : '
         '{loss_video:.3f}').format(epoch=epoch, loss_ave=loss_ave, loss_audio=loss_audio, loss_video=loss_video)
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
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

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
        ('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}')
        .format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v)
    )
    return acc, acc_a, acc_v


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_fixed_batch(batch, device, max_samples=64):
    spectrogram, image, y = batch

    spectrogram = spectrogram[:max_samples]
    image = image[:max_samples]
    y = y[:max_samples]

    if y.dim() > 1:
        labels = torch.argmax(y, dim=1).long()
    else:
        labels = y.long()

    spectrogram = spectrogram.unsqueeze(1).float().to(device)
    image = image.float().to(device)
    labels = labels.to(device)

    return spectrogram, image, labels


@torch.no_grad()
def extract_normalized_embeddings(model, spectrogram, image):
    """
    参考 Mind the Gap：
    提取两个模态 embedding，并做 L2 normalize 后用于 UMAP 可视化。
    """
    a_feature = model.audio_encoder(spectrogram)   # [B, D]
    v_feature = model.video_encoder(image)         # [B, D]

    a_feature = F.normalize(a_feature, dim=1)
    v_feature = F.normalize(v_feature, dim=1)

    return a_feature.detach().cpu().numpy(), v_feature.detach().cpu().numpy()


def compute_gap_distance(audio_emb, video_emb):
    """
    参考论文 4.2 节：
    gap 向量 = 两种模态中心之差
    gap 距离 = 两中心的欧氏距离
    """
    a_center = audio_emb.mean(axis=0)
    v_center = video_emb.mean(axis=0)
    gap_vec = a_center - v_center
    gap_dist = np.linalg.norm(gap_vec)
    return gap_vec, gap_dist


def project_embeddings_umap(audio_emb, video_emb, random_state=42):
    """
    拼接后一起做 UMAP，保证两种模态在同一 2D 空间里。
    """
    all_emb = np.concatenate([audio_emb, video_emb], axis=0)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, all_emb.shape[0] - 1)),
        min_dist=0.1,
        metric='cosine',
        random_state=random_state,
    )
    proj = reducer.fit_transform(all_emb)
    n = audio_emb.shape[0]
    audio_2d = proj[:n]
    video_2d = proj[n:]
    return audio_2d, video_2d


def plot_modality_gap_umap(audio_2d, video_2d, epoch, gap_dist, save_dir):
    """
    参考 Mind the Gap Figure 1(b):
    - 两种模态点画在同一 2D UMAP 空间
    - 同一样本用灰线连接
    - 同时保存 PNG 和 PDF
    """
    ensure_dir(save_dir)
    n = audio_2d.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2))

    for i in range(n):
        ax.plot(
            [audio_2d[i, 0], video_2d[i, 0]],
            [audio_2d[i, 1], video_2d[i, 1]],
            color='gray',
            alpha=0.35,
            linewidth=1.0,
            zorder=1
        )

    ax.scatter(
        audio_2d[:, 0], audio_2d[:, 1],
        c='red', s=80, alpha=0.75, label='Audio', zorder=2
    )
    ax.scatter(
        video_2d[:, 0], video_2d[:, 1],
        c='blue', marker='x', s=110, alpha=0.75, label='Video', zorder=3
    )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'Modality gap (epoch={epoch}, dist={gap_dist:.4f})')
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()

    save_path_png = os.path.join(save_dir, f'epoch_{epoch:03d}_modality_gap_umap.png')
    save_path_pdf = os.path.join(save_dir, f'epoch_{epoch:03d}_modality_gap_umap.pdf')

    plt.savefig(save_path_png, dpi=220, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close(fig)

    return save_path_png, save_path_pdf


def save_epoch_modality_gap(epoch,
                            model,
                            fixed_batch,
                            save_dir,
                            max_samples=64,
                            logger=None):
    """
    每个 epoch 保存一张参考 Mind the Gap 风格的 modality gap 图
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    try:
        spectrogram, image, labels = prepare_fixed_batch(
            fixed_batch, device=device, max_samples=max_samples
        )

        audio_emb, video_emb = extract_normalized_embeddings(model, spectrogram, image)
        gap_vec, gap_dist = compute_gap_distance(audio_emb, video_emb)
        audio_2d, video_2d = project_embeddings_umap(audio_emb, video_emb, random_state=epoch + 42)

        save_path_png, save_path_pdf = plot_modality_gap_umap(
            audio_2d, video_2d, epoch, gap_dist, save_dir
        )

        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_audio_emb.npy'), audio_emb)
        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_video_emb.npy'), video_emb)
        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_gap_vec.npy'), gap_vec)
        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_gap_dist.npy'), np.array([gap_dist]))

        logger.info(f'[ModalityGap] Saved UMAP figure to: {save_path_png}')
        logger.info(f'[ModalityGap] Saved UMAP figure to: {save_path_pdf}')
        logger.info(f'[ModalityGap] gap distance = {gap_dist:.6f}')

    except Exception as e:
        if logger is not None:
            logger.info(f'[ModalityGap] Failed at epoch {epoch}: {e}')
        else:
            print(f'[ModalityGap] Failed at epoch {epoch}: {e}')
    finally:
        if was_training:
            model.train()


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--gap_save_dir', type=str,
                        default='/data/zyh/NeurIPS24-LFM/_figure/modality_gap',
                        help='Directory to save modality gap figures.')
    parser.add_argument('--gap_num_samples', type=int, default=64,
                        help='Number of fixed-batch samples used for modality gap plotting.')

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

    # ----- SET DATALOADER -----
    train_dataset = VADataset(config, mode='train')
    test_dataset = VADataset(config, mode='test')

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['train']['num_workers'], pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
        num_workers=cfg['test']['num_workers'], pin_memory=True
    )

    fixed_batch = next(iter(train_loader))
    ensure_dir(args.gap_save_dir)

    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(
        model.parameters(), lr=lr_adjust,
        momentum=config['train']['optimizer']['momentum'],
        weight_decay=config['train']['optimizer']['wc']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config['train']['lr_scheduler']['patience'], 0.1
    )

    best_acc = 0
    cls_k = []

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        save_epoch_modality_gap(
            epoch=epoch,
            model=model,
            fixed_batch=fixed_batch,
            save_dir=args.gap_save_dir,
            max_samples=args.gap_num_samples,
            logger=logger,
        )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )