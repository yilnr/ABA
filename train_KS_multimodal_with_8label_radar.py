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
# from dataset.VGGSoundDataset import VGGSound,SemiVGGSound
import random
import re
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def parse_int_list(s):
    """Parse comma-separated integers, e.g., '3,6,9'."""
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(',') if x.strip() != ""]


def build_selected_class_names(class_names_str, selected_classes, num_classes):
    """
    Build display names for selected classes.

    Supported formats:
    1) Empty string: use Class {id}.
    2) Length == num_classes: treat as global class-name list and index by label id.
    3) Length == len(selected_classes): treat as names corresponding to selected classes.
    """
    if class_names_str is None or str(class_names_str).strip() == "":
        return [f"Class {c}" for c in selected_classes]

    names = [x.strip().strip('"').strip("'") for x in str(class_names_str).split(',')]
    names = [x for x in names if x != ""]

    if len(names) == num_classes:
        return [names[c] if c < len(names) else f"Class {c}" for c in selected_classes]

    if len(names) == len(selected_classes):
        return names

    print(
        f"[Radar Warning] Got {len(names)} class names, but num_classes={num_classes} "
        f"and selected classes={len(selected_classes)}. Use default names instead."
    )
    return [f"Class {c}" for c in selected_classes]


def select_radar_classes(y_true,
                         num_classes,
                         radar_num_classes=8,
                         radar_class_ids="",
                         radar_sample_mode="frequent",
                         radar_seed=0):
    """
    Select a subset of labels for radar visualization.

    - If radar_class_ids is provided, use these labels directly.
    - Otherwise select radar_num_classes labels from labels appearing in y_true.
    - radar_sample_mode: frequent / random / first.
    """
    explicit_ids = parse_int_list(radar_class_ids)
    present_classes = sorted(set(int(x) for x in y_true))

    if explicit_ids is not None:
        selected = [c for c in explicit_ids if 0 <= c < num_classes and c in present_classes]
        if len(selected) == 0:
            raise ValueError(
                f"No valid radar_class_ids found. Given={explicit_ids}, present={present_classes}."
            )
        return selected

    if radar_num_classes is None or radar_num_classes <= 0:
        return present_classes

    k = min(int(radar_num_classes), len(present_classes))

    if radar_sample_mode == "frequent":
        counter = Counter(int(x) for x in y_true)
        selected = [c for c, _ in counter.most_common(k)]
        return sorted(selected)

    if radar_sample_mode == "random":
        rng = np.random.default_rng(radar_seed)
        selected = rng.choice(present_classes, size=k, replace=False).tolist()
        return sorted(int(x) for x in selected)

    if radar_sample_mode == "first":
        return present_classes[:k]

    raise ValueError(f"Unknown radar_sample_mode: {radar_sample_mode}")


def compute_selected_class_acc(y_true, y_pred, selected_classes):
    """
    Class-wise recall/accuracy on selected labels:
    acc_c = #correct samples whose ground-truth is c / #samples whose ground-truth is c.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    values = []
    for cls in selected_classes:
        mask = (y_true == cls)
        if mask.sum() == 0:
            values.append(np.nan)
        else:
            values.append((y_pred[mask] == cls).mean())
    return np.asarray(values, dtype=np.float32)


def plot_8label_radar_epoch(y_true,
                            pred_multi,
                            pred_audio,
                            pred_video,
                            epoch,
                            save_dir='/data/zyh/NeurIPS24-LFM/_figure/leida_ks',
                            selected_classes=None,
                            class_names=None):
    """
    Save a class-wise radar chart for selected labels.

    Curves:
        Multi: fusion branch.
        Audio: audio branch.
        Video: video branch.

    Axes:
        Average + selected labels.
    """
    ensure_dir(save_dir)

    if selected_classes is None or len(selected_classes) == 0:
        raise ValueError("selected_classes must be non-empty for radar visualization.")
    if class_names is None:
        class_names = [f"Class {c}" for c in selected_classes]

    multi_cls = compute_selected_class_acc(y_true, pred_multi, selected_classes)
    audio_cls = compute_selected_class_acc(y_true, pred_audio, selected_classes)
    video_cls = compute_selected_class_acc(y_true, pred_video, selected_classes)

    multi_avg = np.nanmean(multi_cls)
    audio_avg = np.nanmean(audio_cls)
    video_avg = np.nanmean(video_cls)

    categories = ["Average"] + class_names
    multi_values = np.concatenate([[multi_avg], np.nan_to_num(multi_cls, nan=0.0)])
    audio_values = np.concatenate([[audio_avg], np.nan_to_num(audio_cls, nan=0.0)])
    video_values = np.concatenate([[video_avg], np.nan_to_num(video_cls, nan=0.0)])

    num_axes = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    multi_values = np.concatenate([multi_values, multi_values[:1]])
    audio_values = np.concatenate([audio_values, audio_values[:1]])
    video_values = np.concatenate([video_values, video_values[:1]])

    fig = plt.figure(figsize=(6.2, 5.8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, multi_values, linewidth=2.2, marker='o', markersize=4.5,
            label='Multi', color='#D65F5F')
    ax.fill(angles, multi_values, color='#D65F5F', alpha=0.14)

    ax.plot(angles, audio_values, linewidth=1.8, marker='o', markersize=4.0,
            label='Audio', color='#4C78A8')
    ax.fill(angles, audio_values, color='#4C78A8', alpha=0.08)

    ax.plot(angles, video_values, linewidth=1.8, marker='o', markersize=4.0,
            label='Video', color='#59A14F')
    ax.fill(angles, video_values, color='#59A14F', alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linewidth=0.8, alpha=0.55)

    ax.set_title(f'KineticsSounds Class-wise Performance (Epoch {epoch})', fontsize=12, pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.22, 1.14), fontsize=9, frameon=False)

    selected_suffix = '_'.join(str(c) for c in selected_classes)
    png_path = os.path.join(save_dir, f'epoch_{epoch:03d}_8label_radar_{selected_suffix}.png')
    pdf_path = os.path.join(save_dir, f'epoch_{epoch:03d}_8label_radar_{selected_suffix}.pdf')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return png_path, pdf_path

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
        loss_fusion = criterion( logits_ratio * result_a + logits_ratio * result_v, y).mean()

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
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , Average loss_audio : {loss_audio:.3f},Average loss_video : \
                 {loss_video:.3f}').format(epoch=epoch, loss_ave=loss_ave, loss_audio = loss_audio, loss_video =loss_video))

    return model


def val(epoch, val_loader, model, logger,
        radar_save_dir=None,
        radar_num_classes=8,
        radar_class_ids='',
        radar_sample_mode='frequent',
        radar_seed=0,
        radar_class_names=''):

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

        if radar_save_dir is not None:
            num_classes = len(soft_pred[0]) if len(soft_pred) > 0 else max(label_list) + 1
            selected_classes = select_radar_classes(
                y_true=label_list,
                num_classes=num_classes,
                radar_num_classes=radar_num_classes,
                radar_class_ids=radar_class_ids,
                radar_sample_mode=radar_sample_mode,
                radar_seed=radar_seed,
            )
            class_names = build_selected_class_names(
                class_names_str=radar_class_names,
                selected_classes=selected_classes,
                num_classes=num_classes,
            )
            radar_png, radar_pdf = plot_8label_radar_epoch(
                y_true=label_list,
                pred_multi=pred_list,
                pred_audio=pred_list_a,
                pred_video=pred_list_v,
                epoch=epoch,
                save_dir=radar_save_dir,
                selected_classes=selected_classes,
                class_names=class_names,
            )
            logger.info(f'[Radar] Selected classes: {selected_classes}')
            logger.info(f'[Radar] Saved to: {radar_png}')
            logger.info(f'[Radar] Saved to: {radar_pdf}')

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                              f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    return acc, acc_a, acc_v
    

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument(
        '--radar_save_dir',
        type=str,
        default='/data/zyh/NeurIPS24-LFM/_figure/leida_ks',
        help='Directory to save per-epoch 8-label radar charts.'
    )
    parser.add_argument(
        '--radar_num_classes',
        type=int,
        default=8,
        help='Number of labels selected for radar visualization when --radar_class_ids is not provided.'
    )
    parser.add_argument(
        '--radar_class_ids',
        type=str,
        default='',
        help='Explicit label ids for radar chart, e.g., "0,3,5,8,12,16,20,25". This overrides --radar_num_classes.'
    )
    parser.add_argument(
        '--radar_sample_mode',
        type=str,
        default='frequent',
        choices=['frequent', 'random', 'first'],
        help='How to select labels when --radar_class_ids is not provided.'
    )
    parser.add_argument(
        '--radar_seed',
        type=int,
        default=0,
        help='Random seed for selecting labels when --radar_sample_mode random is used.'
    )
    parser.add_argument(
        '--radar_class_names',
        type=str,
        default='',
        help='Comma-separated class names. Can be full class-name list or selected-class-name list.'
    )

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


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    val_batch = next(iter(train_loader))


    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=config['train']['optimizer']['momentum'],
                          weight_decay=config['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0
    cls_k = []
    
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

        acc, acc_a, acc_v = val(
            epoch,
            test_loader,
            model,
            logger,
            radar_save_dir=args.radar_save_dir,
            radar_num_classes=args.radar_num_classes,
            radar_class_ids=args.radar_class_ids,
            radar_sample_mode=args.radar_sample_mode,
            radar_seed=args.radar_seed,
            radar_class_names=args.radar_class_names,
        )
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