#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
import json
import random
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, average_precision_score
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


def compute_module_grad_norm(module: nn.Module) -> float:
    sq_sum = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq_sum += float(torch.sum(g * g).item())
    return float(sq_sum ** 0.5)


def smooth_curve(values, momentum=0.85):
    if len(values) == 0:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(momentum * smoothed[-1] + (1.0 - momentum) * v)
    return smoothed


def save_learning_dynamics_figure(history, save_path: str):
    epochs = np.arange(len(history['audio_grad']))

    audio_grad = np.asarray(history['audio_grad'], dtype=np.float32)
    video_grad = np.asarray(history['video_grad'], dtype=np.float32)
    audio_grad_s = np.asarray(smooth_curve(history['audio_grad']), dtype=np.float32)
    video_grad_s = np.asarray(smooth_curve(history['video_grad']), dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6))

    ax.plot(epochs, audio_grad_s, label='Audio branch', linewidth=2.4)
    ax.plot(epochs, video_grad_s, label='Video branch', linewidth=2.4)
    ax.scatter(epochs, audio_grad, s=12, alpha=0.35)
    ax.scatter(epochs, video_grad, s=12, alpha=0.35)

    diff = audio_grad_s - video_grad_s
    ax.fill_between(
        epochs,
        audio_grad_s,
        video_grad_s,
        where=diff >= 0,
        alpha=0.10,
        interpolate=True,
    )
    ax.fill_between(
        epochs,
        audio_grad_s,
        video_grad_s,
        where=diff < 0,
        alpha=0.10,
        interpolate=True,
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Gradient Norm')
    ax.set_title('Modality Learning Dynamics on KineticsSounds')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_learning_dynamics_with_acc_figure(history, save_path: str):
    epochs = np.arange(len(history['audio_grad']))

    audio_grad = np.asarray(history['audio_grad'], dtype=np.float32)
    video_grad = np.asarray(history['video_grad'], dtype=np.float32)
    audio_grad_s = np.asarray(smooth_curve(history['audio_grad']), dtype=np.float32)
    video_grad_s = np.asarray(smooth_curve(history['video_grad']), dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(epochs, audio_grad_s, label='Audio branch', linewidth=2.4)
    ax.plot(epochs, video_grad_s, label='Video branch', linewidth=2.4)
    ax.scatter(epochs, audio_grad, s=12, alpha=0.35)
    ax.scatter(epochs, video_grad, s=12, alpha=0.35)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Gradient Norm')
    ax.set_title('Gradient Dynamics')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(epochs, history['acc_a'], label='Audio acc.', linewidth=2.4)
    ax.plot(epochs, history['acc_v'], label='Video acc.', linewidth=2.4)
    ax.plot(epochs, history['acc'], label='Multi acc.', linewidth=2.4, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Performance')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def train_audio_video(epoch, train_loader, model, optimizer, logger, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    grad_audio = Averager()
    grad_video = Averager()
    grad_cls_a = Averager()
    grad_cls_v = Averager()

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

        grad_audio.add(compute_module_grad_norm(model.audio_encoder))
        grad_video.add(compute_module_grad_norm(model.video_encoder))
        grad_cls_a.add(compute_module_grad_norm(model.cls_a))
        grad_cls_v.add(compute_module_grad_norm(model.cls_v))

        optimizer.step()

        tl.add(loss.item())
        tl_a.add(loss_a.item())
        tl_v.add(loss_v.item())

    loss_ave = tl.item()
    loss_audio = tl_a.item()
    loss_video = tl_v.item()

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.4f}, '
         'Average loss_audio:{loss_audio:.4f}, Average loss_video:{loss_video:.4f}, '
         'Grad(audio_enc):{g_a:.4f}, Grad(video_enc):{g_v:.4f}, '
         'Grad(audio_cls):{g_ca:.4f}, Grad(video_cls):{g_cv:.4f}').format(
            epoch=epoch,
            loss_ave=loss_ave,
            loss_audio=loss_audio,
            loss_video=loss_video,
            g_a=grad_audio.item(),
            g_v=grad_video.item(),
            g_ca=grad_cls_a.item(),
            g_cv=grad_cls_v.item(),
        )
    )

    grad_stats = {
        'audio_grad': grad_audio.item(),
        'video_grad': grad_video.item(),
        'audio_cls_grad': grad_cls_a.item(),
        'video_cls_grad': grad_cls_v.item(),
        'loss': loss_ave,
        'loss_audio': loss_audio,
        'loss_video': loss_video,
    }
    return model, grad_stats


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
         'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(
            epoch=epoch, f1=f1, acc=acc, mAP=mAP,
            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v,
        )
    )
    return acc, acc_a, acc_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--plot_dir', type=str, default='/data/zyh/NeurIPS24-LFM/_figure/_learning_dynamics/ks')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg['train']['lr_scheduler']['patience'], 0.1)

    best_acc = 0.0
    save_dir = '/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks'
    os.makedirs(save_dir, exist_ok=True)

    time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = os.path.join(args.plot_dir, time_tag)
    os.makedirs(plot_dir, exist_ok=True)

    history = defaultdict(list)

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model, grad_stats = train_audio_video(epoch, train_loader, model, optimizer, logger, logits_ratio)
        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        history['audio_grad'].append(float(grad_stats['audio_grad']))
        history['video_grad'].append(float(grad_stats['video_grad']))
        history['audio_cls_grad'].append(float(grad_stats['audio_cls_grad']))
        history['video_cls_grad'].append(float(grad_stats['video_cls_grad']))
        history['loss'].append(float(grad_stats['loss']))
        history['loss_audio'].append(float(grad_stats['loss_audio']))
        history['loss_video'].append(float(grad_stats['loss_video']))
        history['acc'].append(float(acc))
        history['acc_a'].append(float(acc_a))
        history['acc_v'].append(float(acc_v))

        save_learning_dynamics_figure(
            history,
            os.path.join(plot_dir, 'learning_dynamics_grad_only.png')
        )
        save_learning_dynamics_with_acc_figure(
            history,
            os.path.join(plot_dir, 'learning_dynamics_grad_and_acc.png')
        )

        with open(os.path.join(plot_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump({k: list(v) for k, v in history.items()}, f, indent=2)

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, 'multi_KS_best_model_grad_dynamics.pth')
            )
            logger.info('Find a better model and save it!')

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth')
            )

    logger.info(f'Gradient dynamics figures saved to: {plot_dir}')
