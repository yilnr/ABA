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

# ===== Added for CSV / TensorBoard / loss-curve visualization =====
import csv
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


def Alignment_Feature(a_f, v_f):
    a_f = F.normalize(a_f, p=2, dim=-1)
    v_f = F.normalize(v_f, p=2, dim=-1)
    similarity_matrix = torch.matmul(a_f, v_f.T)

    similarity_matrix = similarity_matrix / 0.07

    labels = torch.arange(a_f.size(0)).to(a_f.device)
    loss_a_to_v = F.cross_entropy(similarity_matrix, labels)
    loss_v_to_a = F.cross_entropy(similarity_matrix.T, labels)
    loss = (loss_a_to_v + loss_v_to_a) / 2.0
    return loss


def getAlpha_Learnable_Fitted(epoch):
    # Alpha with Learnable learning is fitted with functions
    coef_alpha1 = [2.04623704e-01, 3.35472727e-03, 1.22989557e-04, -2.92947416e-06, 2.23835486e-08, -5.39717505e-11]
    coef_alpha2 = [7.95376296e-01, -3.35472727e-03, -1.22989557e-04, 2.92947416e-06, -2.23835486e-08, 5.39717505e-11]
    alpha1 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha1))
    alpha2 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha2))
    return [alpha1, alpha2]


# ===== Added helper functions =====
def build_run_dir(cfg, exp_id=None):
    """Create one run folder under /data/zyh/NeurIPS24-LFM/_tensorboard_runs."""
    root = Path("/data/zyh/NeurIPS24-LFM/_tensorboard_runs")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = cfg.get("seed", "NA")
    name_parts = ["ks_feature_alignment", timestamp, f"seed{seed}"]
    if exp_id is not None:
        name_parts.append(str(exp_id))
    run_dir = root / "_".join(name_parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_csv(csv_path):
    header = [
        "epoch",
        "loss", "loss_fusion", "loss_a", "loss_v", "loss_alignment",
        "val_f1_multi", "val_acc_multi", "val_mAP_multi",
        "val_f1_audio", "val_acc_audio", "val_mAP_audio",
        "val_f1_video", "val_acc_video", "val_mAP_video",
        "lr",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
    return header


def append_csv_row(csv_path, header, row):
    safe_row = {k: row.get(k, "") for k in header}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(safe_row)


def write_tensorboard(writer, epoch, train_metrics, val_metrics, lr):
    """Write all losses and validation metrics to TensorBoard."""
    # One combined loss chart in TensorBoard.
    writer.add_scalars(
        "Loss/All",
        {
            "loss": train_metrics["loss"],
            "loss_fusion": train_metrics["loss_fusion"],
            "loss_a": train_metrics["loss_a"],
            "loss_v": train_metrics["loss_v"],
            "loss_alignment": train_metrics["loss_alignment"],
        },
        epoch,
    )

    # Individual loss scalars for flexible filtering.
    for key, value in train_metrics.items():
        writer.add_scalar(f"Loss/{key}", value, epoch)

    # Metric groups.
    writer.add_scalars(
        "Accuracy/All",
        {
            "multi": val_metrics["val_acc_multi"],
            "audio": val_metrics["val_acc_audio"],
            "video": val_metrics["val_acc_video"],
        },
        epoch,
    )
    writer.add_scalars(
        "F1/All",
        {
            "multi": val_metrics["val_f1_multi"],
            "audio": val_metrics["val_f1_audio"],
            "video": val_metrics["val_f1_video"],
        },
        epoch,
    )
    writer.add_scalars(
        "mAP/All",
        {
            "multi": val_metrics["val_mAP_multi"],
            "audio": val_metrics["val_mAP_audio"],
            "video": val_metrics["val_mAP_video"],
        },
        epoch,
    )

    # Individual metric scalars.
    for key, value in val_metrics.items():
        writer.add_scalar(f"Metrics/{key}", value, epoch)

    writer.add_scalar("Train/lr", lr, epoch)
    writer.flush()


def update_loss_curve(csv_path, run_dir):
    """Save loss/loss_fusion/loss_a/loss_v/loss_alignment curves into one PNG and one PDF."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return

        plt.figure(figsize=(8.5, 5.2))
        plt.plot(df["epoch"], df["loss"], linewidth=2.0, label="loss")
        plt.plot(df["epoch"], df["loss_fusion"], linewidth=2.0, label="loss_fusion")
        plt.plot(df["epoch"], df["loss_a"], linewidth=2.0, label="loss_a")
        plt.plot(df["epoch"], df["loss_v"], linewidth=2.0, label="loss_v")
        plt.plot(df["epoch"], df["loss_alignment"], linewidth=2.0, label="loss_alignment")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
        plt.savefig(run_dir / "loss_curves.pdf", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[Warning] Failed to update loss curve figure: {e}")


def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio):
    model.train()
    tl = Averager()
    tl_a = Averager()
    tl_v = Averager()
    tl_fusion = Averager()
    tl_alignment = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)
        loss_alignment = Alignment_Feature(f_a, f_v)
        loss_a = criterion(result_a, y).mean()
        loss_v = criterion(result_v, y).mean()
        loss_fusion = criterion(logits_ratio * result_a + logits_ratio * result_v, y).mean()

        loss = loss_a + loss_v + loss_fusion + loss_alignment

        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_a.add(loss_a.item())
        tl_v.add(loss_v.item())
        tl_fusion.add(loss_fusion.item())
        tl_alignment.add(loss_alignment.item())

    train_metrics = {
        "loss": tl.item(),
        "loss_fusion": tl_fusion.item(),
        "loss_a": tl_a.item(),
        "loss_v": tl_v.item(),
        "loss_alignment": tl_alignment.item(),
    }

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: '
         'loss:{loss:.4f}, loss_fusion:{loss_fusion:.4f}, '
         'loss_a:{loss_a:.4f}, loss_v:{loss_v:.4f}, '
         'loss_alignment:{loss_alignment:.4f}').format(
            epoch=epoch,
            **train_metrics,
        )
    )

    return model, train_metrics


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

    val_metrics = {
        "val_f1_multi": float(f1),
        "val_acc_multi": float(acc),
        "val_mAP_multi": float(mAP),
        "val_f1_audio": float(f1_a),
        "val_acc_audio": float(acc_a),
        "val_mAP_audio": float(mAP_a),
        "val_f1_video": float(f1_v),
        "val_acc_video": float(acc_v),
        "val_mAP_video": float(mAP_v),
    }

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        ('Epoch {epoch:d}: '
         'f1:{val_f1_multi:.4f},acc:{val_acc_multi:.4f},mAP:{val_mAP_multi:.4f},'
         'f1_a:{val_f1_audio:.4f},acc_a:{val_acc_audio:.4f},mAP_a:{val_mAP_audio:.4f},'
         'f1_v:{val_f1_video:.4f},acc_v:{val_acc_video:.4f},mAP_v:{val_mAP_video:.4f}').format(
            epoch=epoch,
            **val_metrics,
        )
    )
    return val_metrics


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')

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

    # ----- SET TENSORBOARD / CSV -----
    run_dir = build_run_dir(cfg, exp_id=exp_id)
    writer = SummaryWriter(log_dir=str(run_dir))
    csv_path = run_dir / "metrics.csv"
    csv_header = init_csv(csv_path)
    logger.info(f"TensorBoard and CSV directory: {run_dir}")
    logger.info(f"Metrics CSV path: {csv_path}")

    # ----- SET DATALOADER -----
    # Keep the original data-loading behavior. If your dataset config is only stored in cfg,
    # you can replace `config` with `cfg` in the following two lines.
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

    try:
        for epoch in range(cfg['train']['epoch_dict']):
            logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            model, train_metrics = train_audio_video(
                epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio
            )

            val_metrics = val(epoch, test_loader, model, logger)

            # ----- WRITE CSV AND TENSORBOARD -----
            row = {"epoch": epoch, **train_metrics, **val_metrics, "lr": current_lr}
            append_csv_row(csv_path, csv_header, row)
            write_tensorboard(writer, epoch, train_metrics, val_metrics, current_lr)
            update_loss_curve(csv_path, run_dir)

            acc = val_metrics["val_acc_multi"]
            acc_a = val_metrics["val_acc_audio"]
            acc_v = val_metrics["val_acc_video"]

            m_name = cfg['visual']['name'] + '_' + cfg['text']['name']

            # if epoch % 10 == 0:
            #     torch.save(
            #         model.state_dict(),
            #         f'/root/autodl-tmp/zhengyuhan/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            #     )

            ### TODO:before
            # if acc > best_acc:
            #     best_acc = acc
            #     print('Find a better model and save it!')
            #     logger.info('Find a better model and save it!')
            #     m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
            #     torch.save(model.state_dict(), '/data/lxe/multimodel/NeurIPS24-LFM-main/KS_model/multi_KS_best_model.pth')
    finally:
        writer.close()
        logger.info(f"Finished. TensorBoard files and CSV are saved in: {run_dir}")
