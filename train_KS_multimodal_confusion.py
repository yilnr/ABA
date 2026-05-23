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
# from collections import defaultdict
from collections import defaultdict, Counter
# from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


# def plot_confusion_matrix_epoch(y_true,
#                                 y_pred,
#                                 epoch,
#                                 save_dir='/data/zyh/NeurIPS24-LFM/_figure/confusion_matrix',
#                                 num_classes=None,
#                                 normalize=False):
#     """
#     Save confusion matrix for each epoch.
#     y_true: list[int]
#     y_pred: list[int]
#     normalize: if True, row-normalized confusion matrix; otherwise raw counts.
#     """
#     ensure_dir(save_dir)

#     if num_classes is None:
#         num_classes = max(max(y_true), max(y_pred)) + 1

#     labels = list(range(num_classes))
#     cm = confusion_matrix(y_true, y_pred, labels=labels)

#     if normalize:
#         cm_show = cm.astype(np.float32)
#         row_sum = cm_show.sum(axis=1, keepdims=True) + 1e-8
#         cm_show = cm_show / row_sum * 100.0
#     else:
#         cm_show = cm

#     fig, ax = plt.subplots(figsize=(5.2, 4.6))

#     im = ax.imshow(cm_show, interpolation='nearest', cmap='Blues')
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_title(f'Confusion Matrix (epoch={epoch})', fontsize=12)
#     ax.set_xlabel('Predicted label', fontsize=10)
#     ax.set_ylabel('True label', fontsize=10)

#     ax.set_xticks(np.arange(num_classes))
#     ax.set_yticks(np.arange(num_classes))
#     ax.set_xticklabels(labels, fontsize=8)
#     ax.set_yticklabels(labels, fontsize=8)

#     # 数字标注
#     threshold = cm_show.max() * 0.55 if cm_show.max() > 0 else 0
#     for i in range(num_classes):
#         for j in range(num_classes):
#             value = cm_show[i, j]
#             if normalize:
#                 text = f'{value:.1f}'
#             else:
#                 text = f'{int(value)}'
#             ax.text(
#                 j, i, text,
#                 ha='center',
#                 va='center',
#                 fontsize=7,
#                 color='white' if value > threshold else 'black'
#             )

#     plt.tight_layout()

#     suffix = 'norm' if normalize else 'count'
#     png_path = os.path.join(save_dir, f'epoch_{epoch:03d}_confusion_matrix_{suffix}.png')
#     pdf_path = os.path.join(save_dir, f'epoch_{epoch:03d}_confusion_matrix_{suffix}.pdf')

#     plt.savefig(png_path, dpi=300, bbox_inches='tight')
#     plt.savefig(pdf_path, bbox_inches='tight')
#     plt.close(fig)

#     return png_path, pdf_path
def select_classes_for_cm(y_true,
                          y_pred,
                          num_classes=None,
                          cm_num_classes=None,
                          cm_class_ids=None,
                          cm_sample_mode='random',
                          cm_sample_seed=0):
    """
    Select a subset of classes for confusion matrix visualization.

    cm_class_ids:
        Explicit class ids, e.g. "0,3,5,8".
        If provided, this has higher priority than cm_num_classes.

    cm_num_classes:
        Number of classes to visualize.

    cm_sample_mode:
        random: randomly sample classes from classes appearing in y_true.
        frequent: select classes with the most samples in y_true.
        first: select the first K classes.
    """
    if num_classes is None:
        all_classes = sorted(list(set(y_true) | set(y_pred)))
    else:
        all_classes = list(range(num_classes))

    present_classes = sorted(list(set(y_true)))

    if cm_class_ids is not None and cm_class_ids.strip() != "":
        selected_classes = [int(x) for x in cm_class_ids.split(",")]
        selected_classes = [c for c in selected_classes if c in all_classes]
        return selected_classes

    if cm_num_classes is None or cm_num_classes <= 0 or cm_num_classes >= len(present_classes):
        return present_classes

    if cm_sample_mode == 'random':
        rng = np.random.default_rng(cm_sample_seed)
        selected_classes = rng.choice(
            present_classes,
            size=cm_num_classes,
            replace=False
        ).tolist()
        selected_classes = sorted(selected_classes)

    elif cm_sample_mode == 'frequent':
        counter = Counter(y_true)
        selected_classes = [c for c, _ in counter.most_common(cm_num_classes)]
        selected_classes = sorted(selected_classes)

    elif cm_sample_mode == 'first':
        selected_classes = present_classes[:cm_num_classes]

    else:
        raise ValueError(f"Unknown cm_sample_mode: {cm_sample_mode}")

    return selected_classes


def plot_confusion_matrix_epoch(y_true,
                                y_pred,
                                epoch,
                                save_dir='/data/zyh/NeurIPS24-LFM/_figure/confusion_matrix',
                                num_classes=None,
                                normalize=False,
                                cm_num_classes=None,
                                cm_class_ids=None,
                                cm_sample_mode='random',
                                cm_sample_seed=0,
                                include_other=True):
    """
    Save confusion matrix for each epoch.

    y_true: list[int]
    y_pred: list[int]
    normalize:
        If True, row-normalized confusion matrix.
    cm_num_classes:
        Number of classes sampled for visualization.
    cm_class_ids:
        Explicit class ids, e.g. "0,3,5,8".
    include_other:
        If True, predictions outside selected classes are counted into an "Other" column.
        This avoids hiding errors where selected classes are misclassified as unselected classes.
    """
    ensure_dir(save_dir)

    selected_classes = select_classes_for_cm(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        cm_num_classes=cm_num_classes,
        cm_class_ids=cm_class_ids,
        cm_sample_mode=cm_sample_mode,
        cm_sample_seed=cm_sample_seed
    )

    if len(selected_classes) == 0:
        raise ValueError("No classes selected for confusion matrix visualization.")

    selected_set = set(selected_classes)
    class_to_row = {c: i for i, c in enumerate(selected_classes)}
    class_to_col = {c: i for i, c in enumerate(selected_classes)}

    if include_other:
        # Rows: selected true classes.
        # Columns: selected predicted classes + Other.
        cm = np.zeros((len(selected_classes), len(selected_classes) + 1), dtype=np.int64)

        for t, p in zip(y_true, y_pred):
            if t not in selected_set:
                continue

            row = class_to_row[t]
            if p in selected_set:
                col = class_to_col[p]
            else:
                col = len(selected_classes)  # Other column
            cm[row, col] += 1

        x_labels = [str(c) for c in selected_classes] + ['Other']
        y_labels = [str(c) for c in selected_classes]

    else:
        # Standard square confusion matrix.
        # Note: predictions outside selected classes will be ignored by sklearn.
        cm = confusion_matrix(y_true, y_pred, labels=selected_classes)
        x_labels = [str(c) for c in selected_classes]
        y_labels = [str(c) for c in selected_classes]

    if normalize:
        cm_show = cm.astype(np.float32)
        row_sum = cm_show.sum(axis=1, keepdims=True) + 1e-8
        cm_show = cm_show / row_sum * 100.0
    else:
        cm_show = cm

    n_rows, n_cols = cm_show.shape
    fig_w = max(5.2, 0.45 * n_cols)
    fig_h = max(4.6, 0.45 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm_show, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f'Confusion Matrix (epoch={epoch})', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, fontsize=8)

    threshold = cm_show.max() * 0.55 if cm_show.max() > 0 else 0
    for i in range(n_rows):
        for j in range(n_cols):
            value = cm_show[i, j]
            if normalize:
                text = f'{value:.1f}'
            else:
                text = f'{int(value)}'

            ax.text(
                j, i, text,
                ha='center',
                va='center',
                fontsize=7,
                color='white' if value > threshold else 'black'
            )

    plt.tight_layout()

    suffix = 'norm' if normalize else 'count'
    class_suffix = f'{len(selected_classes)}cls'
    png_path = os.path.join(save_dir, f'epoch_{epoch:03d}_confusion_matrix_{class_suffix}_{suffix}.png')
    pdf_path = os.path.join(save_dir, f'epoch_{epoch:03d}_confusion_matrix_{class_suffix}_{suffix}.pdf')

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


def val(epoch,
        val_loader,
        model,
        logger,
        save_cm_dir=None,
        num_classes=None,
        cm_num_classes=None,
        cm_class_ids=None,
        cm_sample_mode='random',
        cm_sample_seed=0,
        cm_normalize=False):

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

    if save_cm_dir is not None:
        cm_png, cm_pdf = plot_confusion_matrix_epoch(
            y_true=label_list,
            y_pred=pred_list,
            epoch=epoch,
            save_dir=save_cm_dir,
            num_classes=num_classes,
            normalize=cm_normalize,
            cm_num_classes=cm_num_classes,
            cm_class_ids=cm_class_ids,
            cm_sample_mode=cm_sample_mode,
            cm_sample_seed=cm_sample_seed,
            include_other=True
        )
        logger.info(f'[ConfusionMatrix] Saved to: {cm_png}')
        logger.info(f'[ConfusionMatrix] Saved to: {cm_pdf}')

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
    '--cm_save_dir',
    type=str,
    default='/data/zyh/NeurIPS24-LFM/_figure/confusion_matrix_12',
    help='Directory to save confusion matrix figures.'
    )
    parser.add_argument(
    '--cm_num_classes',
    type=int,
    default=12,
    help='Number of classes sampled for confusion matrix visualization. Set <=0 to use all present classes.'
)

    parser.add_argument(
        '--cm_class_ids',
        type=str,
        default='',
        help='Explicit class ids for confusion matrix, e.g., "0,3,5,8". If set, this overrides --cm_num_classes.'
    )

    parser.add_argument(
        '--cm_sample_mode',
        type=str,
        default='random',
        choices=['random', 'frequent', 'first'],
        help='How to select classes when --cm_class_ids is not set.'
    )

    parser.add_argument(
        '--cm_sample_seed',
        type=int,
        default=0,
        help='Random seed for selecting confusion matrix classes.'
    )

    parser.add_argument(
        '--cm_normalize',
        action='store_true',
        help='Use row-normalized confusion matrix.'
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
            save_cm_dir=args.cm_save_dir,
            num_classes=cfg['setting']['num_class'],
            cm_num_classes=args.cm_num_classes,
            cm_class_ids=args.cm_class_ids,
            cm_sample_mode=args.cm_sample_mode,
            cm_sample_seed=args.cm_sample_seed,
            cm_normalize=args.cm_normalize
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