# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# from collections import defaultdict
# import os
# import warnings
# import json
# import numpy as np
# import argparse
# import random
# import re
# from collections import defaultdict

# import matplotlib.pyplot as plt

# import torch
# torch.autograd.set_detect_anomaly(True)
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from torch.nn import functional as F

# from sklearn.metrics import f1_score, average_precision_score
# from tqdm import tqdm

# from utils.min_norm_solvers import MinNormSolver
# from data.template import config
# from dataset.KS import VADataset
# from model.AudioVideo import AVClassifier
# from utils.utils import (
#     create_logger,
#     Averager,
#     deep_update_dict,
# )
# from utils.tools import GSPlugin, weight_init

# warnings.filterwarnings("ignore")


# def compute_mAP(outputs, labels):
#     y_true = labels.cpu().detach().numpy()
#     y_pred = outputs.cpu().detach().numpy()
#     AP = []
#     for i in range(y_true.shape[1]):
#         AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
#     return np.mean(AP)


# def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio):
#     model.train()
#     tl = Averager()
#     tl_a = Averager()
#     tl_v = Averager()
#     criterion = nn.CrossEntropyLoss(reduction='none').cuda()

#     for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
#         image = image.float().cuda()
#         y = y.cuda()
#         spectrogram = spectrogram.unsqueeze(1).float().cuda()
#         optimizer.zero_grad()

#         result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

#         loss_a = criterion(result_a, y).mean()
#         loss_v = criterion(result_v, y).mean()
#         loss_fusion = criterion(logits_ratio * result_a + logits_ratio * result_v, y).mean()

#         loss = loss_a + loss_v + loss_fusion
#         loss.backward()
#         optimizer.step()

#         tl.add(loss.item())
#         tl_a.add(loss_a.item())
#         tl_v.add(loss_v.item())

#     loss_ave = tl.item()
#     loss_audio = tl_a.item()
#     loss_video = tl_v.item()

#     logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     logger.info(
#         ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , '
#          'Average loss_audio : {loss_audio:.3f},Average loss_video : {loss_video:.3f}')
#         .format(epoch=epoch, loss_ave=loss_ave, loss_audio=loss_audio, loss_video=loss_video)
#     )

#     return model


# def val(epoch, val_loader, model, logger):
#     model.eval()
#     pred_list = []
#     pred_list_a = []
#     pred_list_v = []
#     label_list = []
#     soft_pred = []
#     soft_pred_a = []
#     soft_pred_v = []
#     one_hot_label = []

#     with torch.no_grad():
#         for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
#             label_list = label_list + torch.argmax(y, dim=1).tolist()
#             one_hot_label = one_hot_label + y.tolist()

#             image = image.cuda()
#             y = y.cuda()
#             spectrogram = spectrogram.unsqueeze(1).float().cuda()

#             result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)

#             soft_pred_a = soft_pred_a + (F.softmax(result_a, dim=1)).tolist()
#             soft_pred_v = soft_pred_v + (F.softmax(result_v, dim=1)).tolist()
#             soft_pred = soft_pred + (F.softmax(0.5 * result_a + 0.5 * result_v, dim=1)).tolist()

#             pred = (F.softmax(0.5 * result_a + 0.5 * result_v, dim=1)).argmax(dim=1)
#             pred_a = (F.softmax(result_a, dim=1)).argmax(dim=1)
#             pred_v = (F.softmax(result_v, dim=1)).argmax(dim=1)

#             pred_list = pred_list + pred.tolist()
#             pred_list_a = pred_list_a + pred_a.tolist()
#             pred_list_v = pred_list_v + pred_v.tolist()

#         f1 = f1_score(label_list, pred_list, average='macro')
#         f1_a = f1_score(label_list, pred_list_a, average='macro')
#         f1_v = f1_score(label_list, pred_list_v, average='macro')

#         correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
#         correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
#         correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)

#         acc = correct / len(label_list)
#         acc_a = correct_a / len(label_list)
#         acc_v = correct_v / len(label_list)

#         mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
#         mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
#         mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

#     logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     logger.info(
#         ('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},'
#          'f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},'
#          'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}')
#         .format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
#                 f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
#                 f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v)
#     )
#     return acc, acc_a, acc_v


# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)


# def prepare_fixed_batch(batch, device, max_samples=32):
#     """
#     从固定 batch 中取前 max_samples 个样本，用于每个 epoch 统计维度贡献分布。
#     """
#     spectrogram, image, y = batch

#     spectrogram = spectrogram[:max_samples]
#     image = image[:max_samples]
#     y = y[:max_samples]

#     if y.dim() > 1:
#         labels = torch.argmax(y, dim=1).long()
#     else:
#         labels = y.long()

#     spectrogram = spectrogram.unsqueeze(1).float().to(device)
#     image = image.float().to(device)
#     labels = labels.to(device)

#     return spectrogram, image, labels


# @torch.no_grad()
# def extract_unimodal_features(model, spectrogram, image):
#     """
#     直接提取 audio / video 特征，不依赖 model.forward 返回值结构。
#     """
#     a_feature = model.audio_encoder(spectrogram)   # [B, D]
#     v_feature = model.video_encoder(image)         # [B, D]
#     return a_feature, v_feature


# @torch.no_grad()
# def approximate_shapley_contribution(features, labels, classifier, num_perm=32):
#     """
#     用 permutation sampling 近似每一维特征的 Shapley-style contribution。

#     features:   [B, D]
#     labels:     [B]
#     classifier: nn.Linear(D, C)
#     returns:    [D]
#     """
#     device = features.device
#     B, D = features.shape

#     weight = classifier.weight.detach()   # [C, D]
#     bias = classifier.bias.detach()       # [C]
#     sample_idx = torch.arange(B, device=device)

#     contrib = torch.zeros(D, device=device)

#     for _ in range(num_perm):
#         perm = torch.randperm(D, device=device)

#         # 空子集仅保留 bias
#         current_logits = bias.unsqueeze(0).repeat(B, 1)   # [B, C]
#         prev_score = F.softmax(current_logits, dim=1)[sample_idx, labels]  # [B]

#         for d in perm:
#             current_logits = current_logits + features[:, d].unsqueeze(1) * weight[:, d].unsqueeze(0)
#             current_score = F.softmax(current_logits, dim=1)[sample_idx, labels]  # [B]

#             marginal_gain = (current_score - prev_score).mean()
#             contrib[d] += marginal_gain
#             prev_score = current_score

#     contrib = contrib / float(num_perm)
#     return contrib.detach().cpu().numpy()


# def plot_contribution_distribution(audio_contrib, video_contrib, epoch, save_dir):
#     ensure_dir(save_dir)

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))

#     axes[0].hist(audio_contrib, bins=40, alpha=0.85)
#     axes[0].axvline(np.mean(audio_contrib), linestyle='--', linewidth=1.5,
#                     label=f"mean={np.mean(audio_contrib):.4e}")
#     axes[0].set_title(f'Audio contribution distribution (epoch={epoch})')
#     axes[0].set_xlabel('Contribution')
#     axes[0].set_ylabel('Count')
#     axes[0].legend()

#     axes[1].hist(video_contrib, bins=40, alpha=0.85)
#     axes[1].axvline(np.mean(video_contrib), linestyle='--', linewidth=1.5,
#                     label=f"mean={np.mean(video_contrib):.4e}")
#     axes[1].set_title(f'Video contribution distribution (epoch={epoch})')
#     axes[1].set_xlabel('Contribution')
#     axes[1].set_ylabel('Count')
#     axes[1].legend()

#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_contribution_distribution.png')
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     plt.close(fig)

#     np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_audio_contrib.npy'), audio_contrib)
#     np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_video_contrib.npy'), video_contrib)

#     return save_path


# def save_epoch_feature_contribution_distribution(epoch,
#                                                  model,
#                                                  fixed_batch,
#                                                  save_dir,
#                                                  num_perm=32,
#                                                  max_samples=32,
#                                                  logger=None):
#     """
#     每个 epoch 保存一次 audio / video 各维度 contribution 分布。
#     """
#     was_training = model.training
#     model.eval()
#     device = next(model.parameters()).device

#     try:
#         spectrogram, image, labels = prepare_fixed_batch(
#             fixed_batch, device=device, max_samples=max_samples
#         )

#         a_feature, v_feature = extract_unimodal_features(model, spectrogram, image)

#         audio_contrib = approximate_shapley_contribution(
#             a_feature, labels, model.cls_a, num_perm=num_perm
#         )
#         video_contrib = approximate_shapley_contribution(
#             v_feature, labels, model.cls_v, num_perm=num_perm
#         )

#         save_path = plot_contribution_distribution(
#             audio_contrib, video_contrib, epoch, save_dir
#         )

#         if logger is not None:
#             logger.info(f'[Contribution] Saved distribution figure to: {save_path}')
#             logger.info(f'[Contribution] Audio mean={audio_contrib.mean():.6e}, std={audio_contrib.std():.6e}')
#             logger.info(f'[Contribution] Video mean={video_contrib.mean():.6e}, std={video_contrib.std():.6e}')

#     except Exception as e:
#         if logger is not None:
#             logger.info(f'[Contribution] Failed at epoch {epoch}: {e}')
#         else:
#             print(f'[Contribution] Failed at epoch {epoch}: {e}')
#     finally:
#         if was_training:
#             model.train()


# if __name__ == '__main__':
#     # ----- LOAD PARAM -----
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str,
#                         default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
#     parser.add_argument('--contrib_save_dir', type=str,
#                         default='/data/zyh/NeurIPS24-LFM/_figure/visualization_shapley',
#                         help='Directory to save feature contribution distribution figures.')
#     parser.add_argument('--contrib_num_perm', type=int, default=32,
#                         help='Number of random permutations for Shapley approximation.')
#     parser.add_argument('--contrib_num_samples', type=int, default=32,
#                         help='Number of samples from the fixed batch for contribution estimation.')

#     args = parser.parse_args()
#     cfg = config

#     with open(args.config, "r") as f:
#         exp_params = json.load(f)

#     cfg = deep_update_dict(exp_params, cfg)

#     # ----- SET SEED -----
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed_all(cfg['seed'])
#     random.seed(cfg['seed'])
#     np.random.seed(cfg['seed'])
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

#     # ----- SET LOGGER -----
#     local_rank = cfg['train']['local_rank']
#     logits_ratio = cfg['train']['logits_ratio']
#     logger, log_file, exp_id = create_logger(cfg, local_rank)

#     # ----- SET DATALOADER -----
#     train_dataset = VADataset(config, mode='train')
#     test_dataset = VADataset(config, mode='test')

#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=cfg['train']['batch_size'],
#         shuffle=True,
#         num_workers=cfg['train']['num_workers'],
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=cfg['test']['batch_size'],
#         shuffle=False,
#         num_workers=cfg['test']['num_workers'],
#         pin_memory=True
#     )

#     # 固定一个 batch，用来每个 epoch 统计 contribution 分布
#     fixed_batch = next(iter(train_loader))

#     # ----- MODEL -----
#     model = AVClassifier(config=cfg)
#     model = model.cuda()
#     model.apply(weight_init)

#     lr_adjust = config['train']['optimizer']['lr']

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=lr_adjust,
#         momentum=config['train']['optimizer']['momentum'],
#         weight_decay=config['train']['optimizer']['wc']
#     )

#     scheduler = optim.lr_scheduler.StepLR(
#         optimizer,
#         config['train']['lr_scheduler']['patience'],
#         0.1
#     )

#     best_acc = 0
#     cls_k = []

#     ensure_dir(args.contrib_save_dir)

#     for epoch in range(cfg['train']['epoch_dict']):
#         logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

#         scheduler.step()
#         model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

#         acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

#         # 每个 epoch 统计一次 feature contribution distribution
#         save_epoch_feature_contribution_distribution(
#             epoch=epoch,
#             model=model,
#             fixed_batch=fixed_batch,
#             save_dir=args.contrib_save_dir,
#             num_perm=args.contrib_num_perm,
#             max_samples=args.contrib_num_samples,
#             logger=logger,
#         )

#         m_name = cfg['visual']['name'] + '_' + cfg['text']['name']

#         if epoch % 10 == 0:
#             torch.save(
#                 model.state_dict(),
#                 f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
#             )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
import json
import numpy as np
import argparse
import random

import matplotlib.pyplot as plt

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
        ('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , '
         'Average loss_audio : {loss_audio:.3f},Average loss_video : {loss_video:.3f}')
        .format(epoch=epoch, loss_ave=loss_ave, loss_audio=loss_audio, loss_video=loss_video)
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
        ('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},'
         'f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},'
         'f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}')
        .format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v)
    )
    return acc, acc_a, acc_v


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_fixed_batch(batch, device, max_samples=32):
    """
    从固定 batch 中取前 max_samples 个样本，用于每个 epoch 统计维度贡献。
    """
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
def extract_unimodal_features(model, spectrogram, image):
    """
    直接提取 audio / video 特征，不依赖 model.forward 返回值结构。
    """
    a_feature = model.audio_encoder(spectrogram)   # [B, D]
    v_feature = model.video_encoder(image)         # [B, D]
    return a_feature, v_feature


@torch.no_grad()
def approximate_shapley_contribution(features, labels, classifier, num_perm=32):
    """
    用 permutation sampling 近似每一维特征的 Shapley-style contribution。

    features:   [B, D]
    labels:     [B]
    classifier: nn.Linear(D, C)
    returns:    [D]
    """
    device = features.device
    B, D = features.shape

    weight = classifier.weight.detach()   # [C, D]
    bias = classifier.bias.detach()       # [C]
    sample_idx = torch.arange(B, device=device)

    contrib = torch.zeros(D, device=device)

    for _ in range(num_perm):
        perm = torch.randperm(D, device=device)

        # 空子集仅保留 bias
        current_logits = bias.unsqueeze(0).repeat(B, 1)   # [B, C]
        prev_score = F.softmax(current_logits, dim=1)[sample_idx, labels]  # [B]

        for d in perm:
            current_logits = current_logits + features[:, d].unsqueeze(1) * weight[:, d].unsqueeze(0)
            current_score = F.softmax(current_logits, dim=1)[sample_idx, labels]  # [B]

            marginal_gain = (current_score - prev_score).mean()
            contrib[d] += marginal_gain
            prev_score = current_score

    contrib = contrib / float(num_perm)
    return contrib.detach().cpu().numpy()


def plot_contribution_learning_progress(audio_history,
                                        video_history,
                                        epoch,
                                        save_dir,
                                        active_thresh=0.0):
    """
    新的可视化方式：
    1) Audio / Video 维度贡献热力图（横轴epoch，纵轴维度）
    2) Active dimension ratio 曲线
    """
    ensure_dir(save_dir)

    audio_hist = np.asarray(audio_history)   # [E, D]
    video_hist = np.asarray(video_history)   # [E, D]

    # 按当前最后一个 epoch 的贡献从大到小排序，便于看“哪些维度后来学好了”
    audio_order = np.argsort(audio_hist[-1])[::-1]
    video_order = np.argsort(video_hist[-1])[::-1]

    audio_map = audio_hist[:, audio_order].T   # [D, E]
    video_map = video_hist[:, video_order].T   # [D, E]

    # 统一颜色范围，保证两个模态可比较
    vmax = max(
        np.percentile(np.abs(audio_map), 98),
        np.percentile(np.abs(video_map), 98),
        1e-8
    )
    vmin = -vmax

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.2], hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    im1 = ax1.imshow(audio_map, aspect='auto', origin='lower', cmap='coolwarm',
                     vmin=vmin, vmax=vmax)
    ax1.set_title(f'Audio dimension contribution map (epoch={epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dimension rank')

    im2 = ax2.imshow(video_map, aspect='auto', origin='lower', cmap='coolwarm',
                     vmin=vmin, vmax=vmax)
    ax2.set_title(f'Video dimension contribution map (epoch={epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dimension rank')

    # 公共 colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.025, pad=0.02)
    cbar.set_label('Contribution')

    epochs = np.arange(audio_hist.shape[0])
    audio_active_ratio = (audio_hist > active_thresh).mean(axis=1)
    video_active_ratio = (video_hist > active_thresh).mean(axis=1)

    ax3.plot(epochs, audio_active_ratio, marker='o', linewidth=2, label='Audio')
    ax3.plot(epochs, video_active_ratio, marker='s', linewidth=2, label='Video')
    ax3.set_title('Ratio of positively contributing dimensions')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Active-dim ratio')
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(alpha=0.3)
    ax3.legend()

    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_contribution_progress.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

    return save_path


def save_epoch_feature_contribution_progress(epoch,
                                             model,
                                             fixed_batch,
                                             save_dir,
                                             audio_history,
                                             video_history,
                                             num_perm=32,
                                             max_samples=32,
                                             active_thresh=0.0,
                                             logger=None):
    """
    每个 epoch：
    1) 计算当前 audio / video 各维度 contribution
    2) 追加到历史
    3) 画“维度逐渐学好”的进展图
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    try:
        spectrogram, image, labels = prepare_fixed_batch(
            fixed_batch, device=device, max_samples=max_samples
        )

        a_feature, v_feature = extract_unimodal_features(model, spectrogram, image)

        audio_contrib = approximate_shapley_contribution(
            a_feature, labels, model.cls_a, num_perm=num_perm
        )
        video_contrib = approximate_shapley_contribution(
            v_feature, labels, model.cls_v, num_perm=num_perm
        )

        audio_history.append(audio_contrib)
        video_history.append(video_contrib)

        # 保存当前 epoch 的 contribution 向量
        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_audio_contrib.npy'), audio_contrib)
        np.save(os.path.join(save_dir, f'epoch_{epoch:03d}_video_contrib.npy'), video_contrib)

        # 保存整个历史
        np.save(os.path.join(save_dir, 'audio_contrib_history.npy'), np.asarray(audio_history))
        np.save(os.path.join(save_dir, 'video_contrib_history.npy'), np.asarray(video_history))

        save_path = plot_contribution_learning_progress(
            audio_history=audio_history,
            video_history=video_history,
            epoch=epoch,
            save_dir=save_dir,
            active_thresh=active_thresh,
        )

        if logger is not None:
            logger.info(f'[ContributionProgress] Saved figure to: {save_path}')
            logger.info(
                f'[ContributionProgress] Audio mean={audio_contrib.mean():.6e}, '
                f'active_ratio={(audio_contrib > active_thresh).mean():.4f}'
            )
            logger.info(
                f'[ContributionProgress] Video mean={video_contrib.mean():.6e}, '
                f'active_ratio={(video_contrib > active_thresh).mean():.4f}'
            )

    except Exception as e:
        if logger is not None:
            logger.info(f'[ContributionProgress] Failed at epoch {epoch}: {e}')
        else:
            print(f'[ContributionProgress] Failed at epoch {epoch}: {e}')
    finally:
        if was_training:
            model.train()

    return audio_history, video_history


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--contrib_save_dir', type=str,
                        default='/data/zyh/NeurIPS24-LFM/_figure/visualization_shapley_progress',
                        help='Directory to save feature contribution progress figures.')
    parser.add_argument('--contrib_num_perm', type=int, default=32,
                        help='Number of random permutations for Shapley approximation.')
    parser.add_argument('--contrib_num_samples', type=int, default=32,
                        help='Number of samples from the fixed batch for contribution estimation.')
    parser.add_argument('--active_thresh', type=float, default=0.0,
                        help='Threshold for defining whether a dimension is active / learned.')

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
        dataset=train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True
    )

    # 固定一个 batch，用来每个 epoch 统计 contribution
    fixed_batch = next(iter(train_loader))

    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr_adjust,
        momentum=config['train']['optimizer']['momentum'],
        weight_decay=config['train']['optimizer']['wc']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        config['train']['lr_scheduler']['patience'],
        0.1
    )

    best_acc = 0
    cls_k = []

    ensure_dir(args.contrib_save_dir)

    audio_history = []
    video_history = []

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        # 每个 epoch 统计一次 feature contribution progress
        audio_history, video_history = save_epoch_feature_contribution_progress(
            epoch=epoch,
            model=model,
            fixed_batch=fixed_batch,
            save_dir=args.contrib_save_dir,
            audio_history=audio_history,
            video_history=video_history,
            num_perm=args.contrib_num_perm,
            max_samples=args.contrib_num_samples,
            active_thresh=args.active_thresh,
            logger=logger,
        )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )