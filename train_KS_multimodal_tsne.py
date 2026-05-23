#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_KS_multimodal_tsne_trainset_8classes.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data.template import config
from dataset.KS import VADataset
from model.AudioVideo import AVClassifier
from utils.utils import create_logger, Averager, deep_update_dict
from utils.tools import weight_init

# -------------------------------
# TSNE helper functions
# -------------------------------
def _soft_pastel_palette(n_classes: int):
    base = plt.cm.Set3(np.linspace(0, 1, max(n_classes, 3)))
    return [to_rgba(c, alpha=0.68) for c in base[:n_classes]]

def _prepare_tsne_features(features: np.ndarray, max_points: int = 1200, seed: int = 42):
    if features.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(features.shape[0], size=max_points, replace=False)
        return features[indices], indices
    return features, None

def _run_tsne(features: np.ndarray, seed: int = 42):
    feats = features.astype(np.float32)
    if feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)
    if feats.shape[1] > 50:
        pca_dim = min(50, feats.shape[1], feats.shape[0])
        feats = PCA(n_components=pca_dim, random_state=seed).fit_transform(feats)
    perplexity = min(30, max(5, (feats.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        random_state=seed,
    )
    return tsne.fit_transform(feats)

def save_epoch_tsne(epoch: int, audio_features: np.ndarray, video_features: np.ndarray, labels: np.ndarray, save_dir: str, selected_classes=None, max_points: int = 1200, seed: int = 42):
    """
    Save t-SNE visualizations using only selected_classes if provided.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if selected_classes is not None:
        mask = np.isin(labels, selected_classes)
        audio_features = audio_features[mask]
        video_features = video_features[mask]
        labels = labels[mask]

    audio_features, audio_idx = _prepare_tsne_features(audio_features, max_points=max_points, seed=seed)
    labels_audio = labels[audio_idx] if audio_idx is not None else labels

    video_features, video_idx = _prepare_tsne_features(video_features, max_points=max_points, seed=seed)
    labels_video = labels[video_idx] if video_idx is not None else labels

    audio_emb = _run_tsne(audio_features, seed=seed)
    video_emb = _run_tsne(video_features, seed=seed)

    n_classes = int(max(labels_audio.max(), labels_video.max())) + 1
    palette = _soft_pastel_palette(n_classes)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 11.0), dpi=220)
    plt.subplots_adjust(hspace=0.20)

    panels = [(axes[0], audio_emb, labels_audio, 'Audio'), (axes[1], video_emb, labels_video, 'Video')]
    for ax, emb, lbs, title in panels:
        ax.set_facecolor('white')
        for cls in selected_classes if selected_classes is not None else range(n_classes):
            mask = lbs == cls
            if np.any(mask):
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=60,
                    c=[palette[cls]],
                    edgecolors='none'
                )
        ax.set_title(title, fontsize=22, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color('black')

    fig.suptitle(f'Epoch {epoch}', fontsize=18, y=0.995)
    fig.savefig(save_dir / f'epoch_{epoch:03d}_tsne.png', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(save_dir / f'epoch_{epoch:03d}_tsne.pdf', bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)

# -------------------------------
# Visualization on training set
# -------------------------------
def visualize_train_tsne(epoch, train_loader, model, tsne_dir, selected_classes, tsne_max_points=1200, tsne_seed=42):
    model.eval()
    feat_a_list = []
    feat_v_list = []
    labels_list = []

    with torch.no_grad():
        for spectrogram, image, y in tqdm(train_loader):
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            image = image.float().cuda()
            y = y.cuda()

            _, _, _, f_a, f_v = model(spectrogram, image)
            feat_a_list.append(f_a.detach().cpu().reshape(f_a.size(0), -1).numpy())
            feat_v_list.append(f_v.detach().cpu().reshape(f_v.size(0), -1).numpy())
            labels_list.append(torch.argmax(y, dim=1).cpu().numpy())

    audio_features = np.concatenate(feat_a_list, axis=0)
    video_features = np.concatenate(feat_v_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    save_epoch_tsne(epoch, audio_features, video_features, labels, tsne_dir, selected_classes=selected_classes, max_points=tsne_max_points, seed=tsne_seed)

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')
    parser.add_argument('--tsne_root', type=str, default='/data/zyh/NeurIPS24-LFM/_tsne/trainset8')
    parser.add_argument('--tsne_max_points', type=int, default=1200)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        exp_params = json.load(f)
    cfg = deep_update_dict(exp_params, config)

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    logger, log_file, exp_id = create_logger(cfg, 0)

    train_dataset = VADataset(cfg, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    tsne_dir = os.path.join(args.tsne_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(tsne_dir, exist_ok=True)
    logger.info(f'TSNE images will be saved to: {tsne_dir}')

    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    optimizer = optim.SGD(model.parameters(), lr=cfg['train']['optimizer']['lr'],
                          momentum=cfg['train']['optimizer']['momentum'],
                          weight_decay=cfg['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, cfg['train']['lr_scheduler']['patience'], 0.1)
    cls_k = []

    selected_classes = list(range(8))  # 仅使用训练集的8个类

    for epoch in range(cfg['train']['epoch_dict']):
        scheduler.step()
        # Train for one epoch
        model.train()
        for spectrogram, image, y in tqdm(train_loader):
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            image = image.float().cuda()
            y = y.cuda()
            optimizer.zero_grad()
            result_b, result_a, result_v, f_a, f_v = model(spectrogram, image)
            loss = F.cross_entropy(result_b, torch.argmax(y, dim=1).cuda())
            loss.backward()
            optimizer.step()

        # Visualize TSNE using training set with 8 classes
        visualize_train_tsne(epoch, train_loader, model, tsne_dir, selected_classes=selected_classes, tsne_max_points=args.tsne_max_points, tsne_seed=cfg['seed'])

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(tsne_dir, f'epoch_{epoch:03d}_model.pth'))