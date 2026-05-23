#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import warnings
from PIL import Image
import json
import numpy as np
import argparse
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from matplotlib import cm
from PIL import Image
import numpy as np
from utils.min_norm_solvers import MinNormSolver
from tqdm import tqdm
warnings.filterwarnings("ignore")

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


# =========================
# Visualization: Grad-CAM
# =========================
def get_last_visual_layer(model):
    """
    默认取 visual ResNet 的最后一个 block 作为 Grad-CAM 层
    """
    if hasattr(model, "video_encoder") and hasattr(model.video_encoder, "video_net"):
        video_net = model.video_encoder.video_net
        if hasattr(video_net, "layer4"):
            return video_net.layer4[-1]
    raise AttributeError("Cannot find target visual layer for Grad-CAM. Please check model.video_encoder.video_net.layer4")


class VisualGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        def _save_grad(grad):
            self.gradients = grad

        output.register_hook(_save_grad)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

    @torch.enable_grad()
    def generate(self, audio_tensor, video_tensor, class_idx=-1):
        """
        audio_tensor: [1, 1, Ha, Wa] 或与训练时一致的单样本音频shape
        video_tensor: [1, 3, T, H, W]
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(audio_tensor, video_tensor)
        logits_v = extract_visual_logits_from_outputs(outputs)

        if logits_v.dim() != 2:
            raise RuntimeError(f"Visual logits shape is invalid: {logits_v.shape}")

        if class_idx is None or class_idx < 0:
            class_idx = int(logits_v.argmax(dim=1).item())

        score = logits_v[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")

        acts = self.activations
        grads = self.gradients

        if acts.dim() != 4 or grads.dim() != 4:
            raise RuntimeError(f"Expected 4D activations/grads, got acts={acts.shape}, grads={grads.shape}")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.mean(dim=0, keepdim=True)
        cam = F.interpolate(
            cam,
            size=video_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        pred_prob = torch.softmax(logits_v, dim=1)[0, class_idx].item()

        if was_training:
            self.model.train()

        return cam, class_idx, pred_prob


def load_visual_image_as_video_tensor(
    img_path,
    input_h,
    input_w,
    fps,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    把单张图复制成一个“伪视频”，形状必须是 [1, 3, T, H, W]
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((input_w, input_h), Image.BILINEAR)

    orig_np = np.asarray(img).astype(np.float32) / 255.0

    tensor = TF.to_tensor(img)                      # [3, H, W]
    tensor = TF.normalize(tensor, mean=mean, std=std)

    # 正确做法：扩成 [1, 3, 1, H, W]，再沿 T 维复制
    video_tensor = tensor.unsqueeze(0).unsqueeze(2)   # [1, 3, 1, H, W]
    video_tensor = video_tensor.repeat(1, 1, fps, 1, 1)  # [1, 3, fps, H, W]
    video_tensor = video_tensor.cuda(non_blocking=True)

    return orig_np, video_tensor


# def save_visualization_figure(orig_np, cam, save_path, epoch, pred_cls, pred_prob, alpha=0.45):
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#     axes[0].imshow(orig_np)
#     axes[0].set_title("Input", fontsize=13)
#     axes[0].axis("off")

#     axes[1].imshow(orig_np)
#     axes[1].imshow(cam, cmap="jet", alpha=alpha)
#     axes[1].set_title(f"epoch = {epoch} | cls = {pred_cls} | p = {pred_prob:.3f}", fontsize=13)
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)

def save_visualization_figure(orig_np, cam, save_path, epoch=None, pred_cls=None, pred_prob=None, alpha=0.45):
    """
    只保存热力图叠加后的结果图，不拼接原图，不加边框、不加标题
    orig_np: [H, W, 3], 范围 [0,1]
    cam:     [H, W], 范围 [0,1]
    """
    heatmap = cm.jet(cam)[..., :3].astype(np.float32)   # [H, W, 3], 去掉 alpha 通道
    overlay = (1 - alpha) * orig_np + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    out = (overlay * 255).astype(np.uint8)
    Image.fromarray(out).save(save_path)

def extract_visual_logits_from_outputs(outputs):
    """
    兼容不同 forward 返回格式，目标是拿到 visual branch 的 logits
    常见情况：
      - 5个返回值: result_b, result_a, result_v, f_a, f_v
      - 4个返回值: result_b, result_a, result_v, xxx
      - 3个返回值: result_b, result_a, result_v
      - 2个返回值: 默认取最后一个
      - 单个Tensor: 直接返回
    """
    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 3:
            return outputs[2]
        elif len(outputs) == 2:
            return outputs[-1]
        elif len(outputs) == 1:
            return outputs[0]
        else:
            raise RuntimeError("Model outputs is empty.")
    elif torch.is_tensor(outputs):
        return outputs
    else:
        raise RuntimeError(f"Unsupported model outputs type: {type(outputs)}")

def run_visualization(
    epoch,
    model,
    vis_img_path,
    vis_out_dir,
    fps,
    input_h,
    input_w,
    audio_shape,
    logger=None,
    target_class=-1,
    alpha=0.45,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    os.makedirs(vis_out_dir, exist_ok=True)

    orig_np, video_tensor = load_visual_image_as_video_tensor(
        vis_img_path,
        input_h=input_h,
        input_w=input_w,
        fps=fps,
        mean=mean,
        std=std,
    )

    # 构造一个与训练时一致shape的 dummy audio
    dummy_audio = torch.zeros(audio_shape, dtype=torch.float32).cuda(non_blocking=True)

    # orig_save_path = os.path.join(vis_out_dir, "input_image.png")
    # if not os.path.exists(orig_save_path):
    #     Image.fromarray((orig_np * 255).astype(np.uint8)).save(orig_save_path)

    target_layer = get_last_visual_layer(model)
    cam_generator = VisualGradCAM(model, target_layer)

    try:
        cam, pred_cls, pred_prob = cam_generator.generate(
            audio_tensor=dummy_audio,
            video_tensor=video_tensor,
            class_idx=target_class
        )
        save_path = os.path.join(vis_out_dir, f"epoch_{epoch:03d}.png")
        save_visualization_figure(
            orig_np=orig_np,
            cam=cam,
            save_path=save_path,
            epoch=epoch,
            pred_cls=pred_cls,
            pred_prob=pred_prob,
            alpha=alpha,
        )
        msg = f"[Visualization] Saved: {save_path}"
        print(msg)
        if logger is not None:
            logger.info(msg)
    finally:
        cam_generator.remove()


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
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , Average loss_audio : {loss_audio:.3f},Average loss_video : \
                 {loss_video:.3f}').format(epoch=epoch, loss_ave=loss_ave, loss_audio=loss_audio, loss_video=loss_video))

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

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(
        epoch=epoch, f1=f1, acc=acc, mAP=mAP,
        f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
        f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))

    return acc, acc_a, acc_v


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/zyh/NeurIPS24-LFM/data/kinetics_sound.json')

    # visualization args
    parser.add_argument('--vis_img_path', type=str, default='/data/zyh/NeurIPS24-LFM/_figure/sample/violin.jpg',
                        help='指定要做可视化的那张图片路径')
    parser.add_argument('--vis_out_dir', type=str, default='/data/zyh/NeurIPS24-LFM/_figure/visualization_worse',
                        help='可视化结果保存目录')
    parser.add_argument('--vis_every', type=int, default=1,
                        help='每隔多少个 epoch 做一次可视化，默认每个 epoch 都做')
    parser.add_argument('--vis_target_class', type=int, default=-1,
                        help='指定可视化类别；-1 表示使用当前预测类别')
    parser.add_argument('--vis_alpha', type=float, default=0.45,
                        help='热力图叠加透明度')
    parser.add_argument('--vis_mean', nargs=3, type=float, default=[0.485, 0.456, 0.406],
                        help='图像归一化 mean')
    parser.add_argument('--vis_std', nargs=3, type=float, default=[0.229, 0.224, 0.225],
                        help='图像归一化 std')

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

    # 用一个 batch 来拿到网络真实输入尺寸
    val_batch = next(iter(train_loader))
    sample_spec = val_batch[0]
    sample_image = val_batch[1]

    vis_input_h, vis_input_w = sample_image.shape[-2], sample_image.shape[-1]

    # 根据训练时真实spectrogram shape构造单样本 audio shape
    # 训练里是: spectrogram = spectrogram.unsqueeze(1).float().cuda()
    # 所以这里单样本最终shape应为 [1, 1, H, W]
    if sample_spec.dim() == 3:
        vis_audio_shape = (1, 1, sample_spec.shape[-2], sample_spec.shape[-1])
    elif sample_spec.dim() == 4:
        vis_audio_shape = (1, sample_spec.shape[1], sample_spec.shape[-2], sample_spec.shape[-1])
    else:
        raise RuntimeError(f"Unsupported spectrogram shape: {sample_spec.shape}")

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

    os.makedirs(args.vis_out_dir, exist_ok=True)

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k, logits_ratio)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        # 每个 epoch 保存一次可视化
        if args.vis_img_path and (epoch % args.vis_every == 0):
            try:
                run_visualization(
                    epoch=epoch,
                    model=model,
                    vis_img_path=args.vis_img_path,
                    vis_out_dir=args.vis_out_dir,
                    fps=cfg['fps'],
                    input_h=vis_input_h,
                    input_w=vis_input_w,
                    audio_shape=vis_audio_shape,
                    logger=logger,
                    target_class=args.vis_target_class,
                    alpha=args.vis_alpha,
                    mean=tuple(args.vis_mean),
                    std=tuple(args.vis_std),
                )
            except Exception as e:
                msg = f"[Visualization] Failed at epoch {epoch}: {str(e)}"
                print(msg)
                print(traceback.format_exc())
                logger.info(msg)
                logger.info(traceback.format_exc())

        m_name = cfg['visual']['name'] + '_' + cfg['text']['name']

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f'/data/zyh/NeurIPS24-LFM/_bestmodel_all_dataset/ks/multi_KS_best_model_{epoch}_{acc}_{acc_a}_{acc_v}.pth'
            )

        ### TODO:before
        # if acc > best_acc:
        #     best_acc = acc
        #     print('Find a better model and save it!')
        #     logger.info('Find a better model and save it!')
        #     m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
        #     torch.save(model.state_dict(), '/data/lxe/multimodel/NeurIPS24-LFM-main/KS_model/multi_KS_best_model.pth')