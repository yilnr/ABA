import torch
import torch.nn as nn
import torch.nn.functional as F
from .Resnet import resnet18


class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')

    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512


        self.cls_a = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        self.cls_v = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        # self.cls_b = nn.Linear(self.hidden_dim * 2 , config['setting']['num_class'])

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)

        result_a = self.cls_a(a_feature)
        result_v = self.cls_v(v_feature)
        # result_b = self.cls_b(torch.cat((a_feature, v_feature), dim=1))
        result_b = result_v + result_a

        return result_b, result_a, result_v, a_feature, v_feature

    def getFeature(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature
