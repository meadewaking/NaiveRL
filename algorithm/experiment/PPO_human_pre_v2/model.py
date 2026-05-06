import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class AtariToImageNet(nn.Module):
    def __init__(self, in_channels):
        super(AtariToImageNet, self).__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self._init_projection(in_channels)

    def _init_projection(self, in_channels):
        with torch.no_grad():
            self.proj.weight.zero_()
            if in_channels >= 3:
                start = in_channels - 3
                for out_idx in range(3):
                    self.proj.weight[out_idx, start + out_idx, 0, 0] = 1.0
            else:
                self.proj.weight[:, 0, 0, 0] = 1.0

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float()
        x = x / 255.0
        return self.proj(x)


def _load_torchvision_backbone(name, pretrained):
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "PPO_human_pre_v2 uses a public torchvision backbone by default. "
            "Install torchvision first. If the machine is offline, keep torchvision "
            "installed and set config['backbone_pretrained']=False to skip weight download."
        ) from exc

    if name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        return net.features, net.classifier[2].in_features

    if name == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_v2_s(weights=weights)
        return net.features, net.classifier[1].in_features

    raise ValueError("Unsupported backbone_name: {}".format(name))


class Model(nn.Module):
    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        if pretrained is None:
            pretrained = config.get('backbone_pretrained', True)
        self.resize = int(config.get('backbone_resize', 0) or 0)
        self.imagenet_normalize = config.get('imagenet_normalize', True)
        self.input_adapter = AtariToImageNet(config['frame_stack'])
        self.backbone, backbone_dim = _load_torchvision_backbone(
            config['backbone_name'],
            pretrained,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_dim = config.get('feature_dim', 512)
        self.fc = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_pi = nn.Linear(hidden_dim, config['act_dim'])
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def feature(self, x):
        x = self.input_adapter(x)
        if self.resize > 0 and (x.shape[-2] != self.resize or x.shape[-1] != self.resize):
            x = F.interpolate(x, size=(self.resize, self.resize), mode='bilinear', align_corners=False)
        if self.imagenet_normalize:
            x = (x - self.imagenet_mean) / self.imagenet_std
        x = self.backbone(x)
        x = self.pool(x).reshape(x.shape[0], -1)
        return self.fc(x)

    def logits(self, x):
        x = self.feature(x)
        return self.fc_pi(x)

    def pi(self, x):
        return F.softmax(self.logits(x), dim=-1)

    def v(self, x):
        x = self.feature(x)
        return self.fc_v(x)

    def logits_v(self, x):
        x = self.feature(x)
        return self.fc_pi(x), self.fc_v(x)

    def pi_v(self, x):
        logits, v = self.logits_v(x)
        return F.softmax(logits, dim=-1), v


class Teacher(Model):
    pass
