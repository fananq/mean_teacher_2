import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """内容编码器 (Ec): 提取人格特征"""

    def __init__(self, input_dim=2048, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x): return self.net(x)


class StyleEncoder(nn.Module):
    """风格编码器 (Es): 提取光照、模糊度等环境特征"""

    def __init__(self, input_dim=2048, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x): return self.net(x)


class DomainDiscriminator(nn.Module):
    """域辨别器 (D): 区分源域和目标域"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, z_s): return self.net(z_s)


class DisentangledResNet(nn.Module):
    def __init__(self, num_outputs=5, is_teacher=False):
        super().__init__()
        backbone = models.resnet50(weights='DEFAULT')
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.is_teacher = is_teacher
        # 共同拥有内容编码器
        self.content_encoder = ContentEncoder()

        # 【核心约束】教师模型(T1, T2)被物理剥夺了风格编码器和域辨别器
        if not self.is_teacher:
            self.style_encoder = StyleEncoder()
            self.domain_discriminator = DomainDiscriminator()

        self.regressor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_outputs), nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        orig_feat = torch.flatten(self.features(x), 1)

        z_c = self.content_encoder(orig_feat)
        pred = self.regressor(z_c)

        if self.is_teacher:
            if return_features: return pred, z_c
            return pred

        z_s = self.style_encoder(orig_feat)
        domain_pred = self.domain_discriminator(z_s)

        if return_features: return pred, z_c, z_s, domain_pred
        return pred