import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# --- [A. 子模块定义] ---

class ContentEncoder(nn.Module):
    """
    内容编码器 (Ec): 提取与任务相关(人格)的特征
    结构: 2048 -> 512 -> 256
    """
    def __init__(self, input_dim=2048, hidden_dim=512, out_dim=256):
        super(ContentEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) # 输出 z_c
        )

    def forward(self, x):
        return self.net(x)


class StyleEncoder(nn.Module):
    """
    风格编码器 (Es): 提取与域相关(光照/模糊)的特征
    结构: 镜像于 Ec, 但参数不共享
    """
    def __init__(self, input_dim=2048, hidden_dim=512, out_dim=256):
        super(StyleEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) # 输出 z_s
        )

    def forward(self, x):
        return self.net(x)

class DomainDiscriminator(nn.Module):
    """
    域鉴别器 (D): 判断风格特征 z_s 来自源域(0)还是目标域(1)
    结构: 256 -> 128 -> 1 (Sigmoid)
    """
    def __init__(self, input_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # 输出概率: 1代表Target, 0代表Source
        )

    def forward(self, z_s):
        return self.net(z_s)


# --- [B. 主模型定义] ---

class DisentangledResNet(nn.Module):
    """
    主骨干 + 双流解耦头
    这是学生模型和教师模型的本体
    """

    def __init__(self, num_outputs=5):
        super(DisentangledResNet, self).__init__()

        # 1. 骨干网络 (Shared Encoder E)
        # 使用 ResNet50 以获得 2048 维特征
        backbone = models.resnet50(weights='DEFAULT')
        # 去掉最后的全连接层, 保留卷积部分
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # 2. 双流解耦器
        self.content_encoder = ContentEncoder(input_dim=2048)
        self.style_encoder = StyleEncoder(input_dim=2048)

        # 3. [修改] 概率回归头
        # 先经过公共层
        self.regressor_base = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 头 A: 预测均值 (mu)
        self.head_mu = nn.Sequential(
            nn.Linear(128, num_outputs),
            nn.Sigmoid()
        )
        # 头 B: 预测标准差/不确定性 (sigma)
        self.head_sigma = nn.Sequential(
            nn.Linear(128, num_outputs),
            nn.Softplus()  # 保证输出 > 0
        )

    def forward(self, x):
        # 1. 提取底层特征 (B, 2048, 1, 1) -> (B, 2048)
        f = self.features(x)
        f = torch.flatten(f, 1)

        # 2. 解耦
        z_c = self.content_encoder(f)  # 内容特征
        z_s = self.style_encoder(f)  # 风格特征

        # 概率预测
        feat = self.regressor_base(z_c)
        mu = self.head_mu(feat)
        sigma = self.head_sigma(feat)

        # 返回四个值: 均值, 标准差, 内容特征, 风格特征
        return mu, sigma, z_c, z_s


def create_model(ema=False):
    model = DisentangledResNet()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model