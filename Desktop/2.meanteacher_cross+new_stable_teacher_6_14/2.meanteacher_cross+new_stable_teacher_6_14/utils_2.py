import torch
import numpy as np
import torch.nn.functional as F


def update_ema_variables(model, ema_model, alpha, global_step, current_uncertainty=0.0):
    """
    动态 EMA 更新
    current_uncertainty: 当前批次的平均不确定性 (scalar)
    策略: 不确定性越高 -> Alpha 越大 (接近1) -> 更新越慢
    """
    # 简单的映射策略：将不确定性 (sigma) 限制在 0-1 之间作为惩罚项
    # 假设 sigma 通常 < 1 (因为输出经过 sigmoid 是 0-1, sigma 不会太大)
    uncertainty_factor = np.clip(current_uncertainty, 0.0, 1.0)

    # 动态 Alpha 公式:
    # Base_Alpha + (1 - Base_Alpha) * Uncertainty
    # 如果 U=0, Alpha = Base (e.g. 0.99) -> 正常更新
    # 如果 U=1, Alpha = 1.0 -> 停止更新
    current_alpha = alpha + (1.0 - alpha) * uncertainty_factor

    # Rampup 预热 (训练初期强制快速更新)
    current_alpha = min(1 - 1 / (global_step + 1), current_alpha)

    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(current_alpha).add_(param.data, alpha=1 - current_alpha)


def get_current_consistency_weight(epoch, weight, rampup_length):
    """计算当前的一致性损失权重 (随时间增加)"""
    return weight * sigmoid_rampup(epoch, rampup_length)


def sigmoid_rampup(current, rampup_length):
    """Sigmoid 形状的预热曲线"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def orthogonality_loss(z_c, z_s):
    """
    L3: 计算内容特征和风格特征的余弦相似度
    目标: 让相似度趋近于 0 (相互垂直/无关)
    """
    # 计算余弦相似度 (-1 到 1)
    cosine_sim = F.cosine_similarity(z_c, z_s, dim=1)
    # 我们希望它趋近 0，所以取绝对值的平均
    return torch.mean(torch.abs(cosine_sim))