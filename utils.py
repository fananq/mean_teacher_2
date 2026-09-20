import torch
import numpy as np
import torch.nn.functional as F

# 【绿图部分机制】: EMA 仅作用于 Teacher 2，Teacher 1 永不调用此函数！
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        # 获取学生模型的所有参数字典
        student_params = dict(model.named_parameters())
        
        # 遍历老师模型的所有参数
        for name, ema_param in ema_model.named_parameters():
            # 只有当老师的参数名也存在于学生模型中时，才进行 EMA 更新
            # 这会自动跳过学生独有的 style_encoder 和 domain_discriminator
            if name in student_params:
                student_param = student_params[name]
                ema_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
#    with torch.no_grad():
#        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_current_consistency_weight(epoch, weight, rampup_length):
    if rampup_length == 0: return 1.0
    current = np.clip(epoch, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return weight * float(np.exp(-5.0 * phase * phase))

# --- 四大 Loss ---
def orthogonality_loss(z_c, z_s):
    z_c_norm = F.normalize(z_c, dim=1)
    z_s_norm = F.normalize(z_s, dim=1)
    cosine_sim = torch.sum(z_c_norm * z_s_norm, dim=1)
    return torch.mean(cosine_sim ** 2)

def domain_classification_loss(domain_pred, domain_target):
    return F.binary_cross_entropy(domain_pred, domain_target)

def anchor_feature_consistency_loss(z_c_student, z_c_anchor, temperature=0.5):
    """
    [终极改进版 L5]: 基于 KL散度 与 皮尔逊相关的 拓扑关系知识蒸馏
    (Relational Knowledge Distillation with Pearson Correlation)
    """
    # =====================================================================
    # 改进 1: 均值中心化 (Mean-Centering) -> 真正实现“平移不变性 (Shift Invariance)”
    # =====================================================================
    # 减去 Batch 的均值。数学上，中心化后的余弦相似度 = 皮尔逊相关系数！
    # 这样学生模型就可以在特征空间中自由“平移”，只要相对分布形态不变，Loss 就不会惩罚。
    z_c_student_centered = z_c_student - z_c_student.mean(dim=0, keepdim=True)
    z_c_anchor_centered = z_c_anchor - z_c_anchor.mean(dim=0, keepdim=True)

    # L2 归一化
    z_stu_norm = F.normalize(z_c_student_centered, p=2, dim=1)
    z_anc_norm = F.normalize(z_c_anchor_centered, p=2, dim=1)

    # =====================================================================
    # 改进 2: 温度系数缩放 (Temperature Scaling)
    # =====================================================================
    # 计算关系矩阵并除以温度系数。这能放大样本间微小的关系差异，让网络学得更清晰。
    sim_stu = torch.matmul(z_stu_norm, z_stu_norm.t()) / temperature
    sim_anc = torch.matmul(z_anc_norm, z_anc_norm.t()) / temperature

    # =====================================================================
    # 改进 3: 剔除对角线 (Masking Diagonal)
    # =====================================================================
    # 自己和自己的相似度永远是最大的，这会掩盖样本间的相对关系，必须置为 -无穷大
    N = sim_stu.size(0)
    mask = torch.eye(N, dtype=torch.bool, device=sim_stu.device)
    sim_stu = sim_stu.masked_fill(mask, -1e9)
    sim_anc = sim_anc.masked_fill(mask, -1e9)

    # =====================================================================
    # 改进 4: KL 散度概率匹配 (KL-Divergence) -> 真正实现“相对排名 (Relative Rank)”
    # =====================================================================
    # 我们用 Softmax 将相似度矩阵转化为“谁是我最近的邻居”的概率分布。
    # KL 散度不再强求绝对的余弦角度相等，而是要求“A把B当邻居的概率”在两个模型中保持一致！
    prob_anc = F.softmax(sim_anc, dim=1)          # Teacher 作为指导概率 (Target)
    log_prob_stu = F.log_softmax(sim_stu, dim=1)  # Student 作为预测对数概率 (Input)

    # KL_Div(Student || Teacher)
    loss_topology = F.kl_div(log_prob_stu, prob_anc, reduction='batchmean')
    
    return loss_topology

'''
#
# 【绿图新增约束】: 拓扑关系保持锚点防守
def anchor_feature_consistency_loss(z_c_student, z_c_anchor):
    """
    L5: 拓扑关系保持损失 (Topology Relationship Preservation Loss)
    不再约束单个样本的绝对空间位置，而是保持 mini-batch 内样本间的语义拓扑结构不变。
    """
    # 1. 沿着特征维度 (dim=1) 对特征进行 L2 归一化
    # 这一步是为了后续通过矩阵乘法直接得到余弦相似度
    z_c_student_norm = F.normalize(z_c_student, p=2, dim=1)
    z_c_anchor_norm = F.normalize(z_c_anchor, p=2, dim=1)
    
    # 2. 计算成对的相似度矩阵 (Pairwise Similarity Matrix)
    # (N, D) @ (D, N) -> (N, N) 关系矩阵
    sim_matrix_student = torch.matmul(z_c_student_norm, z_c_student_norm.t())
    sim_matrix_anchor = torch.matmul(z_c_anchor_norm, z_c_anchor_norm.t())
    
    # 3. 最小化两个关系矩阵之间的距离
    # 理论上是计算 Frobenius 范数，工程实现上使用 MSE Loss 是最优选
    # 因为 MSE(A, B) = ||A - B||^2_F / N^2，这在数学上等价于最小化 Frobenius 范数
    # 且除了 N^2 之后可以保证梯度不受 Batch Size 大小的剧烈影响，使训练更加稳定。
    return F.mse_loss(sim_matrix_student, sim_matrix_anchor)
'''
'''
def anchor_feature_consistency_loss(z_c_student, z_c_anchor):
    """
    [改进版] 拓扑关系保持损失 (Relational Topology Preservation Loss)
    """
    # 1. L2 归一化
    z_c_student_norm = F.normalize(z_c_student, p=2, dim=1)
    z_c_anchor_norm = F.normalize(z_c_anchor, p=2, dim=1)

    # 2. 计算相似度矩阵 (N x N)
    sim_stu = torch.matmul(z_c_student_norm, z_c_student_norm.t())
    sim_anc = torch.matmul(z_c_anchor_norm, z_c_anchor_norm.t())

    # 3. 【核心修正】剔除对角线 (Self-similarity)
    # 矩阵对角线元素永远是1，不仅无法提供有效梯度，还会严重稀释Loss
    N = sim_stu.size(0)
    # 生成对角线掩码 (True 代表是对角线)
    mask = torch.eye(N, dtype=torch.bool, device=sim_stu.device)
    
    # 利用 ~mask 提取所有非对角线元素，变成 1D 张量
    sim_stu_off_diag = sim_stu[~mask]
    sim_anc_off_diag = sim_anc[~mask]

    # 4. 【核心修正】使用 Smooth L1 替代 MSE
    # 相似度矩阵的分布存在离群值，Smooth L1 能有效防止梯度爆炸和过度拉扯
    return F.smooth_l1_loss(sim_stu_off_diag, sim_anc_off_diag)
'''
