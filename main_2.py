import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset  # <--- 导入 random_split
from torchvision import transforms
import numpy as np

# 导入我们自己写的文件
import model as net_builder
import dataset as my_dataset
import utils


def main():
    # 1. 参数设置
    parser = argparse.ArgumentParser(description='Mean Teacher for Personality Regression')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--epochs', default=50, type=int, help='总Epoch数')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch Size')
    parser.add_argument('--consistency', default=10.0, type=float, help='一致性损失的最大权重')
    parser.add_argument('--consistency-rampup', default=10, type=int, help='多少个Epoch达到最大权重')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA 衰减率')

    # 路径设置
    parser.add_argument('--labeled-csv', type=str, default='../meanteacher_original/data/labeled.csv')
    parser.add_argument('--unlabeled-csv', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/unlabeled.csv')
    parser.add_argument('--image-root', type=str, default='../meanteacher_original/data/images')

    parser.add_argument('--labeled-image-root', type=str, default=None,
                        help='有标签数据的图片文件夹路径 (如果不填，则默认使用 --image-root)')
    parser.add_argument('--unlabeled-image-root', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/images',
                        help='无标签数据的图片文件夹路径 (如果不填，则默认使用 --image-root)')

    parser.add_argument('--gpu', default='0', type=str, help='指定使用的GPU ID，例如 0 或 1')

    args = parser.parse_args()


    # --- 逻辑处理：确定最终使用的路径 ---
    # 如果用户在命令行指定了 --labeled-image-root，就用指定的；否则用通用的 --image-root
    real_labeled_root = args.labeled_image_root if args.labeled_image_root is not None else args.image_root
    real_unlabeled_root = args.unlabeled_image_root if args.unlabeled_image_root is not None else args.image_root

    print(f"==> Labeled 图片路径: {real_labeled_root}")
    print(f"==> Unlabeled 图片路径: {real_unlabeled_root}")
    # ====================================================


    # 【修改】设置设备
    # 这种方式最稳妥，它会只让程序“看到”你指定的显卡，程序内部依然认为它是 cuda:0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if torch.cuda.is_available():
        # 因为设置了 CUDA_VISIBLE_DEVICES，所以这里直接用 'cuda' 即可
        # PyTorch 会自动把它映射到你可见的那张卡上
        device = torch.device("cuda")
        print(f"==> 使用 GPU: {args.gpu} (PyTorch 识别为 cuda:0)")
    else:
        device = torch.device("cpu")
        print("==> 未检测到 GPU，使用 CPU")

    # 2. 构建模型
    print("==> 创建学生、教师和鉴别器模型...")
    student_model = net_builder.create_model(ema=False).to(device)
    teacher_model = net_builder.create_model(ema=True).to(device)

    # [新增] 域鉴别器 (独立于学生教师)
    domain_discriminator = net_builder.DomainDiscriminator().to(device)

    # 优化器: 同时优化 学生模型 和 鉴别器
    optimizer = optim.Adam(
        list(student_model.parameters()) + list(domain_discriminator.parameters()),
        lr=args.lr
    )

    criterion = nn.MSELoss()
    criterion_domain = nn.BCELoss()  # [新增] 用于鉴别器的二分类 Loss

    # 3. 数据准备
    # 训练增强：随机裁剪、翻转等
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 定义有标签数据集
    if not os.path.exists(args.labeled_csv):
        print("错误: 找不到 labeled.csv")
        return

    labeled_dataset = my_dataset.PersonalityDataset(
        args.labeled_csv, real_labeled_root, transform=[train_transform, train_transform], is_unlabeled=False
    )

    # 2. 定义无标签数据集
    if os.path.exists(args.unlabeled_csv):
        unlabeled_dataset = my_dataset.PersonalityDataset(
            args.unlabeled_csv, real_unlabeled_root, transform=[train_transform, train_transform], is_unlabeled=True
        )
    else:
        # 【修改建议】直接报错，不要自动回退
        raise FileNotFoundError(f"错误：找不到无标签数据文件 -> {args.unlabeled_csv}")


    # 3. 划分训练集和验证集 (只从有标签数据里分)
    train_size = int(0.9 * len(labeled_dataset))
    val_size = len(labeled_dataset) - train_size

    # 注意：这里我们不能直接用 random_split 得到的 Subset 给 Sampler 用，因为 Subset 隐藏了原始索引
    # 为了简单起见，我们手动切分 indices
    all_indices = np.random.RandomState(42).permutation(len(labeled_dataset))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # 创建真正的 Dataset 对象 (Subset)
    train_labeled_set = torch.utils.data.Subset(labeled_dataset, train_indices)
    val_set = torch.utils.data.Subset(labeled_dataset, val_indices)

    # 4. 构建 Sampler 需要的索引
    # 混合后的 Dataset = [train_labeled_set, unlabeled_dataset]
    # 所以 labeled 索引是 0 ~ len(train_labeled_set)
    # unlabeled 索引是 len(train_labeled_set) ~ 结尾
    labeled_idxs = list(range(len(train_labeled_set)))
    unlabeled_idxs = list(range(len(train_labeled_set), len(train_labeled_set) + len(unlabeled_dataset)))

    # 设定比例：假设 Batch Size=32，我们想要 16个有标签，16个无标签
    labeled_batch_size = args.batch_size // 2
    unlabeled_batch_size = args.batch_size - labeled_batch_size

    batch_sampler = my_dataset.TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, unlabeled_batch_size
    )

    # 5. 创建最终的 Loader
    # 核心：使用 ConcatDataset 把两者物理连接
    train_loader = DataLoader(
        ConcatDataset([train_labeled_set, unlabeled_dataset]),
        batch_sampler=batch_sampler,
        num_workers=2,
        pin_memory=True
    )

    # 验证集 Loader (普通的)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"==> 数据准备完毕: Labeled={len(train_labeled_set)}, Unlabeled={len(unlabeled_dataset)}")
    # --- [修改结束] ---

    # 4. 开始训练
    global_step = 0
    best_val_loss = float('inf')  # 记录最佳 Loss

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        global_step = train_one_epoch(
        train_loader,
        student_model, teacher_model, domain_discriminator, # 传入鉴别器
        optimizer, criterion, criterion_domain,             # 传入新Loss
        epoch, global_step, args, device
        )
        # --- [核心修改 2: 验证与保存] ---
        # 验证
        val_loss = validate(val_loader, teacher_model, criterion, device)

        print(f"Epoch {epoch + 1} 完成. 耗时: {time.time() - start_time:.1f}s | Val Loss: {val_loss:.6f}")

        # 保存最新模型
        torch.save(teacher_model.state_dict(), 'latest_model.pth')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(teacher_model.state_dict(), 'best_model.pth')
            print(f"    ==> 👑 发现更优模型 (Val Loss: {best_val_loss:.6f}), 已保存 best_model.pth")


def train_one_epoch(train_loader, student, teacher, discriminator, optimizer, criterion, criterion_domain, epoch,
                    global_step, args, device):
    student.train()
    teacher.train()
    discriminator.train()

    cons_weight = utils.get_current_consistency_weight(epoch, args.consistency, args.consistency_rampup)

    # 权重超参数 (根据论文经验值设定，可微调)
    lambda_style = 0.1  # 风格分类权重的系数
    lambda_ortho = 0.1  # 正交解耦权重的系数

    labeled_bs = args.batch_size // 2

    # 直接解包三个变量：img_student(强增强), img_teacher(弱增强), targets(标签)
    for i, (img_student, img_teacher, targets) in enumerate(train_loader):

        # 此时 img_student 和 img_teacher 都是 4D Tensor [Batch, 3, 224, 224]
        inp_strong = img_student.to(device)  # 给学生
        inp_weak = img_teacher.to(device)  # 给教师
        targets = targets.to(device)

        # ===========================
        # 1. Forward Pass (前向传播)
        # ===========================

        # 模型现在返回: mu(均值), sigma(不确定性), z_c(内容), z_s(风格)
        mu_s, sigma_s, zc_s, zs_s = student(inp_strong)

        with torch.no_grad():
            mu_t, sigma_t, _, _ = teacher(inp_weak)

            # ===========================
            # 2. Loss Calculation
            # ===========================

            # --- A. L2: 概率监督回归损失 (Probabilistic Supervised Loss) ---
            # 公式: sum( (y-mu)^2 / 2sigma^2 + 0.5 * log(sigma^2) )
        mu_s_sup = mu_s[:labeled_bs]
        sigma_s_sup = sigma_s[:labeled_bs]
        target_sup = targets[:labeled_bs]

            # 加上 1e-6 防止除以 0
        var_s_sup = sigma_s_sup ** 2 + 1e-6

            # 高斯负对数似然 (Gaussian NLL)
        loss_nll = 0.5 * (torch.log(var_s_sup) + (target_sup - mu_s_sup) ** 2 / var_s_sup)
        loss_sup = loss_nll.mean()

            # --- B. L4: 不确定性感知一致性损失 (Uncertainty-Aware Consistency) ---
            # 权重 w = 1 / (sigma_teacher^2 + epsilon)
            # 如果教师很确信 (sigma小)，权重就大；教师不确信，权重就小
        mu_s_unsup = mu_s[labeled_bs:]
        mu_t_unsup = mu_t[labeled_bs:]
        sigma_t_unsup = sigma_t[labeled_bs:]

        var_t_unsup = sigma_t_unsup ** 2 + 1e-6
        uncertainty_weight = 1.0 / var_t_unsup

        squared_diff = (mu_s_unsup - mu_t_unsup) ** 2
            # 加权 MSE
        loss_cons = (uncertainty_weight * squared_diff).mean() * cons_weight

            # --- C. L1: 域鉴别损失 (Domain Classification Loss) ---
        domain_labels_source = torch.zeros(labeled_bs, 1).to(device)
        domain_labels_target = torch.ones(labeled_bs, 1).to(device)

        d_out_source = discriminator(zs_s[:labeled_bs])
        d_out_target = discriminator(zs_s[labeled_bs:])

        loss_d_source = criterion_domain(d_out_source, domain_labels_source)
        loss_d_target = criterion_domain(d_out_target, domain_labels_target)
        loss_style = (loss_d_source + loss_d_target) * lambda_style

            # --- D. L3: 正交解耦损失 (Orthogonal Disentanglement Loss) ---
        loss_ortho = utils.orthogonality_loss(zc_s, zs_s) * lambda_ortho

            # Total Loss
        loss_total = loss_sup + loss_cons + loss_style + loss_ortho

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        global_step += 1

            # --- [E. 动态 EMA 更新] ---
            # 计算当前学生模型的平均不确定性 (作为调整 Alpha 的依据)
            # 你也可以用 teacher 的 uncertainty，或者两者的平均
        batch_uncertainty = torch.mean(sigma_s).item()

        utils.update_ema_variables(student, teacher, args.ema_decay, global_step,
                                    current_uncertainty=batch_uncertainty)

        if i % 20 == 0:
            print(f"Step [{i}/{len(train_loader)}] Total: {loss_total.item():.3f} | "
                f"NLL: {loss_sup.item():.3f} | Cons: {loss_cons.item():.3f} | "
                f"Style: {loss_style.item():.3f} | Ortho: {loss_ortho.item():.3f}")

    return global_step


# --- [核心修改 3: 验证函数] ---
def validate(val_loader, model, criterion, device):
    """
    运行验证集，返回平均 Loss
    """
    model.eval()  # 极其重要：关闭 Dropout，锁定 BN 统计量
    total_loss = 0.0

    with torch.no_grad():  # 不记录梯度
        for i, (images, _, targets) in enumerate(val_loader):
            # images: 验证集图片
            # _: 忽略第二个 augment 版本，验证时不需要
            # targets: 真实标签

            images = images.to(device)
            targets = targets.to(device)

            # 【修改点】模型现在返回 (pred, z_c, z_s)，我们需要解包
            # 我们只关心预测准确度，所以只取第一个值
            # 解包: 只取 mu 进行验证
            mu, _, _, _ = model(images)

            loss = criterion(mu, targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

if __name__ == '__main__':
    main()
