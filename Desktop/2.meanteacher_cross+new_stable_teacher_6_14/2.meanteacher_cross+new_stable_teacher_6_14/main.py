import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset  # <--- 导入 random_split
from torchvision import transforms
import numpy as np
import random


# 导入我们自己写的文件
import model as net_builder
import dataset as my_dataset
import utils


def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 CUDA 算子确定性（会牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    # 1. 参数设置
    parser = argparse.ArgumentParser(description='Mean Teacher for Personality Regression')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--pretrain-epochs', default=10, type=int, help='阶段一: 源域预训练')  # <--- 新增
    parser.add_argument('--epochs', default=40, type=int, help='阶段二: 目标域自适应 (或叫 adapt-epochs)')  # <--- 新增
    parser.add_argument('--batch-size', default=32, type=int, help='Batch Size')
    parser.add_argument('--consistency', default=10.0, type=float, help='一致性损失的最大权重')
    parser.add_argument('--consistency-rampup', default=10, type=int, help='多少个Epoch达到最大权重')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA 衰减率')

    # 路径设置
    parser.add_argument('--labeled-csv', type=str, default='../meanteacher_original/data/test.csv')
    parser.add_argument('--labeled-image-root', type=str,
                        default='../meanteacher_original/data/images', help='有标签图片文件夹')
    parser.add_argument('--unlabeled-image-root', type=str,
                        default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/images',
                        help='无标签图片文件夹')
    parser.add_argument('--unlabeled-csv', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/unlabeled.csv')
    parser.add_argument('--image-root', type=str, default='../meanteacher_original/data/images')

    #parser.add_argument('--labeled-image-root', type=str, default=None,
    #                    help='有标签数据的图片文件夹路径 (如果不填，则默认使用 --image-root)')
    #parser.add_argument('--unlabeled-image-root', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/images',
    #                    help='无标签数据的图片文件夹路径 (如果不填，则默认使用 --image-root)')

    parser.add_argument('--gpu', default='0', type=str, help='指定使用的GPU ID，例如 0 或 1')


    args = parser.parse_args()

    # 【新增】设置随机种子
    set_seed(1000)


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
    print("==> 创建三网阵列: Student(双流), Teacher1(锚点/绿图), Teacher2(EMA/红图)")
    student_model = net_builder.DisentangledResNet(is_teacher=False).to(device)
    teacher1_anchor = net_builder.DisentangledResNet(is_teacher=True).to(device)
    teacher2_ema = net_builder.DisentangledResNet(is_teacher=True).to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    criterion_domain = nn.BCELoss()

    # =======================================================
    # 数据加载器定义 (DataLoaders)
    # =======================================================
    print("==> 2. 准备数据加载器...")

    # 1. 定义数据增强 (Transforms)
    # Mean Teacher 的精髓：弱增强给老师，强增强给学生
    train_transform = [
        transforms.Compose([  # 弱增强
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([  # 强增强
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]

    # 2. 读取原始的 数据集 (利用你 dataset.py 里的 PersonalityDataset)
    full_labeled_dataset = my_dataset.PersonalityDataset(
        csv_file=args.labeled_csv,
        root_dir=args.labeled_image_root,
        transform=train_transform,
        is_unlabeled=False
    )

    unlabeled_dataset = my_dataset.PersonalityDataset(
        csv_file=args.unlabeled_csv,
        root_dir=args.unlabeled_image_root,
        transform=train_transform,
        is_unlabeled=True
    )

    # 3. 划分验证集 (从有标签数据中拆出 10% 做验证)
    train_size = int(0.9 * len(full_labeled_dataset))
    val_size = len(full_labeled_dataset) - train_size
    train_labeled_set, val_set = random_split(full_labeled_dataset, [train_size, val_size])

    # ---------------------------------------------------------
    # 核心：构造混合的 train_loader (前半源域，后半目标域)
    # ---------------------------------------------------------

    # 4. 把训练用的 有标签 和 无标签 拼成一个大 Dataset
    train_dataset = ConcatDataset([train_labeled_set, unlabeled_dataset])

    # 5. 获取它们在合并后的 Dataset 中的索引分布
    labeled_indices = list(range(len(train_labeled_set)))
    unlabeled_indices = list(range(len(train_labeled_set), len(train_labeled_set) + len(unlabeled_dataset)))

    # 6. 使用 dataset.py 里的采样器 (这里假设类名叫 TwoStreamBatchSampler)
    # 它的作用是：每次抽取 batch_size 个样本，其中一半从 labeled_indices 抽，一半从 unlabeled_indices 抽
    batch_sampler = my_dataset.TwoStreamBatchSampler(
        primary_indices=labeled_indices,  # 前半部分：有标签 (Source)
        secondary_indices=unlabeled_indices,  # 后半部分：无标签 (Target)
        batch_size=args.batch_size,  # 总大小
        secondary_batch_size=args.batch_size // 2  # 占据一半
    )

    # 7. 最终定义 Loader
    # 训练集使用刚才定义的混合采样器
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2)

    # 验证集使用普通的 DataLoader，顺序读取即可
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(
        f"==> 数据准备完毕: 训练有标签={len(train_labeled_set)}, 无标签={len(unlabeled_dataset)}, 验证集={len(val_set)}")
    # =======================================================
    # 数据加载器定义结束
    # =======================================================


    # =======================================================
    # 【阶段一】：源域监督预训练 (为复制锚点打基础)
    # =======================================================
    print(f"\n>>> 阶段一: 源域监督预训练 ({args.pretrain_epochs} Epochs) <<<")
    for epoch in range(args.pretrain_epochs):
        student_model.train()
        for i, (img_student, img_teacher, targets) in enumerate(train_loader):
            # 预训练阶段，只使用前半个 Batch (Source 域)
            labeled_bs = args.batch_size // 2
            inp_strong = img_student[:labeled_bs].to(device)
            target_sup = targets[:labeled_bs].to(device)

            pred_stu, zc_stu, zs_stu, d_pred = student_model(inp_strong, return_features=True)

            # Loss 计算
            # --- Loss 计算 ---
        # 1. 监督回归损失 (Loss 2)
            loss_sup = criterion(pred_stu, target_sup)

        # 2. 正交解耦损失 (Loss 3)
            loss_ortho = utils.orthogonality_loss(zc_stu, zs_stu) * 0.1

        # 3. 【新增】风格分类损失 (Loss 1)
        # 在第一阶段，所有数据都来自源域，所以标签全为 0
            target_domain = torch.zeros_like(d_pred).to(device)
            loss_style = criterion_domain(d_pred, target_domain) * 0.1 # 建议给予 0.1 的权重系数

        # 总损失合成
            loss_total = loss_sup + loss_ortho + loss_style

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()


    # =======================================================
    # 【分水岭】：全盘复制，化作冰雕 (绿图的核心指令)
    # =======================================================
    print("\n>>> 分水岭: 复制学生知识给 T1 和 T2 <<<")
    # strict=False 自动剥离风格编码器和域辨别器
    teacher1_anchor.load_state_dict(student_model.state_dict(), strict=False)
    teacher2_ema.load_state_dict(student_model.state_dict(), strict=False)

    # 绿图要求：T1 完全冻结，化作内容锚点
    for param in teacher1_anchor.parameters(): param.requires_grad = False
    teacher1_anchor.eval()

    # T2 准备接收 EMA，不需要梯度，但保持 train 模式更新 BN
    for param in teacher2_ema.parameters(): param.requires_grad = False
    teacher2_ema.train()

    # =======================================================
    # 【阶段二】：目标域自适应与多重防守
    # =======================================================
    print(f"\n>>> 阶段二: 引入目标域探索 ({args.epochs} Epochs) <<<")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        global_step = adapt_one_epoch(
            train_loader, student_model, teacher1_anchor, teacher2_ema,
            optimizer, criterion, criterion_domain, epoch, global_step, args, device
        )

        # 验证 T2 (EMA 模型泛化最好)
        val_loss = validate(val_loader, teacher2_ema, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(teacher2_ema.state_dict(), 'best_model.pth')
            print(f"    ==> 👑 发现更优模型 (Val Loss: {best_val_loss:.6f})")


def adapt_one_epoch(train_loader, student, t1_anchor, t2_ema, optimizer, criterion, criterion_domain, epoch,
                    global_step, args, device):
    student.train()

    cons_weight = utils.get_current_consistency_weight(epoch, args.consistency, args.consistency_rampup)
    lambda_style = 0.1
    lambda_ortho = 0.1
    lambda_anchor = 2.0#1.0  # T1 的拉扯力度

    labeled_bs = args.batch_size // 2

    for i, (img_student, img_teacher, targets) in enumerate(train_loader):
        inp_strong = img_student.to(device)
        inp_weak = img_teacher.to(device)
        targets = targets.to(device)

        # 1. 学生前向 (处理整个 Batch：源域 + 目标域)
        #pred_stu, zc_stu, zs_stu, d_pred_stu = student(inp_strong, return_features=True)

        # ==================== 修复 BN 域偏移 ====================
        # 不要把源域和目标域拼在一起 forward！分两次输入！
        
        # 1. 源域 (前一半) 前向传播
        pred_sup, zc_sup, zs_sup, d_pred_sup = student(inp_strong[:labeled_bs], return_features=True)
        
        # 2. 目标域 (后一半) 前向传播
        pred_u, zc_u, zs_u, d_pred_u = student(inp_strong[labeled_bs:], return_features=True)

        # 拼接结果以兼容后面的计算
        pred_stu = torch.cat([pred_sup, pred_u], dim=0)
        zc_stu = torch.cat([zc_sup, zc_u], dim=0)
        zs_stu = torch.cat([zs_sup, zs_u], dim=0)
        d_pred_stu = torch.cat([d_pred_sup, d_pred_u], dim=0)
        # ========================================================


        with torch.no_grad():
            # 2. T1 (绿图锚点) 前向：只看后半段目标域图片，提取权威内容特征
            _, zc_t1_anchor = t1_anchor(inp_weak[labeled_bs:], return_features=True)

            # 3. T2 (红图EMA) 前向：只看后半段目标域图片，提供伪标签
            pred_t2_ema, _ = t2_ema(inp_weak[labeled_bs:], return_features=True)

        # ==================== Loss 计算 ====================

        # A. 监督 Loss (只看前半段 Source)
        loss_sup = criterion(pred_stu[:labeled_bs], targets[:labeled_bs])

        # B. T2 EMA 一致性探索 (看后半段 Target)
        loss_cons = criterion(pred_stu[labeled_bs:], pred_t2_ema) * cons_weight

        # C. T1 锚点防守 (看后半段 Target) -> 【绿图的新增要求】
        #loss_anchor = utils.anchor_feature_consistency_loss(zc_stu[labeled_bs:], zc_t1_anchor) * lambda_anchor

# C. T1 锚点防守 -> 改为守护源域 (Source) 的拓扑结构！
        # 让 T1 看着它最擅长的源域图片 (前半段)，为学生稳住大后方的拓扑基本盘
        loss_anchor = utils.anchor_feature_consistency_loss(zc_stu[:labeled_bs], zc_t1_anchor) * lambda_anchor

        # D. 域辨别 Loss (辨别前半段为 0，后半段为 1)
        domain_labels = torch.cat([
            torch.zeros(labeled_bs, 1),
            torch.ones(len(inp_strong) - labeled_bs, 1)
        ]).to(device)
        loss_style = criterion_domain(d_pred_stu, domain_labels) * lambda_style

        # E. 正交解耦 Loss (全局)
        loss_ortho = utils.orthogonality_loss(zc_stu, zs_stu) * lambda_ortho

        # ==================== 反向传播 ====================
        loss_total = loss_sup + loss_cons + loss_anchor + loss_style + loss_ortho

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 【核心约束】：只允许 T2 跟着学生变化，T1 不动！
        global_step += 1
        utils.update_ema_variables(student, t2_ema, args.ema_decay, global_step)

    return global_step


def validate(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, _, targets) in enumerate(val_loader):
            outputs = model(images.to(device), return_features=False)
            total_loss += criterion(outputs, targets.to(device)).item()
    return total_loss / len(val_loader)

if __name__ == '__main__':
    main()
