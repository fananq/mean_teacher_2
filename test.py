import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 引入项目文件
import model as net_builder
import dataset as my_dataset
import pandas as pd


def test():
    parser = argparse.ArgumentParser(description='Test Personality Regression Model')
    parser.add_argument('--test-csv', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/test.csv' )#required=True, help='测试集CSV路径')
    parser.add_argument('--image-root', type=str, default='../meanteacher_original/data/elea_dataset/elea_dataset/my_data/images', help='图片根目录')

    parser.add_argument('--model-path', type=str, default='best_model.pth', help='模型权重路径')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> 使用设备: {device}")

    # 1. 加载模型结构
    # 注意：Mean Teacher 论文建议测试时使用 Teacher (EMA) 模型的参数，因为它更稳定。
    # 但由于我们保存的是 student 的 state_dict (见 main.py)，
    # 如果你也保存了 teacher，这里可以用 teacher。
    print("==> 加载模型...")
    #model = net_builder.create_model(ema=False)  # 结构必须和训练时一致
    model = net_builder.DisentangledResNet(is_teacher=True)

    # 加载权重
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        # 如果保存的是整个 checkpoint 字典，取 'state_dict'；如果是直接保存的模型参数，直接load
        # 根据你 main.py 的写法: torch.save(student_model.state_dict(), ...)
        model.load_state_dict(checkpoint)
        print(f"==> 成功加载权重: {args.model_path}")
    else:
        print(f"Error: 找不到权重文件 {args.model_path}")
        return

    model.to(device)
    model.eval()  # 切换到评估模式 (关闭 Dropout, 锁定 BatchNorm)

    # 2. 准备测试数据
    # 测试时不要用 RandomCrop/Flip，要用确定的 CenterCrop
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # 居中裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = my_dataset.PersonalityDataset(
        args.test_csv, args.image_root, transform=test_transform, is_unlabeled=False
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. 开始推理
    all_preds = []
    all_targets = []

    print("==> 开始推理...")
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images = images.to(device)

            # Forward
            # 【关键修正】模型现在返回 4 个值: mu, sigma, z_c, z_s
            # 我们只需要预测均值 mu
            mu= model(images)

            # 收集结果 (转回 CPU 并转为 numpy)
            all_preds.append(mu.cpu().numpy())
            all_targets.append(labels.numpy())

    # 拼接所有 batch 的结果
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 4. 计算指标
    # 大五人格通常对应 5 个维度：O, C, E, A, N
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

    print("\n" + "=" * 40)
    print("       测试结果评估 (Regression)       ")
    print("=" * 40)

    # 总体指标
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"总体 MSE (均方误差): {mse:.4f}")
    print(f"总体 MAE (平均绝对误差): {mae:.4f}")
    print("-" * 40)
    print(f"平均每项预测偏差: {(mae * 100):.2f}% (假设满分是1)")
    print("-" * 40)

    # 分维度指标
    print(f"{'Trait':<20} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 45)

    num_traits = all_targets.shape[1]
    for i in range(num_traits):
        # 计算每一列(每一个性格维度)的误差
        trait_mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        trait_mse = mean_squared_error(all_targets[:, i], all_preds[:, i])

        # 防止维度越界（如果你没有5个维度）
        name = trait_names[i] if i < len(trait_names) else f"Dim {i}"
        print(f"{name:<20} | {trait_mae:.4f}     | {trait_mse:.4f}")

    print("=" * 40)

    # 5. (可选) 保存详细预测结果到 CSV 以便人工查看
    save_path = "test_predictions.csv"
    df = pd.DataFrame(all_preds, columns=trait_names[:all_preds.shape[1]])
    # 也可以把真实标签拼上去对比
    for i in range(num_traits):
        df[f"True_{trait_names[i]}"] = all_targets[:, i]

    df.to_csv(save_path, index=False)
    print(f"\n详细预测结果已保存至: {save_path}")


if __name__ == '__main__':
    test()
