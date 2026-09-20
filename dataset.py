import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
import numpy as np
from torch.utils.data.sampler import Sampler

class PersonalityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_unlabeled=False):
        """
        Args:
            csv_file: CSV 路径 (格式: filename, score1, score2, score3, score4, score5)
            root_dir: 图片文件夹路径
            transform: 预处理函数
            is_unlabeled: 标记是否为无标签数据
        """
        # 读取 CSV，假设没有表头(header=None)，如果有表头请改为 header=0
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_unlabeled = is_unlabeled

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. 读取图片路径 (假设文件名在第0列)
        img_name = str(self.data_frame.iloc[idx, 0])
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一张全黑图片防止崩溃
            image = Image.new('RGB', (224, 224))

        # 2. 读取标签 (假设分数在第1到5列)
        if not self.is_unlabeled:
            # 确保数据是 float32 类型
            # 这里的 shape 是 (5,)
            labels = self.data_frame.iloc[idx, 1:6].values.astype('float32')
        else:
            # 【修正】无标签数据必须和有标签数据形状一致 (5,)
            # 不能只写 -1.0，要生成一个全是 -1 的数组
            labels = np.full(5, -1.0, dtype=np.float32)

            # 3. 双流增强 (Mean Teacher 核心)
        # 如果 transform 是列表，说明我们要生成两张不同的增强图
        if self.transform and isinstance(self.transform, list):
            img1 = self.transform[0](image)  # 给学生
            img2 = self.transform[1](image)  # 给教师
            return img1, img2, labels

        # 普通验证/测试模式
        elif self.transform:
            img = self.transform(image)
            return img, labels

        return image, labels


def get_infinite_iter(dataloader):
    """无限循环迭代器，解决有标签和无标签数据量不一致的问题"""
    while True:
        for batch in dataloader:
            yield batch


class TwoStreamBatchSampler(Sampler):
    """
    双流采样器：在一个Batch中混合有标签和无标签数据
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # 只要 primary (有标签) 数据没遍历完，就一直继续
        primary_iter = iterate_once(self.primary_indices)
        # secondary (无标签) 数据无限循环
        secondary_iter = iterate_eternally(self.secondary_indices)

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)