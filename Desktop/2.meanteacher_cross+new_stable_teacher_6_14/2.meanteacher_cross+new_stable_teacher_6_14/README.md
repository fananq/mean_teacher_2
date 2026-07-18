请在你的电脑上创建一个新文件夹，例如命名为 personality_project。结构如下：
personality_project/
│
├── data/                    # 存放图片和CSV
│   ├── images/              # 图片文件夹 (放你的jpg/png)
│   ├── labeled.csv          # 有标签数据的CSV
│   └── unlabeled.csv        # 无标签数据的CSV (如有)
│
├── dataset.py               # [新代码] 数据加载与预处理
├── model.py                 # [新代码] 神经网络模型
├── utils.py                 # [新代码] 辅助函数 (EMA, Rampup)
├── main.py                  # [新代码] 训练主程序
└── requirements.txt         # [新代码] 依赖库




为了让程序跑起来，你需要准备 CSV 文件。

1. data/labeled.csv 格式： 假设你的大五人格分数是 0 到 1 之间（如果是 1-100，请先除以 100）。 CSV 不要包含表头，内容如下：

代码段

img_001.jpg,0.1,0.2,0.3,0.4,0.5
img_002.jpg,0.9,0.8,0.7,0.6,0.1
...
2. data/unlabeled.csv 格式： 因为代码会统一读取，所以哪怕没有标签，也要填满 5 列数字（填 0 或者 -1 都可以，反正代码不会用）。

代码段

img_100.jpg,0,0,0,0,0
img_101.jpg,0,0,0,0,0
...
3. 图片位置： 把所有图片（无论是 labeled 还是 unlabeled）都放在 data/images/ 文件夹里。




运行代码
打开终端 (Terminal)，进入 personality_project 文件夹，激活你的环境，然后运行：

Bash

# 安装依赖
pip install -r requirements.txt

# 开始训练
# 假设你有 1000 张有标签图，10000 张无标签图
# consistency-rampup 设为 10，表示前 10 个 epoch 慢慢增加无标签数据的权重
python main.py --labeled-csv data/labeled.csv --unlabeled-csv data/unlabeled.csv --epochs 50 --batch-size 32
