# 图像描述生成项目 (Image Captioning)

这是一个基于深度学习的图像描述生成项目，支持多种编码器架构（CLIP、ViT、ResNet等）。

## 环境要求

- Python 3.7+
- PyTorch
- CLIP
- Transformers
- Torchvision

## 安装依赖

```bash
pip install torch torchvision
pip install clip
pip install transformers
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

## 项目设置

### 需要自行创建的目录和文件

由于以下文件较大或包含个人配置，Git仓库中不包含这些文件，需要自行创建：

```
项目根目录/
├── data/                    # 数据目录（需要创建）
│   ├── train.json          # 训练数据
│   ├── val.json            # 验证数据
│   └── test.json           # 测试数据
├── model_new/              # 模型保存目录（需要创建）
├── vit_models/             # ViT模型目录（需要创建）
│   └── vit_base/
│       └── models--google--vit-base-patch16-224-in21k/
├── logs/                   # 日志目录（需要创建）
└── flickr8k_aim3/         # 原始数据集目录（需要创建）
```


### 下载预训练模型

1. **ViT模型**：
   - 下载ViT-Base模型到`vit_models/vit_base/`目录
   - 或者修改`encoder.py`中的模型路径

2. **数据集**：
   - 将Flickr8k数据集放在`flickr8k_aim3/`目录
   - 或者修改`generate_json_data.py`中的数据路径

## 使用方法

### 1. 数据预处理

首先将下载好的数据集放在根目录下，把`generate_json_data.py`中的`args.split_type`参数指定为对应数据集的原始json文件，然后运行：

```bash
python generate_json_data.py
```

处理好的文件会被保存到`./data`文件夹之下。

### 2. 模型训练

注意各种参数的配置。当需要用到ViT作为encoder的时候，需要自己下载vit_models，这里不提供配置文件。将下载好的vit模型文件地址复制到`encoder.py`对应的部分中。

基础训练：
```bash
python train.py
```

带参数的训练：
```bash
python train.py --batch-size 64 --epochs 20 --patience 5 --lr 2e-4 --step-size 8 --alpha-c 1.0 --log-interval 100 --data data --network clip_vit --hidden_dim 512 --tf --is_finetune False
```

训练好的模型会保存到`model_new`文件夹之下。

### 3. 生成描述和可视化

提供decoder注意力可视化灰度图：

```bash
python generate_caption.py --model model_new/clip_vit_512_frozen_10.pth --network clip_vit --hidden_dim 512 --is_finetune False
```

### 4. 模型评估

```bash
python eval.py --model model_new/clip_vit_512_frozen_10.pth --network clip_vit --hidden_dim 512 --is_finetune False
```

## 支持的编码器

- `clip_rn50`: CLIP ResNet-50
- `clip_vit`: CLIP ViT-B/32
- `vit_base`: ViT Base
- `resnet152`: ResNet-152

## 项目结构

```
├── encoder.py              # 编码器实现
├── decoder_lstm.py         # LSTM解码器
├── attention.py            # 注意力机制
├── dataset.py              # 数据集处理
├── train.py                # 训练脚本
├── eval.py                 # 评估脚本
├── generate_caption.py     # 生成描述
├── generate_json_data.py   # 数据预处理
├── utils.py                # 工具函数
├── requirements.txt        # 依赖包列表
├── .gitignore             # Git忽略文件配置
└── readme.md               # 项目说明
```

## 注意事项

- 确保在运行代码前创建了必要的目录
- 模型文件较大，训练完成后会保存在`model_new/`目录
- 日志文件会保存在`logs/`目录
- 如果使用ViT编码器，需要下载相应的预训练模型