# 模型存储目录

## 文件夹作用

此文件夹用于存储训练好的图像描述生成模型文件，包含不同架构和配置的模型权重。

## 文件命名规则

模型文件名格式：`{encoder}_{hidden_dim}_{finetune_status}_{epoch}.pth`

- **encoder**: 编码器类型 (clip_vit, clip_rn50, resnet152)
- **hidden_dim**: 隐藏层维度 (512, 1024, 2048)
- **finetune_status**: 微调状态 (frozen, finetune)
- **epoch**: 训练轮数

## 模型类型说明

### CLIP模型
- `clip_vit_*.pth`: 基于CLIP ViT的模型
- `clip_rn50_*.pth`: 基于CLIP ResNet-50的模型

### ResNet模型
- `resnet152_*.pth`: 基于ResNet-152的模型

### 可视化结果
- `*_caption_result.png`: 模型生成描述的可视化结果
- `*_caption_result_vit.png`: 使用ViT编码器的可视化结果

## 使用说明

1. **模型加载**：在训练和推理脚本中指定模型文件路径
2. **模型选择**：根据任务需求选择合适的模型架构和配置
3. **结果查看**：查看对应的PNG文件了解模型效果

## 注意事项

- 模型文件较大，建议定期清理不需要的模型
- 确保模型文件与代码版本兼容
- 不同配置的模型性能可能差异较大，建议对比测试 