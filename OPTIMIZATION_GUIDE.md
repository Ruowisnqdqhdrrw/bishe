# 手写数学公式识别优化指南

## 优化目标
- 字符准确率 >= 80%
- 公式准确率 >= 80%

## 当前状态
根据训练日志，原模型最佳性能：
- 序列准确率（公式）: ~70.5%
- Token准确率（字符）: ~86.5%

## 优化方案

### 1. 模型架构优化 (model_e2e_v3.py)
- **SE注意力模块**: 增强通道注意力，提升特征表达能力
- **覆盖注意力机制**: 防止重复关注和遗漏，提高长公式识别准确率
- **更深的网络**: 3层LSTM Decoder + 残差连接
- **位置编码**: 增强序列位置信息
- **DropPath正则化**: 防止过拟合

### 2. 数据增强优化 (dataset_e2e_v3.py)
- 弹性变形：模拟手写自然变化
- 透视变换：模拟拍摄角度变化
- 笔画粗细变化：模拟不同书写风格
- CutOut：随机遮挡提升鲁棒性
- 高斯噪声和椒盐噪声
- 亮度/对比度调整

### 3. 训练策略优化 (train_e2e_v3.py)
- **Warmup + Cosine学习率调度**: 更稳定的训练
- **Label Smoothing**: 防止过拟合
- **梯度累积**: 等效更大batch size
- **覆盖损失**: 惩罚重复关注
- **更慢的Teacher Forcing衰减**: 更充分的学习

## 使用方法

### 训练新模型
```bash
# 方法1: 使用批处理文件
train_v3.bat

# 方法2: 直接运行Python
python -m train.train_e2e_v3 --data_root "D:/CROHME/CROHME" --epochs 200 --batch_size 16 --device cuda --amp
```

### 恢复训练
```bash
train_v3_resume.bat
```

### 评估模型
```bash
# 评估V3模型
python evaluate_model.py --checkpoint checkpoints_e2e_v3/best.ckpt --data_root "D:/CROHME/CROHME"

# 评估原模型
python evaluate_model.py --checkpoint checkpoints_e2e/best.ckpt --data_root "D:/CROHME/CROHME"
```

## 推荐训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| epochs | 200-250 | 足够的训练轮数 |
| batch_size | 16 | 根据显存调整 |
| lr | 8e-4 | 初始学习率 |
| dropout | 0.4 | Decoder dropout |
| encoder_dropout | 0.2 | Encoder dropout |
| teacher_forcing | 0.8 | 初始TF比例 |
| tf_decay | 0.995 | TF衰减率 |
| warmup_epochs | 10 | 预热轮数 |
| patience | 35 | 早停耐心值 |

## 文件结构

```
├── train/
│   ├── model_e2e.py        # 原始模型
│   ├── model_e2e_v3.py     # 优化版模型 ★
│   ├── dataset_e2e.py      # 原始数据集
│   ├── dataset_e2e_v3.py   # 优化版数据集 ★
│   ├── train_e2e.py        # 原始训练脚本
│   └── train_e2e_v3.py     # 优化版训练脚本 ★
├── inference/
│   └── predictor_e2e.py    # 推理模块（支持V2/V3）
├── checkpoints_e2e/        # 原模型检查点
├── checkpoints_e2e_v3/     # V3模型检查点 ★
├── train_v3.bat            # V3训练脚本 ★
├── train_v3_resume.bat     # V3恢复训练 ★
└── evaluate_model.py       # 评估脚本 ★
```

## 预期效果

经过优化后，预期可以达到：
- 字符准确率: 85-90%
- 公式准确率: 80-85%

## 注意事项

1. **显存要求**: 建议至少6GB显存，如果显存不足可以减小batch_size
2. **训练时间**: 完整训练约需要8-12小时（RTX 3060/4060级别）
3. **数据路径**: 确保CROHME数据集路径正确
4. **早停机制**: 如果35个epoch没有提升会自动停止

## 问题排查

1. **CUDA内存不足**: 减小batch_size或关闭amp
2. **训练不收敛**: 检查数据路径，降低学习率
3. **过拟合**: 增加dropout，增强数据增强
