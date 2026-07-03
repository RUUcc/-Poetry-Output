# 基于字符级循环神经网络的古诗生成

## 项目简介

神经网络与深度学习期末课程设计。使用字符级 RNN/LSTM/GRU 语言模型，训练后给定前缀自动续写古诗。

## 目录结构

```
代码实现/
├── main/
│   ├── poetry_clean.ipynb    # 主 Notebook（数据处理、模型、训练、评估）
│   ├── loss_curve_all.png    # 四模型 Loss 汇总图
│   └── loss_curve_4models.png
├── poetry_clean_brief.txt     # 训练数据（3,000 首古诗，实际使用）
├── poetry_clean.txt           # 完整数据（20,000 首古诗）
└── README.md
```

## 数据流程

```
古诗文本 → 字符级词表 → Token 编码 → 滑动窗口 → DataLoader → 模型训练 → 贪心解码生成
```

### 数据集

- `poetry_clean_brief.txt`：3,000 首古诗，140,714 个 token
- 词表大小：4,041 个汉字
- 特殊 Token：`<PAD>=0` `<UNK>=4042` `<BOS>=4043` `<EOS>=4044`

### 滑动窗口

| 参数 | 值 |
|------|-----|
| SEQ_LEN | 64 |
| STRIDE | 32 |
| 样本总数 | 4,396 |
| 训练/测试 | 90% / 10% |
| BATCH_SIZE | 128 |

## 模型架构

所有模型共享：`Embedding(4044, 128)` → RNN/LSTM/GRU → `Linear(128, 4044)`

| 模型 | 类型 | 层数 | 参数量 |
|------|------|------|--------|
| CharRNN | RNN | 1 | 1,072,589 |
| CharLSTM | LSTM | 1 | 1,171,661 |
| CharGRU | GRU | 1 | 1,138,637 |
| CharLSTM2Layer | LSTM | 2 (dropout=0.3) | 1,303,757 |

## 训练配置

| 参数 | 值 |
|------|-----|
| EMBED_DIM | 128 |
| HIDDEN_SIZE | 128 |
| 损失函数 | CrossEntropyLoss (ignore_index=PAD) |
| 优化器 | Adam (lr=0.001) |
| 梯度裁剪 | max_norm=1.0 |
| Epoch | 20 |

## 实验结果

### Loss 对比

| 模型 | Train Loss | Test Loss | 参数量 |
|------|-----------|-----------|--------|
| CharRNN | 0.0775 | 0.0822 | 1,072,589 |
| CharLSTM | 0.0845 | 0.0872 | 1,171,661 |
| CharGRU | 0.0799 | 0.0844 | 1,138,637 |
| CharLSTM2Layer | 0.0929 | 0.0936 | 1,303,757 |

Test Loss 排名：**RNN < GRU < LSTM < LSTM2Layer**

### 生成效果

使用贪心解码（greedy decoding），给定前缀自动续写，遇到 `<EOS>` 停止。受限于小数据集（3,000 首）和较小模型（hidden=128），生成结果偏短。

## 运行方式

1. 安装依赖：`pip install torch numpy matplotlib`
2. 打开 `main/poetry_clean.ipynb`
3. 按顺序运行所有 Cell（Cell 1 → Cell 13）

## 技术决策

- **Adam lr=0.001**：自适应学习率适配 RNN 梯度不稳定和 Embedding 稀疏更新
- **STRIDE=32**：相比 STRIDE=16 减少一半样本量，加速 CPU 训练
- **EMBED_DIM=HIDDEN_SIZE=128**：满足 ≥128 要求的最小值，平衡容量与速度
- **数据集降级**：从 20,000 首降至 3,000 首以加速迭代

## 关键技术点

- 字符级语言建模（逐字预测，非词级）
- `ignore_index=<PAD>` 避免填充位置污染 loss 计算
- 梯度裁剪防止 RNN 梯度爆炸
- 贪心解码 + 早停（遇到 `<EOS>` 停止）
