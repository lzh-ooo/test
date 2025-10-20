# MateConv 预训练优化版 - 使用说明

## 📚 目录

1. [快速开始](#快速开始)
2. [数据处理](#数据处理)
3. [模型训练](#模型训练)
4. [模型推理](#模型推理)
5. [高级功能](#高级功能)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers datasets jsonlines tqdm pyyaml numpy
```

### 目录结构

```
Labelimg/
├── model/                          # 模型代码目录（需要您提供）
│   ├── model.py                   # Transformer 模型
│   ├── LMConfig.py               # 模型配置
│   └── dataset.py                # 数据集类
├── dataset/                       # 数据目录
│   ├── mobvoi_seq_monkey_general_open_corpus.jsonl  # 原始数据
│   └── pretrain_data.bin         # 处理后的数据
├── out/                          # 输出目录
│   └── *.pth                     # 模型检查点
├── config.yaml                   # 配置文件
├── optimized_data_processing.py  # 优化的数据处理
├── optimized_pretrain.py         # 优化的训练脚本
├── optimized_inference.py        # 优化的推理脚本
├── train.py                      # 训练启动脚本
├── inference.py                  # 推理启动脚本
└── utils.py                      # 工具函数
```

---

## 📊 数据处理

### 方法 1: 使用 Python 脚本

```python
from optimized_data_processing import OptimizedDataProcessor

# 创建处理器
processor = OptimizedDataProcessor(
    tokenizer_path='/path/to/tokenizer',
    max_length=512,
    chunk_size=50000,
    save_interval=1000000
)

# 验证和清洗数据
processor.validate_and_clean_jsonl(
    input_path='./dataset/raw_data.jsonl',
    output_path='./dataset/cleaned_data.jsonl'
)

# 处理数据集
processor.process_dataset(
    input_path='./dataset/cleaned_data.jsonl',
    output_path='./dataset/pretrain_data.bin',
    resume=True  # 支持断点续传
)

# 保存统计信息
processor.save_statistics('./dataset/data_stats.json')
```

### 方法 2: 直接运行主函数

```bash
python optimized_data_processing.py
```

### 数据格式要求

输入数据应为 JSONL 格式，每行一个 JSON 对象：

```json
{"text": "这是第一条文本数据"}
{"text": "这是第二条文本数据"}
```

---

## 🎯 模型训练

### 方法 1: 使用配置文件（推荐）

**1. 编辑配置文件 `config.yaml`**

```yaml
data:
  tokenizer_path: "/path/to/tokenizer"
  train_data_path: "./dataset/pretrain_data.bin"
  val_data_path: null

training:
  epochs: 15
  batch_size: 32
  learning_rate: 3.0e-4
  optimizer_type: "adamw"
  lr_scheduler: "cosine"
  use_ema: true
  early_stopping: true
```

**2. 启动训练**

```bash
python train.py --config config.yaml
```

### 方法 2: 使用命令行参数

```bash
python train.py \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --optimizer_type adamw \
    --lr_scheduler cosine \
    --use_ema \
    --early_stopping \
    --out_dir out
```

### 方法 3: 分布式训练

**单机多卡训练：**

```bash
# 使用 torchrun (PyTorch >= 1.10)
torchrun --nproc_per_node=2 train.py \
    --config config.yaml \
    --ddp

# 或使用 python -m torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py --config config.yaml --ddp
```

### 恢复训练

```bash
python train.py \
    --config config.yaml \
    --resume \
    --resume_from out/checkpoint_step_5000.pth
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 15 |
| `--batch_size` | 批次大小 | 32 |
| `--learning_rate` | 学习率 | 3e-4 |
| `--optimizer_type` | 优化器 (adamw/adam) | adamw |
| `--lr_scheduler` | 学习率调度器 (cosine/linear) | cosine |
| `--use_ema` | 使用 EMA | False |
| `--grad_clip` | 梯度裁剪 | 1.0 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--dtype` | 数据类型 (float32/float16/bfloat16) | bfloat16 |
| `--early_stopping` | 启用早停 | False |
| `--patience` | 早停耐心值 | 5 |

---

## 💬 模型推理

### 方法 1: 单次生成

```bash
python inference.py \
    --model_path out/pretrain_512.pth \
    --prompt "中国" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

### 方法 2: 流式输出

```bash
python inference.py \
    --model_path out/pretrain_512.pth \
    --prompt "长江、" \
    --stream
```

### 方法 3: 交互式对话

```bash
python inference.py \
    --model_path out/pretrain_512.pth \
    --interactive
```

交互式对话示例：

```
🤖 交互式对话模式
================================================================================
输入 'exit'、'quit' 或 'q' 退出
输入 'clear' 清空对话历史
================================================================================

👤 用户: 你好
🤖 助手: 你好！有什么可以帮助你的吗？

👤 用户: 介绍一下中国
🤖 助手: 中国是一个历史悠久的国家...

👤 用户: exit
👋 再见！
```

### 方法 4: 批量生成

**1. 创建提示文件 `prompts.txt`：**

```
中国
长江、黄河
你好，好久不见
```

**2. 运行批量生成：**

```bash
python inference.py \
    --model_path out/pretrain_512.pth \
    --batch_file prompts.txt
```

### 方法 5: 使用 Python API

```python
from transformers import AutoTokenizer
from model.model import Transformer
from model.LMConfig import LMConfig
from optimized_inference import TextGenerator, GenerationConfig

# 加载模型
lm_config = LMConfig()
model = Transformer(lm_config)
model.load_state_dict(torch.load('out/pretrain_512.pth'))

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('/path/to/tokenizer')

# 创建生成器
generator = TextGenerator(model, tokenizer)

# 生成配置
gen_config = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    max_new_tokens=100,
    repetition_penalty=1.2
)

# 单次生成
result = generator.generate("中国", gen_config)
print(result)

# 流式生成
def callback(token):
    print(token, end="", flush=True)

result = generator.generate(
    "长江、",
    gen_config,
    stream=True,
    callback=callback
)

# 批量生成
prompts = ["中国", "长江、", "你好"]
results = generator.batch_generate(prompts, gen_config)

# 多轮对话
history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "介绍一下中国"}
]
response = generator.chat(history, gen_config)
```

### 采样参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--temperature` | 温度参数，越高越随机 | 0.7-1.0 |
| `--top_k` | Top-K 采样，保留概率最高的 k 个词 | 40-50 |
| `--top_p` | Top-P 采样，保留累积概率达到 p 的词 | 0.85-0.95 |
| `--repetition_penalty` | 重复惩罚，越高越不重复 | 1.0-1.3 |
| `--no_repeat_ngram_size` | 禁止重复的 n-gram 大小 | 0-3 |
| `--greedy` | 使用贪心解码（最确定） | - |

**采样策略建议：**

- **创意写作**: `temperature=1.0, top_k=50, top_p=0.95`
- **日常对话**: `temperature=0.8, top_k=40, top_p=0.9`
- **事实性回答**: `temperature=0.5, top_k=20, top_p=0.85` 或使用 `--greedy`

---

## 🔧 高级功能

### 1. 使用 Wandb 跟踪训练

```bash
pip install wandb
wandb login

python train.py \
    --config config.yaml \
    --use_wandb \
    --wandb_project "MateConv-Pretrain" \
    --wandb_run_name "exp1"
```

### 2. 梯度累积（模拟大 Batch）

```bash
# 等效于 batch_size=128 (32 * 4)
python train.py \
    --batch_size 32 \
    --accumulation_steps 4
```

### 3. 混合精度训练

```bash
# 使用 BF16（推荐在 Ampere 架构 GPU 上使用）
python train.py --dtype bfloat16

# 使用 FP16（通用）
python train.py --dtype float16

# 使用 FP32（最高精度但最慢）
python train.py --dtype float32
```

### 4. 模型 EMA（提升生成质量）

```bash
python train.py --use_ema
```

### 5. 验证集和早停

```yaml
# config.yaml
data:
  val_data_path: "./dataset/val_data.bin"

training:
  early_stopping: true
  patience: 5
  eval_interval: 500
```

### 6. 自定义学习率调度

```bash
# 余弦退火（推荐）
python train.py --lr_scheduler cosine

# 线性衰减
python train.py --lr_scheduler linear

# 多项式衰减
python train.py --lr_scheduler polynomial
```

---

## 📈 性能优化建议

### 训练加速

1. **使用混合精度训练**: `--dtype bfloat16` 或 `--dtype float16`
2. **增大 batch size**: 在显存允许的情况下尽可能大
3. **使用梯度累积**: 显存不足时使用 `--accumulation_steps`
4. **多卡训练**: 使用 `--ddp` 启用分布式训练
5. **减少日志频率**: `--log_interval 100`

### 显存优化

1. **降低 batch size**: `--batch_size 16`
2. **使用梯度累积**: `--accumulation_steps 4`
3. **使用混合精度**: `--dtype float16`
4. **减少最大序列长度**: 修改配置文件中的 `max_seq_len`

### 生成加速

1. **使用 KV Cache**: `use_cache=True`（默认启用）
2. **减少生成长度**: `--max_new_tokens 50`
3. **使用贪心解码**: `--greedy`
4. **批量生成**: 使用 `batch_generate()` 而不是循环调用

---

## ❓ 常见问题

### Q1: 训练时显存不足怎么办？

**A:** 尝试以下方法：
```bash
python train.py \
    --batch_size 8 \
    --accumulation_steps 4 \
    --dtype float16
```

### Q2: 如何恢复中断的训练？

**A:**
```bash
python train.py \
    --resume \
    --resume_from out/checkpoint_interrupted.pth
```

### Q3: 生成的文本重复怎么办？

**A:** 增加重复惩罚和使用 ngram 阻止：
```bash
python inference.py \
    --repetition_penalty 1.3 \
    --no_repeat_ngram_size 3
```

### Q4: 如何使生成更有创意/更保守？

**A:** 调整温度参数：
- **更有创意**: `--temperature 1.2`
- **更保守**: `--temperature 0.5` 或 `--greedy`

### Q5: 训练速度太慢怎么办？

**A:**
1. 使用混合精度: `--dtype bfloat16`
2. 增大 batch size: `--batch_size 64`
3. 使用多卡训练: `--ddp`
4. 减少验证频率: `--eval_interval 1000`

### Q6: 如何查看训练进度？

**A:** 使用 Wandb：
```bash
python train.py --use_wandb --wandb_project "my-project"
```
然后访问 https://wandb.ai 查看可视化

---

## 📝 完整示例

### 示例 1: 从零开始训练

```bash
# 1. 处理数据
python optimized_data_processing.py

# 2. 训练模型
python train.py \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --use_ema \
    --use_wandb

# 3. 测试生成
python inference.py \
    --model_path out/best_model.pth \
    --prompt "中国" \
    --stream
```

### 示例 2: 分布式训练 + 混合精度

```bash
torchrun --nproc_per_node=4 train.py \
    --config config.yaml \
    --ddp \
    --dtype bfloat16 \
    --batch_size 64 \
    --use_ema \
    --use_wandb
```

### 示例 3: 交互式对话

```bash
python inference.py \
    --model_path out/best_model.pth \
    --interactive \
    --temperature 0.8 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --stream
```

---

## 🎉 优化总结

相比原始代码，本优化版本提供了：

### 数据处理优化
- ✅ 合并清洗和预处理步骤
- ✅ 添加断点续传功能
- ✅ 详细的数据统计和可视化
- ✅ 更好的错误处理

### 训练优化
- ✅ 多种优化器选择 (AdamW, Adam)
- ✅ 灵活的学习率调度 (Cosine, Linear, Polynomial)
- ✅ 模型 EMA 提升生成质量
- ✅ 验证集和早停机制
- ✅ 完善的检查点管理
- ✅ Wandb 集成
- ✅ 分布式训练支持

### 推理优化
- ✅ 多种采样策略 (Greedy, Top-K, Top-P, Temperature)
- ✅ KV Cache 加速生成
- ✅ 批量生成支持
- ✅ 流式输出
- ✅ 交互式对话
- ✅ 重复惩罚和 ngram 阻止

### 代码质量
- ✅ 清晰的代码结构
- ✅ 详细的注释
- ✅ 配置文件支持
- ✅ 完善的错误处理
- ✅ 易于扩展

---

## 📞 支持

如有问题，请查看代码注释或提交 Issue。

祝训练顺利！🚀

