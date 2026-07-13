---
title: "LLMs 微调对齐实战"
date: 2026-07-13T13:27:00+08:00
author: "dengkoicat"
tags: ["RL", "LLM","AI","Fine-Tuning"]
categories: ["LLM"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## Model

模型文件结构：

```text
LLM-Model/
│
├── 模型配置
│   ├── config.json 定义模型架构，如层数、隐藏维度、注意力头、词表大小和上下文长度
│   └── generation_config.json 定义默认生成参数，如 temperature、top_p、do_sample 和特殊 token ID
│
├── 模型权重
│   ├── model.safetensors 保存模型训练得到的完整权重，通常用于较小模型
│   ├── model-00001-of-000xx.safetensors
│   ├── ...
│   ├── model-000xx-of-000xx.safetensors 分片保存较大模型的权重参数
│   └── model.safetensors.index.json 记录每个模型参数位于哪个权重分片中
│
├── 分词器
│   ├── tokenizer.json  保存完整词表、分词规则以及 token 与 ID 的映射
│   ├── tokenizer_config.json  保存分词器类型、最大长度、特殊 token 和 padding 配置。
│   ├── special_tokens_map.json  定义 BOS、EOS、PAD、UNK 等特殊 token。
│   ├── vocab.json 保存词表，常见于 BPE 分词器
│   ├── merges.txt 保存 BPE 的 token 合并规则
│   └── tokenizer.model 保存 SentencePiece 分词模型，部分模型使用
│
├── 对话模板
│   └── chat_template.jinja 将 system、user、assistant、tool 等消息转换成模型实际输入格式。
│
├── 自定义模型代码（可选）
    ├── configuration_xxx.py 定义自定义模型配置类   
    ├── modeling_xxx.py 定义模型网络结构和前向计算逻辑
    └── tokenization_xxx.py 定义自定义分词器逻辑
```

## Dataset

数据集文件结构：

```text
Text-Dataset/
│
├── README.md 数据集说明
├── LICENSE 数据集许可证
├── train.jsonl 训练数据
├── validation.jsonl 验证数据
└── test.jsonl 测试数据
```

### 统一约定

生产中建议将 prompt 统一表示为消息列表，避免在数据层拼接 Qwen、Llama 等模型的聊天模板：

```json
{
  "prompt": [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "解释什么是过拟合。"}
  ]
}
```

文中的 `chosen`、`rejected`、`completion` 都是 **assistant 的纯文本回答**。不同框架也可能要求将它们表示为 assistant 消息列表；这只是序列化差异，语义不变。

{{< figure
    src="relation.png"
    caption="Fig. 1. 主观偏好任务的常见生产链路。对于数学、代码等可验证任务，PPO、GRPO、GSPO 中的 RM 可以被确定性奖励函数替代。"
    align="center"
    width="90%"
>}}



### SFT

SFT（Supervised Fine-Tuning）需要“上下文 + 标准回答”。推荐格式：

```json
{"messages":[
  {"role":"system","content":"你是一个专业的客服助手。"},
  {"role":"user","content":"订单什么时候发货？"},
  {"role":"assistant","content":"订单通常会在付款后 48 小时内发出。"}
]}
```

多轮 SFT 只需要继续追加消息：

```json
{"messages":[
  {"role":"user","content":"什么是 LoRA？"},
  {"role":"assistant","content":"LoRA 是一种参数高效微调方法。"},
  {"role":"user","content":"它的主要优点是什么？"},
  {"role":"assistant","content":"它只训练少量新增参数，因此显存和存储成本较低。"}
]}
```

旧式 Alpaca 数据也很常见：

```json
{"instruction":"解释什么是 LoRA。","input":"","output":"LoRA 是一种参数高效微调方法。"}
```

其语义等价于：`instruction + input` 是 user 内容，`output` 是 assistant 内容。

### Reward Model（RM）

RM 的任务不是生成文本，而是对候选回答排序或打分。它训练所需的数据是偏好对：同一个 prompt 下，一条 `chosen` 优于一条 `rejected`。

```json
{
  "prompt": [
    {"role":"user","content":"请简要解释什么是过拟合。"}
  ],
  "chosen": "过拟合是模型过度记住训练数据，导致它在新数据上的泛化能力下降。",
  "rejected": "过拟合就是模型训练得很久。"
}
```

RM 训练完成后，对任意 `(prompt, completion)` 输出一个标量奖励：

```text
reward = RM(prompt, completion)
```

PPO、GRPO、GSPO 在主观偏好任务中依赖这个分数作为优化目标。因此，偏好对不是直接喂给它们的策略训练数据，而是先用于训练 RM。

### DPO

DPO 使用的输入与 RM 相同，仍然是偏好对：

```json
{
  "prompt": [
    {"role":"user","content":"给出一条节水建议。"}
  ],
  "chosen": "优先修复漏水点，并使用节水型器具。",
  "rejected": "多用水可以保证生活质量。"
}
```

差别在于：DPO **不需要先单独训练 RM**。它直接根据 `chosen` 与 `rejected` 的相对偏好更新策略模型。

数据质量要求：

1. `chosen` 和 `rejected` 必须回答同一个 prompt。
2. `chosen` 应在正确性、安全性、帮助性或风格上显著更优。
3. 不要把“回答更长”当作“回答更好”。

### PPO

PPO 的策略优化数据通常只有 prompt：

```json
{
  "prompt": [
    {"role":"user","content":"用三句话解释光合作用。"}
  ]
}
```

训练过程为：

```text
读取 prompt
-> 当前策略在线生成 completion
-> RM(prompt, completion) 计算奖励
-> PPO 根据奖励更新策略
```

因此 PPO 需要两份不同的数据：

```text
RM 训练数据：prompt + chosen + rejected
PPO 策略数据：prompt
```

若使用可验证任务，也可以用规则函数代替 RM，例如检查数学最终答案或代码单元测试是否通过。

### GRPO/GSPO

GRPO/GSPO 的策略训练数据与 PPO 类似，通常也是 prompt：

```json
{
  "prompt": [
    {"role":"user","content":"计算 37 乘以 19，并只输出最终整数。"}
  ]
}
```

与 PPO 的关键差别是：对于同一个 prompt，GRPO/GSPO 会一次生成多条 completion，并根据一组奖励的相对高低进行优化。

主观偏好任务的奖励来自 RM：

```text
completions = policy.generate(prompt, n=G)
rewards = [RM(prompt, completion) for completion in completions]
```

可验证任务通常在数据中提供真值，并由规则奖励函数读取：

```json
{
  "prompt": [
    {"role":"user","content":"计算 37 乘以 19。请把最终答案放在 \\boxed{} 中。"}
  ],
  "answer": "703"
}
```

此时奖励可以是：答案正确得 1 分，格式正确得 0.1 分。`answer`、`solution`、`ground_truth` 等字段名必须与奖励函数读取的参数一致。


## Post-Training

实践中，SFT 和 DPO 通常使用 MS-Swift：它对 Qwen、ModelScope 数据集和常见微调流程支持完善，命令行简单，适合单机或中小规模集群训练。PPO、GRPO、GSPO 则更常使用 verl，因为这类在线强化学习需要高吞吐地生成多条回答、调用奖励模型评分，并在多机多卡环境中协调训练与 rollout；verl 对这类分布式 RL 流程更偏工程化。

### MS-Swift 实践

#### SFT

| 模块 | 内容 |
| :--- | :--- |
|   **LLM**  | Qwen2.5-0.5B-Instruct |
|   **Dataset**   | alpaca-gpt4-data-zh，csv格式 |
|   **PEFT**  | ✗ |

SFT 全参数微调：

```bash
swift sft \
  --model "${MODEL_PATH}" \  # 模型目录
  --tuner_type full \        # 全参更新
  --dataset "${DATA_PATH}" \ # 数据集目录
  --torch_dtype bfloat16 \   # BF16
  --num_train_epochs 3 \     # EPOCH
  --learning_rate 1e-5 \     # 学习率
  --per_device_train_batch_size 2 \   # 每张 GPU 每个微批次处理 2 条样本
  --gradient_accumulation_steps 8 \   # 连续计算 8 个微批次的梯度后，才执行一次优化器更新
  --max_length 2048 \                 # 最多保留 2048 个 token
  --split_dataset_ratio 0.01 \        # 1% 验证集
  --output_dir "${OUTPUT_DIR}"
```
- 总步数 48330/(2 x 8 x 1) x 3 = 9063 
$$\text{Total Steps} = \left\lceil \frac{\text{训练集样本数}}{\text{全局 Batch Size}} \right\rceil \times \text{训练轮数 (Epochs)}$$

