---
title: "LoRA"
date: 2026-06-14
description: "LoRA,a parameter-efficient fine-tuning (PEFT) method."
slug: "lora"
tags:
  - LoRA
categories:
  - AI
---

## Core idea

继续来讲一下 **LoRA** 微调，全称 **Low-Rank Adaptation**。  
**LoRA** 的核心思想是不直接更新原模型的大矩阵，而是对 Linear层额外训练两个低秩小矩阵 A 和 B，用它们来近似参数更新量。  
使用 **LoRA** 就可以让微调训练的参数量急剧减少，在大模型中一般训练方阵 $\textbf{W}_q,\textbf{W}_v,\textbf{W}_k,\textbf{W}_o$  
$$\textbf{W} \in \mathbb{R}^{N \times N}$$
**LoRA** 就是用两个低秩矩阵 $\textbf{B} \in \mathbb{R}^{N \times r},\textbf{A} \in \mathbb{R}^{r \times N}$ 来代替 $\textbf{W}$ 全参训练  
训练时冻结原来的参数 $\textbf{W}$ ，只训练 $\textbf{A},\textbf{B}$  
$$B\cdot A\cdot x = \Delta\textbf{W}\cdot x$$
所以 LoRA 不是一种新的训练目标，而是一种参数高效微调方式，或者说是一种低秩参数化的更新方式。*MiniMind* 中 **LoRA** 接在 **SFT** 之后，用来适配医疗，考试和 LLM 自我认知场景。  

用 *GPT-3* 举例，$\textbf{W}_{Q,V} \in \mathbb{R}^{12288 \times 12288}$  ，如果小任务中 $\text{rank}(r) = 14$，

| Method | $\textbf{W}_{Q}$ | $\textbf{W}_{V}$ | params         |
| ------ | ---------------- | ---------------- | -------------- |
| Full   | 150M             | 150M             | 300M           |
| LoRA   | 0.34M            | 0.34M            | 0.68M $0.23\%$ |

## Code

>*MiniMind-Based PyTorch* Implementation

训练数据和 **SFT** 基本一致
```json
{"conversations": 
	[{"role": "user", 
	  "content": "上班一族，经常坐在电脑前半天都不动，经常脖子疼、腰疼和腕关节疼想得到的帮助：最近买了一个低周波治疗器，请问可以经常使用吗？多长时间使用一次较好"},
	 {"role": "assistant", 
	  "content": "长时间坐在电脑前确实可能导致脖子、腰部和腕关节疼痛。这是因为长时间保持同一姿势可能导致肌肉紧张和关节压力增加。低周波治疗器可以帮助缓解这些疼痛，它通过电刺激来放松肌肉，减轻疼痛。
	  \n\n低周波治疗器的使用频率和时间取决于你的疼痛程度和个人舒适度。一般来说，每次使用时间可以在10-30分钟之间，每天可以使用1-2次。但是，如果你在使用过程中感到不适，应立即停止使用，并咨询医生。
	  \n\n此外，除了使用低周波治疗器，你还可以采取以下措施来缓解疼痛：\n\n1. 定时休息：每隔一小时起身走动5-10分钟，做一些伸展运动。\n\n2. 调整坐姿：确保电脑屏幕位于眼睛水平线上方，腰部和背部得到支撑，脚平放在地上。\n\n3. 做一些针对颈部、腰部和腕部的伸展和强化运动。\n\n4. 保持良好的生活习惯，包括健康饮食和充足的睡眠。\n\n5. 如果疼痛持续不减或者加重，应及时就医。\n\n请记住，低周波治疗器只是缓解疼痛的一种工具，最重要的是调整生活习惯和工作习惯，预防疼痛的发生。"}]}
```

1. 定义 **LoRA** 模块   
$$\text{LoRA}(x) = B(Ax)$$
- $\textbf{A}$ ，高斯初始化，保证 $\textbf{B}$ 可以收到梯度，防止整个 **LoRA** 分支为 0  
- $\textbf{B} = 0$，保证训练开始时 $\textbf{B} \cdot \textbf{A} \cdot x= 0$，刚开始训练时初始等价原模型  
```python
class LoRA(nn.Module):

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

2. 插入 **LoRA**
 - 遍历模型，找到所有 Linear 层，且 $IN_{dim} = OUT_{dim}$   
 - 主流 *PEFT* 一般会指定模块名字，如 `q_proj` `v_proj`   
$$y = Wx+BAx$$
```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) 
	        and module.in_features == module.out_features:

            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

3. 冻结原参数
  -  只把 LoRA 参数加入optimizer
```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)
# 统计参数
total_params = sum(p.numel() for p in model.parameters())
lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)

...
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

## Train vs Inference

训练阶段，原始权重 $\textbf{W}$ 被冻结，只有 **LoRA** 的 $\textbf{A} \textbf{B}$ 两个矩阵参与更新： 
$$y = Wx + BAx$$

推理阶段有两种方式： 
1. 保留 **LoRA** 分支，推理时仍然计算 $Wx + BAx$  
2. 将 **LoRA** 权重合并回原权重： 
$$W' = W + BA$$
合并后前向就重新变成普通 Linear：
$$y = W'x$$
这样推理时不需要额外的 LoRA 分支。


