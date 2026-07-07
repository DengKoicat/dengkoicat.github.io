---
title: "大语言模型对齐：PPO 篇"
date: 2026-07-06T09:44:00+08:00
author: "dengkoicat"
tags: ["Deep Learning", "Reinforcement Learning", "LLM"]
categories: ["Deep Learning", "LLM"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 前言

大语言模型对齐（alignment）相关概念很多，例如 **SFT**、**RLHF**、**PPO**、**DPO**、**GRPO**、**GSPO** 等。它们并不是同一层级的东西，可以先按“训练阶段”和“优化方法”来理解：

```text
预训练 Base Model
        ↓
SFT：监督微调，让模型学会按指令回答
        ↓
偏好 / 强化对齐阶段
        ├── RLHF：先训练奖励模型，再用强化学习优化模型
        │       └── PPO：经典 RLHF 优化算法
        │
        ├── DPO：不显式训练奖励模型，也不跑在线 RL，直接用偏好数据优化模型
        ├── GRPO：去掉 Critic 的组相对策略优化，常用于推理增强
        └── GSPO：序列级策略优化，提升大模型 RL 训练稳定性
```

简单来说，SFT 解决“模型会不会按指令回答”，后续的偏好对齐解决“模型的回答是否更符合人类偏好或任务奖励”。

本文只展开 **PPO 在 RLHF 中的作用**。DPO、GRPO、GSPO 这里仅作为定位：它们都属于偏好对齐或强化对齐路线，但训练目标、数据使用方式、是否需要 Reward Model、是否需要在线采样都不同。

{{<figure
    src="aligned-against-baseline.png"
    caption="Fig. 1. 在 InstructGPT 实验中，SFT 和 PPO 训练后的模型都优于 GPT(Prompt) 基线。Image source: Training language models to follow instructions with human feedback, OpenAI, 2022."
    align="center"
    width="90%"
>}}

## SFT

SFT（Supervised Fine-Tuning）是 LLM 对齐训练的第一步，本质上是有监督训练。经过 Pretrain 的 LLM 只是一个会预测下一个 token 的 Base Model；SFT 的目的就是让这个 Base Model 变成一个可以**遵循指令**的 Instruct Model。

首先，我们需要整理高质量的指令数据集。每条样本通常包含用户输入和理想回答，这就是 SFT 的监督源。模型在训练过程中会学习“在这种指令下应该如何回答”，包括回答格式、语气、领域规范和安全边界。

SFT 和 Pretrain 的核心都是 next-token cross entropy，但二者有区别：

1. Pretrain 面向大规模通用语料，目标是学习语言、知识和模式；SFT 面向指令-回答数据，目标是学习交互方式。
2. Pretrain 通常对每个 token 算 loss；SFT 通常只对 assistant 的回答部分算 loss，用户提问部分只作为条件上下文。
3. Pretrain 常是大规模全参训练；SFT 在工程中可以全参微调，也可以使用 LoRA / QLoRA 等参数高效微调方法。

### SFT 数据集

训练数据本质是一个 $(x,y)$，其中 $x$ 是输入指令，$y$ 是理想输出。多轮对话数据通常写成 conversation 格式：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "你背后的模型是哪个版本？它由谁开发？"
    },
    {
      "role": "assistant",
      "content": "我是由 jingyaogong 开发的高效小参数 AI 模型。"
    },
    {
      "role": "user",
      "content": "你模型的训练数据来源是什么？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据涵盖多领域，确保覆盖广泛，但具体细节不公开。"
    }
  ]
}
```

如果把用户输入记为 $x$，assistant 回答记为 $y=(y_1,\dots,y_T)$，SFT 的损失可以写成：

$$
\mathcal{L}_{\text{SFT}}(\theta) =-\sum_{t=1}^{T}\log \pi_\theta(y_t \mid x, y_{\lt t})
$$

更贴近工程实现的写法会加入 mask，只在 assistant token 上计算 loss：

$$
\mathcal{L}_{\text{SFT}}(\theta) =-\sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid y_{\lt t})
$$

其中 $m_t=1$ 表示该 token 属于 assistant 回答，需要计入损失；$m_t=0$ 表示该 token 属于 prompt、system message 或其他不需要监督的位置。

### SFT 优缺点

SFT 的优势：

- 训练稳定，成本相对低，少量高质量样本就能明显改善模型行为。
- 能提升指令遵循、对话格式、回答风格和领域规范。
- 数据和训练流程直观，便于调试和复现。

SFT 的缺点：

- 训练结果高度依赖数据集质量，低质量样本会让模型学到坏格式、坏习惯甚至错误知识。
- 本质是模仿标准答案，不能直接表达“两个回答哪个更好”的偏好信息。
- 如果学习率、训练轮数或数据分布控制不好，可能破坏 Base Model 原有能力，出现灾难性遗忘。
- 对安全、伦理、复杂推理和多目标权衡的提升有限，因为这些问题往往不是单一标准答案能完全覆盖的。

## PPO 在 RLHF 中解决什么问题

已经有一个会回答问题的模型，如何通过人类偏好继续优化它，让它生成更符合人类意图的答案？这就是 RLHF（Reinforcement Learning from Human Feedback）的目标。

比如 LLM 知道 `1+1=2`，也知道“一加一等于二”。如果某个应用场景更偏好简洁、符号化的答案，SFT 只能让模型模仿训练集中出现的写法，却很难稳定表达“这个回答比另一个回答更好”。偏好对齐要解决的正是这个问题。

典型 RLHF 可以分成三步：

1. 先训练一个经过 SFT 的初始策略模型 $\pi_{\text{SFT}}$。
2. 收集同一 prompt 下多个回答的人类偏好排序，训练奖励模型 $r_\phi(x,y)$。
3. 用 PPO 微调语言模型，让它在不偏离参考模型太远的前提下获得更高奖励。

PPO 在这里不是“让模型学标准答案”，而是把语言模型看成一个策略（Policy），让它通过奖励信号调整输出分布。

### Reward Model 与 Bradley-Terry

RLHF 中的 Reward Model（RM）负责给模型回答打分。它不是直接从“正确答案”里学出来的，而是从人类偏好比较中学出来的。

假设同一个 prompt $x$ 下有两个回答：

$$
y_w \succ y_l
$$

其中 $y_w$ 是人类更喜欢的回答，$y_l$ 是人类不太喜欢的回答。Reward Model 给两个回答打分：

$$
r_\phi(x,y_w),\quad r_\phi(x,y_l)
$$

Bradley-Terry 模型是一种经典的成对比较概率模型。它关心的不是两个回答的绝对分数，而是分数差：

$$
P(y_w \succ y_l|x) = \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

对应损失为：

$$
\mathcal{L}_{RM}(\phi) = -\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
$$

这个 loss 的直觉很简单：让偏好回答 $y_w$ 的分数高于 rejected 回答 $y_l$。训练完成后，Reward Model 通常会被冻结，作为 PPO 阶段的奖励函数近似。

需要注意：Reward Model 学到的是人类偏好的近似，不是真理本身。如果偏好数据有偏、标注标准不稳定，或者 Reward Model 泛化不好，后续 PPO 就可能放大这些问题。

### LLM 如何对应强化学习

在 RLHF 中，LLM 可以被建模成一个策略：

| 强化学习概念 | LLM 中的对应物 |
| :--- | :--- |
| 状态 $s_t$ | prompt 加上当前已经生成的 token，即 $(x,y_{\lt t})$ |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 策略 $\pi_\theta(a_t \mid s_t)$ | 当前 LLM 的 token 概率分布 |
| 轨迹 $\tau$ | 一次完整生成 $(x,y)$ |
| 奖励 $r_\phi(x,y)$ | Reward Model 对完整回答的评分 |

给定 prompt $x$，LLM 按照自回归方式生成回答：

$$
y=(y_1,\dots,y_T)
$$

整段回答的概率可以写成：

$$
\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t|x,y_{\lt t})
$$

如果只看奖励最大化，目标可以写成：

$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta(\cdot|x)} \left[ r_\phi(x,y) \right]
$$

但直接最大化 Reward Model 分数会出问题。模型可能钻 RM 的空子，生成高分但怪异、啰嗦、谄媚、重复或分布崩坏的回答。这就是 reward hacking。

因此，LLM 的 PPO 通常不是只最大化 RM 分数，而是最大化带 KL 约束的目标：

$$
\max_\theta \mathbb{E}_{y\sim \pi_\theta} \left[ r_\phi(x,y) - \beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x) \right) \right]
$$

其中：

- $\pi_{\mathrm{ref}}$ 通常是 SFT 后冻结的参考模型。
- $r_\phi(x,y)$ 是 Reward Model 给完整回答的分数。
- $\beta$ 控制 KL 约束强度。$\beta$ 越小，模型越追求奖励，变化越大；$\beta$ 越大，模型越保守。

这也是 RLHF 中 PPO 的核心矛盾：既要让模型变得更符合偏好，又不能让它偏离原来的语言能力和指令遵循能力太远。

### KL 惩罚如何落到 token 上

完整计算两个语言模型分布之间的 KL 很贵，因为每一步都涉及整个词表分布。工程上常用 sampled token 上的 log-prob difference 作为近似 KL 惩罚。

生成第 $t$ 个 token 后，可以写成：

$$
r_t^{KL} = -\beta \left[ \log \pi_\theta(y_t|x,y_{\lt t}) - \log \pi_{\mathrm{ref}}(y_t|x,y_{\lt t}) \right]
$$

如果当前模型比参考模型更强烈地倾向这个 token，括号内为正，KL 惩罚为负；如果当前模型没有明显偏离参考模型，惩罚较小。

最终 token-level reward 常见形式是：

$$
r_t = \begin{cases} r_t^{KL}, & t \lt T \\ r_\phi(x,y)+r_t^{KL}, & t=T \end{cases}
$$

也就是说，中间 token 主要承受 KL 惩罚；回答结束时，再把 Reward Model 对完整回答的分数加到最后一步。

更直观地看，一条回答的总奖励大致是：

$$
R(x,y) = r_\phi(x,y) - \beta \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t|x,y_{\lt t}) - \log \pi_{\mathrm{ref}}(y_t|x,y_{\lt t}) \right]
$$

这里要纠正一个常见误解：KL 不是“奖励模型给的奖励”，而是约束项。它的作用是防止当前策略为了追求 RM 高分而偏离参考模型太远。

### Value Model

PPO 是 Actor-Critic 风格的方法。Actor 是当前要优化的语言模型策略 $\pi_\theta$；Critic 是 Value Model，用来估计当前状态未来能拿到多少回报。

Value Model 预测：

$$
V_\psi(s_t) \approx \mathbb{E} \left[ \sum_{k=0}^{T-t}\gamma^k r_{t+k} \mid s_t \right]
$$

其中 $s_t=(x,y_{\lt t})$。对于 LLM 来说，状态不是图像或棋盘，而是“prompt + 当前已经生成的 token 序列”。

工程上，Value Model 通常不会从随机权重开始训练，而是在语言模型 backbone 后面接一个 Value Head：

- Actor Head 输出词表分布，可以理解为 $d_{\text{model}} \times |V|$。
- Value Head 输出一个标量，可以理解为 $d_{\text{model}} \times 1$。

拿到一条轨迹的奖励序列 $[r_1,r_2,\dots,r_T]$ 后，可以计算每个状态的经验回报：

$$
G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

最直接的 Value Loss 是均方误差：

$$
L_t^{\text{unclip}}(\psi) = \left( V_\psi(s_t)-G_t \right)^2
$$

很多 PPO 实现会使用 value clipping。记录更新前的旧预测值 $V_{\text{old}}(s_t)$，限制新 value 不要一次变化太大：

$$
V_{\psi}^{\text{clip}}(s_t) = V_{\text{old}}(s_t) + \text{clip} \left( V_\psi(s_t)-V_{\text{old}}(s_t), -\epsilon, \epsilon \right)
$$

对应 clipped value loss：

$$
L_t^{\text{clip}}(\psi) = \left( V_{\psi}^{\text{clip}}(s_t)-G_t \right)^2
$$

最终 Value Loss 常写成：

$$
L_{V}(\psi) = \frac{1}{2} \max \left( L_t^{\text{unclip}}(\psi), L_t^{\text{clip}}(\psi) \right)
$$

这里用 max 是一种保守策略：如果 clipped 和 unclipped 两个版本中有一个误差更大，就按更大的误差惩罚，避免 value function 更新过猛。



### Advantage

只知道 reward 还不够。PPO 真正更新策略时，关心的是某个动作比当前状态下的平均水平好多少，这就是 Advantage。

最简单的优势函数可以写成：

$$
A_t = G_t - V_\psi(s_t)
$$

如果 $A_t>0$，说明这个 token 选择带来的回报高于 Value Model 的预期，应该提高它的概率；如果 $A_t<0$，说明这个 token 选择低于预期，应该降低它的概率。

实际 PPO 中常用 GAE（Generalized Advantage Estimation）来降低方差。先定义 TD residual：

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

再计算：

$$
A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$

其中：

- $\gamma$ 是折扣因子。
- $\lambda$ 控制 bias-variance trade-off。
- $\lambda$ 越接近 1，越接近完整 Monte Carlo return，偏差小但方差大。
- $\lambda$ 越接近 0，越依赖一步 TD，方差小但偏差大。

在 LLM RLHF 中，奖励通常比较稀疏：RM 分数主要在回答结束时给出，中间 token 主要是 KL 惩罚。因此 Advantage 的估计质量会直接影响训练稳定性。

### PPO-Clip

PPO 的核心是限制新旧策略之间的变化幅度。先定义重要性采样比率：

$$
\rho_t(\theta) = \frac{ \pi_\theta(y_t|s_t) }{ \pi_{\theta_{\text{old}}}(y_t|s_t) }
$$

如果 $\rho_t(\theta)>1$，说明新策略比旧策略更倾向于生成这个 token；如果 $\rho_t(\theta)<1$，说明新策略降低了这个 token 的概率。

普通 policy gradient 会直接最大化：

$$
\rho_t(\theta)A_t
$$

但这样可能导致策略一步更新太大。PPO-Clip 的目标函数为：

$$
L_t^{\text{PPO}}(\theta) = \min \left( \rho_t(\theta)A_t,\, \text{clip} \left( \rho_t(\theta), 1-\epsilon, 1+\epsilon \right) A_t \right)
$$

clip 的作用要分情况理解：

- 如果 $A_t>0$，说明这个 token 比预期好，模型应该提高它的概率；但 $\rho_t$ 最多提高到 $1+\epsilon$，超过就不再给额外收益。
- 如果 $A_t<0$，说明这个 token 比预期差，模型应该降低它的概率；但 $\rho_t$ 最多降低到 $1-\epsilon$，超过也不再给额外收益。

这就是 PPO 里的 proximal：每次更新只允许策略在旧策略附近移动，避免一次梯度更新把模型推崩。

### PPO 的完整 Loss

实际训练中，PPO 往往把 policy loss、value loss、entropy bonus、KL penalty 组合起来。若按“最小化 loss”的写法，可以写成：

$$
L(\theta,\psi) = - \mathbb{E} \left[ L_t^{\text{PPO}}(\theta) \right] + c_v L_V(\psi) - c_e H(\pi_\theta) + \beta D_{\mathrm{KL}} \left( \pi_\theta \| \pi_{\mathrm{ref}} \right)
$$

其中：

- 第一项是 policy loss，用来提高高 Advantage token 的概率，降低低 Advantage token 的概率。
- 第二项是 value loss，用来训练 Critic。
- 第三项是 entropy bonus，用来鼓励探索，避免策略过早变得过于确定。
- 第四项是 KL penalty，用来约束当前模型不要偏离参考模型太远。

{{<figure
    src="ppo.png"
    caption="Fig. 2. PPO 过程"
    align="center"
    width="90%"
>}}





## 参考

[1] Long Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.

[2] John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[3] Hugging Face. [Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/zh/rlhf).
