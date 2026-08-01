---
title: "大语言模型对齐"
date: 2026-07-06T09:44:00+08:00
author: "dengkoicat"
tags: ["AI", "LLM", "Post-training", "RLHF", "PPO", "DPO", "GRPO", "GSPO"]
categories: ["技术博客", "LLM"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

这篇博客想梳理大语言模型对齐阶段最常见的几条路线：**PPO、DPO、GRPO 和 GSPO**。

它们经常一起出现，但并不在同一个层级上。更准确地说，SFT 是让模型学会“按指令回答”的第一步；RLHF / DPO / GRPO / GSPO 则是在此基础上，让模型的输出更符合人类偏好、任务奖励或可验证标准。

一个简化的对齐流程如下：

```text
预训练 Base Model
        ↓
SFT：监督微调，让模型学会遵循指令
        ↓
偏好 / 强化对齐阶段
        ├── RLHF：先训练奖励模型，再用强化学习优化策略
        │       └── PPO：经典 RLHF 策略优化算法
        │       └── GRPO：去掉 Critic，用组内相对奖励估计 Advantage
        │       └── GSPO：把策略比率从 token 级改成 sequence 级
        │
        └── DPO：不显式训练奖励模型，直接用偏好对优化策略
```

本文的重点不是罗列名词，而是把每个方法的目标函数拆开：先看它想解决什么问题，再看公式为什么长成这样。

{{<figure
    src="aligned-against-baseline.png"
    caption="Fig. 1. 在 InstructGPT 实验中，SFT 和 PPO 训练后的模型都优于 GPT(Prompt) 基线。Image source: Training language models to follow instructions with human feedback, OpenAI, 2022."
    align="center"
    width="90%"
>}}

## 符号

| 符号 | 含义 |
| :--- | :--- |
| $x$ | 用户输入，也就是 prompt |
| $y$ | 模型生成的完整回答 |
| $y_t$ | 回答中的第 $t$ 个 token |
| $y_{\lt t}$ | 第 $t$ 个 token 之前已经生成的前缀 |
| $\pi(y \mid x; \boldsymbol{\theta})$ | 当前要训练的策略模型，也就是 Actor |
| $\pi_{\mathrm{ref}}(y \mid x)$ | 参考模型，通常是冻结的 SFT 模型 |
| $\pi(y \mid x; \boldsymbol{\theta}_{\mathrm{old}})$ | 策略更新前的旧模型，用于计算 PPO / GRPO 的概率比 |
| $r(x,y;\boldsymbol{\phi})$ | 奖励模型对回答 $y$ 的打分 |
| $v(s_t;\mathbf{w})$ | Value Model / Critic，对状态 $s_t$ 的价值估计 |
| $D_{\mathrm{KL}}(P\|Q)$ | KL 散度，用来衡量两个概率分布的差异 |
| $\beta$ | KL 约束或偏好差异项的权重 |
| $\sigma(z)$ | Sigmoid 函数，$\sigma(z)=\frac{1}{1+e^{-z}}$ |
| $y_w, y_l$ | 偏好数据中的 chosen response 与 rejected response |
| $u_t$ | 从第 $t$ 步开始的实际回报 |
| $\hat{A}_t$ | token 级 Advantage 估计，表示当前动作比平均水平好多少 |
| $A_i$ | 第 $i$ 条回答的组相对 Advantage |
| $G$ | 同一个 prompt 下采样的回答数量 |

## SFT：对齐的起点

SFT（Supervised Fine-Tuning）是大语言模型对齐的第一步。预训练模型只会预测下一个 token；SFT 的作用是把它变成能理解指令、按格式回答、遵守基本交互规范的 Instruct Model。

SFT 数据通常是一组输入输出对：

```json
{
  "messages": [
    {"role": "user", "content": "解释什么是过拟合。"},
    {"role": "assistant", "content": "过拟合是模型过度记住训练数据，导致它在新数据上的泛化能力下降。"}
  ]
}
```

如果把用户输入记为 $x$，assistant 回答记为 $y=(y_1,\dots,y_T)$，SFT 的基本损失为：

$$ \mathcal{L}_{\mathrm{SFT}}(\boldsymbol{\theta}) =-\sum_{t=1}^{T}\log \pi(y_t \mid x,y_{\lt t};\boldsymbol{\theta}) $$

工程实现中通常会加 mask，只在 assistant 的回答 token 上计算 loss：

$$ \mathcal{L}_{\mathrm{SFT}}(\boldsymbol{\theta}) =-\sum_{t=1}^{T}m_t\log \pi(y_t \mid x,y_{\lt t};\boldsymbol{\theta}) $$

其中 $m_t=1$ 表示该 token 属于 assistant 回答，需要监督；$m_t=0$ 表示该 token 是 prompt、system message 或其他上下文，不计入训练损失。

SFT 很稳定，也很直观。但它本质上是在模仿标准答案，无法直接表达“两个回答哪个更好”。例如两个回答都正确，但一个更简洁、更安全、更符合语气要求，SFT 很难显式建模这种偏好差异。偏好对齐要解决的正是这个问题。

## RLHF 与 PPO

RLHF（Reinforcement Learning from Human Feedback）的经典流程可以分成三步：

1. 用 SFT 得到一个初始策略模型 $\pi_{\mathrm{ref}}$。
2. 收集同一 prompt 下多个回答的人类偏好排序，训练奖励模型 $r(x,y;\boldsymbol{\phi})$。
3. 用 PPO 优化策略模型 $\pi(\cdot;\boldsymbol{\theta})$，让它获得更高奖励，同时不要偏离参考模型太远。

### 奖励模型

奖励模型的训练数据不是标准答案，而是偏好对：

$$ (x,y_w,y_l) $$

其中 $y_w$ 是人类更喜欢的回答，$y_l$ 是人类不太喜欢的回答。奖励模型给两个回答分别打分：

$$ r(x,y_w;\boldsymbol{\phi}),\quad r(x,y_l;\boldsymbol{\phi}) $$

我们希望 $r(x,y_w;\boldsymbol{\phi})$ 比 $r(x,y_l;\boldsymbol{\phi})$ 更大。Bradley-Terry 模型把这个想法写成概率：

$$ P(y_w \succ y_l \mid x) =\sigma\left(r(x,y_w;\boldsymbol{\phi})-r(x,y_l;\boldsymbol{\phi})\right) $$

对应的负对数似然损失为：

$$ \mathcal{L}_{\mathrm{RM}}(\boldsymbol{\phi}) =-\mathbb{E}_{(x,y_w,y_l)} \left[ \log\sigma\left(r(x,y_w;\boldsymbol{\phi})-r(x,y_l;\boldsymbol{\phi})\right) \right] $$

这个式子的直觉很清楚：如果 chosen 的分数明显高于 rejected，Sigmoid 接近 1，loss 变小；如果 rejected 分数更高，loss 会变大。

训练完成后，奖励模型通常被冻结，后续 PPO 把它当作奖励函数来使用。

### RLHF 的优化目标

在 RLHF 中，语言模型可以被看成一个策略：

| 强化学习概念 | LLM 中的对应物 |
| :--- | :--- |
| 状态 $s_t$ | prompt 加上已经生成的 token，即 $(x,y_{\lt t})$ |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 策略 $\pi(a_t \mid s_t;\boldsymbol{\theta})$ | 当前模型的 token 概率分布 |
| 轨迹 $\tau$ | 一次完整生成 $(x,y)$ |
| 奖励 $r(x,y;\boldsymbol{\phi})$ | 奖励模型对完整回答的评分 |

给定 prompt $x$，语言模型自回归生成回答 $y=(y_1,\dots,y_T)$，整段回答的概率为：

$$ \pi(y \mid x;\boldsymbol{\theta}) =\prod_{t=1}^{T}\pi(y_t \mid x,y_{\lt t};\boldsymbol{\theta}) $$

如果只追求奖励最大化，可以写成：

$$ \max_{\boldsymbol{\theta}} \mathbb{E}_{y\sim\pi(\cdot\mid x;\boldsymbol{\theta})} \left[ r(x,y;\boldsymbol{\phi}) \right] $$

但直接最大化奖励很危险。模型可能学会钻 Reward Model 的空子，生成高分但奇怪、重复、啰嗦或分布崩坏的回答，这就是 **reward hacking**。

因此 RLHF 通常加入参考模型 KL 约束，在追求高奖励的同时，不让策略跑得太远：

$$ \max_{\boldsymbol{\theta}} \mathbb{E}_{y\sim\pi(\cdot\mid x;\boldsymbol{\theta})} \left[ r(x,y;\boldsymbol{\phi}) -\beta D_{\mathrm{KL}} \left( \pi(\cdot\mid x;\boldsymbol{\theta}) \| \pi_{\mathrm{ref}}(\cdot\mid x) \right) \right] $$

这个目标有两个方向：

- $r(x,y;\boldsymbol{\phi})$：鼓励模型生成奖励更高的回答。
- $D_{\mathrm{KL}}$：限制模型不要离 SFT 参考模型太远。

$\beta$ 控制约束强度。它越大，模型越保守；它越小，模型越敢追求奖励。

这就是 RLHF 想优化的目标。接下来的问题是：怎么求解它？PPO 给出的答案是 Actor-Critic + clipped objective。

### PPO 怎么优化这个目标

PPO 是 Actor-Critic 风格的方法。Actor 是当前策略模型 $\pi(\cdot;\boldsymbol{\theta})$，Critic 是 Value Model $v(s;\mathbf{w})$。

**为什么需要 Value Model？** 直接对策略做 policy gradient 的问题是方差太大——同一个 token 在不同采样中可能拿到差别很大的奖励，导致梯度信号噪声很高。Value Model 的作用是提供一个 baseline：它估计当前状态的"平均水平"，让 Advantage 只关注"比平均好多少"，从而降低方差。

Value Model 估计当前状态未来能拿到多少回报：

$$ v(s_t;\mathbf{w}) \approx \mathbb{E} \left[ \sum_{k=0}^{T-t}\gamma^k r_{t+k} \mid s_t \right] $$

有了 $v(s;\mathbf{w})$，就可以计算 Advantage 的采样估计：

$$ \hat{A}_t=u_t-v(s_t;\mathbf{w}) $$

其中 $u_t$ 是从第 $t$ 步开始的实际回报。如果 $\hat{A}_t>0$，说明当前 token 比平均水平更好，应该提高它的概率；如果 $\hat{A}_t<0$，说明它低于预期，应该降低它的概率。实际 PPO 中常用 GAE（Generalized Advantage Estimation）通过多步 TD residual 的加权组合来进一步平滑 Advantage 估计，降低方差。

有了 Advantage 估计，下一步是用它更新策略。但直接做 policy gradient（最大化 $\rho_t(\boldsymbol{\theta})\hat{A}_t$）可能一步更新太大，新策略跑偏。PPO 的做法是 **clip**：

$$ L_t^{\mathrm{PPO}}(\boldsymbol{\theta}) = \min \left( \rho_t(\boldsymbol{\theta})\hat{A}_t,\; \mathrm{clip}(\rho_t(\boldsymbol{\theta}),1-\epsilon,1+\epsilon)\hat{A}_t \right) $$

其中 $\rho_t(\boldsymbol{\theta}) = \frac{\pi(y_t\mid s_t;\boldsymbol{\theta})} {\pi(y_t\mid s_t;\boldsymbol{\theta}_{\mathrm{old}})}$ 是新旧策略的重要性采样比率。

分情况看会更容易理解：

- 如果 $\hat{A}_t>0$，说明这个 token 比预期好，模型应该提高它的概率；但 $\rho_t$ 超过 $1+\epsilon$ 后，不再给额外收益。
- 如果 $\hat{A}_t<0$，说明这个 token 比预期差，模型应该降低它的概率；但 $\rho_t$ 低于 $1-\epsilon$ 后，也不再给额外收益。

这就是 PPO 里的 proximal：每次更新只允许策略在旧策略附近移动。

最后还有一个工程问题：前面定义的 KL 约束目标是在整段回答上的，但 PPO 是逐 token 更新的。实际做法是把 KL 惩罚折算成 token-level reward：

$$ r_t^{\mathrm{KL}} =-\beta \left[ \log\pi(y_t\mid x,y_{\lt t};\boldsymbol{\theta}) -\log\pi_{\mathrm{ref}}(y_t\mid x,y_{\lt t}) \right] $$

每个 token 承受 KL 惩罚；回答结束时，再把 Reward Model 对完整回答的分数加到最后一步：

$$ r_t= \begin{cases} r_t^{\mathrm{KL}}, & t \lt T \\ r(x,y;\boldsymbol{\phi})+r_t^{\mathrm{KL}}, & t=T \end{cases} $$

把上面这些组件合在一起，PPO 的完整训练目标为：

$$ \mathcal{L}_{\mathrm{PPO}}(\boldsymbol{\theta},\mathbf{w}) = -\frac{1}{T}\sum_{t=1}^{T} L_t^{\mathrm{PPO}}(\boldsymbol{\theta}) + c_1 \mathcal{L}_{\mathrm{VF}}(\mathbf{w}) - c_2 H\big(\pi(\cdot;\boldsymbol{\theta})\big) $$

其中：

- **策略项** $-\frac{1}{T}\sum_{t} L_t^{\mathrm{PPO}}(\boldsymbol{\theta})$：PPO-Clip 目标的均值，取负是因为优化器做最小化。
- **Value loss** $\mathcal{L}_{\mathrm{VF}}(\mathbf{w}) = \left(v(s_t;\mathbf{w}) - u_t\right)^2$：让 Critic 的价值估计逼近实际回报。
- **熵正则** $H\big(\pi(\cdot;\boldsymbol{\theta})\big) = -\sum_{a}\pi(a\mid s_t;\boldsymbol{\theta})\log\pi(a\mid s_t;\boldsymbol{\theta})$：鼓励探索，防止策略过早坍缩。

$c_1$ 和 $c_2$ 分别是 Value loss 和熵正则的权重系数。KL 约束不放在 loss 里，而是通过 token-level reward 间接生效。

{{<figure
    src="ppo.png"
    caption="Fig. 2. PPO 过程"
    align="center"
    width="90%"
>}}

## DPO：直接偏好优化

PPO 的流程很完整，但也很重。它需要训练 Reward Model，需要在线采样，需要 Value Model，需要估计 Advantage，还要小心 reward hacking 和 KL 控制。

DPO（Direct Preference Optimization）的出发点更直接：

> 既然最终想让 chosen response 比 rejected response 更可能出现，为什么一定要显式训练 Reward Model，再跑一轮强化学习？

DPO 直接使用偏好对 $(x,y_w,y_l)$ 优化策略模型，不再单独训练奖励模型，也不需要在线 RL 采样。

### 从 KL 约束目标出发

先看 RLHF 中常见的 KL 约束优化目标：

$$ \max_{\pi} \mathbb{E}_{y\sim\pi(\cdot\mid x)} \left[ r(x,y) -\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} \right] $$

固定一个 prompt $x$，这个优化问题的最优策略可以写成：

$$ \pi^*(y\mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{1}{\beta}r(x,y)\right) $$

其中 $Z(x)$ 是归一化项：

$$ Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{1}{\beta}r(x,y)\right) $$

这个式子说明：如果某个回答的奖励 $r(x,y)$ 更高，那么最优策略 $\pi^*$ 就应该比参考模型更倾向于生成它。

对最优策略公式取对数：

$$ \log\pi^*(y\mid x) = \log\pi_{\mathrm{ref}}(y\mid x) +\frac{1}{\beta}r(x,y) -\log Z(x) $$

整理后得到奖励函数的等价表示：

$$ r(x,y) = \beta \left[ \log\frac{\pi^*(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} +\log Z(x) \right] $$

这一步是 DPO 的关键。它告诉我们：奖励可以用“最优策略相对于参考模型的 log-prob 比值”表示出来。

### 用当前策略替代最优策略

真实的 $\pi^*$ 不可得。DPO 的做法是用当前要训练的策略 $\pi(\cdot;\boldsymbol{\theta})$ 替代它，从而得到隐式奖励：

$$ r_{\boldsymbol{\theta}}(x,y) = \beta \left[ \log\frac{\pi(y\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y\mid x)} +\log Z(x) \right] $$

偏好建模仍然采用 Bradley-Terry 形式：

$$ P(y_w\succ y_l\mid x) = \sigma \left( r_{\boldsymbol{\theta}}(x,y_w)-r_{\boldsymbol{\theta}}(x,y_l) \right) $$

把 $r_{\boldsymbol{\theta}}$ 代入，两个回答对应同一个 prompt $x$，所以都有相同的 $\log Z(x)$，差值中会被抵消：

$$ \begin{aligned} r_{\boldsymbol{\theta}}(x,y_w)-r_{\boldsymbol{\theta}}(x,y_l) &= \beta \left[ \log\frac{\pi(y_w\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_w\mid x)} +\log Z(x) \right] \\ &\quad - \beta \left[ \log\frac{\pi(y_l\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_l\mid x)} +\log Z(x) \right] \\ &= \beta\log\frac{\pi(y_w\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi(y_l\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_l\mid x)} \end{aligned} $$

于是 DPO 的损失函数为：

$$ \mathcal{L}_{\mathrm{DPO}}(\boldsymbol{\theta}) = - \mathbb{E}_{(x,y_w,y_l)} \left[ \log\sigma \left( \beta\log\frac{\pi(y_w\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi(y_l\mid x;\boldsymbol{\theta})} {\pi_{\mathrm{ref}}(y_l\mid x)} \right) \right] $$

也可以写得更直观一点：

$$ \mathcal{L}_{\mathrm{DPO}}(\boldsymbol{\theta}) = - \log\sigma \left( \beta \left[ \log\pi(y_w\mid x;\boldsymbol{\theta})-\log\pi_{\mathrm{ref}}(y_w\mid x) \right] - \beta \left[ \log\pi(y_l\mid x;\boldsymbol{\theta})-\log\pi_{\mathrm{ref}}(y_l\mid x) \right] \right) $$

这个 loss 在做两件事：

1. 提高当前模型相对于参考模型在 $y_w$ 上的概率。
2. 降低当前模型相对于参考模型在 $y_l$ 上的概率。

所以 DPO 不是简单模仿 chosen response，而是在优化 chosen 和 rejected 的相对偏好。

$\beta$ 控制策略偏离参考模型的强度。沿着前面的 KL 约束视角看，较大的 $\beta$ 表示更强的 KL 约束，模型通常更保守、更接近参考模型；较小的 $\beta$ 允许模型更大幅度地偏离参考模型。

{{<figure
    src="dpo.png"
    caption="Fig. 3. DPO 过程"
    align="center"
    width="90%"
>}}

### DPO 的优缺点

DPO 的优势很明显：

- 不需要单独训练 Reward Model。
- 不需要在线采样和强化学习循环。
- 训练形式接近监督学习，实现简单，稳定性通常更好。

但它也有边界：

- 强依赖偏好数据质量，chosen / rejected 标注有噪声时会直接影响模型。
- 它是离线方法，不像 PPO / GRPO 那样能让当前模型在线探索新回答。
- 对数学、代码、工具调用等可验证任务，如果奖励可以由环境或测试直接给出，在线 RL 方法可能更灵活。

## GRPO：组相对策略优化

GRPO（Group Relative Policy Optimization）更接近 PPO，而不是 DPO。它仍然让模型在线生成回答，再根据奖励更新策略。

它对 PPO 的关键简化是：**去掉 Critic / Value Model**。

PPO 用 $v(s_t;\mathbf{w})$ 估计某个状态的平均回报，再用 $u_t-v(s_t;\mathbf{w})$ 估计 Advantage。GRPO 不训练 Value Model，而是在同一个 prompt 下采样多条回答，用这组回答的相对好坏估计 Advantage。

### 组内采样

对同一个 prompt $x$，从旧策略中采样 $G$ 条回答：

$$ \{y_1,y_2,\dots,y_G\} \sim \pi(\cdot\mid x;\boldsymbol{\theta}_{\mathrm{old}}) $$

然后用奖励模型或规则奖励函数打分：

$$ r_i=r(x,y_i),\quad i=1,\dots,G $$

如果某条回答的奖励高于组内平均，它就是相对好的回答；如果低于组内平均，它就是相对差的回答。

GRPO 的组相对 Advantage 通常写成：

$$ A_i= \frac{ r_i-\mathrm{mean}(r_1,\dots,r_G) }{ \mathrm{std}(r_1,\dots,r_G) } $$

其中 $A_i$ 是第 $i$ 条回答相对于同组回答的好坏。

这个设计非常适合数学、代码、推理等任务。比如同一道题采样 8 个答案，有的正确、有的错误，GRPO 不需要为每个中间状态训练价值函数，只需要比较同组回答的最终奖励。

### GRPO 目标函数

GRPO 仍然保留 PPO-Clip 的思想。对第 $i$ 条回答中的第 $t$ 个 token，定义概率比：

$$ \rho_{i,t}(\boldsymbol{\theta}) = \frac{ \pi(y_{i,t}\mid x,y_{i,\lt t};\boldsymbol{\theta}) }{ \pi(y_{i,t}\mid x,y_{i,\lt t};\boldsymbol{\theta}_{\mathrm{old}}) } $$

GRPO 的策略优化目标可以写成：

$$ \mathcal{L}_{\mathrm{GRPO}}(\boldsymbol{\theta}) = - \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[ \min \left( \rho_{i,t}(\boldsymbol{\theta})A_i,\; \mathrm{clip}(\rho_{i,t}(\boldsymbol{\theta}),1-\epsilon,1+\epsilon)A_i \right) -\beta D_{\mathrm{KL}}\big(\pi(\cdot \mid x,y_{i,\lt t};\boldsymbol{\theta})\|\pi_{\mathrm{ref}}(\cdot \mid x,y_{i,\lt t})\big) \right] $$

它和 PPO 的核心差异在 Advantage：

- PPO 的 Advantage 估计来自 $u_t-v(s_t;\mathbf{w})$，需要 Value Model。
- GRPO 的 Advantage 来自同组回答的奖励归一化，不需要 Value Model。

{{<figure
    src="grpo.png"
    caption="Fig. 4. GRPO 过程"
    align="center"
    width="90%"
>}}

GRPO 的优点是节省了 Value Model 的显存和计算开销，也减少了一部分训练复杂度。代价是它需要同一个 prompt 下采样多条回答；如果组内回答差异太小，或者奖励函数区分度不足，Advantage 估计就会变弱。

## GSPO：序列级策略优化

GRPO 去掉了 Value Model，但仍然有一个粒度不一致的问题：

- reward 通常是 sequence-level 的，也就是对整段回答打分；
- Advantage $A_i$ 也是 sequence-level 的；
- 但概率比 $\rho_{i,t}$ 和 clipping 仍然是 token-level 的。

这会导致一个现象：明明奖励评价的是整段回答，训练却在每个 token 上分别做 ratio 和 clip。某些 token 的概率波动可能放大梯度噪声，影响训练稳定性。

GSPO（Group Sequence Policy Optimization）的核心改动是：**把策略比率从 token 级改成 sequence 级**。

### 序列级重要性比

对第 $i$ 条回答 $y_i$，GSPO 定义 sequence-level ratio：

$$ s_i(\boldsymbol{\theta}) = \left( \frac{\pi(y_i\mid x;\boldsymbol{\theta})} {\pi(y_i\mid x;\boldsymbol{\theta}_{\mathrm{old}})} \right)^{\frac{1}{|y_i|}} $$

由于语言模型是自回归的：

$$ \pi(y_i\mid x;\boldsymbol{\theta}) = \prod_{t=1}^{|y_i|} \pi(y_{i,t}\mid x,y_{i,\lt t};\boldsymbol{\theta}) $$

所以 $s_i(\boldsymbol{\theta})$ 可以写成：

$$ s_i(\boldsymbol{\theta}) = \exp \left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{ \pi(y_{i,t}\mid x,y_{i,\lt t};\boldsymbol{\theta}) }{ \pi(y_{i,t}\mid x,y_{i,\lt t};\boldsymbol{\theta}_{\mathrm{old}}) } \right) $$

这里的 $\frac{1}{|y_i|}$ 是长度归一化。它避免长回答因为 token 更多而导致 log-ratio 累积过大。

直观地说：

- GRPO 看的是“每个 token 的概率变化”。
- GSPO 看的是“整段回答平均意义上的概率变化”。

### GSPO 目标函数

GSPO 保留组相对 Advantage $A_i$，但用 sequence-level ratio 做 clipping：

$$ \mathcal{J}_{\mathrm{GSPO}}(\boldsymbol{\theta}) = \frac{1}{G} \sum_{i=1}^{G} \min \left( s_i(\boldsymbol{\theta})A_i,\; \mathrm{clip}(s_i(\boldsymbol{\theta}),1-\epsilon,1+\epsilon)A_i \right) $$

如果按最小化 loss 写，并加上 KL 正则防止策略偏离参考模型太远：

$$ \mathcal{L}_{\mathrm{GSPO}}(\boldsymbol{\theta}) = - \frac{1}{G} \sum_{i=1}^{G} \left[ \min \left( s_i(\boldsymbol{\theta})A_i,\; \mathrm{clip}(s_i(\boldsymbol{\theta}),1-\epsilon,1+\epsilon)A_i \right) - \beta D_{\mathrm{KL}}\big(\pi(\cdot \mid x;\boldsymbol{\theta})\|\pi_{\mathrm{ref}}(\cdot \mid x)\big) \right] $$

当 $A_i>0$ 时，说明这条回答比同组平均更好，GSPO 会提高整段回答的生成概率；当 $A_i<0$ 时，说明这条回答较差，GSPO 会降低整段回答的生成概率。

{{<figure
    src="gspo.png"
    caption="Fig. 5. GSPO 过程"
    align="center"
    width="90%"
>}}

GSPO 的价值在于让优化粒度和奖励粒度更一致。对于数学、代码、长链路推理这类整段回答打分的任务，这种 sequence-level 更新通常更符合奖励信号的形态。

它的代价是牺牲了一部分 token-level 的细粒度控制。如果一条回答整体奖励不错，但内部某些 token 很差，GSPO 不会像 token-level 方法那样精细地区分每个 token。

## 方法对比

| 方法 | 核心思想 | 数据来源 | 是否在线采样 | 是否需要奖励模型 / 奖励函数 | 是否需要 Value Model | Advantage 计算 | Ratio / Clip 粒度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PPO | 先训练奖励模型，再用强化学习优化策略 | prompt + 在线生成回答 | 需要 | 需要 | 需要 | $\hat{A}_t=u_t-v(s_t;\mathbf{w})$，常配合 GAE | token-level |
| DPO | 直接用偏好对优化 chosen / rejected 的相对概率 | prompt + chosen + rejected | 不需要 | 不需要显式奖励模型 | 不需要 | 不显式计算 Advantage | sequence-level 偏好对 |
| GRPO | 去掉 Critic，用同组回答的相对奖励估计 Advantage | prompt + 在线生成多条回答 | 需要 | 需要奖励模型或规则奖励 | 不需要 | $A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}$ | token-level |
| GSPO | 在 GRPO 基础上，把 ratio 改成 sequence-level | prompt + 在线生成多条回答 | 需要 | 需要奖励模型或规则奖励 | 不需要 | 组相对 Advantage | sequence-level |

从工程角度看：

| 方法 | 训练成本 | 工程复杂度 | 适合场景 | 不太适合的场景 |
| :--- | :--- | :--- | :--- | :--- |
| PPO | 最高 | 最高 | 通用 RLHF、安全对齐、多目标偏好优化 | 资源有限、快速迭代、小模型实验 |
| DPO | 最低 | 最低 | 偏好对齐、回答风格优化、低成本 post-training | 需要在线探索、环境交互或可验证奖励的任务 |
| GRPO | 中高 | 中等 | 数学、代码、推理等可验证奖励任务 | 奖励函数弱、组内样本差异小的任务 |
| GSPO | 中高 | 中等偏高 | 序列级奖励、长链路推理、大规模 RL 训练 | 需要 token 级精细控制的任务 |

## 总结

如果只用一句话概括这几种方法：

- **SFT** 让模型学会按指令回答。
- **PPO** 通过奖励模型和在线强化学习继续优化模型，但流程最重。
- **DPO** 把偏好学习变成离线监督式优化，简单稳定，成本低。
- **GRPO** 去掉 PPO 中的 Value Model，用组内相对奖励估计 Advantage。
- **GSPO** 进一步把 GRPO 的 token-level ratio 改成 sequence-level ratio，让优化粒度更贴近整段回答奖励。

实际选择时，可以先看奖励信号来自哪里：

如果只有高质量偏好对，DPO 往往是最省事的选择；如果有可验证奖励函数，并且希望模型在线探索更好的解法，GRPO / GSPO 更自然；如果要做通用人类偏好 RLHF，并且资源充足，PPO 仍然是经典基线。


## LLMs 微调对齐实践
### Model

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

### Dataset

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

#### 统一约定

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



#### SFT

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

#### Reward Model（RM）

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

#### DPO

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

#### PPO

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

#### GRPO/GSPO

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


### Post-Training

实践中，SFT 和 DPO 通常使用 MS-Swift：它对 Qwen、ModelScope 数据集和常见微调流程支持完善，命令行简单，适合单机或中小规模集群训练。PPO、GRPO、GSPO 则更常使用 verl，因为这类在线强化学习需要高吞吐地生成多条回答、调用奖励模型评分，并在多机多卡环境中协调训练与 rollout；verl 对这类分布式 RL 流程更偏工程化。

#### MS-Swift 实践

##### SFT

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



## 参考文献

[1] Long Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.

[2] John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[3] Paul F. Christiano et al. [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741), 2017.

[4] Rafael Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.

[5] Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.

[6] Chujie Zheng et al. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), 2025.
