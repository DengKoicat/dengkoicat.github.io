---
title: "Reinforcement Learning for LLMs"
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

## RLHF

### SFT

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

### PPO

RLHF（Reinforcement Learning from Human Feedback）的经典流程可以分成三步：

1. 用 SFT 得到一个初始策略模型 $\pi_{\mathrm{ref}}$。
2. 收集同一 prompt 下多个回答的人类偏好排序，训练奖励模型 $r(x,y;\boldsymbol{\phi})$。
3. 用 PPO 优化策略模型 $\pi(\cdot;\boldsymbol{\theta})$，让它获得更高奖励，同时不要偏离参考模型太远。

#### 奖励模型

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

#### RLHF 的优化目标

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

#### PPO 怎么优化这个目标

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

### DPO：直接偏好优化

PPO 的流程很完整，但也很重。它需要训练 Reward Model，需要在线采样，需要 Value Model，需要估计 Advantage，还要小心 reward hacking 和 KL 控制。

DPO（Direct Preference Optimization）的出发点更直接：

> 既然最终想让 chosen response 比 rejected response 更可能出现，为什么一定要显式训练 Reward Model，再跑一轮强化学习？

DPO 直接使用偏好对 $(x,y_w,y_l)$ 优化策略模型，不再单独训练奖励模型，也不需要在线 RL 采样。

#### 从 KL 约束目标出发

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

#### 用当前策略替代最优策略

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

#### DPO 的优缺点

DPO 的优势很明显：

- 不需要单独训练 Reward Model。
- 不需要在线采样和强化学习循环。
- 训练形式接近监督学习，实现简单，稳定性通常更好。

但它也有边界：

- 强依赖偏好数据质量，chosen / rejected 标注有噪声时会直接影响模型。
- 它是离线方法，不像 PPO / GRPO 那样能让当前模型在线探索新回答。
- 对数学、代码、工具调用等可验证任务，如果奖励可以由环境或测试直接给出，在线 RL 方法可能更灵活。

### GRPO

GRPO（Group Relative Policy Optimization）更接近 PPO，而不是 DPO。它仍然让模型在线生成回答，再根据奖励更新策略。

它对 PPO 的关键简化是：**去掉 Critic / Value Model**。

PPO 用 $v(s_t;\mathbf{w})$ 估计某个状态的平均回报，再用 $u_t-v(s_t;\mathbf{w})$ 估计 Advantage。GRPO 不训练 Value Model，而是在同一个 prompt 下采样多条回答，用这组回答的相对好坏估计 Advantage。

#### 组内采样

对同一个 prompt $x$，从旧策略中采样 $G$ 条回答：

$$ \{y_1,y_2,\dots,y_G\} \sim \pi(\cdot\mid x;\boldsymbol{\theta}_{\mathrm{old}}) $$

然后用奖励模型或规则奖励函数打分：

$$ r_i=r(x,y_i),\quad i=1,\dots,G $$

如果某条回答的奖励高于组内平均，它就是相对好的回答；如果低于组内平均，它就是相对差的回答。

GRPO 的组相对 Advantage 通常写成：

$$ A_i= \frac{ r_i-\mathrm{mean}(r_1,\dots,r_G) }{ \mathrm{std}(r_1,\dots,r_G) } $$

其中 $A_i$ 是第 $i$ 条回答相对于同组回答的好坏。

这个设计非常适合数学、代码、推理等任务。比如同一道题采样 8 个答案，有的正确、有的错误，GRPO 不需要为每个中间状态训练价值函数，只需要比较同组回答的最终奖励。

#### GRPO 目标函数

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

### GSPO

GRPO 去掉了 Value Model，但仍然有一个粒度不一致的问题：

- reward 通常是 sequence-level 的，也就是对整段回答打分；
- Advantage $A_i$ 也是 sequence-level 的；
- 但概率比 $\rho_{i,t}$ 和 clipping 仍然是 token-level 的。

这会导致一个现象：明明奖励评价的是整段回答，训练却在每个 token 上分别做 ratio 和 clip。某些 token 的概率波动可能放大梯度噪声，影响训练稳定性。

GSPO（Group Sequence Policy Optimization）的核心改动是：**把策略比率从 token 级改成 sequence 级**。

#### 序列级重要性比

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

#### GSPO 目标函数

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

### 方法对比

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


## 分布式训练
### 为什么需要分布式训练

训练大模型最直接的瓶颈不是「能不能把数据读进去」，而是单张 GPU 同时放不下训练过程中必须常驻或临时保留的状态。推理时主要需要模型参数和当前生成所需的 KV cache；训练时还要保存前向传播的中间激活、反向传播产生的梯度，以及优化器为了更新参数而维护的额外状态。因此，一个看起来只是「30B 参数」的模型，全参数训练时显存开销会远大于权重文件本身。

以 30B 参数、FP16 权重、AdamW 全参数训练为例，如果每个参数本体用 FP16 保存，权重约占 $30\times10^9\times2\text{B}\approx60\text{GB}$；反向传播后还要有一份 FP16 梯度，约 60GB；AdamW 会为每个参数维护 fp32 的一阶、二阶动量，共约 240GB；混合精度训练里还经常保留 fp32 master weights，约 120GB。仅模型状态就接近：

$$
\text{Memory}_{\text{model state}} \approx N_{\text{param}} \times (2 + 2\text{-}4 + 8 + 0\text{-}4)\ \text{bytes}
$$

在这个例子里，未做 ZeRO/FSDP 切分时，仅这些状态就约为 $60+60+240+120=480\text{GB}$。再加上前向传播保存的 activations、attention 临时 buffer、通信 buffer、算子 workspace，训练时需要的显存会继续上升。一个简化的显存构成如下：

| 显存来源 | 估算方式 | 30B FP16 示例 | 大致占比 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| 参数 weights | $30B \times 2\text{B}$ | 60GB | 约 12% | FP16/bf16 参数本体，前向、反向、更新都要访问 |
| 梯度 gradients | $30B \times 2\text{B}$ | 60GB | 约 12% | backward 后产生，通常与参数量同阶 |
| AdamW 优化器状态 | $30B \times 8\text{B}$ | 240GB | 约 47% | 一阶动量 $m$ 和二阶动量 $v$ 通常是 fp32，各 4 bytes |
| FP32 master weights | $30B \times 4\text{B}$ | 120GB | 约 24% | 混合精度训练中常见，用于更稳定地更新参数 |
| 激活 activations | 与 batch size、sequence length、hidden size、layer 数相关 | 约几十 GB | 约 5%+ | 前向时保存，反向时用于链式求导；checkpointing 可以用重算换显存 |
| Attention/KV、通信 buffer、算子 workspace | 与序列长度、并行策略、kernel 实现相关 | 不固定 | 不固定 | 长上下文 attention、all-reduce/all-gather、FlashAttention 或 fused kernel 都可能产生临时显存 |
| 当前 batch 数据 | token ids、labels、attention mask | 通常远小于 1GB | 约 0% | 数据集本体通常在磁盘或 CPU 内存中，GPU 只接收当前 micro-batch |
| RLHF/GRPO/PPO 额外开销 | policy/reference/reward/critic、rollout 样本、logprobs | 视算法而定 | 视算法而定 | 在线 RL 训练可能同时保留多个模型或中间结果，比 SFT 更吃显存和吞吐 |

因此，分布式训练的核心目标不是简单地「多几张卡就更快」，而是把这些显存和计算压力拆开：数据并行把 batch 分到多张卡上提升吞吐；张量并行把单层矩阵乘法切开；流水线并行把不同层放到不同 GPU；ZeRO/FSDP 则把参数、梯度、优化器状态切片存储，避免每张卡都保存完整副本。对于大模型 post-training，尤其是 PPO/GRPO/GSPO 这类需要在线生成、评分、再更新的流程，分布式训练几乎是让实验能跑起来的前提。

### 数据并行
#### DDP

最朴素的数据并行（DP / `DataParallel`）思路是：把一个 batch 切到多张 GPU 上，每张卡各自前向和反向，最后再把结果汇总。但这种做法通常由单个进程调度，多卡之间的输入分发、输出收集和梯度聚合容易集中到 GPU_0 上，导致主卡显存更高、通信和 Python 调度开销更明显。假设模型参数 / 梯度大小为 $\Psi$，GPU 数为 $N$，DP 中 GPU_0 大致需要接收 $(N-1)\Psi$ 的梯度，再向其他 GPU 同步 $(N-1)\Psi$ 的参数或梯度结果；其他 GPU 通常只传出约 $\Psi$ 梯度、传入约 $\Psi$ 参数。因此 DP 的通信和负载集中在主卡上，卡数一多，GPU 利用率反而不稳定。

DDP（Distributed Data Parallel）把每张 GPU 放在独立进程里，每个进程持有一份模型副本、处理自己的 mini-batch，并在反向传播时通过 all-reduce 同步梯度。这样通信可以和 backward 部分重叠，主卡瓶颈也会小很多，所以实际训练中单机多卡和多机多卡一般优先使用 DDP，而不是 DP。

{{< figure
    src="ddp-ring-allreduce.png"
    caption="Fig. 6. DDP 中常见的 Ring-AllReduce：先通过 Scatter-Reduce 让每个梯度分块在环上累加，再通过 All-Gather 把累加后的分块广播给所有 GPU。"
    align="center"
    width="70%"
>}}

Ring-AllReduce 可以理解为把完整梯度切成多个 chunk，例如 $A,B,C$。在 `Scatter-Reduce` 阶段，每张 GPU 每一轮只把一个 chunk 发给右邻居、从左邻居接收一个 chunk，并把收到的 chunk 累加到本地对应分块上；经过 $N-1$ 轮后，每个 chunk 都会在某一张 GPU 上得到全局求和结果 $\sum_i g_i[\text{chunk}]$。接着进入 `All-Gather` 阶段，这些已经求和的 chunk 继续沿环传递，再经过 $N-1$ 轮，所有 GPU 都拿到完整的全局梯度。最后通常除以 `world_size` 得到平均梯度，每张卡再用相同的梯度执行 optimizer step，所以各卡模型参数保持一致。

从通信量看，Ring-AllReduce 会把大小为 $\Psi$ 的梯度切成 $N$ 份，每份大小为 $\Psi/N$。对每张 GPU 来说，`Scatter-Reduce` 阶段需要传入 / 传出 $(N-1)\Psi/N \approx \Psi$，`All-Gather` 阶段也需要传入 / 传出 $(N-1)\Psi/N \approx \Psi$，所以总传入 / 传出约为 $2(N-1)\Psi/N \approx 2\Psi$。这个通信量基本不随 GPU 数线性增长，也没有 DP 那种 GPU_0 集中瓶颈。


#### ZeRO

DDP 解决了 DP 的主卡瓶颈，但它仍然有一个明显问题：每张 GPU 都保存完整的参数、梯度和优化器状态。ZeRO（Zero Redundancy Optimizer）的思路是把这些“每张卡重复存一份”的模型状态切开，让数据并行不只并行计算，也并行存储。

ZeRO 的三个阶段是累积关系：

- ZeRO-1：切分 optimizer states。
- ZeRO-2：在 ZeRO-1 基础上继续切分 gradients。
- ZeRO-3：在 ZeRO-2 基础上继续切分 parameters。

{{<figure
    src="https://www.microsoft.com/en-us/research/wp-content/uploads/2020/02/DeepSpeed-Image-1.png"
    caption="Fig. 7. ZeRO 三个阶段对参数、梯度和优化器状态的切分方式。Image source: [Microsoft Research, ZeRO & DeepSpeed, 2020](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)."
    align="center"
    width="70%"
>}}

沿用前面 30B 参数、FP16 权重、AdamW 的估算：参数 60GB，梯度 60GB，AdamW 状态 240GB，FP32 master weights 120GB。若数据并行度为 $N$，只看模型状态，不计 activation 和临时 buffer，则各阶段单卡显存大致为：

| 方式 | 单卡模型状态显存 | $N=8$ | $N=64$ | 主要代价 |
| :--- | :--- | :--- | :--- | :--- |
| DDP | $P+G+O+M$ | 480GB | 480GB | 每卡完整复制，显存不随卡数下降 |
| ZeRO-1 | $P+G+\frac{O+M}{N}$ | 165GB | 125.6GB | optimizer step 前后需要同步参数更新结果 |
| ZeRO-2 | $P+\frac{G+O+M}{N}$ | 112.5GB | 66.6GB | 梯度用 reduce-scatter / all-gather 形式通信 |
| ZeRO-3 | $\frac{P+G+O+M}{N}$ | 60GB | 7.5GB | 前向和反向时需要按层 all-gather 参数 |

这个表里最值得注意的是：ZeRO-1 对 AdamW 这类优化器很有效，因为优化器状态本来就是最大头；ZeRO-3 的显存下降最彻底，但它把参数也切开了，计算到某一层时必须临时把这一层参数聚合出来，所以通信调度会更复杂。

##### ZeRO-1

ZeRO-1 只切分优化器状态。每张 GPU 仍然持有完整参数 $P$ 和完整梯度 $G$，但 AdamW 的一阶动量 $m$、二阶动量 $v$，以及常见的 FP32 master weights 会按数据并行 rank 分片保存。

如果用普通 DDP，每张卡都要保存完整的 $O+M=360\text{GB}$ 优化器相关状态；ZeRO-1 在 8 卡时把这部分降到 $360/8=45\text{GB}$，单卡模型状态从 480GB 降到约 165GB。这个收益已经很明显，但它仍然保留完整参数和完整梯度，所以对于更大的模型，只靠 ZeRO-1 往往不够。

工程上，ZeRO-1 的侵入性相对小，通信量也接近 DDP。它适合“模型参数本身还能放下，但 AdamW 状态放不下”的场景，比如中等规模全参数 SFT 或 DPO。

##### ZeRO-2

ZeRO-2 在 ZeRO-1 的基础上继续切分梯度。反向传播过程中，各卡不再长期保存完整梯度，而是通过 reduce-scatter 把梯度归约到对应分片上：哪个 rank 负责哪部分参数的 optimizer state，它就只保留那部分参数的梯度。

这一步把 $G$ 也从每卡完整复制变成 $\frac{G}{N}$。在 30B、8 卡的例子里，单卡模型状态约为：

$$
P+\frac{G+O+M}{N}=60+\frac{60+240+120}{8}=112.5\text{GB}
$$

ZeRO-2 的特点是性价比很高：它比 ZeRO-1 多省一份梯度显存，但参数仍然完整保存在每张卡上，因此前向和反向的参数访问方式比较简单。很多 RLHF / GRPO 训练如果还要同时放 reference model、reward model 或 rollout engine，会优先尝试 ZeRO-2，再配合 activation checkpointing、bf16 和更小的 micro-batch。

##### ZeRO-3

ZeRO-3 把参数也切分了。每张 GPU 只保存 $\frac{1}{N}$ 的参数、梯度和优化器状态；执行某一层前向或反向时，再临时 all-gather 出当前层需要的完整参数，用完后释放或重新分片。

{{<figure
    src="ZeRO-3.png"
    caption="Fig. 8. ZeRO-3 的一次训练流程：模型状态长期按 rank 分片保存，前向 / 反向时临时聚合当前计算所需参数，反向后再把梯度归约并重新切回各自负责的分片。"
    align="center"
    width="90%"
>}}

因此 ZeRO-3 的显存下降几乎随数据并行度线性缩放。30B 模型在 64 卡下，模型状态从 DDP 的 480GB/卡降到约 7.5GB/卡，这才让“单卡完全放不下参数本体”的训练变得可行。代价是通信更多，尤其是 layer-wise parameter all-gather 会进入训练主路径；如果网络带宽弱、batch 太小或 overlap 做得不好，吞吐可能明显下降。

直观选择可以按下面的规则：

- 模型能放下，主要是 optimizer state 太大：优先 ZeRO-1。
- 参数能放下，但梯度和优化器状态太大：优先 ZeRO-2。
- 参数本身也放不下，或者还要叠加 RLHF 多模型开销：考虑 ZeRO-3 / FSDP。

所以 ZeRO 的本质不是“让训练一定更快”，而是用通信换显存。对于 post-training，显存省下来以后可以换成更大的模型、更长的 sequence、更大的 rollout batch，或者同时容纳 Actor / Reference / Critic / Reward Model，这通常比单纯追求单步吞吐更重要。


#### FSDP

FSDP（Fully Sharded Data Parallel）可以理解为 PyTorch 原生的 ZeRO-3 风格数据并行。它同样把 parameters、gradients、optimizer states 切到不同 rank 上，计算时再把当前 FSDP unit 需要的参数临时 all-gather 出来；反向传播结束后，通过 reduce-scatter 让每张卡只保留自己负责的梯度分片，optimizer step 也只更新本地参数分片。

{{<figure
    src="https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png"
    caption="Fig. 9. FSDP 的前向 / 反向流程：每个 FSDP unit 在计算前 all-gather 参数，计算后释放完整参数，反向时再 reduce-scatter 梯度。Image source: [PyTorch Tutorials, Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)."
    align="center"
    width="90%"
>}}

FSDP 和 ZeRO-3 的直觉非常接近，但工程入口不同：ZeRO 通常出现在 DeepSpeed 训练栈里，通过 optimizer/runtime 配置打开不同 stage；FSDP 则是 PyTorch `torch.distributed` 里的模块级 sharding API，通常按 transformer block 或更大的 module 粒度包起来。包得太粗，每次 all-gather 的完整参数更大，峰值显存更高；包得太细，通信次数更多，调度开销也会上升，所以实际训练里常按 layer/block 做 auto-wrap 或手动 sharding。

##### FSDP1

早期 PyTorch FSDP 一般指 `FullyShardedDataParallel` 这个 wrapper API。它会把一个 FSDP unit 里的多个参数 flatten 成 `FlatParameter`，这样通信 bucket 更规整，也方便在 eager mode 下做 all-gather / reduce-scatter overlap。

问题也来自 flatten：多个原始参数被拼成一个大参数以后，参数级别的行为会变复杂，比如只冻结其中一部分参数、给不同参数用不同 dtype / optimizer 规则、保存和加载 sharded state dict 等，都需要额外处理。因此 FSDP1 在已有代码里仍然常见，但新项目里不一定是最顺手的默认选项。

##### FSDP2

FSDP2 是 PyTorch 新一代 FSDP API，核心入口从 wrapper 变成了 composable 的 `fully_shard(module)`。它通常从内到外使用：先对每个 transformer layer 调用 `fully_shard`，最后再对 root model 调用一次，让非 layer 参数也进入 sharding 体系。

更关键的变化是，FSDP2 用 `DTensor` 表示被切分的参数，并通过 `DeviceMesh` 描述设备拓扑。这样做的好处是参数仍然更像“原来的参数”，而不是被揉进一个 `FlatParameter` 里；state dict、meta-device 初始化、冻结部分参数、以及和 TP / PP / CP 这类并行方式组合时，抽象会更清楚。PyTorch 当前教程也把 FSDP1 标为 deprecated，新代码一般优先看 FSDP2。

| 维度 | FSDP1 | FSDP2 |
| :--- | :--- | :--- |
| API 形态 | `FullyShardedDataParallel(module)` wrapper | `fully_shard(module)` composable API |
| 参数表示 | 多个参数 flatten 成 `FlatParameter` | sharded parameter 以 `DTensor` 表示 |
| 设备拓扑 | 主要围绕 process group 配置 | 通过 `DeviceMesh` 更自然地表达 1D/2D/多维并行 |
| 组合能力 | 可用，但和 TP、checkpoint、参数冻结等组合时更容易遇到边角复杂度 | 更适合和 TP、DCP、meta init、混合精度策略组合 |
| 新项目倾向 | 维护老代码时常见 | 新 PyTorch 分布式训练优先考虑 |

因此选择时可以这样理解：如果训练栈已经围绕 DeepSpeed、HuggingFace Trainer 或 verl 的 ZeRO 配置组织，继续用 ZeRO 会更直接；如果希望尽量留在 PyTorch 原生 `torch.distributed` 体系里，并且后面可能叠加 tensor parallel、distributed checkpoint 或 TorchTitan 风格的训练代码，FSDP2 会更贴近新的工程方向。


### 模型并行

#### Tensor Parallel

Tensor Parallel（TP，张量并行）切的是「层内部的矩阵」，不是 batch，也不是完整的模型状态副本。对于一个很宽的 Transformer layer，单个 linear / attention projection 的权重矩阵本身就可能很大；TP 会把这些矩阵沿输入维或输出维切到多个 GPU 上，让每张卡只做这一层的一部分 GEMM。

{{<figure
    src="https://docs.pytorch.org/tutorials/_images/megatron_lm.png"
    caption="Fig. 10. Megatron-LM 风格的 Tensor Parallel：MLP 中第一层 linear 做 column parallel，第二层 linear 做 row parallel；Self-Attention 中 Q/K/V 和 attention heads 可以按 head 分到不同 GPU。Image source: [PyTorch Tutorials, Large Scale Transformer model training with Tensor Parallel](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html), adapted from [Megatron-LM](https://arxiv.org/abs/1909.08053)."
    align="center"
    width="90%"
>}}

以两层 MLP 为例，第一层权重 $A$ 可以按输出维切成 $A=[A_1,A_2,\dots,A_T]$，每张 GPU 计算自己的 $XA_i$，中间的 GeLU 也可以各自独立做；第二层权重 $B$ 再按输入维切成 $\begin{bmatrix}B_1;B_2;\dots;B_T\end{bmatrix}$，每张卡计算 $Y_iB_i$，最后用一次 all-reduce 把各卡部分结果加起来。这样两层 linear 中间不需要先把完整 activation 拼出来，通信点被压到 block 边界附近。

Self-Attention 更适合 TP：multi-head attention 的不同 head 本来就是相对独立的。Q/K/V projection 可以按 head 或 hidden 维度切开，每张卡负责一部分 head 的 attention 计算；attention output projection 再用 row parallel，把各卡结果归约回完整 hidden 表示。Megatron-LM 的设计重点就在这里：尽量让大矩阵乘法本地完成，只在必要位置做少量 collective。

和 ZeRO/FSDP 相比，TP 的侧重点不同：

| 方式 | 切分对象 | 主要节省 | 主要通信 | 更适合 |
| :--- | :--- | :--- | :--- | :--- |
| ZeRO/FSDP | 参数、梯度、优化器状态这些 model states | 训练状态显存 | 参数 all-gather、梯度 reduce-scatter | 模型状态太大、需要扩大数据并行规模 |
| Tensor Parallel | 单层内部的权重矩阵和 activation | 单层权重、activation、GEMM 计算量 | layer 内 all-reduce / all-gather / reduce-scatter | hidden size 很大、单层矩阵本身放不下或算不动 |

TP 的代价是通信更贴近每个 Transformer block 的主路径。DDP/ZeRO 的通信通常围绕梯度同步或参数分片展开，而 TP 在每一层的 attention / MLP 里都可能有 collective，所以它非常依赖 GPU 间带宽和拓扑。实践中 TP 通常优先放在单机 NVLink / NVSwitch 内做，例如 2、4、8 路 TP；跨节点继续增大 TP degree 往往会被网络延迟拖住。

TP 还有两个工程限制：第一，hidden size、attention heads、vocab size 往往需要能被 `tensor_model_parallel_size` 整除，否则要 padding 或改切分策略；第二，TP degree 太大以后，每张卡上的 GEMM 变小，kernel 利用率和 CPU 调度开销都会变差。因此 TP 不是越大越好，常见做法是把 TP 控制在节点内部，再和 DP / ZeRO / FSDP、Pipeline Parallel 组合起来：

$$
\text{world size}=\text{DP}\times\text{TP}\times\text{PP}\times\text{CP}
$$

比如 64 张 GPU 可以设成 TP=4、PP=2、DP=8：TP 负责把单层宽矩阵切开，PP 负责把层深度切开，DP/FSDP 负责处理 batch 和模型状态。对 post-training 来说，如果只是 optimizer state 太大，优先考虑 ZeRO/FSDP；如果单个 attention/MLP block 已经成为显存或计算瓶颈，再引入 TP 会更合理。

#### Pipeline Parallel

Pipeline Parallel（PP，流水线并行）切的是「模型深度」。它把连续的 transformer layers 分成多个 stage，例如 48 层模型切成 4 段，每张 GPU 或每组 GPU 负责 12 层。前向时 activation 从前一个 stage 传到后一个 stage，反向时 activation gradient 再反向传回来；每个 stage 只保存自己那段层的参数、梯度、优化器状态和激活。

如果直接把一个 batch 串行跑过所有 stage，GPU 利用率会很差：第 1 个 stage 算完后才能把结果交给第 2 个 stage，第 2 个 stage 工作时第 1 个 stage 又闲着。因此 PP 通常会把 batch 再切成多个 micro-batches，让不同 micro-batch 同时处在不同 stage 上，就像生产线一样填满各段计算。

{{<figure
    src="https://developer-blogs.nvidia.com/wp-content/uploads/2021/03/interleaved_1F1B_schedule-1-625x288.png"
    caption="Fig. 11. Pipeline Parallel 的 1F1B 和 interleaved 1F1B schedule：蓝色是 forward，绿色是 backward，灰色是 pipeline bubble。Interleaving 让每张设备持有多个 model chunks，可以缩短 bubble，但会增加 stage 边界通信。Image source: [NVIDIA Technical Blog, Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)."
    align="center"
    width="90%"
>}}

最简单的 GPipe schedule 是先让所有 micro-batches 做完 forward，再统一做 backward。它容易理解，但 activation 会一直保存到对应 backward 才能释放，所以显存压力随 micro-batch 数 $m$ 上升。1F1B（one-forward-one-backward）则在 warm-up 之后交替执行 forward 和 backward：某个 stage 只要有 backward 可以做，就尽快反传并释放 activation，因此峰值 activation 显存通常低很多。

PP 的核心代价是 pipeline bubble。设 pipeline stage 数为 $p$，micro-batch 数为 $m$，不考虑 interleaving 时，开头填充流水线和结尾排空流水线都会产生空闲时间，bubble fraction 可以粗略写成：

$$
\text{bubble fraction}\approx\frac{p-1}{m}
$$

所以 PP 想跑得满，$m$ 通常要明显大于 $p$。但 micro-batch 也不能无限加：micro-batch 太小会让单次 GEMM 变小、kernel 利用率下降；micro-batch 太多又会增加调度开销，并可能影响 optimizer step 的 batch 语义。实际训练里常用 gradient accumulation 把多个 micro-batches 聚成一个 global batch，再在 pipeline flush 后统一更新参数。

Interleaved 1F1B 进一步把每张设备上的连续层切成多个 virtual pipeline stages / model chunks。例如原来每张 GPU 负责 8 层，可以拆成两个 4 层 chunk，按交错顺序安排在 pipeline 中。这样等价于让流水线更细，bubble 大致可以再除以 chunk 数 $v$：

$$
\text{bubble time}\approx\frac{(p-1)(t_f+t_b)}{v}
$$

代价是 stage 边界变多，activation 和 activation gradient 的点对点通信也变多。因此 PP 常被放在跨节点维度：TP 适合节点内高带宽互联，PP 的通信只发生在相邻 stage 之间，消息是 activation 而不是整层参数 all-reduce，更容易跨节点扩展。

和 TP 的区别可以这样看：

| 方式 | 切分维度 | 通信位置 | 主要收益 | 主要风险 |
| :--- | :--- | :--- | :--- | :--- |
| Tensor Parallel | 层内部宽度 | 每个 Transformer block 内部 collective | 解决单层矩阵太宽、单层计算太重 | TP degree 过大导致小 GEMM 和频繁通信 |
| Pipeline Parallel | 层深度 | 相邻 stage 之间 send / recv activation | 解决模型太深、整模型放不下，适合跨节点扩展 | pipeline bubble、stage 负载不均、micro-batch 调参复杂 |

因此，PP 更适合「层数很多」或「单机放不下完整模型深度」的场景。对 LLM 训练常见的组合是 TP 放在节点内、PP 跨节点切层、DP/FSDP/ZeRO 在最外层扩 batch 和切模型状态。对 post-training 来说，如果 rollout batch 不大、sequence 很长，PP 的 bubble 可能更难摊薄；如果训练的是很深的大模型，并且已经有足够的 micro-batches 或 gradient accumulation，PP 才更容易体现收益。

## LlamaFactory

LLaMA-Factory 更适合做 SFT、LoRA、QLoRA、DPO 这类离线微调和快速实验。它对 Qwen 等常见模型、HuggingFace / ModelScope 数据集、本地数据集、LoRA 合并与量化、Web UI 和命令行训练流程支持完整，工程门槛低，适合单机或中小规模集群上快速验证数据、模板、超参和微调方法。

### SFT

简单用 4090 上手验证一下 SFT (LoRA)，通过 webui 配置训练参数得到命令行训练。训练数据是 Alpaca 格式，不管是 Alpaca 还是 ShareGPT，LF 都会转换为一个统一的格式，然后 `--template qwen` 转换为 Qwen Chat Template。

LoRA 设置为 `--lora_target all`，表示对模型中的所有线性层应用 LoRA。rank 越大，可训练参数越多，表达能力也更强；常用 rank 为 $rank \in \[4,8,16,32\]$，太大边际收益小。lora_alpha 是 LoRA 的缩放系数，常见经验是设为 rank 的 2 倍，表示 LoRA 分支输出会按 2 倍系数缩放后再叠加到原始权重上。

| 参数 | 说明 |
| :--- | :--- |
| --stage sft | 训练阶段是监督微调，让模型学习指令数据里的回答方式。 |
| --finetuning_type lora | 使用 LoRA 做参数高效微调，只训练低秩适配器，不更新全部模型参数。 |
| --model_name_or_path Qwen2.5-0.5B-Instruct | 基座模型路径，这里用本地的 Qwen2.5-0.5B-Instruct。 |
| --template qwen | 对话模板，需要和 Qwen 系列模型的 chat template 对齐。 |
| --dataset_dir cvalues-sft / --dataset cvalue | 数据集目录和数据集名称，对应 LLaMA-Factory 的数据集配置。 |
| --cutoff_len 2048 | 单条样本的最大 token 长度，超过后会截断。 |
| --per_device_train_batch_size 8 | 单卡 micro batch size 稍微设大一点，目的是尽量吃满 GPU，提高训练吞吐；前提是显存能放下。 |
| --gradient_accumulation_steps 2 | 梯度累积步数设小一点，每 2 个 micro batch 更新一次；单卡等效为 8 * 2 = 16 个样本更新一次。 |
| --learning_rate 5e-05 | LoRA 参数的学习率，通常可以比全参微调更大。 |
| --num_train_epochs 3.0 | 训练 3 个 epoch。 |
| --lr_scheduler_type cosine / --warmup_ratio 0.03 | 前 3% 训练步数做 warmup，让学习率从小到大逐步升高，避免训练初期不稳定；之后用 cosine 衰减，让后期更新更平滑。 |
| --bf16 True | 使用 bf16 训练，适合支持 bf16 的 GPU。 |
| --flash_attn auto | 自动启用可用的 FlashAttention，降低显存并提升长序列吞吐。 |
| --lora_rank 8 / --lora_alpha 16 / --lora_dropout 0 | LoRA 的秩、缩放系数和 dropout。 |
| --lora_target all | 尽量对模型中的可训练线性层都插入 LoRA adapter。 |

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /public/home/test120/models-datasets/models/Qwen2.5-0.5B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir /public/home/test120/models-datasets/datasets/cvalues-sft \
    --dataset cvalue \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_ratio 0.03 \
    --packing False \
    --enable_thinking False \
    --report_to none \
    --output_dir /public/home/test120/Dev/SFT-cvalue-LF/train_logs \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```



{{<figure
    src="lf-sft.png"
    caption="Fig. 12. SFT 过程中的 train_loss。"
    align="center"
    width="90%"
>}}



## VeRL

veRL 更适合做 PPO、GRPO、GSPO 等 RLHF / RLVR 强化学习后训练，尤其是多机多卡的大规模训练。虽然 LLaMA-Factory 也支持 PPO、奖励模型训练等 RLHF 流程，但在线强化学习通常要高吞吐生成多条回答、调用奖励模型或规则奖励打分，并在 actor、reference model、reward model、rollout engine 和 trainer 之间协调资源；veRL 对这类分布式 RL 训练流程更偏工程化，能更自然地接入 FSDP / Megatron-LM、vLLM / SGLang 等训练和推理基础设施。

### PPO

[快速入门：在 GSM8K 数据集上进行 PPO 训练](https://verl.org.cn/en/latest/start/quickstart.html)

入门级别的 PPO 训练教程，数据集 OpenAI/GSM8K，模型 Qwen2.5-0.5B-Instruct。

HuggingFace 原始 GSM8K 的 parquet 不能直接用，还要把原始 GSM8K 的字段结构变成 verl 训练需要的字段结构：
1. 它包含计算 RL 奖励所需的字段
2. 读取速度更快


```python
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

```json
------------ 原始 ------------
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}

------------ VeRL ------------

{
  "data_source": "openai/gsm8k",
  "prompt": [
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\"."
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "72"
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
  }
}


```

在这次实验中，Qwen2.5-0.5B 有这些职责：
- Actor，训练，PPO 优化目标
- Rollout，不训练，其实和 Actor 权重一样，就是 Actor 的 copy
- Reference，冻结，参考模型
- Critic，训练，proj head

```text
                Qwen2.5-0.5B
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Actor       Reference     Critic
        训练          冻结          训练
          │
          ↓
       Rollout
     vLLM生成回答
```

最终的的 VeRL 训练命令行，一些关键的参数：
- train_batch_size=128，$\pi_{old}$ 一次 rollout 的数量，如果小的话方差不稳定
- ppo_mini_batch_size=64，64 条轨迹更新一次参数
- ppo_micro_batch_size_per_gpu=4，gpu 一次处理 4 条轨迹 

```BASH
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

下面这组 TensorBoard 曲线来自一次单卡 smoke test：`train_batch_size=256`，`ppo_mini_batch_size=64`，actor / critic 的 `ppo_micro_batch_size_per_gpu=1`，`kl_coef=0.001`，训练 3 个 epoch，并在训练前先跑一次验证。它不是上面示例命令的完整替代，而是用相同 GSM8K 数据和 Qwen2.5-0.5B-Instruct 做的一次快速可行性检查。

{{<figure
    src="ppo-smoke-test-tensorboard.png"
    caption="Fig. 12. PPO smoke test 的 TensorBoard 曲线汇总：上排是验证分数、critic reward 和 critic score，下排是 actor KL、PPO clip fraction 和梯度范数。"
    align="center"
    width="100%"
>}}

这组曲线可以放在一起看：`val-core/open.../mean@1` 从接近 0 上升到约 0.53，说明经过 PPO 后，模型在 GSM8K 验证集上的 pass@1 类指标有了明显提升；`critic/rewards/mean` 和 `critic/score/mean` 基本同步从 0 附近涨到 0.62 左右，说明 reward/score 信号和最终验证指标方向一致，不是只在训练奖励上虚高。

Actor 侧的三个指标主要看训练是否稳定。`actor/ppo_kl` 保持在 $10^{-4}$ 量级，和 `kl_coef=0.001` 对应，说明策略没有明显偏离 reference；`actor/pg_clipfrac` 最终约 0.004，表示被 PPO clip 截断的 token 比例很低，策略更新没有频繁撞到 clip 边界；`actor/grad_norm` 大致在 1.9 到 2.1 之间波动，没有持续爆炸。整体看，这次 A01 单卡实验更像是一个稳定收敛的 sanity check：reward、score 和验证指标都在升，KL 和 clipfrac 仍然很小。



## 参考文献

[1] Long Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.

[2] John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[3] Paul F. Christiano et al. [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741), 2017.

[4] Rafael Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.

[5] Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.

[6] Chujie Zheng et al. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), 2025.

[7] Samyam Rajbhandari et al. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2020.
