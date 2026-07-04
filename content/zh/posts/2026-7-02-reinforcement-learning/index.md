---
title: "Reinforement Learning"
date: 2026-07-02T21:27:00+08:00
author: "dengkoicat"
tags: ["Deep Learning", "Reinforcement Learning","AI"]
categories: ["Deep Learning"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 基本概论

| 符号 | 含义 |
| :--- | :--- |
|   **state** $s$。  | 状态 |
|   **Action** $a$   | 动作 |
|   **Policy** $\pi$  | 决策函数，根据状态做出决策到控制 Agent 的动作（马里奥上下左右）|
|   **reward** $R$ | 奖励，最影响强化学习的结果 |
|   **state transition** |状态转移 $\text{old_state} \rightarrow \text{new_state}$,一般是随机的，随机性从环境来 |
| **Return** | 回报，把t时刻以后的奖励加起来|


### 决策函数

**Policy function** $\pi: (s, a) \in [0, 1]$,就是一个 *PDF* :

$$\pi(a \mid s) = \mathbb{P}(A = a \mid S = s).$$

强化学习就是学习这个 **Policy function** $\pi$, 来指导 Agent 观测 $\mathcal{s}$ 来做出动作 $\mathcal{a}$。这个 $\pi$ 使得Agent 动作有随机性，
这样才不会被“对手”算出来下一步动作, 所以 $\pi$ 最好是一个PDF。

{{< figure
    src="policy.png"
    caption="Fig. 1. Policy函数举例，马里奥可以上左右移动。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

### Agent 与环境交互

马里奥就是 Agent，环境就是游戏界面，马里奥根据环境来做决策，然后到下一个状态。吃到金币给奖励，赢了游戏给大奖励，与之对应，输了游戏给大惩罚。

{{< figure
    src="agent-environment.png"
    caption="Fig. 2. Agent和环境交互例子。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

### RL 中的随机性

* **动作随机性**：给定状态 $s$ ，Agent的状态可能是随机的，比如
    * $\pi(\text{"left"} \mid s) = 0.2$,
    * $\pi(\text{"right"} \mid s) = 0.1$,
    * $\pi(\text{"up"} \mid s) = 0.7$.

* **状态随机性**：给定状态 $S = s,A = a$ ，$S\rightarrow S'$ 环境转移是随机的，系统会随机抽样到下一个状态
    * $S' = P(\cdot \mid s,a)$


### RL 过程

* 游戏界面 (**state** $s_1$)
* 根据 $\pi$ 做出 (**action** $a_1$)
* 得到一个新的游戏界面 (**state** $s_2$) 和 **reward** $r_1$
* 根据 $\pi$ 做出 (**action** $a_2$)
* ....

{{< figure
    src="rl-tra.png"
    caption="Fig. 3. RL 轨迹链路示意。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

(**state**,**action**,**reward**)轨迹：
$$s_1,a_1,r_1,s_2,a_2,r_2,\dots,s_T,a_T,r_T$$

### Return \& Reward

**Return** 回报，把t时刻以后的奖励加起来:
* $U_t = R_t+R_{t+1}+R_{t+2}+\dots$ 

*eg*, $R_t$ 和 $R_{t+1}$ 同样重要吗？
* 下面哪个你最想要？
    * 立刻给你 80
    * 一年后给你 100

很明显，肯定是现在 80 最好，一年后的 100 块不如现在的 80。所有 $R_{t+1}$ 不如当前的 $R{t}$ 重要
，一般要给 $R_{t+1}$ 打个折扣，比如八折。

**Discounted Return** 折扣回报率。
* $\gamma \in [0,1]$，折扣率 (hyper-parameter)
* $U_t=R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2} + \gamma ^3 R_{t+3}+\dots$

在 $t$ 步，回报 $U_t$ 是随机的。
* 随机性来源两方面：
    * **action** 是随机的，$\mathbb{P}(A = a \mid S = s) = \pi(a \mid s)$
    * **state** 是随机的，$\mathbb{P}(S' = s' \mid S = s,A = a  ) = \pi(s' \mid s,a)$
* 对于 $i \ge t$，**reward** $R_i$ 取决于 $S_i$ 和 $A_i$。
* 因此，给定 $s_t$，回报率 $U_t$ 依赖于这两个随机变量 $A_t,A_{t+1},A_{t+2},\dots$ and $S_{t+1},S_{t+2},\dots$ 
$$A,S \rightarrow R \rightarrow U$$

### 价值函数
折扣回报 $U_t = R_t+\gamma R_{t+1} + \gamma ^2 R_{t+2} + \gamma ^3 R_{t+3}+\dots$，是一个随机变量，在 $t$ 时刻并不知道具体是什么，那如何评估呢？可以对 $U_t$ 求期望，这样就可以消除随机变量。

$Q_\pi(s,a)$，动作价值函数
$$Q_\pi(s_t,a_t) = \mathbb{E}[U_t \mid S_t = s_t,A_t = a_t]$$
动作价值函数 $Q_\pi(s,a)$ 的直观意义就是，用 **Policy** 函数 $\pi$ 在 $s_t$ 做出动作 $a_t$ 是好还是坏。  已知 **Policy** 函数 $\pi$，$Q_\pi(s,a)$ 就会给当前状态下，所有动作 $a$ 打分，这样就可以知道动作的好坏。

$Q^\star(s_t,a_t)$，最大化动作价值函数
$$Q^\star(s_t, a_t) = \max_\pi Q_\pi(s_t, a_t).$$
$Q^\star$ 直接根据 $s_t$ 对 $a_t$ 做评价，比如玩超级玛丽的时候，马里奥当前向上走“好不好”、“胜算有多大”。有了 $Q^\star$ 之后，Agent就可以根据 $Q^\star$ 对动作的评价来做决策，比如 $Q^\star$ 认为往上跳分数最高，Agent就会往上跳。

假如我们在 $Q_\pi(s_t,a_t)$ 不对 $\pi$ 求期望，而是对 $A$ 求期望呢？对 $A$ 求期望就可以把 $A$ 消除掉，是不是就可以得到环境的评估函数？

$V(s)$，状态价值函数 
$$V_\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \sum_{a} \pi(a \mid s_t) \cdot Q_\pi(s_t, a)$$
$V_\pi(s)_t$ 的直观意义就是能够得到局势信息，根据游戏界面，我们就可以知道当前局势是不是“有利”的，是快赢了还是快输了，用于评估状态。

当然我们还可以在 $V_\pi(s)_t$ 对 $s$ 求期望得到 $\mathbb{E}_S[V\pi{S}]$，可以评估 $\pi$ 的好坏。


### How does AI control the agent?

* 假设我们有一个好的 **Policy** 函数 $\pi(a \mid s)$。
    * 观测 **State** $s_t$，
    * 随机采样得到动作：$a_t \sim \pi(a \mid s_t)$

* 假设我们知道最优动作价值函数 $Q^\star(s_t, a_t)$
    * 观测 **State** $s_t$，
    * 根据 $Q^\star(s_t, a_t)$ 选择最好的 $a_t$：$a_t = \text{argmax}_{a}Q^\star(s_t, a_t)$

所有 **RL** 的目的就是学习这两个函数，有了这两个函数之一，我们就可以让 Agent 对 $s_t$ 来做出决策 $a_t$。


## 价值学习,Value-Based RL

### Deep Q-Network

> 强化学习的目标：赢下游戏（最大化 **reward**）。
>
> 如果我们知道 $Q^\star(s, a)$，
> 很明显最好的 **action** 就是 $a\star = \text{argmax}_{a}Q^\star(s, a)$，
>
> 这样我们就可以利用 $Q^\star(s, a)$ 来做决策，虽然在某一时刻不一定
> 是最好的，但也代表了一个趋势。它不一定追求“即时奖励最大”，而是追求“长远累积回报最大”。

但是问题是，*我们不知道 $Q^\star(s, a)$ 是什么*。
谁也不是先知，很难预测股票市场的 $Q^\star(s, a)$，但是对于超级玛丽，Agent 玩了几百万场次后也和先知没有区别了。用自己学习到的经验近似先知（$Q^\star(s, a)$）。

**DQN** 就是用神经网络来近似 $Q^\star(s, a)$,
$$Q(s, a;w) \approx Q^\star(s, a)$$


{{< figure
    src="dqn-input-output.png"
    caption="Fig. 4. DQN的输入在不同任务下可能不同，对于超级玛丽来说，游戏画面就是输入，用卷积层将输入变成向量，全连接层映射到输出向量（score action）。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

DQN的输入在不同任务下可能不同，对于超级玛丽来说，游戏画面就是输入，输出就是移动的打分。

{{< figure
    src="game-with-dqn.png.png"
    caption="Fig. 5. 将 DQN 应用在游戏里）。 (Image source: [Shusen Wang YouTube, 2019](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU/))"
    align="center"
    width="90%"
>}}

* 将 DQN 应用在游戏里
    * 观测到游戏画面内容 $s_t$
    * 根据 $a_t = argmax_aQ(s_t, a;w)$ 做出这一步的决策
    * 得到奖励 $r_t$，状态转移 $s_{t+1} \sim p(\cdot \mid s_t, a_t)$
    * ...


### Temporal Difference (TD) Learning

> 驾车从成都到西安需要的时间
>
> 模型可能估计要 10 个小时，这个值可能不是准确的，甚至完全偏移。不难想到如果非常多的人用了这个模型，用数据去更新模型，从而让模型更准确。
>
> 问题就是 *如何更新模型*


1. 出发前让模型做一个预测： $q = Q(w)$，e.g., $q=10$。
2. 实际上我从成都到西安，驾车只需要两个小时，这个两小时就是真实值 $y$，$y=2$。
3. $\mathcal{loss}= L = \frac{1}{2}(q-y)^2$，记录损失
4. 让模型对 $L$ 在 $w$ 求偏导 $\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial q}{\partial \mathbf{w}} \cdot \frac{\partial L}{\partial q} = (q - y) \cdot \frac{\partial Q(\mathbf{w})}{\partial \mathbf{w}}$
5. 用梯度下降更新模型参数 $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \left. \frac{\partial L}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_t}$

这样的话问题又开了，我们必须完成整个旅途才能更新模型。*能不能在中途或者出发前更新呢*？

能不能在成都->西安中途去更新模型呢，比如到达汉中的时候就去更新模型，有的兄弟有的，**TD** 算法就是来解决这个问题的。

* 模型预测 $q=10$，但实际上 6个小时(actual) 就到汉中了
* 模型这个时候更新权重，并且预测 汉中->西安 2个小时(estimate)
* 更新估计值 $6 + 2 = 8$，这个 8小时 就是 **TD Target**
* 预测值 8个小时比 10个小时 更可靠，因为有了真实观测值介入

所以，**TD** 更新权重的方法是，
1. **TD Target** $y = 8$
2. $L=\frac{1}{2}(Q(W)-y)^2$，$Q(W)-y$ 称为 **TD error** $\delta$
3. $\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial q}{\partial \mathbf{w}} \cdot \frac{\partial L}{\partial q} = (10-8) \cdot \frac{\partial Q(\mathbf{w})}{\partial \mathbf{w}}$
4. 用梯度下降更新模型参数 $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \left. \frac{\partial L}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_t}$

### TD Learning for DQN

在 成都->西安，我们有这个表达式：
$$T_{CD->XI'AN} \approx T_{CD->HZ} + T_{HZ->XI'AN}$$
其中，$T_{CD->XI'AN}$ 模型总预测，$T_{CD->HZ}$ 实际时间，$T_{HZ->XI'AN}$ 模型估计。 

在深度**RL**中：
$$Q(s_t,a_t;\mathbf{w})\approx r_t+\gamma \cdot Q(s_{t+1},a_{t+1};\mathbf{w})$$
其中，$r_t$ t时刻真实的奖励，相当于 $T_{CD->HZ}$ 实际时间，$\gamma \cdot Q(s_{t+1},a_{t+1};\mathbf{w})$ 是 **DQN** 在$t+1$时刻做的估计。
为什么 **RL** 有这种公式呢？

回顾一下折扣回报，
$$
\begin{aligned}
U_t &= R_t + \gamma R_{t+1} + \gamma ^2 R_{t+2} + \gamma ^3 R_{t+3} + \dots \\
&= R_t + \gamma \cdot(R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} + \dots) \\
&= R_t + \gamma \cdot U_{t+1}
\end{aligned}
$$

将 **TD** 应用到 **DQN**:
* **DQN** 的输出，$Q(s_t,a_t;\mathbf{w})$，是 $\mathbb{E}[U_t]$
* **DQN** 的输出，$Q(s_{t+1},a_{t+1};\mathbf{w})$，是 $\mathbb{E}[U_{t+1} ]$
* 由 $U_t = R_t + \gamma \cdot U_{t+1}$ 得到，$Q(s_t,a_t;\mathbf{w}) \approx \mathbb{E}[R_t + \gamma\cdot Q(s_{t+1},a_{t+1};\mathbf{w})]$
* 有了预测值 $Q(s_t,a_t;\mathbf{w_t})$
* **TD Target**:
$$
\begin{aligned}
y_t &= r_t + \gamma \cdot Q(s_{t+1}, a_{t+1}; \mathbf{w}_t) \\
&= r_t + \gamma \cdot \max_{a} Q(s_{t+1}, a; \mathbf{w}_t).
\end{aligned}
$$

* **Loss**: $L_t = \frac{1}{2} [Q(s_t, a_t; \mathbf{w}) - y_t]^2$
* **Gradient descent**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \left. \frac{\partial L_t}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_t}$

