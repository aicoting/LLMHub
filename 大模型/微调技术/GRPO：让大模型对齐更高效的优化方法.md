# 📚 微调系列文章
[一文了解微调技术的发展与演进](https://zhuanlan.zhihu.com/p/1939080284374022103)  
[一文搞懂 LoRA 如何高效微调大模型](https://zhuanlan.zhihu.com/p/1939447022114567022)  
[LoRA详细步骤解析](https://zhuanlan.zhihu.com/p/1939807872113410970)	  
[一文搞懂如何用 QLoRA 高效微调大语言模型](https://zhuanlan.zhihu.com/p/1939997552779978284)  
[一文理解 AdaLoRA 动态低秩适配技术](https://zhuanlan.zhihu.com/p/1940347806129845834)  
[一文理解提示微调（Prefix Tuning/Prompt Tuning/P Tuning）](https://zhuanlan.zhihu.com/p/1940892127459547050)  
[RLHF （基于人类反馈的强化学习）的核心原理](https://zhuanlan.zhihu.com/p/1941259638084469752)  
[一文理解监督微调(SFT)在大语言模型训练中的作用](https://zhuanlan.zhihu.com/p/1944692406898393889)  
[一文理解 PPO 的核心机制与大模型中的应用](https://zhuanlan.zhihu.com/p/1945428431652238524)  
[DPO是怎么通过偏好数据直接优化大模型的？](https://zhuanlan.zhihu.com/p/1946134061836907801)



在大语言模型（LLM）的训练与对齐过程中，如何在保证模型性能的同时提升训练效率，一直是学界和工业界关注的焦点。常见的方法有 **PPO（Proximal Policy Optimization）** 和 **DPO（Direct Preference Optimization）**，但它们各自也有一些局限性。最近提出的 **GRPO（Group Relative Policy Optimization）**，则在这方面提供了一种新的思路。本文将从概念、原理、对比和应用场景四个角度，带你了解这一方法。

阅读本文时，请带着这三个问题思考：

1. **GRPO和PPO有什么区别？**
2. **GRPO 的基本原理是什么？**
3. **GRPO有哪些应用场景？**



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/aicoting/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

---

## 1. 什么是 GRPO？
GRPO，全称 **Group Relative Policy Optimization**，是一种用于训练大模型的 **对齐优化方法**。它的核心目标是：在不依赖复杂的价值函数（value model）的情况下，通过分组比较和相对奖励机制来指导模型学习用户偏好。

<font style="color:rgb(77, 77, 77);">GRPO最初在 </font>[<font style="color:rgb(78, 161, 219);">DeepSeekMath</font>](https://arxiv.org/abs/2402.03300)<font style="color:rgb(77, 77, 77);"> 中提出，用于提升模型在开放域数学问题上的推理能力，后扩展至 </font>[<font style="color:rgb(78, 161, 219);">DeepSeek-R1</font>](https://arxiv.org/abs/2501.12948)<font style="color:rgb(77, 77, 77);"> 等通用推理模型。结合</font><font style="color:rgb(78, 161, 219) !important;">LoRA</font><font style="color:rgb(77, 77, 77);">（低秩适配）技术，GRPO可在消费级GPU上微调小模型，降低了RLHF门槛。</font>

相比传统的 RLHF（基于强化学习的对齐方式），GRPO 不需要额外训练一个 reward model，而是通过 **成组样本比较** 的方式来决定哪些输出更优，从而引导模型参数的更新。这使得它在效率、稳定性和实现难度上，都有一定优势。

---

## 2. GRPO 的核心原理
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1755674685665-1bedd3e0-bd29-4969-929b-b509057626f0.png)

GRPO 的关键思想是：

+ **成组比较**：不再是单一候选的奖励评估，而是将多个模型输出放在一起进行比较。
+ **相对奖励**：并不是给绝对分数，而是通过组内排序来决定哪个输出更好。
+ **优化目标**：利用这些相对偏好，更新策略，使模型更倾向于生成高质量的输出。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1755675324005-fad4f98c-d6c7-44ca-864c-7a7313eb7b48.png)

具体来说，假设一个 prompt 有多个 candidate responses，人工或自动打分系统可以对这些响应进行排序。GRPO 会根据排序关系，将更好的响应作为正样本，较差的作为负样本，然后通过相对优势来更新模型。这种方式避免了 **PPO 那种需要精确 reward function 的问题**，也减少了 DPO 在单对比较下的局限性。

**示例代码**

```python
# === GRPO 伪代码示例 ===
# 输入：prompt，模型 policy πθ
# 输出：更新后的 policy

for prompt in dataset:
    # 1. 生成一组候选答案（group）
    candidates = [policy.generate(prompt) for _ in range(GROUP_SIZE)]
    
    # 2. 计算每个候选的 reward（如基于偏好模型或规则）
    rewards = [reward_fn(c) for c in candidates]
    
    # 3. 相对奖励：组内减去平均值，突出相对好坏
    baseline = mean(rewards)
    relative_rewards = [r - baseline for r in rewards]
    
    # 4. 计算旧策略概率 (log prob)
    old_log_probs = [old_policy.log_prob(c) for c in candidates]
    new_log_probs = [policy.log_prob(c) for c in candidates]
    
    # 5. 构造目标函数 (类似 PPO，但基于组内相对奖励)
    ratios = [exp(new - old) for new, old in zip(new_log_probs, old_log_probs)]
    objective = mean([
        min(ratio * rel_r,
            clip(ratio, 1-ε, 1+ε) * rel_r)
        for ratio, rel_r in zip(ratios, relative_rewards)
    ])
    
    # 6. 更新参数
    policy.update(objective)

```

---

## 3. GRPO 与 PPO、DPO 的对比
+ **PPO**
    - 优点：稳定的强化学习优化方法，广泛应用于 RLHF。
    - 局限：需要训练额外的 reward model，成本高，容易引入偏差。
+ **DPO**
    - 优点：直接基于用户偏好优化，不需要 reward model。
    - 局限：主要依赖成对比较（pairwise），在多候选场景下效率不足。
+ **GRPO**
    - 优点：结合了两者的优点，支持 **组内比较**，更好地利用反馈信息；不需要额外 reward model。
    - 价值：在数据利用率、训练稳定性、效率上都有提升。

---

## 4. GRPO 的应用场景
GRPO 的应用潜力主要体现在 **大模型对齐** 的关键任务上：

+ **对话系统优化**：通过用户反馈，将模型回答引导得更自然、更符合人类偏好。
+ **内容生成**：在文本生成、摘要、翻译等任务中，利用组比较方式高效筛选高质量结果。
+ **多模态模型训练**：不仅限于文本，还能扩展到图像-文本生成或视频生成的对齐中。
+ **低成本对齐场景**：对于缺乏大规模 reward model 训练资源的团队，GRPO 提供了一条更轻量化的路径。

随着大模型逐渐走向应用落地，GRPO 这种相对简洁高效的优化方法，有望成为主流的对齐工具之一。

---

最后我们回答一下文章开头提出的三个问题：

1. **GRPO和PPO有什么区别？**

PPO训练和维护单独的value model和reward model进行强化学习；GRPO抛弃了独立的value model，通过分组比较和相对奖励机制来指导模型学习用户偏好，GRPO 不需要额外训练一个 reward model，而是通过 **成组样本比较** 的方式来决定哪些输出更优，从而引导模型参数的更新。

2. **GRPO 的基本原理是什么？**

GRPO 会根据排序关系，将更好的响应作为正样本，较差的作为负样本，然后通过相对优势来更新模型。

3. **GRPO 有哪些应用场景？**

**<font style="color:rgb(0, 0, 0);">GRPO适用于计算资源受限</font>**<font style="color:rgb(0, 0, 0);">，</font>**<font style="color:rgb(0, 0, 0);">存在客观评价标准</font>**<font style="color:rgb(0, 0, 0);">（代码、数学、科学等领域，可以通过程序化、确定性的方式来评估生成内容的质量）和</font>**<font style="color:rgb(0, 0, 0);">需要提升模型推理能力</font>**<font style="color:rgb(0, 0, 0);">的应用场景。</font>

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">coting</font><font style="color:rgb(25, 27, 31);">！</font>

<font style="color:rgb(25, 27, 31);">以上内容部分参考了相关开源文档与社区资料。非常感谢，如有侵权请联系删除！</font>





