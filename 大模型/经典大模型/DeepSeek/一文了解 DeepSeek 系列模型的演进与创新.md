# <font style="color:rgb(51, 51, 51);">一文了解 DeepSeek 系列模型的演进与创新</font>
<font style="color:rgb(51, 51, 51);">近年来，DeepSeek 团队在大语言模型（LLM）领域持续发力，围绕模型架构、专家路由、推理效率、训练方法等方面不断优化，推出了一系列性能强劲的开源模型。本文对 DeepSeek 系列的关键论文进行了梳理，帮助大家快速了解其技术演进路径与核心创新。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754489996572-a5992414-3a5f-45f0-b1b0-5928833877b2.png)

---

## <font style="color:rgb(51, 51, 51);">1. </font>[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)<font style="color:rgb(51, 51, 51);">（2024年1月）</font>
<font style="color:rgb(51, 51, 51);">作为 DeepSeek 系列的首个基础模型，DeepSeek LLM 基于 Transformer 架构，并在推理效率和训练调度上做出优化：</font>

+ <font style="color:rgb(51, 51, 51);">引入 </font>**分组查询注意力（GQA）**<font style="color:rgb(51, 51, 51);">，有效降低推理成本；</font>
+ <font style="color:rgb(51, 51, 51);">支持 </font>**多步学习率调度器**<font style="color:rgb(51, 51, 51);">，提升训练效率；</font>
+ <font style="color:rgb(51, 51, 51);">在预训练和对齐阶段提出创新方法，为后续模型打下基础。</font>



---

## <font style="color:rgb(51, 51, 51);">2. </font>[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)<font style="color:rgb(51, 51, 51);">（2024年1月）</font>


![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754405960635-7b7eebb7-6ff2-4148-af97-48cc2ef443c5.png)

<font style="color:rgb(51, 51, 51);">DeepSeekMoE 聚焦于混合专家（MoE）结构的高效利用，提出了两个关键策略：</font>

+ **细粒度专家分割（Fine-Grained Expert Segmentation）**<font style="color:rgb(51, 51, 51);">：提高专家模块的可组合性；</font>
+ **共享专家隔离（Shared Expert Isolation）**<font style="color:rgb(51, 51, 51);">：提升专家之间的独立性，避免干扰；</font>

<font style="color:rgb(51, 51, 51);">在不增加计算开销的前提下，实现了更灵活、高性能的专家调用方式。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754405928704-69f138e4-b411-4437-b10c-529b744cbe7b.png)

---

## <font style="color:rgb(51, 51, 51);">3. </font>[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)<font style="color:rgb(51, 51, 51);">（2024年5月）</font>
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754406011619-8d322877-5496-4521-945b-ce6aad140eb7.png)

<font style="color:rgb(51, 51, 51);">DeepSeek-V2 在 DeepSeekMoE 的基础上进一步优化性能与成本：</font>

+ <font style="color:rgb(51, 51, 51);">创新引入 </font>**多头潜在注意力（MLA）**<font style="color:rgb(51, 51, 51);">，大幅减少推理过程中的 KV 缓存；</font>
+ <font style="color:rgb(51, 51, 51);">延续 MoE 架构优势，在推理效率显著提升的同时，降低整体训练成本。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754405992570-1634d532-227f-448a-a150-6f77387aaae0.png)

---

## <font style="color:rgb(51, 51, 51);">4. </font>[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)<font style="color:rgb(51, 51, 51);">（2024年12月）</font>
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754406045265-3a79136b-9aef-455e-ad03-97f2a2c2726d.png)

<font style="color:rgb(51, 51, 51);">DeepSeek-V3 是目前该系列中规模最大、性能最强的模型：</font>

+ <font style="color:rgb(51, 51, 51);">总参数量达 </font>**671B**<font style="color:rgb(51, 51, 51);">，每个 token 激活 </font>**37B**<font style="color:rgb(51, 51, 51);"> 参数；</font>
+ <font style="color:rgb(51, 51, 51);">采用 </font>**无辅助损失的负载均衡策略**<font style="color:rgb(51, 51, 51);"> 和 </font>**多令牌预测（MTP）**<font style="color:rgb(51, 51, 51);"> 训练目标；</font>
+ <font style="color:rgb(51, 51, 51);">支持 </font>**FP8 混合精度训练**<font style="color:rgb(51, 51, 51);">，在保证性能的同时大幅降低训练资源消耗。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754406079817-4f595cb5-658b-4182-a4b8-db5685550eb7.png)

---

## <font style="color:rgb(51, 51, 51);">5. </font>[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)<font style="color:rgb(51, 51, 51);">（2025年1月）</font>
<font style="color:rgb(51, 51, 51);">DeepSeek-R1 旨在进一步提升模型的</font>**推理能力**<font style="color:rgb(51, 51, 51);">，核心策略包括：</font>

+ <font style="color:rgb(51, 51, 51);">基于 DeepSeek-V3-Base 进行强化学习优化；</font>
+ <font style="color:rgb(51, 51, 51);">引入 </font>**冷启动数据集**<font style="color:rgb(51, 51, 51);"> 和 </font>**多阶段训练流程**<font style="color:rgb(51, 51, 51);">；</font>
+ <font style="color:rgb(51, 51, 51);">显著提升模型在复杂任务中的可读性与逻辑性。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754406093735-f9f5e4c0-375c-47d5-835e-8ca382b6c10a.png)

---

## <font style="color:rgb(51, 51, 51);">6. </font>[Distilling Reasoning Capabilities from DeepSeek-R1 to Smaller Models](https://github.com/deepseek-ai/DeepSeek-R1)<font style="color:rgb(51, 51, 51);">（2025年1月）</font>
<font style="color:rgb(51, 51, 51);">为降低大模型使用门槛，团队发布了基于 DeepSeek-R1 的蒸馏模型：</font>

+ <font style="color:rgb(51, 51, 51);">推理能力被成功迁移至更小模型，如 Qwen、LLaMA 等；</font>
+ <font style="color:rgb(51, 51, 51);">蒸馏后的模型在多个评测任务中超越同类开源模型，在保持轻量的同时具备强大推理性能。</font>

---

## <font style="color:rgb(51, 51, 51);">结语</font>
<font style="color:rgb(51, 51, 51);">DeepSeek 系列不仅在大模型架构上持续创新，还在高效推理、专家分配、推理能力增强等方面提出了系统性的解决方案。从基础模型到混合专家，再到强化学习与知识蒸馏，展现了一个完整的大模型演进路径，为开源社区带来了极具参考价值的技术成果。</font>

<font style="color:rgb(51, 51, 51);">如果你正在研究大语言模型，DeepSeek 系列无疑是值得深入学习与关注的重要项目。</font>

