**📚**** DeepSeek系列文章**

[一文了解 DeepSeek 系列模型的演进与创新](https://zhuanlan.zhihu.com/p/1936913619192361679)

[一文搞懂DeepSeek LLM](https://zhuanlan.zhihu.com/p/1937265830216857540)

[DeepSeekMoE 架构解析](https://zhuanlan.zhihu.com/p/1937606952491410576)



在大模型快速演进的时代，Mixture-of-Experts（MoE）模型成为提升性能与效率的重要路径。然而，如何让 MoE 模型既拥有足够的表达能力，又能在推理阶段保持轻量，仍是一个核心挑战。

2024 年 5 月，DeepSeek 团队发布了重磅模型 —— [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)，在 DeepSeekMoE 的基础上做出关键优化，提出**多头潜在注意力（MLA）**机制，并全面提升推理效率与训练经济性。

在阅读这篇文章前，我们建议你带着以下三个问题思考：

1. **DeepSeek-V2 在 MoE 架构上做了哪些关键改进？**
2. **MLA（Multi-head Latent Attention）是如何帮助减少推理开销的？**
3. **相比 Mixtral 等 MoE 模型，DeepSeek-V2 在效率与性能之间是如何平衡的？**



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

---

## 一、设计目标
DeepSeek-V2 的出发点是构建一个**比肩 SOTA 模型性能**、但同时**具备推理高效与训练经济性**的 MoE 架构。为此，他们结合了 DeepSeekMoE 的专家机制和一种新颖的注意力优化方式——**Multi-head Latent Attention（MLA）**。

其模型构成如下：

| 模型 | 总参数量 | 激活参数量 | MoE 层数 | 每层专家数 | 路由策略 |
| --- | --- | --- | --- | --- | --- |
| DeepSeekMoE 145B | 145B | 36.4B | 48 | 64 | Top-2 |
| **DeepSeek-V2 236B** | 236B | **39B** | 64 | 128 | Top-2 + MLA |


---

## 二、MLA
传统 MoE 架构中的瓶颈之一，是每个 token 都需要独立路由、独立构建 Key/Value 缓存。在大型模型上，这会导致推理成本成倍上升。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754487728457-7af04b4b-884b-49ac-93b7-71a5a21fbd9e.png)





MLA（Multi-head Latent Attention）机制提出了一个折中方案：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754487847279-8144a1a9-2fb6-484b-abd3-d74f20191411.png)

+ **引入 latent slots（潜在注意力槽）**：一个共享的、低维的 latent 空间，供多个 token 共享；
+ 在每个注意力层，将 token 的 Query 与 latent 槽交互，而非与其他所有 token 交互；
+ **KV 缓存不再随 token 增长而线性扩张**，从而极大减小推理过程中的内存和算力需求。



论文中的实验证明：

+ MLA 能将 KV 缓存开销减少 **约 50%**；
+ 在保留模型性能的同时，**大幅提升了推理吞吐量与延迟控制能力**；
+ 特别适合实际部署场景，如 API、移动端、嵌入式等。

---

## 三、训练与性能


在预训练阶段，DeepSeek-V2 结合了 DeepSeekMoE 的以下优势：

+ **细粒度专家分割（FGES）** + **共享专家隔离（SEI）**
+ MoE 层采用 Top-2 路由器，提升专家选择多样性
+ 支持 FP8 精度训练，进一步优化训练成本



在多个基准测试上，DeepSeek-V2 表现稳健，尤其在逻辑推理、数学、代码生成等方面优势明显：

## ![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754487986838-53792af3-2cd1-49f6-b9d8-7070bdbd8618.png)
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754487705030-ee02cbb5-f16f-4652-aa68-0a8d09e65984.png)

## 四、经济性分析
相比传统 Dense 大模型，MoE 模型天然具有“激活参数少、FLOPs 可控”的优势。DeepSeek-V2 在此基础上：

+ **激活参数仅为 39B（相当于一个中型模型）**
+ 每个 token 的计算 FLOPs 显著低于 GPT-4、GPT-3.5 等闭源模型

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754487951138-023e208c-eb78-41a6-91cb-4f50f681dec5.png)

+ 训练与部署成本都更加经济可控，适合大规模实际落地应用

---

## 📌 结语
DeepSeek-V2 并不仅仅是 DeepSeekMoE 的“增大版”，它展示了如何通过结构性创新（如 MLA），在提升性能的同时控制推理成本。对于想将大模型真正落地到产品、服务中的研究者和工程师来说，DeepSeek-V2 提供了一个 **更强、更稳、更经济的 MoE 路线图**。



最后我们回答一下文章开头提出的三个问题：

**1. DeepSeek-V2 在 MoE 架构上做了哪些关键改进？**  
在 DeepSeekMoE 的基础上，继续采用细粒度专家划分和共享专家隔离，同时引入 MLA 机制来进一步减少 KV 缓存压力，提升推理效率。

**2. MLA 是如何帮助减少推理开销的？**  
MLA 通过引入 latent attention slots，让多个 token 共享一组 KV 信息，避免每个 token 都独立存储 KV，从而显著减小推理中的显存和计算需求。

**3. 相比 Mixtral 等模型，DeepSeek-V2 在效率与性能之间如何平衡？**  
在保持与 Mixtral 类似的计算成本下，DeepSeek-V2 取得更强性能表现，且具有更优的推理效率和内存利用率，展现了 MoE 架构在工业场景下的实用潜力。

---

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">算法coting</font><font style="color:rgb(25, 27, 31);">！</font>



## 📚 推荐阅读
[一文了解 DeepSeek 系列模型的演进与创新](https://zhuanlan.zhihu.com/p/1936913619192361679)

[一文搞懂DeepSeek LLM](https://zhuanlan.zhihu.com/p/1937265830216857540)

