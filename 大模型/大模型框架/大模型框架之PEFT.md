📚** 大模型框架系列文章**

[大模型工程框架生态全览](https://zhuanlan.zhihu.com/p/1946500640349094644)

[深入 LangChain：大模型工程框架架构全解析](https://zhuanlan.zhihu.com/p/1946599445497095365)

[手把手带你使用LangChain框架从0实现RAG](https://zhuanlan.zhihu.com/p/1946857016162252076)

[深入 vLLM：高性能大模型推理框架解析](https://zhuanlan.zhihu.com/p/1947248904983811905)

[知识管理与 RAG 框架全景：从 LlamaIndex 到多框架集成](https://zhuanlan.zhihu.com/p/1947256018003277719)

[大模型微调框架之TRL](https://zhuanlan.zhihu.com/p/1947619721609458855)

在大模型浪潮下，如何让模型在 **低成本、低门槛** 的条件下完成特定任务的适配，是开发者和研究者共同关注的问题。  
Hugging Face 推出的 **PEFT（Parameter-Efficient Fine-Tuning）框架**，为这一挑战提供了系统化的解决方案。这篇文章就是带大家看一下PEFT框架到底是何方神圣。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1756285089281-1606264c-2507-4293-9772-715513559218.png)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/algcoting/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案。

**1.PEFT的工作原理是什么？**

**2.PEFT支持哪些微调方法？**

**3.PEFT有什么优势？**

---

## 1. 什么是 PEFT？
**PEFT** 是 Hugging Face 推出的一个 **参数高效微调框架**，通过只训练模型的一小部分额外参数（如 LoRA adapter），避免对整个大模型进行全量更新。

+ **降低显存和计算需求**
+ **减少存储开销**（adapter 通常只有几 MB）
+ **性能接近全量微调**
+ **支持多种方法**（LoRA、Prefix Tuning、Prompt Tuning、IA³ 等）

PEFT 框架将这些方法统一封装，方便开发者在 Hugging Face 生态内快速调用。

---

## 2. 框架功能与生态集成
PEFT 与 Hugging Face 的多个核心库深度集成：

+ **Transformers**：  
提供 `get_peft_model`、`add_adapter`、`load_adapter` 等接口，可以在 NLP 模型上快速添加和切换 adapter。
+ **Diffusers**：  
适用于 Stable Diffusion 等扩散模型，显著降低训练显存占用。
+ **Accelerate**：  
支持分布式训练和推理，轻松运行在 GPU、TPU、Apple Silicon 等不同硬件环境。
+ **TRL（Transformer Reinforcement Learning）**：  
可结合 RLHF、DPO 等方法，应用于大模型对齐和强化学习微调。

---

## 3. 快速上手示例
以 **Qwen2.5-3B-Instruct** 为例，只需几行代码即可使用 LoRA 微调：

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 定义 LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
)

# 将模型包装为 PEFT 模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

输出显示：仅需训练 **0.1% 的参数**！

训练完成后保存 adapter：

```python
model.save_pretrained("qwen2.5-3b-lora")
```

加载推理：

```python
from transformers import AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model = PeftModel.from_pretrained(model, "qwen2.5-3b-lora")

inputs = tokenizer("写一首关于春天的诗", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4. 应用场景
+ **NLP**：文本分类、问答、摘要、机器翻译等下游任务。
+ **CV**：图像分类、检测、分割等视觉任务。
+ **扩散模型**：Stable Diffusion 的个性化微调（如 DreamBooth）。
+ **RLHF**：大语言模型的偏好对齐与强化学习。

---

## 5. 总结
作为 Hugging Face 生态中的重要组成部分，**PEFT 框架**为大模型的落地提供了新的可能：

+ 在 **计算与存储资源有限** 的环境下完成微调；
+ 在 **多任务、多领域** 场景下快速切换 adapter；
+ 与 **Transformers、Diffusers、Accelerate、TRL** 等库无缝集成；
+ 支持 **量化（如 QLoRA）**，进一步降低硬件门槛。

未来，随着大模型规模不断扩大，**PEFT 框架将会成为大模型训练与部署的标配工具**。



最后我们来回答一下文章开头提出的三个问题：

**1. PEFT 的工作原理是什么？**  
PEFT（参数高效微调）的工作原理是在保持大模型 **绝大多数参数冻结不动** 的情况下，只训练一小部分额外参数（如低秩矩阵、前缀向量或适配层）。这些额外参数通过与原有权重的结合，能够有效调整模型的表达能力，使模型适应下游任务，同时避免大规模参数更新带来的计算和存储开销。

**2. PEFT 支持哪些微调方法？**  
PEFT 框架目前支持多种主流的参数高效微调方法，包括 **LoRA（Low-Rank Adaptation）**、**Prefix Tuning**、**Prompt Tuning（软提示）**、**P-Tuning v2**、**IA³（Infused Adapter）** 等。这些方法通过不同的方式在模型中插入少量可训练参数，实现对大模型的快速适配。

**3. PEFT 有什么优势？**  
PEFT 的主要优势是 **高效与灵活**：一方面，它显著减少了显存、计算和存储需求，使得消费级 GPU 就能完成大模型微调；另一方面，PEFT 训练得到的 adapter 文件通常只有几 MB，便于存储和快速切换，避免灾难性遗忘。此外，实验结果表明，PEFT 在性能上能够接近甚至媲美全量微调，是大模型落地应用的重要工具。

关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号 **coting**！

以上内容部分参考了相关开源文档与社区资料。非常感谢，如有侵权请联系删除！

