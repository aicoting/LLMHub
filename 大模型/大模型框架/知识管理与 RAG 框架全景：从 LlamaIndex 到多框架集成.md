📚** 大模型框架系列文章**

[大模型工程框架生态全览](https://zhuanlan.zhihu.com/p/1946500640349094644)

[深入 LangChain：大模型工程框架架构全解析](https://zhuanlan.zhihu.com/p/1946599445497095365)

[手把手带你使用LangChain框架从0实现RAG](https://zhuanlan.zhihu.com/p/1946857016162252076)

[深入 vLLM：高性能大模型推理框架解析](https://zhuanlan.zhihu.com/p/1947248904983811905)

****

在大模型工程中，**知识管理与检索增强生成（RAG, Retrieval-Augmented Generation）** 是提升模型准确性和实用性的关键。通过将文档、向量索引、长期记忆和多数据源结合，大模型能够在复杂任务中实现知识增强生成。

前面我已经介绍了RAG的概念，工作流程，并且用LangChain框架实现了一个小小的demo，除了LangChain框架，还有很多优秀的RAG框架。

本篇文章就让我们来看一下 LlamaIndex和Haystack 这两个框架，我简单的介绍一下架构设计，以及多框架集成和知识库动态管理实践，同时提供示例代码帮助你快速理解并上手做自己的小demo。

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/algcoting/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案：

1. **LlamaIndex 和 Haystack 的核心架构设计和使用方法是什么？**
2. **多框架（LangChain + LlamaIndex + vLLM）集成实践如何实现？**
3. **知识库动态更新、长期记忆设计和多数据源整合有哪些最佳实践？**

---

## 1. LlamaIndex 架构解析
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1755140223422-246d2ae6-681b-4f45-ba83-69d91f87cb8a.png)

LlamaIndex 是一个**面向大模型的向量索引与文档管理框架**，其核心功能包括：

+ 文档导入和预处理
+ 文档向量化与索引构建
+ 查询检索与结果聚合

#### 示例代码：构建向量索引
```python
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
# LlamaIndex 将文档向量化存储，可用于高效知识检索，支撑 RAG 流程。
# 读取本地文档
documents = SimpleDirectoryReader('docs/').load_data()

# 构建向量索引
index = GPTVectorStoreIndex.from_documents(documents)

# 查询
query = "Explain the capital of France."
response = index.query(query)
print(response)
```

---

## 2. Haystack 架构设计
和LlamaIndex类似，Haystack 是一个完整的 **检索增强生成（RAG）框架**，提供了丰富的功能：

+ 多种文档存储和索引（FAISS、Elasticsearch、Milvus 等）
+ 多模型组合（检索器 + 生成器）
+ 多轮对话与知识追踪

#### 示例代码：构建检索器 + 生成器管道
```python
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import FAISSDocumentStore
# Haystack 支持多模型组合和检索增强生成，方便快速搭建 RAG 系统。
# 创建文档存储
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# 添加文档
document_store.write_documents([{"content": "Paris is the capital of France.", "meta": {}}])

# 初始化检索器和生成器
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 构建 RAG 管道
pipeline = ExtractiveQAPipeline(reader, retriever)

# 执行查询
result = pipeline.run(query="Where is Paris?", params={"Retriever": {"top_k": 1}})
print(result['answers'][0].answer)
```

---

## 3. 多框架集成案例：LangChain + LlamaIndex + vLLM
我们之前介绍了LangChain并手把手带你们实现了一个demo，如果再次将 LangChain、LlamaIndex 和 vLLM 集成，可以实现比我们上次更高效的代码，他们各自负责：

+ LangChain 负责任务编排、Agent 调度
+ LlamaIndex 提供向量索引与知识检索
+ vLLM 提供高吞吐量推理能力

#### 示例代码：简单集成
```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import VLLM
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
# 结合多框架，可以实现高性能、知识增强的生成应用。
# 读取文档并创建索引
documents = SimpleDirectoryReader("docs/").load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# 定义 LangChain Prompt
template = PromptTemplate(input_variables=["query", "context"], template="Answer using context: {context}\nQuestion: {query}")
llm = VLLM(model="huggingface/gpt-j-6B")
chain = LLMChain(llm=llm, prompt=template)

# 查询与生成
query = "What is the capital of France?"
context = index.query(query).response
result = chain.run({"query": query, "context": context})
print(result)
```

---

## 4. 知识库动态更新与长期记忆设计
同时LLamaIndex还有一个非常牛的功能，就是可以实现知识库的动态更新和对话的长期记忆，这对于不固定的知识库和需要长期对话的用户可以说是一道照亮他们的光，没错，真神降临！

LlamaIndex支持以下功能：

+ **动态更新**：定期或实时添加新文档到索引
+ **长期记忆**：结合向量数据库和缓存策略，实现多轮任务记忆
+ **策略设计**：根据任务类型和用户偏好，动态调整检索结果和生成逻辑

#### 示例代码：动态添加文档到 LlamaIndex
```python
from llama_index import GPTVectorStoreIndex, Document
# 动态更新保证知识库及时生效，支撑长期对话和多轮任务。
new_doc = Document(text="Berlin is the capital of Germany.")
index.insert(new_doc)

# 查询新文档
response = index.query("What is the capital of Germany?")
print(response)
```

---

## 5. 多数据源整合与跨模态检索
同时，LlamaIndex实现的RAG 系统可支持文本、表格、PDF、图片等多数据源，并统一向量化处理，实现跨模态检索，能够满足绝大部分场景的使用需求。

#### 示例代码：文本 + PDF 集成（伪示例）
```python
from llama_index.readers import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
# 多数据源整合保证模型能够获取更全面的知识，实现跨模态增强生成。
# 读取文本和 PDF
text_docs = SimpleDirectoryReader("text_docs/").load_data()
pdf_docs = SimpleDirectoryReader("pdf_docs/").load_data()

# 合并并创建索引
all_docs = text_docs + pdf_docs
index = GPTVectorStoreIndex.from_documents(all_docs)

# 查询
response = index.query("Explain AI concepts in the PDFs and texts.")
print(response)
```

---

最后，我们回答文章开头的问题

1. **LlamaIndex 和 Haystack 的核心架构和使用方法是什么？**  
LlamaIndex 提供向量索引和文档管理；Haystack 提供检索 + 生成的 RAG 管道，支持多模型组合和多轮对话。
2. **多框架集成实践如何实现？**  
LangChain 负责任务编排，LlamaIndex 提供知识检索，vLLM 提供高吞吐量推理，实现高性能知识增强生成。
3. **知识库动态更新、长期记忆设计和多数据源整合有哪些最佳实践？**  
通过动态插入文档、向量化存储、多数据源整合和缓存策略，实现多轮任务记忆和跨模态检索，保证系统灵活、高效和可扩展。

---

关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号 **coting**！

以上内容参考 LlamaIndex、Haystack 和 LangChain 官方文档及社区资料，如有侵权请联系删除。



