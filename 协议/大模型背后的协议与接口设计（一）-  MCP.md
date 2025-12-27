<font style="color:rgb(25, 27, 31);">这篇文章是本系列的第一篇。</font>

最近MCP讨论热度逐渐攀升，每天都有很多的MCP工具产生，MCP是AI产业的又一大变革，未来还会有更加丰富通过MCP实现的AI产品来解放人类的双手。那么，MCP到底是个什么呢？

下面我分三个部分介绍一下MCP

+ 什么是MCP
+ MCP的核心架构
+ MCP的工作流程

后续还会出MCP的**具体使用部分**

## 一、什么是MCP
MCP，即Model Context Protocal（模型上下文协议），[<font style="color:rgb(9, 64, 142);">Anthropic</font>](https://zhida.zhihu.com/search?content_id=254822599&content_type=Article&match_order=1&q=Anthropic&zhida_source=entity)<font style="color:rgb(9, 64, 142);">（也就是开发Claude的公司） </font>2024年11月25日发布于[Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) 一文中。

MCP是AI领域的一种开放协议，他标准化了应用程序如何为LLM提供上下文，就像`USB`提供了一种将各种外围设备比如键盘、鼠标连接在电脑上一样，MCP提供了将不同数据源和工具连接到AI模型的标准化方式。在没有MCP之前，我们平时在使用chatgpt等大模型的时候，如果数据来自我们电脑中的`网页`，`数据库`，`文件`等内部工具时，我们要通过复制或者截图等方式和大模型进行交互。MCP的出现则解决了这个问题，MCP可以作为一个中介，完成上面的工作，用户只需要发出命令即可，大大提高了工作效率。

想象这样一个场景：当你对AI说"我上周看的论文大概有哪些创新点"，它往往只能“动嘴”而不能“动手”，需要你亲手上传你读过的论文，才能得到AI的回答。MCP 的出现，正是为了让 AI 从“智能回答者”变成“智能执行者”，它不再只是回复操作步骤，而是直接调取你的文件系统，完成分类归档、生成摘要——这就是MCP带来的变革。

下面这张图就非常形象的解释了MCP的作用。

![](https://cdn.nlark.com/yuque/0/2025/gif/28454971/1748311995075-0b2ffdc3-e5ca-4cbe-9a10-56fc4c215c1d.gif)

## 二、MCP的核心架构
MCP模型区别于传统模式的最大不同就是MCP集中处理AI和其他的各种数据来源，不需要用户频繁进行交互，使得工作过程对用户更加透明。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1748312045934-fc9a0e44-5da0-4d8f-92b6-71283f0b3bd6.png)



MCP的核心遵循`客户端-服务器`架构，其中host主机可以连接多个服务器

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1748312300464-ebd104af-82b9-419a-9e25-520f03f7f38b.png)

+ `MCP Host`：希望通过MCP协议访问数据的ChatGPT、Claude等AI工具
+ `MCP Client`：与服务器保持一对一连接的客户端
+ `MCP Server`：轻量级的程序，每个程序都通过标准化的Model Context Protocol公开特定功能
+ `Local Data Source`：MCP服务器可以安全访问的计算机文件、数据库和服务
+ `Remote Service`：MCP服务器可以连接到的Internet上的可用的外部系统

## 三、MCP的工作流程
MCP的工作流程类似HTTP，但是其中还是有些许不同，下面的图片非常形象的描述了MCP的工作流程

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1748312829819-be4e33c8-630a-4349-884f-ebb25e1954b3.png)



可以看到上述步骤中Agent（即客户端）提出请求后，MCP之后的所有步骤对用户都是透明的，最后得到AI工具的回答。

上述内容部分参考[https://cloud.tencent.com/developer/article/2517102](https://cloud.tencent.com/developer/article/2517102)和[https://blog.csdn.net/atbigapp/article/details/146173905](https://blog.csdn.net/atbigapp/article/details/146173905)，非常感谢，如有侵权请联系删除.

