在日常开发中，我们总是希望找到一个既轻量又智能的编程助手。Cursor，作为一款基于 VS Code 打造、深度集成 AI 的现代代码编辑器，正逐渐成为程序员提升效率的新宠。相比传统 IDE，它不仅支持 Copilot 类似的代码补全，还引入了更强大的 AI 对话功能，帮助你快速定位 bug、重构代码、甚至理解复杂项目逻辑。

<font style="color:rgb(25, 27, 31);">这篇博客将带你从零开始，手把手完成 Cursor 的安装与环境配置，并演示如何高效使用它进行日常开发。无论你是前端、后端，还是 AI 算法工程师，相信都能在 Cursor 中找到属于你的提效方式。</font>

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">算法coting</font><font style="color:rgb(25, 27, 31);">！</font>

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

## <font style="color:rgb(25, 27, 31);">安</font>
## 怎么安装
<font style="color:rgb(25, 27, 31);">首先点击右侧链接跳转 Cursor 官方网站下载 Cursor 编辑器：</font>[cursor.com](https://link.zhihu.com/?target=https%3A//www.cursor.com/cn/downloads)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753622596002-e6021644-eba5-4755-b5e9-0bd563f64459.png)

cursor提供免费版，专业版和卓越版，个人开发者免费版就够用了，点击下载按钮。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753622646493-5b53b757-7dfc-4a26-bffd-075fe54afade.png)

<font style="color:rgb(25, 27, 31);">下载好后点击运行进行安装，建议选在非系统盘中。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753623542195-a230d026-f96a-41b2-8fac-0d597c721db6.png)

接下来无脑下一步直到安装完成，运行cursor。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753623614336-52dbca4b-1118-4431-8d93-f0c47604f23c.png)

## 配置中文
<font style="color:rgb(25, 27, 31);">如果你想要中文界面，File-preference-extension，搜索Chinese，安装如下图中文插件，重启cursor就可以运行中文界面啦。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753622753697-1658bfe8-6d43-4782-8fb4-af2542dfbc8f.png)

## 工作面板介绍
下面我们来看一下cursor的主要构造，首先点击右上角的三个框框把所有的区域展示出来，包括下图所示的项目代码区域，工作区域，AI工作区域和控制台区域。项目代码区域主要是浏览项目架构的地方，工作区域用来编写代码，AI工作区域是我们使用AI工具辅助编码的地方，控制台是我们运行命令的地方。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753623032367-b22f658b-2bef-44cd-b7f5-430af0744670.png)

## 实战讲解
比如现在我们想要写一个个人博客网站，可以给出指令，AI就会分析并给出项目架构。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753625153015-841b5d21-4c25-4648-97e6-ecbd69e2434c.png)

然后跟他说根据项目架构编码实现，他会自动创建文件，自动编写demo程序，当然不会一次编完，后续需要我们不断根据需求和大模型交互。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753625122157-403d271f-93cf-41c3-8e27-f1124d2f441f.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753625193412-1230ef59-b006-45bc-a02d-2dc83047c02a.png)

运行之后如果出现问题，在AI工作区域可以直接引用控制台区域的报错。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633028340-0efb5e2b-1636-4381-9098-f97c2764004b.png)



当大模型建议要对某个文件进行修改时，我们可以直接点击文件名跳转到对应文件，然后点击apply按钮应用修改，这个真的很方便，不用自己找文件在哪里了。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633272566-6a32320e-ad3f-4469-b164-6e1cb7926e38.png)

当AI自动修改完毕之后，我们需要点击accept按钮接受更改。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633148996-49e6084b-73e1-468c-876c-a333dbc44303.png)

最后，我们的博客就可以成功运行啦！！！是不是很快，这个过程不到十分钟。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753632953432-1be86ad4-02c6-423c-840d-c552b8e62549.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753632978265-ddefe73c-d73f-4b25-969a-6265f9ba4f2d.png)



![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633001975-d82d52ac-07b2-4e6b-aa54-09ba34674355.png)

我觉得界面有点简陋，所以让大模型帮我美化了一下前端界面。



![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633335548-69d7f576-7a8f-4d80-ae19-b3291add6d04.png)

美化后的结果如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753633370883-8ea8aad2-3fa1-4508-b1b3-1ed1a7c7d7d5.png)



相信你已经成功完成了 Cursor 的安装与初步上手啦。从 AI 辅助编程、代码重构，到对话式调试，Cursor 正在用更智能的方式革新我们的开发体验。

当然，工具只是辅助，真正提升效率的核心仍然是我们对技术的理解与实践。希望你在使用 Cursor 的过程中，不仅能提高开发效率，也能激发更多对代码和 AI 的思考。未来的开发，不只是写代码，更是与 AI 携手共创。

