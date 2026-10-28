# Agent 开发学习资源

资料整理日期：2026-10-28。先按阶段完成一个主资源及对应练习，再用其他资源查漏补缺；不用一次学完所有课程。

本清单根据资源维护者的公开 GitHub README 和课程介绍整理。当前环境可以读取这些 GitHub 材料，但没有直接验证视频平台的播放、地区可用性或注册要求。外部课程和 API 更新较快，请以官方最新文档为准；调用模型 API 前确认费用和额度，密钥只放在安全的环境配置中，不上传仓库。

## 1. Python 和开发基础：按薄弱点补齐

- [Python-100-Days](https://github.com/jackfrued/Python-100-Days)（中文文字、代码练习；作者的[部分配套视频](https://space.bilibili.com/1177252794)）。已有计算机基础可挑选数据结构、函数、文件、异常、类和模块练习；视频链接是作者主页，并非指定课程单集。
- [聪明办法学 Python](https://github.com/datawhalechina/learn-python-the-smart-way)（中文 Notebook 教程与实例）。作为第二参考，完成练习并尝试解释代码；基础足够时可以跳过重复章节。

建议产出：一个能读取 JSON、处理错误并通过单元测试的小工具，以及 Git 提交记录。

## 2. 大模型 API 和提示：开始写应用

- [吴恩达 × OpenAI Prompt Engineering 中文双语视频](https://www.bilibili.com/video/BV1Bo4y1A7FU/)（视频；链接来自 [Datawhale LLM Cookbook](https://github.com/datawhalechina/llm-cookbook) 的 README）。配合教材选择提示、总结和结构化输出示例，先理解输入输出，再亲自写代码。视频平台访问未在此环境验证。
- [Datawhale 面向开发者的大模型手册](https://github.com/datawhalechina/llm-cookbook)（中文文字与 Notebook，复现和翻译系列课程）。先学 Prompt Engineering 和构建问答系统；课程示例的 SDK、模型名称和框架可能过时，代码以所用提供商的最新文档为准。

建议产出：使用环境变量配置密钥的小型命令行应用，记录正常响应、超时或限流等失败的处理，并确保不会提交密钥。模型服务通常需账号和可能产生费用；也可先用模拟响应练习接口与测试。

## 3. Agent 基础：理解工具调用和循环

- [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents)（中文开源教程、代码与案例）。先读第 1、3、4 章，理解模型、ReAct、工具、观察及停止条件；再按项目需要阅读第 8 章记忆与检索、第 12 章评估。第 4 章用手写实现理解流程，再使用框架。
- [Hello-Agents 视频课程导航](https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra13-Hello-Agents视频课录制共创.md)（中文视频列表）。选择 ReAct 和 LangGraph 等对应视频；部分条目标为待发布，具体平台和视频是否可播放未在当前环境验证。
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)（英文课程；[课程仓库](https://github.com/huggingface/agents-course)）。已有 Python 和 LLM 基础后学 Unit 1，理解 agent、工具和 Think–Act–Observe；再参考 Unit 2 的框架内容。课程说明列有 Agentic RAG 和可观测性/评估章节，作为后续补充；可能需要账号或外部模型服务，具体以课程当前说明为准。

建议产出：先实现一个限制工具白名单、调用次数和错误处理的简单 Agent，写出工具调用过程和失败案例。不要直接把框架示例照搬成项目。

## 4. 检索增强生成与评估：构建可检验的项目

- [Datawhale 动手学大模型应用开发（LLM Universe）](https://github.com/datawhalechina/llm-universe)（中文文字与 Notebook）。重点学习文档处理、分块、向量检索、问答和验证迭代；其示例及依赖可能较旧，用它理解流程，并对照当前文档实现。
- [Hugging Face Agents Course：Agentic RAG](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction) 和 [Observability and Evaluation](https://huggingface.co/learn/agents-course/bonus-unit2/introduction)（英文课程）。用来理解如何测量检索和生成，而不只凭单次演示判断系统质量。

建议产出：给知识库问答准备带来源的样本和独立测试集，记录召回、回答正确性、无答案处理、耗时及费用；对比基线并保存失败案例。指标要说明样本量和计算方式，不虚构效果。

## 5. 模型原理：进阶选修

- [Andrej Karpathy：从零构建 GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（英文视频；[课程仓库与代码](https://github.com/karpathy/nn-zero-to-hero)）。适合希望理解自回归语言模型、Transformer 和训练流程时选看；课程仓库建议先具备张量及 PyTorch 基础。视频播放可用性未在当前环境验证。对于以应用开发实习为目标的入门阶段，它不需要排在项目实践之前。

## 使用原则

1. 每次学习后留下能复现的代码、少量笔记和一个自己的例子；不要复制大段课程内容或提交数据、密钥。
2. 每个外部资源保留来源链接并遵守许可证；项目 README 写明自己的贡献。
3. 比较课程和当前官方文档的版本；旧教程的框架 API、模型名称与价格可能已经变化。
4. 课程完成和证书不等同于岗位能力；优先完成可运行、可测试、能解释设计与取舍的作品。

## 6. 开源项目：作为设计参考

- [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch)：分步 Notebook 与配套视频，适合学习检索、索引和生成；先从简单检索基线做起。
- [smolagents](https://github.com/huggingface/smolagents)：阅读工具调用与 Agent 循环实现，理解框架如何控制工具执行；代码执行需要安全隔离。
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research)：多步骤研究 Agent 的进阶架构参考，依赖模型及可能的搜索服务；不要求入门阶段完整部署。
- [Ragas](https://github.com/vibrantlabsai/ragas)：评估工具参考；先理解指标及人工标注，再决定是否引入框架。

这些仓库已核验其公开介绍，未在当前环境运行；请检查各自许可证和依赖。阅读和引用参考项目不等于自己实现了这些项目。
