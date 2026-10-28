# AI Agent 开发实习学习路线

更新：2026-10-28。目标：粤港澳地区 AI Agent 应用开发实习；2027 年 2 月集中投递并争取尽快到岗，按每天至少 4 小时、每周约 28 小时安排，可根据课业弹性调整。Python 和深度学习已有基础，重点加强独立编程与后端基础。投递与学习并行，看到合适岗位可提前申请。

## 岗位能力与方向

优先定位 Agent 应用开发 / LLM 应用研发 / AI 应用工程实习，同时关注大模型算法实习。依据深圳商汤 Agent 实习、深圳平安科技大模型应用实习和香港 CAIR 的岗位样本，优先级依次是：Python 与软件工程；LLM API、结构化输出与工具调用；RAG、评测；FastAPI、数据库和部署；框架与工作流。算法岗可能另需 PyTorch、SFT、推理部署或多模态，按具体岗位补充。这些样本数量有限，不代表所有企业，也不保证岗位仍在招。

- 商汤深圳，大模型算法 agent 实习生（算法倾向）：https://sensetime.jobs.feishu.cn/edu/m/position/7660462461628483866/detail
- 平安科技深圳，大模型应用算法实习生（第三方平台）：https://www.shixiseng.com/intern/inn_wmcezk0oinhe?pcm=pc_SearchList
- 香港 CAIR，AI Agent Engineer / Intern（医疗与模型部署倾向）：https://cair-cas.org.hk/article/603
- 腾讯音乐 AI 全栈工程师（深圳/广州，校招而非实习，仅供能力参照）：https://join.tencentmusic.com/campus/post-details/?id=15037&refer_code=

## 技术主线

Python + Git + HTTP/JSON + pytest → 模型 API 和提示词 → 手写工具调用循环 → RAG 与评测 → LangGraph 工作流 → FastAPI、数据库、Docker、可观测性 → 作品集、简历与面试。先掌握原理和一个框架，再按岗位扩展。模型 API 与部署选择遵循可访问性和预算，不要求购买高价服务。

## 14 周计划

| 周次 | 学习与实践 | 验收成果 |
| --- | --- | --- |
| 1 | Python、虚拟环境、Git、HTTP/JSON 与异常处理；Agent 概念 | 独立完成 JSON 任务统计脚本；记录问题、运行方式和测试 |
| 2 | HTTP、FastAPI 入门；LLM 消息、结构化输出与环境变量 | 实现一个基础接口和命令行助手；无 API 密钥时使用 mock |
| 3 | Function calling、工具参数校验、执行—观察循环 | 手写单 Agent，至少 2 个工具，限制次数并处理失败 |
| 4 | 阅读官方 Agent 入门与 LangGraph，状态、分支、持久化 | 用框架重做一个工作流，解释和手写版区别 |
| 5 | RAG 文档解析、分块、检索、引用 | 基础文档问答，回答显示证据；无证据时拒答 |
| 6 | 稀疏/向量检索、重排、检索评测 | 建 30 条人工标注问题；对比简单基线和改进方案 |
| 7 | FastAPI、Pydantic、pytest、SQL 基础 | 封装问答 API；覆盖成功、无证据和故障场景 |
| 8 | 工作流、工具路由、超时、日志与持久化 | 旗舰项目 MVP，能完成一个真实多步骤任务 |
| 9 | 确认机制、重试、工具权限、提示注入基本防护 | 对外部内容和工具调用设置边界，写失败案例 |
| 10 | 小规模评测、错误分类、成本和延迟 | 用 30–50 条样本生成报告，比较基线并记录真实结果 |
| 11 | Docker、配置、复现文档、简单展示 | 项目可在新环境按 README 启动，提供演示素材 |
| 12 | 项目迭代、开源阅读、简历 | 两个项目整理成可解释的经历，完善 README |
| 13 | 算法题、Python、数据库、网络、Agent 高频面试 | 完成模拟面试；解释自己的技术选择和失败案例 |
| 14 | 岗位投递、面试复盘和缺口补齐 | 按岗位维护投递表，每周复盘并更新项目 |

贯穿全程：每周 3 次算法题练习、一次岗位搜集、一次复盘。时间不够时先保留 Python、单 Agent、RAG 与评测和一个完整项目，再扩展多 Agent 或微调。岗位开放时间不固定，别等 2 月才第一次搜岗位。

## 学习资源（按顺序）

1. 中文主线：[Datawhale Hello-Agents](https://datawhalechina.github.io/hello-agents/)，选读基础、工具、RAG、评测相关章节；[配套代码](https://github.com/datawhalechina/hello-agents)。
2. 入门视频和练习：[Hugging Face Agents Course Unit 1](https://huggingface.co/learn/agents-course/en/unit1/introduction)：工具、消息与 Think–Act–Observe。英文材料，可配合中文教材。
3. 第 3–4 周：[AI Agents in LangGraph](https://www.deeplearning.ai/alpha/short-courses/ai-agents-in-langgraph)，先手写再学习框架；API 以[当前 LangGraph 官方文档](https://docs.langchain.com/oss/python/learn)为准。注册、收费和访问条件以课程页面为准。
4. 第 5–6 周：[Building and Evaluating Advanced RAG](https://www.deeplearning.ai/courses/building-evaluating-advanced-rag)：检索及答案可信性评估。
5. 评测参考：[LangSmith 评测指南](https://docs.langchain.com/langsmith/evaluate-llm-application)。先用本地 JSON 样本和 pytest 即可。

## 项目路线

### 项目 1：手写工具调用 Agent

输入任务，让模型选择计算、查询本地数据等工具；完成参数校验、工具执行、观察结果和循环，设置调用上限，记录错误和日志。先用 mock 测试，之后接入真实模型。README 写清可运行示例、架构和限制。

### 项目 2：有证据的文档问答助手

使用公开且允许使用的材料，完成文档分块、检索、回答及来源标注；无证据拒答，封装 API。建立至少 30 条问题及证据标注，同一批样本比较无检索、基础检索和改进方案，记录引用正确率、检索 Recall@k、无答案拒答表现、延迟和成本。记录分母及计算方法，不预先宣称效果。

### 旗舰项目：招聘岗位研究助手

输入用户提供或获准使用的 JD，提取技术要求、检索本地学习笔记，生成证据可追溯的能力缺口和学习建议；支持多步骤工具调用、人工确认、状态持久化、故障恢复、FastAPI 与日志。先完成本地数据版，避免未经授权的自动抓取和自动投递。基于 30–50 个标注样本评估字段抽取、引用、任务完成、异常与越权行为，展示真实基线和失败案例。可以将项目 2 的检索与评测模块复用进旗舰项目。

开源参考：[LangGraph ReAct 示例](https://github.com/langchain-ai/react-agent)、[STORM](https://github.com/stanford-oval/storm)、[Dify](https://github.com/langgenius/dify)。用于学习架构与取舍；作品必须注明借鉴与许可证，能解释本人改动。

## 第一周：今天就开始

每天约 4 小时，按基础伸缩。无须先买 API。详细日程见 `week-01-python-foundations.md`。

1. 第 1 天（1–2 小时）：确认 Python、Git 可用，创建项目虚拟环境；阅读 Hello-Agents 的 Agent 基础。写下 Agent、工具、模型、工作流的区别。
2. 第 2–3 天（3–4 小时）：复习函数、列表/字典、类型标注、异常、文件读取和 JSON。编写 `task_stats.py`，读取本地 JSON 任务列表，统计总数、已完成数、按类别分组；缺失文件或字段时给出清晰错误。
3. 第 4 天（2 小时）：用 pytest 为正常输入、空列表、异常输入写测试；README 写运行与测试命令。
4. 第 5 天（1–2 小时）：学习 HTTP 的请求/响应和状态码，读 Hugging Face Unit 1 前半，画出文字版模型—工具—结果循环。
5. 第 6–7 天（2–4 小时）：尝试 2 道简单/中等算法题，复盘并提交代码；记录 3 个困惑，以及本周投入时间和完成情况。

验收：能独立运行脚本及测试，解释 JSON 数据如何流转，说明 Agent 工具调用为何需要观察结果。进度慢时先完成脚本和测试，不以视频观看时长充当成果。

## GitHub 使用与安全

仓库以 README、`notes/`、`projects/`、`evaluations/` 分层，学习过程记录真实变化。使用 `.env.example` 说明配置，不提交 `.env`、API key、身份信息、课程受限材料和公司数据。每个项目的 README 含问题、技术结构、运行、测试、评测、局限和参考来源。后续按用户确定的公开范围同步。

## 面试和投递

目标关键词：AI Agent 开发、LLM 应用开发、AI 应用研发、大模型应用、智能体工程、RAG、AI Engineer Intern；地点检索深圳、广州、东莞、香港。同步看企业招聘官网、学校就业中心和招聘平台，核实实习时长、在岗天数、毕业时间、资格和香港工作安排。按岗位要求准备 Python、算法、HTTP、SQL、并发及项目讲解；算法方向补充 Transformer、训练与推理、微调。建立岗位表记录链接、截止日期、状态和复盘。简历成果只写实际测得的数据。

