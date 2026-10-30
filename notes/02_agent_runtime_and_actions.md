# Agent 开发学习笔记 Day 2：工作机制与 Action

> 学习日期：2026-10-31  
> 主题：System Prompt、ReAct、Tool Calling、Runtime、Action  
> 前置笔记：[Day 1：Agent 基础、消息格式、Tool 与 MCP](./01_agent_fundamentals_and_tools.md)

## 0. 与 Day 1 的衔接

Day 1 解决“**模型看到了什么**”：用户输入如何变成 `messages`，工具函数如何经 `@tool` 包装成工具，以及 MCP 如何提供外部能力。Day 2 解决“**模型提出调用后，程序怎样真正执行**”。

```text
Day 1：用户输入 → messages / 工具描述 → LLM
Day 2：LLM 提出 Tool Call → Runtime 校验并执行 → 工具结果写回 messages → LLM 继续
```

这条链路把“一个会回答问题的模型”连接成“一个能使用外部能力完成任务的系统”。

## 1. Agent 的组成

可以用工作模型概括为：

```text
Agent ≈ LLM + 工具 + 指令 + Runtime + 状态
```

- **LLM**：依据当前上下文选择下一步或生成回答。
- **工具**：检索、计算、调用 API、读写受控资源。
- **指令**：规定任务目标、行为约束和可用工具；常以系统消息等形式提供。
- **Runtime**：应用程序中的控制代码，负责调用模型、处理工具调用、执行工具和终止循环。
- **状态**：至少包括消息历史；实际系统还可能保存任务进度、工具输出及错误信息。

这是便于理解的架构式，不是所有框架都统一采用的正式定义。

## 2. System Prompt 与 ReAct

System Prompt 可以描述角色、工具使用规则、输出约束等，但它本身**不会执行 Python 函数**。工具是否可用、参数如何校验、调用是否成功，由 Runtime 和工具实现决定。不同模型 API 对工具的描述可能通过独立的 `tools` 参数传入，不能假定都拼在 System Prompt 中。

**CoT** 指分步骤推理；**ReAct** 强调在推理和外部行动之间交替。典型循环是：

```text
Thought（判断下一步）→ Action（发起工具调用）→ Observation（读取结果）→ 再判断
```

“Thought”是解释工作流程的概念，不意味着应用必须获得或展示模型的完整内部推理文本。可观察、可记录的重点是动作、参数、工具结果和最终回答。

## 3. 从 Tool 到 Tool Call

Day 1 已经用 `@tool` 和 `inspect.signature()` 说明：函数名、描述、参数等信息可以被整理成工具元数据。接下来要区分三个层次：

| 层次 | 含义 | 例子 |
| --- | --- | --- |
| Tool Schema | 向模型描述工具名称、用途和参数 | `get_weather(location: str)` |
| Tool Call | 模型提出要调用的工具及参数 | `get_weather(location="深圳")` |
| Tool Execution | Runtime 调用实际实现并取得结果 | Python 函数查询天气 API |

模型输出的结构由具体 API 决定。下面仅是**概念示例**，不是通用接口格式：

```json
{
  "name": "get_weather",
  "arguments": {"location": "深圳"}
}
```

Day 1 的 `Tool` 对象负责封装元数据及可调用函数；本节进一步说明：还需要 Runtime 把“调用意图”映射到该函数。装饰器的具体行为依框架而异，`@tool` 本身不保证工具已在当前模型请求中注册。

## 4. Runtime：谁在执行循环？

一个最小的 Runtime 可以按以下顺序工作：

1. 把用户输入加入 `messages`，准备可用工具描述。
2. 调用 LLM API。
3. 若模型返回工具调用，记录助手的工具调用消息。
4. 按名称查找允许调用的工具，校验和解析参数。
5. 执行工具，并把成功结果或错误作为工具结果消息写回 `messages`；通常要关联对应的调用 ID。
6. 再次调用 LLM；模型可能继续调用工具，也可能给出最终回答。
7. 达到最终回答、步数限制或其他停止条件时结束。

```python
# 教学伪代码：不同模型 API 的字段名和消息格式不相同
for step in range(MAX_STEPS):
    response = call_llm(messages=messages, tools=tool_schemas)

    if not response.tool_calls:
        return response.content

    messages.append(response.assistant_message)

    for call in response.tool_calls:
        try:
            tool = allowed_tools[call.name]
            arguments = validate_arguments(tool, call.arguments)
            observation = tool(**arguments)
        except Exception as exc:
            observation = {"error": str(exc)}

        messages.append(make_tool_message(call.id, observation))

raise RuntimeError("达到最大执行步数")
```

这里的 `call_llm`、`validate_arguments` 和 `make_tool_message` 是说明职责的占位函数，不是可直接运行的完整实现。实际项目还需要超时、权限、日志、费用和敏感信息处理。

## 5. Action：Agent 如何与环境交互

Action 是一次具体操作，如搜索网页、查询数据、运行受控代码或调用设备接口。Hugging Face Agents Course 用三种方式解释动作的表示和执行：

| 类型 | 动作如何表示 | 执行者 |
| --- | --- | --- |
| JSON Agent | 模型按约定输出 JSON 动作 | 外部解析器和 Runtime |
| Code Agent | 模型输出代码片段 | 受控代码执行环境 |
| Function-calling Agent | 模型使用专门的工具调用接口输出结构化调用 | 应用 Runtime |

课程把 Function-calling Agent 归为 JSON Agent 的子类；实践中两者都涉及结构化动作，但**原生工具调用接口的消息格式由提供商定义**，不能把所有结果当成同一种 JSON 文本解析。Code Agent 的灵活性来自代码组合能力，执行生成代码时需要隔离环境、权限和资源限制。

## 6. Stop and Parse：何时把控制权交给 Runtime？

课程描述的 **Stop and Parse** 可以理解为：

```text
生成完整动作 → 结束当前生成 / 交出控制权 → 解析动作 → 执行 → 返回 Observation
```

对于文本协议，解析器需要识别动作边界并提取字段；对于原生 Function Calling，SDK 往往已把工具调用整理成结构化字段，但 Runtime 仍需校验和执行。**Stop 不等于遇到 JSON 的右大括号就停止**，实际结束方式取决于模型接口、完成状态或协议。

这也连接 Day 1 的“消息结束、助手轮次结束、工具交接”：**工具交接意味着模型请求 Runtime 做事，不等于任务已经完成**。

## 7. 一次完整的天气查询

假设用户问：“深圳现在天气如何？”

```text
① user message：深圳现在天气如何？
② LLM：提出 get_weather(location="深圳")
③ Runtime：校验参数并调用天气工具
④ Tool message：{"temperature": 25, "condition": "晴"}
⑤ LLM：根据工具结果回答“深圳现在 25°C，晴。”
```

第④步就是 **Observation**。它需要与原问题及对应的 Tool Call 一起进入下一次模型调用的上下文。若工具报错，Runtime 也可以把受控错误信息反馈给模型，让模型决定重试、换方法或向用户说明失败。

## 8. 今日结论与下一步

1. **模型提出动作，Runtime 执行动作。**
2. **Tool Schema、Tool Call、Tool Execution 是三个不同环节。**
3. **Agent 的关键闭环是“决策 → 行动 → 反馈 → 再决策”**，而不只是一次模型回答。
4. **终止条件、参数校验和错误处理**是让这个闭环可靠运行的必要部分。

下一步学习 **Observation**：亲手构造一条工具结果消息，观察它如何改变下一次模型调用的输入；再处理一次工具失败和一次多步调用。可将 Day 1 的 `@tool` 示例接入一个最小 Runtime，记录每轮的动作、结果与停止原因。

## 学习来源

- [Hugging Face Agents Course：Thought–Action–Observation](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure)
- [Hugging Face Agents Course：Thought 与 ReAct](https://huggingface.co/learn/agents-course/en/unit1/thoughts)
- [Hugging Face Agents Course：Actions](https://huggingface.co/learn/agents-course/en/unit1/actions)
- [Day 1 笔记](./01_agent_fundamentals_and_tools.md)
