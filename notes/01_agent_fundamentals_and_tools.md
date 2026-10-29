# AI Agent 学习笔记 01：基础概念、Chat Template、Tool 与 MCP

> 学习来源：Hugging Face Agents Course — Unit 1、Datawhale Easy-Vibe   
> 笔记中记录的学习日期：2026-10-29；整理上传：2026-10-30   
> 学习主题：Agent 基础、消息格式、特殊 Token、工具调用、Python 装饰器、MCP

---

## 1. 什么是 AI Agent？

### 1.1 基本定义

AI Agent 是一种能够进行推理（Reasoning）、规划（Planning）并与环境交互（Interacting with its environment）的 AI 系统。

三个核心能力：

- **Reasoning（推理）**：理解任务、分析问题、判断下一步行动。
- **Planning（规划）**：制定完成任务所需的步骤。
- **Interacting with the environment（与环境交互）**：通过工具、API 或其他接口获取信息、执行操作。

例如，开发一个天气查询 Agent：

1. 用户要求查询明天深圳的天气。
2. LLM 理解需求，判断需要获取实时天气信息。
3. Agent 调用天气查询工具。
4. 工具返回天气数据。
5. LLM 根据结果生成自然语言回答。

需要注意：LLM 是 Agent 的重要组成部分，但 Agent 不等于 LLM。Agent 还需要工具接口、执行逻辑和控制流程等组件。

---

## 2. LLM 如何处理对话？

### 2.1 从聊天界面到模型输入

用户看到的是一条条独立的聊天消息，但在底层，聊天系统通常需要将多条消息组织成模型能够处理的输入序列。

例如，应用程序可以维护以下消息列表：

```
messages = [
    {
        "role": "system",
        "content": "你是一名 AI 助教。"
    },
    {
        "role": "user",
        "content": "什么是 AI Agent？"
    },
    {
        "role": "assistant",
        "content": "AI Agent 是能够使用工具完成任务的智能系统。"
    },
    {
        "role": "user",
        "content": "举一个例子。"
    }
]
```

这里的 `messages` 是一个 Python 列表，每个元素代表一条消息。

常见字段：

- `role`：表示消息的角色。
- `content`：表示消息的内容。

常见角色包括：

| Role        | 含义                     |
| ----------- | ---------------------- |
| `system`    | 系统指令，规定模型的行为和任务要求      |
| `user`      | 用户消息                   |
| `assistant` | 助手消息                   |
| `tool`      | 某些聊天协议中用于表示工具返回结果的消息角色 |

不同 API 或协议对消息结构的要求可能不同。

### 2.2 普通文本如何转换成 message？

假设用户输入：

```
user_input = "帮我查询明天深圳的天气。"
```

应用程序可以将它包装成一条消息：

```
message = {
    "role": "user",
    "content": user_input
}
```

这个过程本质上是普通的 Python 数据处理，并不需要 LLM 参与。

当用户发送新消息时，应用程序或 Agent 框架可以将消息追加到历史列表：

```
messages.append({
    "role": "user",
    "content": user_input
})
```

实际应用还需要管理历史消息、避免重复添加，并在必要时控制上下文长度。

### 2.3 什么是 Chat Template？

**Chat Template（聊天模板）是一套将结构化聊天消息转换成特定模型所需输入格式的规则。**

不同模型使用的消息格式和特殊 Token 可能不同。

例如，某些模型使用类似下面的格式：

```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

其中：

- `<|im_start|>`：开始一条消息。
- `user`：表示发言角色。
- `你好`：消息内容。
- `<|im_end|>`：结束当前消息。
- 最后的 `assistant` 前缀：提示模型接下来生成助手消息。

这些标记是格式示例，实际使用哪个标记取决于模型。

### 2.4 在 Transformers 中使用 Chat Template

```
from transformers import AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": "什么是 AI Agent？"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(prompt)
```

重要参数：

- `messages`：结构化的聊天记录。
- `tokenize=False`：返回格式化后的文本，而不是 Token ID。
- `add_generation_prompt=True`：添加适合模型开始生成助手消息的前缀，具体形式由模板决定。

也可以设置 `tokenize=True`，直接通过模板处理流程生成模型输入所需的 Token ID。

完整链路：

```
用户输入
    ↓
应用程序构造 message
    ↓
组合 messages
    ↓
Chat Template 格式化
    ↓
Tokenizer 转换为 Token ID
    ↓
LLM 生成输出
```

如果使用封装好的模型 API 或 Agent 框架，Chat Template 和 Tokenization 通常由底层服务或框架处理。

---

## 3. Special Tokens：特殊 Token 与结束标记

特殊 Token 用于帮助模型识别消息边界、角色信息或控制流程。

### 3.1 常见特殊 Token

| Token              | 常见作用                  |
| ------------------ | --------------------- |
| `<\|im_start\|>`   | 开始一条消息                |
| `<\|im_end\|>`     | 结束一条消息                |
| `<\|end\|>` | Harmony 格式中的消息结束标记 |
| `<\|fim_suffix\|>` | 在 Harmony 格式中表示助手轮次结束 |
| `<\|ghissue\|>`    | 在 Harmony 格式中表示控制权交接  |

注意：这些标记属于特定格式的示例，并非所有模型都使用它们。

### 3.2 消息结束、轮次结束与工具交接的区别

这三个概念尤其重要。

**消息结束（Message End）**

表示当前消息结束，但不一定代表整个助手轮次完成。

**助手轮次结束（Assistant Turn End）**

表示助手当前这一轮回答已经完成。

**控制权交接（Handoff）**

表示模型发出了需要交由工具或其他执行方处理的请求。

以 Harmony 格式为例，`<|end|>` 表示消息结束，`<|fim_suffix|>` 表示助手轮次结束，`<|ghissue|>` 表示控制权交接；`<|im_end|>` 是其他格式中的示例，不应混为同一协议。

不同模型和 API 的具体协议可能不同，不能假设所有模型都采用相同的标记。

### 3.3 Base Model 与 Instruct Model

**Base Model**

主要学习根据已有 Token 预测后续 Token。

**Instruct Model**

通常经过指令微调，使其更适合遵循指令、回答问题和参与对话。

Chat Template 可以按照特定格式组织输入，但它本身不会自动把 Base Model 训练成 Instruct Model。

---

## 4. 什么是 Tool？

### 4.1 基本概念

Tool（工具）是 Agent 可以使用的一项功能，通常由一个函数或其他可执行接口实现。

常见工具：

| Tool          | 功能              |
| ------------- | --------------- |
| Calculator    | 数学计算            |
| Web Search    | 搜索最新网络信息        |
| Retrieval     | 从外部知识库检索信息      |
| API Interface | 访问 GitHub 等外部服务 |
| File System   | 读取或操作文件         |

工具可以补充 LLM 的能力，例如精确计算、获取实时信息或执行外部操作。

### 4.2 一个简单的计算器工具

```
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

result = calculator(3, 4)

print(result)  # 12
```

这里：

- `calculator` 是函数名称。
- `a` 和 `b` 是输入参数。
- `int` 是参数的类型提示。
- `-> int` 是返回值类型提示。
- 文档字符串描述函数的用途。
- `return a * b` 是实际执行逻辑。

类型提示用于声明预期类型，并不会默认强制校验所有传入值。

### 4.3 工具需要哪些信息？

一个工具通常需要以下信息：

1. Name：工具名称。
2. Description：工具功能描述。
3. Arguments：输入参数及其类型。
4. Outputs：输出类型或输出结构。
5. Callable：实际执行操作的函数或接口。

工具描述告诉 LLM 如何选择和使用工具；Callable 才负责真正执行操作。

---

## 5. Python 装饰器 Decorator

### 5.1 什么是装饰器？

装饰器是一种用于扩展或改变函数、类等对象行为的 Python 机制。

常见用途：

- 日志记录
- 计时
- 权限检查
- 缓存
- 注册工具

装饰器经常用于在不直接修改原函数主体的情况下，为函数增加额外功能。

### 5.2 函数可以作为参数传递

```
def say_hello():
    print("你好！")

my_function = say_hello

my_function()
```

这里：

- `say_hello` 是函数对象。
- `say_hello()` 是调用函数。
- `my_function = say_hello` 让两个变量指向同一个函数对象。

理解这一点是理解装饰器的基础。

### 5.3 手动实现一个装饰器

```
def my_decorator(func):
    def wrapper():
        print("开始执行函数")
        func()
        print("函数执行结束")

    return wrapper
```

逐项理解：

- `func`：传入的原函数。
- `wrapper`：包装函数。
- `func()`：在包装函数中执行原函数。
- `return wrapper`：返回包装函数本身，而不是立即执行它。

使用时：

```
def say_hello():
    print("你好！")

say_hello = my_decorator(say_hello)

say_hello()
```

输出：

```
开始执行函数
你好！
函数执行结束
```

执行过程：

1. 将原函数传入 `my_decorator`。
2. 创建 `wrapper`。
3. 返回 `wrapper`。
4. 让 `say_hello` 指向返回的包装函数。
5. 调用 `say_hello()` 时执行包装逻辑。

### 5.4 `@decorator` 语法

```
@my_decorator
def say_hello():
    print("你好！")
```

这基本等价于：

```
def say_hello():
    print("你好！")

say_hello = my_decorator(say_hello)
```

因此：

```
@decorator
def func():
    ...
```

可以理解为：

```
def func():
    ...

func = decorator(func)
```

这是理解 `@tool` 的关键。

### 5.5 `@tool` 的作用

在课程的简化实现中：

```
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

等价于：

```
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

calculator = tool(calculator)
```

`tool` 装饰器会读取函数信息，并将函数包装成课程定义的 `Tool` 对象。

因此，装饰器返回的对象不一定仍然是原始函数。它可以是包装函数，也可以是工具对象，具体取决于装饰器的实现。

---

## 6. `inspect`：自动提取函数信息

在 Agent 工具系统中，我们希望自动读取函数名称、参数类型、返回类型和文档字符串，而不是手动填写每个字段。

Python 的 `inspect` 模块提供了读取函数信息的能力。

### 6.1 获取函数签名

```
import inspect

def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

signature = inspect.signature(calculator)

print(signature)
```

输出：

```
(a: int, b: int) -> int
```

`signature` 是一个 `Signature` 对象，其中包含函数参数及返回值注解等信息。

### 6.2 遍历输入参数

```
arguments = []

for param in signature.parameters.values():
    annotation_name = (
        param.annotation.__name__
        if hasattr(param.annotation, "__name__")
        else str(param.annotation)
    )

    arguments.append((param.name, annotation_name))
```

最终：

```
arguments = [
    ("a", "int"),
    ("b", "int")
]
```

解释：

- `signature.parameters.values()`：取得各个参数对象。
- `param.name`：参数名称。
- `param.annotation`：参数的类型注解。
- `hasattr(obj, "__name__")`：检查对象是否具有 `__name__` 属性。
- `append()`：将参数名称和类型加入列表。

这里使用字符串形式保存类型名称，方便后续生成文本描述。

### 6.3 获取返回值类型

```
return_annotation = signature.return_annotation

if return_annotation is inspect._empty:
    outputs = "No return annotation"
else:
    outputs = (
        return_annotation.__name__
        if hasattr(return_annotation, "__name__")
        else str(return_annotation)
    )
```

如果函数声明了：

```
def calculator(a: int, b: int) -> int:
    return a * b
```

那么 `outputs` 会被设置为 `"int"`。

如果函数没有声明返回类型，则会得到 `"No return annotation"`。

注意：这仅表示函数是否声明返回类型，并不代表函数实际有没有返回值。

### 6.4 提取名称和描述

```
name = func.__name__
description = func.__doc__ or "No description provided."
```

- `func.__name__`：获取函数名称。
- `func.__doc__`：获取函数文档字符串。
- `or`：在文档字符串不存在或为空时，使用默认描述。

因此，合理的函数名、类型提示和文档字符串可以帮助工具框架自动生成描述。

---

## 7. `Tool` 类：把工具信息组织成对象

课程中的简化工具类：

```
from typing import Callable

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        arguments: list,
        outputs: str
    ):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        args_str = ", ".join(
            f"{arg_name}: {arg_type}"
            for arg_name, arg_type in self.arguments
        )

        return (
            f"Tool Name: {self.name}, "
            f"Description: {self.description}, "
            f"Arguments: {args_str}, "
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
```

主要属性：

| 属性            | 作用     |
| ------------- | ------ |
| `name`        | 工具名称   |
| `description` | 工具说明   |
| `func`        | 实际执行函数 |
| `arguments`   | 输入参数信息 |
| `outputs`     | 输出类型说明 |

主要方法：

- `to_string()`：将工具元信息转换成文本。
- `__call__()`：使工具对象可以像函数一样被调用。

例如：

```
calculator_tool = Tool(
    name="calculator",
    description="Multiply two integers.",
    func=calculator,
    arguments=[("a", "int"), ("b", "int")],
    outputs="int"
)

print(calculator_tool.to_string())
print(calculator_tool(3, 4))
```

输出示例：

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
12
```

注意：这只是课程中的简化实现。生产环境中的工具系统通常还需要参数校验、异常处理、权限管理和输入输出 Schema 等机制。

### 7.1 装饰器如何创建 Tool 对象？

课程中的装饰器逻辑可以概括为：

```
import inspect

def tool(func):
    signature = inspect.signature(func)

    arguments = []

    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__
            if hasattr(param.annotation, "__name__")
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))

    return_annotation = signature.return_annotation

    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, "__name__")
            else str(return_annotation)
        )

    description = func.__doc__ or "No description provided."
    name = func.__name__

    return Tool(
        name=name,
        description=description,
        func=func,
        arguments=arguments,
        outputs=outputs
    )
```

执行：

```
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

在这份实现中，`calculator` 最终指向返回的 `Tool` 对象，而原始函数保存在对象的 `func` 属性中。

执行过程：

```
原始 Python 函数
    ↓
@tool
    ↓
inspect.signature()
    ↓
提取参数和返回类型
    ↓
提取函数名和文档字符串
    ↓
创建 Tool 对象
    ↓
返回 Tool 对象
```

需要特别注意：自动生成工具描述并不等于执行工具。`@tool` 负责包装函数和整理信息，真正的工具执行仍需通过函数调用完成。

---

## 8. Agent 如何调用工具？

LLM 本身主要负责根据输入生成输出。它不会因为看到了 Python 函数就自动在应用程序中执行该函数。

通常需要 Agent 程序协调整个流程。

### 8.1 工具调用流程

```
用户提出问题
    ↓
Agent 将问题和可用工具信息提供给 LLM
    ↓
LLM 判断是否需要调用工具
    ↓
LLM 输出工具调用请求及参数
    ↓
Agent 解析请求并执行对应函数
    ↓
工具返回结果
    ↓
Agent 将工具结果交给 LLM
    ↓
LLM 生成最终回答
```

例如，用户询问：

> 3 乘以 4 等于多少？

模型可能决定调用：

```
calculator(a=3, b=4)
```

Agent 程序执行函数后得到 `12`，再将结果交给 LLM，由 LLM 生成自然语言回答。

需要注意：上面的调用表示意。实际系统通常使用结构化工具调用协议，包含工具名称、参数和调用 ID 等信息。

### 8.2 为什么工具结果需要返回给 LLM？

工具负责执行具体操作并返回结果，而 LLM 可以结合用户的原始问题和工具结果生成更自然的回答。

例如：

```
{
  "result": 12
}
```

LLM 可以将其组织成：

> 3 乘以 4 等于 12。

在多步任务中，LLM 还可以根据工具返回结果继续决定下一步行动。

---

## 9. MCP：Model Context Protocol

### 9.1 什么是 MCP？

MCP 全称为 **Model Context Protocol（模型上下文协议）**。

它是一种标准化协议，用于让 AI 应用通过统一的通信方式连接外部工具、数据源和其他上下文能力。

MCP 的目的之一是减少不同 AI 应用与工具提供方之间的重复集成工作。

### 9.2 Tool 与 MCP 的区别

| 对比  | Tool               | MCP                               |
| --- | ------------------ | --------------------------------- |
| 本质  | 一个具体功能或工具接口        | 一套通信协议                            |
| 作用  | 执行某项操作             | 规范应用与工具服务之间的交互                    |
| 例子  | `calculator(a, b)` | 通过 MCP Client 调用 MCP Server 提供的工具 |
| 关注点 | 功能本身               | 工具发现、调用、结果交换等通信机制                 |

可以这样理解：

- Tool 解决“这个功能是什么、如何执行”。
- MCP 解决“AI 应用如何按照统一协议发现和使用外部能力”。

MCP 本身不负责具体业务计算，也不取代 Agent 的规划和决策逻辑。

### 9.3 MCP 的三个核心角色

**MCP Host**

宿主 AI 应用，负责协调模型、客户端以及用户交互。

**MCP Client**

通常位于 Host 内部，负责与 MCP Server 建立连接并交换协议消息。

**MCP Server**

向客户端提供工具、资源或提示模板，并处理相应请求。

一个 Host 可以管理多个 MCP Client，并连接不同的 MCP Server。

### 9.4 MCP 的常见能力

| 能力        | 作用         | 示例                    |
| --------- | ---------- | --------------------- |
| Tools     | 执行操作       | 查询 GitHub Issues、执行计算 |
| Resources | 提供上下文数据    | 读取项目文件、获取文档           |
| Prompts   | 提供可复用的提示模板 | 代码审查模板、研究总结模板         |

### 9.5 实际例子：通过 MCP 查询 GitHub

假设用户要求：

> 查询某个 GitHub 仓库最近的 Issues。

流程可能是：

1. Agent 理解用户需求。
2. MCP Client 与 GitHub MCP Server 通信，获取可用工具信息。
3. 应用将相关工具信息提供给 LLM。
4. LLM 决定调用适合的工具并生成参数。
5. 应用通过 MCP Client 发送调用请求。
6. MCP Server 执行对应操作并返回结果。
7. Agent 将结果交给 LLM，生成最终回答。

注意：MCP Server 可以运行在本地，也可以运行在远程环境。MCP 只规定通信协议，不要求所有工具都部署在远程服务器上。

### 9.6 Tool、装饰器与 MCP 的关系

这三个概念处于不同层次：

- `@tool`：Python 装饰器，用于包装函数或注册工具，具体行为由框架实现。
- `Tool`：具体工具对象或工具接口。
- MCP：连接 AI 应用与外部能力的标准通信协议。

它们可以配合使用，但不能互相等同。并非所有 MCP Server 都需要使用 Python，也并非所有工具都通过 MCP 暴露。

---

## 10. 本次学习的知识结构

```
AI Agent
│
├── LLM 与对话输入
│   ├── message
│   ├── messages
│   ├── Chat Template
│   ├── Tokenizer
│   └── Special Tokens
│       ├── Message End
│       ├── Assistant Turn End
│       └── Tool Handoff
│
├── Tools
│   ├── 函数实现
│   ├── 工具名称与描述
│   ├── 输入参数与输出类型
│   ├── Python 装饰器 @tool
│   ├── inspect.signature()
│   └── Tool 类
│
├── Tool Calling
│   ├── LLM 决定调用工具
│   ├── Agent 执行调用
│   ├── 工具返回结果
│   └── LLM 生成最终回答
│
└── MCP
    ├── Host
    ├── Client
    ├── Server
    ├── Tools
    ├── Resources
    └── Prompts
```

---

## 11. 学习检查清单

- 能够解释 AI Agent 与普通 LLM 的区别。
- 能够说明 `message` 与 `messages` 的区别。
- 能够解释 Chat Template 的作用。
- 能够区分消息结束、助手轮次结束和工具交接。
- 能够编写一个简单的 Python Tool。
- 能够解释 `@decorator` 与 `func = decorator(func)` 的关系。
- 能够解释 `inspect.signature()` 如何提取函数信息。
- 能够解释 `@tool` 如何将函数包装成工具对象。
- 能够描述 Agent 调用工具并将结果交回 LLM 的完整流程。
- 能够解释 Tool 与 MCP 的区别。
- 能够区分 MCP Host、Client 和 Server。

---

## 12. 下一步学习建议

建议按照以下顺序继续：

1. **巩固 Python 装饰器**：练习函数作为参数、闭包、`wrapper` 和 `return`。
2. **实践 Tool Calling**：编写计算器工具，让 Agent 完成“模型决策—程序执行—结果反馈”。
3. **学习 Agent Workflow**：理解 Thought、Action、Observation 的循环。
4. **学习 MCP**：搭建一个简单的 MCP Server，并使用兼容的 MCP Client 调用工具。
5. **完成一个小项目**：构建一个可以查询 GitHub 仓库信息的 Agent。

参考资料：

- [Datawhale Easy-Vibe：AI Agent 原理与工具调用](https://datawhalechina.github.io/easy-vibe/zh-cn/appendix/8-artificial-intelligence/ai-agents.html)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit1/introduction)
- [Messages and Special Tokens](https://huggingface.co/learn/agents-course/en/unit1/messages-and-special-tokens)
- [What are Tools?](https://huggingface.co/learn/agents-course/en/unit1/tools)
- [Understanding AI Agents through the Thought-Action-Observation Cycle](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure)
- [Hugging Face MCP Course](https://huggingface.co/learn/mcp-course/)

