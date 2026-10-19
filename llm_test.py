import usage_tracker
import config
import json
from pathlib import Path

from openai import OpenAI

# 1. 加载配置
import config

config.validate_config()

api_key = config.API_KEY
base_url = config.BASE_URL
model = config.MODEL

if not api_key:
    raise ValueError("没有读取到 SILICONFLOW_API_KEY，请检查 .env 文件。")

if not base_url:
    raise ValueError("没有读取到 SILICONFLOW_BASE_URL。")

if not model:
    raise ValueError("没有读取到 SILICONFLOW_MODEL。")


# 2. 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
    max_retries=1,
)


# 3. 设置系统提示词和对话历史
SYSTEM_PROMPT = """
你是一位耐心、严谨的 Python 编程老师，学生是初学者。
回答问题时要结合完整的对话历史。
当用户使用指代词时，优先参考之前的对话。
回答应准确、简洁，优先控制在 200 到 400 个汉字。
先解释核心概念，再给一个简单的代码例子。
只有用户要求详细讲解时，才展开解释。
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


# 统计文件放在 llm_test.py 同一个目录
USAGE_FILE = Path(__file__).with_name("token_usage.json")

# 本次运行的 Token 统计
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# 统计当前程序运行期间的 Token 用量
def load_usage():
    """读取历史累计用量。"""
    default_usage = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }

    if not USAGE_FILE.exists():
        return default_usage

    try:
        with USAGE_FILE.open("r", encoding="utf-8") as file:
            saved_usage = json.load(file)

        # 检查读取到的数据是否包含所需字段
        for key in default_usage:
            value = saved_usage.get(key)

            if type(value) is not int or value < 0:
                raise ValueError(f"统计文件中的 {key} 不合法")

        return saved_usage

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"读取历史统计失败：{error}")
        print("本次运行将从零开始累计历史用量。")
        return default_usage


def save_usage(usage_data):
    """保存历史累计用量。"""
    with USAGE_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            usage_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

# 程序启动时加载历史数据
lifetime_usage = load_usage()

def trim_history():
    """保留系统提示词和最近 5 轮完整对话。"""

    # 收集完整的问答轮次
    complete_turns = []
    pending_user = None

    for message in messages[1:]:
        if message["role"] == "user":
            pending_user = message

        elif message["role"] == "assistant" and pending_user is not None:
            complete_turns.append((pending_user, message))
            pending_user = None

    # 只保留最近 5 轮
    recent_turns = complete_turns[-5:]

    # 重建历史，始终保留 system prompt
    messages[:] = [messages[0]]

    for user_message, assistant_message in recent_turns:
        messages.append(user_message)
        messages.append(assistant_message)


# 4. 定义流式提问函数
def ask_ai(question):
    messages.append({
        "role": "user",
        "content": question,
    })

    try:
        # stream=True 表示启用流式输出
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            stream=True,
            stream_options={"include_usage": True},
        )

        print("\nAI：", end="", flush=True)

        # 用来保存完整回答
        answer_parts = []

        # 逐段接收并显示模型输出
        
        usage = None

        for chunk in stream:
            # 先读取用量，因为最后一个统计片段可能没有 choices
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)

        # 将所有片段拼成完整回答
        answer = "".join(answer_parts)

        print()  # 回答结束后换行

        if usage is not None:
            usage_tracker.record_usage(usage)

            # 更新跨运行的历史累计用量
            lifetime_usage["total_prompt_tokens"] += usage.prompt_tokens
            lifetime_usage["total_completion_tokens"] += usage.completion_tokens
            lifetime_usage["total_tokens"] += usage.total_tokens

            try:
                save_usage(lifetime_usage)
                print("历史 Token 统计已保存。")
            except OSError as error:
                print(f"保存历史统计失败：{error}")

            print(
                f"本次 Token：输入 {usage.prompt_tokens}，"
                f"输出 {usage.completion_tokens}，"
                f"合计 {usage.total_tokens}"
            )

            print(
                f"累计 Token：输入 {total_prompt_tokens}，"
                f"输出 {total_completion_tokens}，"
                f"合计 {total_tokens}"
            )
        else:
            print("本次响应没有返回 Token 用量统计。")

        if not answer.strip():
            print("模型没有返回有效文本，请重试。")
            messages.pop()
            return

        # 保存完整回答，供下一轮对话使用
        messages.append({
            "role": "assistant",
            "content": answer,
        })

        # 保留系统提示词和最近 10 条对话消息
        trim_history()

    except Exception as error:
        # 请求失败时，移除本轮尚未配对的用户消息
        if messages and messages[-1]["role"] == "user":
            messages.pop()

        print(f"\n调用失败：{type(error).__name__}: {error}")


# 5. 主程序
def main():
    print("AI Python 助手已启动（流式输出模式）。")
    print("输入 exit、quit 或 退出，可以结束程序。")

    while True:
        try:
            question = input("\n你：").strip()

            if question.lower() in ("exit", "quit", "退出"):
                print("\n本次运行 Token 统计：")
                print(f"输入 Token：{total_prompt_tokens}")
                print(f"输出 Token：{total_completion_tokens}")
                print(f"总计 Token：{total_tokens}")

                print("\n历史累计 Token 统计：")
                print(f"输入 Token：{lifetime_usage['total_prompt_tokens']}")
                print(f"输出 Token：{lifetime_usage['total_completion_tokens']}")
                print(f"总计 Token：{lifetime_usage['total_tokens']}")

                usage_tracker.print_usage_summary()
                print("程序已退出。")
                break

            if not question:
                print("问题不能为空，请重新输入。")
                continue

            ask_ai(question)

        except (KeyboardInterrupt, EOFError):
            print("\n程序已退出。")
            break


if __name__ == "__main__":
    main()
