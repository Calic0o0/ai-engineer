
import llm_client
import usage_tracker

import logger as logging_config

SYSTEM_PROMPT = """
你是一位耐心、严谨的 Python 编程老师。
请结合完整的对话历史回答问题。
回答尽量控制在 200-400 个汉字左右，
优先解释核心概念并提供一个代码例子。
只有用户明确要求详细解释时才展开。
"""

# 保存当前对话历史
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


def trim_history():
    """保留系统提示词和最近 5 轮完整对话。"""
    complete_turns = []
    pending_user = None

    for message in messages[1:]:
        if message["role"] == "user":
            pending_user = message
        elif message["role"] == "assistant" and pending_user is not None:
            complete_turns.append((pending_user, message))
            pending_user = None

    recent_turns = complete_turns[-5:]

    messages[:] = [messages[0]]

    for user_message, assistant_message in recent_turns:
        messages.append(user_message)
        messages.append(assistant_message)


def ask_ai(question):
    """发送问题，获取回答并维护对话历史。"""
    messages.append({
        "role": "user",
        "content": question,
    })

    result = llm_client.stream_answer(messages)

    if (
        not result.success
        or not result.answer
        or not result.answer.strip()
    ):
        # 请求失败或没有有效回答时，不保留没有得到回答的问题
        messages.pop()
        return

    answer = result.answer

    messages.append({
        "role": "assistant",
        "content": answer,
    })

    trim_history()


def main():
    logging_config.setup_logging()

    print("AI 编程助手已启动！")
    print("输入 exit、quit 或 退出即可结束程序。")

    try:
        while True:
            question = input("\n你：").strip()

            if question.lower() in ("exit", "quit", "退出"):
                break

            if not question:
                print("请输入一个问题。")
                continue

            print("\nAI：", end="", flush=True)
            ask_ai(question)

    except (KeyboardInterrupt, EOFError):
        print("\n收到退出信号。")

    finally:
        usage_tracker.print_usage_summary()
        print("程序已退出。")


if __name__ == "__main__":
    main()
