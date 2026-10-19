
import llm_client

messages = [
    {
        "role": "system",
        "content": "你是一个简洁的 Python 编程老师。",
    },
    {
        "role": "user",
        "content": "用一句话解释 Python 函数。",
    },
]

answer = llm_client.stream_answer(messages)

if answer:
    print("\n测试成功：已获取完整回答。")
else:
    print("\n测试失败：没有获取到回答。")
