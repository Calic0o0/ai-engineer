
import json
from pathlib import Path

# 统计文件放在当前模块所在目录
USAGE_FILE = Path(__file__).with_name("data") / "token_usage.json"

# 本次程序运行的统计
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0


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

        for key in default_usage:
            value = saved_usage.get(key)

            if type(value) is not int or value < 0:
                raise ValueError(f"统计数据 {key} 不合法")

        return saved_usage

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"读取历史统计失败：{error}")
        print("历史用量暂时按零处理。")
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


# 程序启动时读取历史累计用量
lifetime_usage = load_usage()


def record_usage(usage):
    """记录一次 API 调用的 Token 用量。"""
    global total_prompt_tokens
    global total_completion_tokens
    global total_tokens

    total_prompt_tokens += usage.prompt_tokens
    total_completion_tokens += usage.completion_tokens
    total_tokens += usage.total_tokens

    lifetime_usage["total_prompt_tokens"] += usage.prompt_tokens
    lifetime_usage["total_completion_tokens"] += usage.completion_tokens
    lifetime_usage["total_tokens"] += usage.total_tokens

    try:
        save_usage(lifetime_usage)
        print("历史 Token 统计已保存。")
    except OSError as error:
        print(f"保存历史统计失败：{error}")


def print_usage_summary():
    """显示本次运行和历史累计的 Token 用量。"""
    print("\n本次运行 Token 统计：")
    print(f"输入 Token：{total_prompt_tokens}")
    print(f"输出 Token：{total_completion_tokens}")
    print(f"总计 Token：{total_tokens}")

    print("\n历史累计 Token 统计：")
    print(f"输入 Token：{lifetime_usage['total_prompt_tokens']}")
    print(f"输出 Token：{lifetime_usage['total_completion_tokens']}")
    print(f"总计 Token：{lifetime_usage['total_tokens']}")

    print(f"\n统计文件：{USAGE_FILE}")
