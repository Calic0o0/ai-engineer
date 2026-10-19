
import requests


def get_todo(todo_id):
    url = f"https://jsonplaceholder.typicode.com/todos/{todo_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        print(f"请求失败：{error}")
        return None


todo = get_todo(1)

if todo is not None:
    title = todo.get("title", "没有标题")
    completed = todo.get("completed", False)

    print("Title:", title)
    print("Completed:", completed)

    if completed:
        print("任务已经完成")
    else:
        print("任务尚未完成")
