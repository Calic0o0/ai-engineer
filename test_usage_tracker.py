
import json

import usage_tracker


def test_load_usage_returns_defaults_when_file_missing(
    monkeypatch,
    tmp_path,
):
    missing_file = tmp_path / "token_usage.json"
    monkeypatch.setattr(usage_tracker, "USAGE_FILE", missing_file)

    result = usage_tracker.load_usage()

    assert result == {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }


def test_load_usage_reads_valid_file(monkeypatch, tmp_path):
    usage_file = tmp_path / "token_usage.json"
    expected = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_tokens": 150,
    }
    usage_file.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    monkeypatch.setattr(usage_tracker, "USAGE_FILE", usage_file)

    result = usage_tracker.load_usage()

    assert result == expected


from types import SimpleNamespace


def test_record_usage_accumulates_and_saves(
    monkeypatch,
    tmp_path,
    capsys,
):
    usage_file = tmp_path / "token_usage.json"
    monkeypatch.setattr(usage_tracker, "USAGE_FILE", usage_file)

    # 重置本次运行统计
    monkeypatch.setattr(usage_tracker, "total_prompt_tokens", 0)
    monkeypatch.setattr(usage_tracker, "total_completion_tokens", 0)
    monkeypatch.setattr(usage_tracker, "total_tokens", 0)

    # 使用独立的历史累计数据，避免影响真实统计
    lifetime = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_tokens": 150,
    }
    monkeypatch.setattr(usage_tracker, "lifetime_usage", lifetime)

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    usage_tracker.record_usage(usage)

    assert usage_tracker.total_prompt_tokens == 10
    assert usage_tracker.total_completion_tokens == 5
    assert usage_tracker.total_tokens == 15

    assert lifetime == {
        "total_prompt_tokens": 110,
        "total_completion_tokens": 55,
        "total_tokens": 165,
    }

    saved_data = json.loads(usage_file.read_text(encoding="utf-8"))
    assert saved_data == lifetime

    capsys.readouterr()


def test_load_usage_handles_invalid_json(
    monkeypatch,
    tmp_path,
    capsys,
):
    usage_file = tmp_path / "token_usage.json"

    # 故意写入无效 JSON
    usage_file.write_text(
        '{"total_prompt_tokens": ',
        encoding="utf-8",
    )
    monkeypatch.setattr(usage_tracker, "USAGE_FILE", usage_file)

    result = usage_tracker.load_usage()

    assert result == {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }

    captured = capsys.readouterr()
    assert "读取历史统计失败" in captured.out
    assert "历史用量暂时按零处理" in captured.out


import pytest


@pytest.mark.parametrize(
    "invalid_value",
    [-1, "100", None, 3.5],
)
def test_load_usage_handles_invalid_values(
    monkeypatch,
    tmp_path,
    capsys,
    invalid_value,
):
    usage_file = tmp_path / "token_usage.json"

    data = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_tokens": 150,
    }
    data["total_prompt_tokens"] = invalid_value

    usage_file.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
    monkeypatch.setattr(usage_tracker, "USAGE_FILE", usage_file)

    result = usage_tracker.load_usage()

    assert result == {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }

    captured = capsys.readouterr()
    assert "读取历史统计失败" in captured.out


def test_record_usage_handles_save_error(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        usage_tracker,
        "lifetime_usage",
        {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    def fake_save_usage(usage_data):
        raise OSError("模拟文件写入失败")

    monkeypatch.setattr(
        usage_tracker,
        "save_usage",
        fake_save_usage,
    )

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    # 保存失败时，函数不应向外抛出 OSError
    usage_tracker.record_usage(usage)

    captured = capsys.readouterr()
    assert "保存历史统计失败" in captured.out
    assert "模拟文件写入失败" in captured.out
