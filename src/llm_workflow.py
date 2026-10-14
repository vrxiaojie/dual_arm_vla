"""Run a language instruction and report progress without touching GUI widgets."""

from scripts.send_command import send_command
from src.llm_command import load_config, plan_request


def run_instruction(instruction, env_path, port, emit, *, stream=False):
    """Emit (kind, value) events; return True only when all commands succeed."""
    emit("status", "正在解析")
    try:
        config = load_config(env_path)
        if stream:
            commands, message = plan_request(
                instruction, config,
                on_delta=lambda kind, text: emit("llm_delta", {"channel": kind, "text": text}),
            )
        else:
            commands, message = plan_request(instruction, config)
        emit("message", message or "已生成指令。")
        emit("plan", commands)
        if not commands:
            emit("status", "需要补充信息")
            return False
        for command in commands:
            emit("status", f"执行中：{command}")
            value = int(command) if command in ("1", "2") else command
            response = send_command(
                value, port,
                on_response=lambda item, arg=command: emit("response", (arg, item)),
            )
            status = response.get("status")
            if status in ("scene_ready", "already_at_target"):
                continue
            if status == "completed" and response.get("report", {}).get("result") == "SUCCESS":
                continue
            emit("status", "任务失败")
            return False
        emit("status", "任务完成")
        return True
    except (OSError, ValueError, KeyError) as exc:
        emit("error", str(exc))
        emit("status", "任务失败")
        return False
