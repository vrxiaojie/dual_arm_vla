"""Translate natural-language requests into the existing command arguments."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse


SYSTEM_PROMPT = """你是双臂 MuJoCo 演示的指令规划器。只输出一个 JSON 对象，格式为
{"commands":["参数1","参数2"],"message":"简短说明"}，不要 Markdown、代码或命令前缀。
commands 的每个元素只能是 task1、task2、task3、1、2、start、reset 之一。
程序会把每个参数接在 python scripts/send_command.py 后执行，并按数组顺序等待每步完成。

task1：两只机械臂协同搬运桌上的横杆。先输出 task1 切换并重置场景；
然后输出 1 表示将杆搬到绿色区域，输出 2 表示将杆搬到蓝色区域。
例如“在搬运横杆的场景下，将杆子搬到绿色区域”对应 ["task1","1"]。
task2：两只机械臂抓住箱子两侧把手，把地上的箱子搬进绿色方框。
先输出 task2，再输出 start。没有箱子搬到蓝色区域的动作。
task3：左臂从蓝色区域拿起竖直方块，交给右臂，右臂放入绿色目标区。
先输出 task3，再输出 start。没有反向交接动作。
reset 只重置当前场景的物体和机械臂，不开始搬运。
如果用户只要求切换场景，输出单个 task 参数。如果只要求重置，输出 ["reset"]。
对于没有明确场景的“开始”或“搬运”，以及不支持或含糊的任务，
输出空 commands，并在 message 中简短说明需要什么信息。不要猜测场景。
不要生成坐标、Python 代码、Shell 命令或其他参数。"""

ALLOWED = frozenset(("task1", "task2", "task3", "1", "2", "start", "reset"))
TASKS = frozenset(("task1", "task2", "task3"))


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = ""


def load_config(path: Path) -> LLMConfig:
    """Read simple KEY=VALUE entries without executing the .env file."""
    if not path.is_file():
        raise ValueError(f"Missing {path}; copy .env.example to .env and fill it in")
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if not values.get(key) or values[key].startswith("replace-with-"):
            raise ValueError(f"Set {key} in {path}")
    base_url = values["LLM_BASE_URL"].rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must be an HTTP(S) API root URL")
    effort = values.get("LLM_REASONING_EFFORT", "")
    if effort and effort not in ("minimal", "low", "medium", "high", "xhigh"):
        raise ValueError("LLM_REASONING_EFFORT must be minimal, low, medium, high, xhigh, or blank")
    return LLMConfig(base_url, values["LLM_API_KEY"], values["LLM_MODEL"], effort)


def validate_plan(content: str) -> tuple[list[str], str]:
    """Reject everything except a short, valid sequence of known arguments."""
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not JSON") from exc
    if not isinstance(plan, dict) or set(plan) != {"commands", "message"}:
        raise ValueError("LLM response must contain only commands and message")
    commands, message = plan["commands"], plan["message"]
    if (not isinstance(commands, list) or len(commands) > 2
            or any(type(item) is not str or item not in ALLOWED for item in commands)
            or not isinstance(message, str)):
        raise ValueError("LLM response contains invalid command arguments")
    if len(commands) == 1 and commands[0] not in TASKS | {"reset"}:
        raise ValueError("A motion command must specify and switch its task first")
    if len(commands) == 2 and not (
        commands[0] == "task1" and commands[1] in ("1", "2")
        or commands[0] in ("task2", "task3") and commands[1] == "start"
    ):
        raise ValueError("LLM response contains an unsupported command sequence")
    return commands, message


def plan_request(instruction: str, config: LLMConfig) -> tuple[list[str], str]:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
    }
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort
    endpoint = config.base_url + "/chat/completions"
    http_request = request.Request(
        endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=60) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        raise ValueError(f"LLM API returned HTTP {exc.code}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise ValueError(f"LLM API connection failed: {exc.reason if isinstance(exc, error.URLError) else exc}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM API response has no message content") from exc
    if not isinstance(content, str):
        raise ValueError("LLM API message content must be text")
    return validate_plan(content)
