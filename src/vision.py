"""Two-view final-state inspection; visual verdicts never override motion safety."""

import base64
from contextlib import nullcontext
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np

from src.llm_command import load_config, request_content


SYSTEM_PROMPT = '''你是 MuJoCo 双臂任务的视觉验收员。输入是同一结束时刻的俯视和侧视 RGB 图。
仅根据图像和任务目标判断最终状态，不要假设动作已成功，也不要输出机器人指令。
俯视图用于检查目标区域和物体完整占地；侧视图用于检查物体是否落在支撑面上、
是否仍被夹爪悬持。按任务明确指定的验收对象检查区域占地，已放下且夹爪已松开、离开物体。
相机使用透视投影，高处表面可能比地面投影更大，应结合侧视图判断，证据不足时不要猜测。
明显在区域外、仍悬空或被抓持返回 FAILURE。遮挡、区域边界不可见、两视角矛盾或
无法判断是否释放时返回 UNCERTAIN，不要猜测。静态终态图不能证明搬运轨迹、交接
过程、稳定性或过程中无碰撞，不要声称已验证这些事实。
只输出 JSON，且仅包含以下字段：
{"verdict":"SUCCESS|FAILURE|UNCERTAIN","confidence":0.0,
"reason":"简短中文理由，说明两个视角的证据和局限"}
confidence 是 0 到 1 的数字，表示对此判定的确信程度。'''


def encode_png(pixels):
    """Encode an RGB uint8 array without adding an imaging dependency."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected H×W×3 uint8 RGB pixels")
    height, width, _ = pixels.shape

    def chunk(tag, data):
        return (struct.pack('!I', len(data)) + tag + data
                + struct.pack('!I', zlib.crc32(tag + data)))

    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def task_goal(task, direction):
    if task == 'task1' and type(direction) is int and direction in (1, 2):
        return '将橙色横杆完整放入' + ('绿色' if direction == 1 else '蓝色') + '长条区域，并释放、撤离双臂。'
    if task == 'task2' and direction == 'start':
        return ('将箱子主体（0.60 米立方体）的地面占地完整放在绿色方框内，并释放、撤离双臂。'
                '此场景方框只约束箱体主体，不要求两侧外伸把手及连接件在框内；'
                '请勿仅因把手超出方框就判失败。')
    if task == 'task3' and direction == 'start':
        return '交接任务终态：橙色方块及两端握柄完整放在绿色目标支撑台上，双臂均已释放、撤离。只判断终态。'
    raise ValueError('Unsupported visual task/command')


def validate_verdict(content):
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('Visual LLM response is not JSON') from exc
    if not isinstance(value, dict) or set(value) != {'verdict', 'confidence', 'reason'}:
        raise ValueError('Visual response must contain only verdict, confidence and reason')
    confidence = value['confidence']
    if (value['verdict'] not in ('SUCCESS', 'FAILURE', 'UNCERTAIN')
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(value['reason'], str) or not value['reason'].strip()):
        raise ValueError('Invalid visual verdict fields')
    return value


def evaluate_images(images, task, direction, config):
    if set(images) != {'top_view', 'side_view'}:
        raise ValueError('Both top_view and side_view images are required')
    content = [{'type': 'text', 'text': task_goal(task, direction)}]
    for name, label in (('top_view', '俯视图'), ('side_view', '侧视图')):
        content.append({'type': 'text', 'text': label})
        encoded = base64.b64encode(images[name]).decode('ascii')
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + encoded}})
    payload = {'model': config.model, 'response_format': {'type': 'json_object'}, 'messages': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': content},
    ]}
    if config.reasoning_effort:
        payload['reasoning_effort'] = config.reasoning_effort
    return validate_verdict(request_content(payload, config))


def inspect_final_state(scene, command, directory, env_path=None, capture_lock=None):
    """Save images first; record rendering/API errors independently of motion."""
    result = {'status': 'error', 'images': {}}
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        images = {}
        with capture_lock if capture_lock is not None else nullcontext():
            frames = scene.capture_rgb()
        for name, pixels in frames.items():
            images[name] = encode_png(pixels)
            path = directory / (name + '.png')
            path.write_bytes(images[name])
            result['images'][name] = str(path.resolve())
        result['status'] = 'captured'
        if env_path is not None:
            config = load_config(env_path)
            result.update(evaluate_images(images, scene.task, command.direction, config))
            result.update(status='evaluated', model=config.model)
    except Exception as exc:
        result.update(status='error', error=str(exc))
    return result
