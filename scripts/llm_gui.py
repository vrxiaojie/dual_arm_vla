#!/usr/bin/env python3
"""Local HTML interface for the LLM-to-robot workflow (standard library only)."""

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import urlparse
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.llm_workflow import run_instruction

PAGE = Path(__file__).with_name("llm_gui.html")


class Job:
    def __init__(self, instruction, threshold):
        self.instruction = instruction
        self.threshold = threshold
        self.events = []
        self.condition = threading.Condition()
        self.done = False

    def emit(self, kind, value):
        with self.condition:
            self.events.append({"kind": kind, "value": value})
            if kind == "done":
                self.done = True
            self.condition.notify_all()


def visual_summary(value, threshold):
    """Keep visual acceptance separate from the controller's outcome."""
    result = dict(value)
    result["threshold"] = threshold
    confidence = value.get("confidence")
    if value.get("status") == "error":
        result["acceptance"] = "视觉验收出错"
    elif value.get("status") != "evaluated":
        result["acceptance"] = "仅拍摄图像，未进行视觉判定"
    elif (type(confidence) not in (int, float) or not math.isfinite(confidence)
          or not 0 <= confidence <= 1):
        result["acceptance"] = "置信度无效"
    elif value.get("verdict") == "UNCERTAIN" or confidence < threshold:
        result["acceptance"] = "待确认（不确定或低于阈值）"
    elif value.get("verdict") == "SUCCESS":
        result["acceptance"] = "视觉验收通过"
    else:
        result["acceptance"] = "视觉验收失败"
    return result


class CommandWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, env_path, robot_port):
        super().__init__(address, Handler)
        self.env_path = env_path
        self.robot_port = robot_port
        self.jobs = {}
        self.images = {}
        self.lock = threading.Lock()
        self.active = None

    def start_job(self, instruction, threshold):
        with self.lock:
            if self.active is not None:
                return None
            # Retain recent completed jobs for SSE reconnection.
            if len(self.jobs) >= 20:
                self.jobs.pop(next(iter(self.jobs)))
            job_id = secrets.token_hex(16)
            job = Job(instruction, threshold)
            self.jobs[job_id] = job
            self.active = job_id
        threading.Thread(target=self.worker, args=(job,), daemon=True).start()
        return job_id

    def worker(self, job):
        def emit(kind, value):
            job.emit(kind, value)
            if kind == "response":
                command, response = value
                visual = response.get("report", {}).get("visual_evaluation")
                if isinstance(visual, dict):
                    result = visual_summary(visual, job.threshold)
                    result["command"] = command
                    images = {}
                    for name in ("top_view", "side_view"):
                        path = visual.get("images", {}).get(name)
                        if not isinstance(path, str):
                            continue
                        path = Path(path).resolve()
                        # Expose only camera PNGs returned by the local simulator.
                        if path.name != name + ".png" or not path.is_file():
                            continue
                        token = secrets.token_hex(16)
                        with self.lock:
                            self.images[token] = path
                            if len(self.images) > 80:
                                self.images.pop(next(iter(self.images)))
                        images[name] = "/images/" + token
                    result["images"] = images
                    job.emit("vision", result)
                elif response.get("status") == "completed":
                    job.emit("vision", {"command": command, "images": {},
                                         "acceptance": "未返回视觉数据；请使用 --vision 或 --capture-images 启动模拟器"})

        try:
            run_instruction(job.instruction, self.env_path, self.robot_port, emit, stream=True)
        except Exception as exc:
            job.emit("error", str(exc))
            job.emit("status", "任务失败")
        finally:
            with self.lock:
                self.active = None
            job.emit("done", None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/images/"):
            with self.server.lock:
                image = self.server.images.get(path.removeprefix("/images/"))
            if image is None or not image.is_file():
                self.reply(404, {"error": "图像不存在"})
            else:
                self.reply(200, image.read_bytes(), "image/png")
        elif path.startswith("/events/"):
            with self.server.lock:
                job = self.server.jobs.get(path.removeprefix("/events/"))
            if job is None:
                self.reply(404, {"error": "任务不存在"})
                return
            try:
                cursor = max(0, int(self.headers.get("Last-Event-ID", "-1")) + 1)
            except ValueError:
                self.reply(400, {"error": "无效事件编号"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with job.condition:
                        if cursor >= len(job.events) and not job.done:
                            job.condition.wait(timeout=10)
                        events = job.events[cursor:]
                        done = job.done
                    for event in events:
                        data = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f"id: {cursor}\ndata: {data}\n\n".encode("utf-8"))
                        cursor += 1
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    if done:
                        return
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.reply(404, {"error": "页面不存在"})

    def do_POST(self):
        if self.path != "/run":
            self.reply(404, {"error": "接口不存在"})
            return
        # Reject cross-origin browser submissions to this local robot interface.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.reply(403, {"error": "不允许跨站请求"})
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("需要 JSON 请求")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("请求长度无效")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("需要 JSON 对象")
            instruction = data.get("instruction")
            threshold = data.get("threshold", 0.8)
            if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
                raise ValueError("请输入 1–4000 字的指令")
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("阈值必须在 0 到 1 之间")
        except (ValueError, UnicodeDecodeError) as exc:
            self.reply(400, {"error": str(exc)})
            return
        job_id = self.server.start_job(instruction.strip(), threshold)
        if job_id is None:
            self.reply(409, {"error": "已有任务执行中，请等待完成"})
        else:
            self.reply(202, {"job_id": job_id})


def main(argv=None):
    parser = argparse.ArgumentParser(description="HTML LLM robot command interface")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=8765, help="模拟器命令端口")
    parser.add_argument("--web-port", type=int, default=8080, help="HTML 页面端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)
    with CommandWebServer(("127.0.0.1", args.web_port), args.env, args.port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"双臂指令界面：{url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
