#!/usr/bin/env python3
"""Tkinter interface for the local LLM-to-robot command workflow."""

import argparse
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.llm_workflow import run_instruction


class CommandApp:
    def __init__(self, root, env_path, port):
        self.root = root
        self.env_path = env_path
        self.port = port
        self.events = queue.Queue()
        self.busy = False
        root.title("双臂机械臂自然语言指令")
        root.minsize(650, 520)

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(frame, text="输入自然语言指令").grid(row=0, column=0, sticky="w")
        self.input = tk.Text(frame, height=3, wrap="word")
        self.input.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        self.input.bind("<Control-Return>", self.send)
        self.send_button = ttk.Button(frame, text="发送", command=self.send)
        self.send_button.grid(row=2, column=0, sticky="e")

        self.status = tk.StringVar(value="空闲")
        status_row = ttk.Frame(frame)
        status_row.grid(row=3, column=0, sticky="ew", pady=(8, 8))
        ttk.Label(status_row, text="任务状态：").pack(side="left")
        self.status_label = ttk.Label(status_row, textvariable=self.status, foreground="#226622")
        self.status_label.pack(side="left")

        panels = ttk.Frame(frame)
        panels.grid(row=4, column=0, sticky="nsew")
        panels.columnconfigure(0, weight=1)
        panels.rowconfigure(1, weight=1)
        panels.rowconfigure(3, weight=1)
        ttk.Label(panels, text="文本输出").grid(row=0, column=0, sticky="w")
        self.output = scrolledtext.ScrolledText(panels, height=9, wrap="word", state="disabled")
        self.output.grid(row=1, column=0, sticky="nsew", pady=(4, 10))
        ttk.Label(panels, text="指令解析").grid(row=2, column=0, sticky="w")
        self.plan = scrolledtext.ScrolledText(panels, height=6, wrap="word", state="disabled")
        self.plan.grid(row=3, column=0, sticky="nsew", pady=(4, 0))

        self.input.focus_set()
        self.root.after(100, self._drain_events)

    @staticmethod
    def _append(widget, text, clear=False):
        widget.configure(state="normal")
        if clear:
            widget.delete("1.0", "end")
        widget.insert("end", text + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _set_status(self, value):
        self.status.set(value)
        color = "#aa2222" if value == "任务失败" else "#996000" if value in (
            "正在解析", "需要补充信息") or value.startswith("执行中") else "#226622"
        self.status_label.configure(foreground=color)

    def send(self, _event=None):
        if self.busy:
            return "break"
        instruction = self.input.get("1.0", "end").strip()
        if not instruction:
            self._set_status("请输入指令")
            return "break"
        self.busy = True
        self.send_button.configure(state="disabled")
        self._append(self.output, f"你：{instruction}")
        self._append(self.plan, "等待 LLM 解析…", clear=True)
        self._set_status("正在解析")
        threading.Thread(
            target=self._worker, args=(instruction,), daemon=True
        ).start()
        return "break"

    def _worker(self, instruction):
        try:
            run_instruction(instruction, self.env_path, self.port,
                            lambda kind, value: self.events.put((kind, value)))
        except Exception as exc:
            self.events.put(("error", str(exc)))
            self.events.put(("status", "任务失败"))
        finally:
            self.events.put(("done", None))

    def _drain_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self._set_status(value)
                elif kind == "message":
                    self._append(self.output, f"LLM：{value}")
                elif kind == "plan":
                    lines = [f"{index}. python scripts/send_command.py {arg}"
                             for index, arg in enumerate(value, 1)]
                    self._append(self.plan, "\n".join(lines) if lines else "无可执行指令", clear=True)
                elif kind == "response":
                    command, response = value
                    status = response.get("status", "unknown")
                    detail = response.get("message") or response.get("report", {}).get("failure_reason")
                    self._append(self.output, f"{command}：{status}" + (f"（{detail}）" if detail else ""))
                elif kind == "error":
                    self._append(self.output, f"错误：{value}")
                elif kind == "done":
                    self.busy = False
                    self.send_button.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tkinter LLM robot command interface")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    root = tk.Tk()
    CommandApp(root, args.env, args.port)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
