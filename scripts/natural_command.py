#!/usr/bin/env python3
"""Plan and optionally execute a natural-language robot instruction."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.llm_command import load_config, plan_request
from scripts.send_command import send_command


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use an LLM to choose existing robot commands")
    parser.add_argument("instruction", help="Natural-language instruction, in quotes")
    parser.add_argument("--env", type=Path, default=ROOT / ".env", help="LLM configuration file")
    parser.add_argument("--port", type=int, default=8765, help="Local command server port")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without sending them")
    args = parser.parse_args(argv)
    try:
        commands, message = plan_request(args.instruction, load_config(args.env))
        if message:
            print(message)
        if not commands:
            print("No command sent: request needs clarification.")
            return 2
        for command in commands:
            print(f"python scripts/send_command.py {command}", flush=True)
        if args.dry_run:
            return 0
        for command in commands:
            response = send_command(int(command) if command in ("1", "2") else command, args.port)
            status = response.get("status")
            if status in ("scene_ready", "already_at_target"):
                continue
            if status == "completed" and response.get("report", {}).get("result") == "SUCCESS":
                continue
            print(f"Stopped after {command}: {response}", file=sys.stderr)
            return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"Command error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
