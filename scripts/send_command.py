#!/usr/bin/env python3
"""Standard-library-only command client."""
import argparse
import json
import socket


def send_command(command, port=8765):
    with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
        connection.settimeout(300)
        connection.sendall((json.dumps({'command': command}) + '\n').encode())
        with connection.makefile('rb') as stream:
            response = json.loads(stream.readline())
            print(json.dumps(response, ensure_ascii=False), flush=True)
            if response['status'] == 'accepted':
                response = json.loads(stream.readline())
                print(json.dumps(response, ensure_ascii=False, indent=2), flush=True)
            return response


def main():
    parser = argparse.ArgumentParser(description='Switch task1/task2; task1: 1/2; task2: start; reset current scene')
    parser.add_argument('command', nargs='?', choices=('1', '2', 'task1', 'task2', 'start', 'reset'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    while True:
        try:
            value = str(args.command) if args.command else input('task1/task2=switch | 1=green 2=blue | start=carry box | reset | q=quit > ').strip()
            if value.lower() in ('q', 'quit', 'exit'):
                return 0
            if value not in ('1', '2', 'task1', 'task2', 'start', 'reset'):
                print('Enter task1, task2, 1, 2, start, reset or q.')
                continue
            response = send_command(int(value) if value in ('1', '2') else value, args.port)
            ok = response['status'] in ('already_at_target', 'scene_ready') or (
                response['status'] == 'completed' and response['report']['result'] == 'SUCCESS')
            if args.command:
                return 0 if ok else 1
        except (OSError, ValueError) as exc:
            print(f'Command error: {exc}. Check main program/port; do not retry blindly after a timeout.')
            if args.command:
                return 1
        except (EOFError, KeyboardInterrupt):
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
