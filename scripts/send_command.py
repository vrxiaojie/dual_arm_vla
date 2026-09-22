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
    parser = argparse.ArgumentParser(description='1: blue to green; 2: green to blue')
    parser.add_argument('command', nargs='?', type=int, choices=(1, 2))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    while True:
        try:
            value = str(args.command) if args.command else input('1=green, 2=blue, q=quit > ').strip()
            if value.lower() in ('q', 'quit', 'exit'):
                return 0
            if value not in ('1', '2'):
                print('Please enter 1 or 2 (q to quit).')
                continue
            response = send_command(int(value), args.port)
            ok = response['status'] == 'already_at_target' or (
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
