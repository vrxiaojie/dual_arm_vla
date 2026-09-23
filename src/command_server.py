"""Local JSON command bridge. All simulation work stays on the main thread."""
import json
import queue
import socketserver
import threading
from src.config import TaskCommand


class CommandServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port=8765):
        self.pending = queue.Queue(maxsize=1)
        self.guard = threading.Lock()
        self.state = 'idle'
        super().__init__(('127.0.0.1', port), CommandHandler)

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
        self.server_close()
        self.thread.join()

    def submit(self, direction):
        control = isinstance(direction, str) and direction in ("task1", "task2", "task3", "start", "reset")
        command = direction if control else TaskCommand(direction)
        with self.guard:
            if self.state != 'idle' and not (self.state == 'failed' and direction in ('reset', 'task1', 'task2', 'task3')):
                return None, {'status': self.state, 'message': 'No command accepted'}
            job = (command, threading.Event(), {})
            self.state = 'busy'
            self.pending.put_nowait(job)
            return job, {'status': 'accepted', 'command': direction}

    def complete(self, job, response, failed=False):
        with self.guard:
            job[2].update(response)
            self.state = 'failed' if failed else 'idle'
            job[1].set()


class CommandHandler(socketserver.StreamRequestHandler):
    def send(self, response):
        self.wfile.write((json.dumps(response) + '\n').encode())
        self.wfile.flush()

    def handle(self):
        self.connection.settimeout(5)
        try:
            raw = self.rfile.readline(1025)
            if len(raw) > 1024 or not raw.endswith(b'\n'):
                raise ValueError('Expected one JSON line, at most 1024 bytes')
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError('Expected a JSON object')
            job, response = self.server.submit(request.get('command'))
        except (ValueError, OSError) as exc:
            try:
                self.send({'status': 'invalid', 'message': str(exc)})
            except OSError:
                pass
            return
        try:
            self.send(response)
            if job is not None:
                job[1].wait()
                self.send(job[2])
        except OSError:
            pass  # Disconnecting does not cancel an accepted motion.
