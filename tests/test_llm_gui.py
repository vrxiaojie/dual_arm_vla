import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib import request
from unittest.mock import patch

from scripts.llm_gui import CommandWebServer, visual_summary
from src.llm_command import LLMConfig, plan_request


class HTMLInterfaceTests(unittest.TestCase):
    def test_stream_content_and_reasoning_before_validating_plan(self):
        class Response(io.BytesIO):
            headers = {"Content-Type": "text/event-stream"}

        chunks = [
            {"choices": [{"delta": {"reasoning_content": "选择横杆场景"}}]},
            {"choices": [{"delta": {"content": '{"commands":["task1",'}}]},
            {"choices": [{"delta": {"content": '"1"],"message":"搬运"}'}}]},
        ]
        raw = ''.join('data: ' + json.dumps(c) + '\r\n\r\n' for c in chunks)
        events = []
        with patch('src.llm_command.request.urlopen', return_value=Response(
                (raw + 'data: [DONE]\n\n').encode())) as call:
            plan = plan_request('搬运横杆', LLMConfig('https://example.test/v1', 'key', 'model'),
                                lambda kind, text: events.append((kind, text)))
        self.assertEqual(plan, (["task1", "1"], "搬运"))
        self.assertEqual(events[0], ('reasoning', '选择横杆场景'))
        self.assertEqual(len(events), 3)
        self.assertTrue(json.loads(call.call_args.args[0].data)['stream'])

    def test_visual_threshold_does_not_change_motion_report(self):
        value = dict(status='evaluated', confidence=0.8, verdict='SUCCESS')
        self.assertEqual(visual_summary(value, 0.8)['acceptance'], '视觉验收通过')
        self.assertIn('待确认', visual_summary(value, 0.9)['acceptance'])
        self.assertNotIn('acceptance', value)
        self.assertEqual(visual_summary(dict(value, verdict='FAILURE'), 0.8)['acceptance'], '视觉验收失败')
        self.assertIn('待确认', visual_summary(dict(value, verdict='UNCERTAIN'), 0.8)['acceptance'])
        self.assertEqual(visual_summary(dict(status='error'), 0.8)['acceptance'], '视觉验收出错')

    def test_http_job_sse_and_camera_images(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'top_view.png'
            image.write_bytes(b'fake-png')

            def workflow(instruction, env, port, emit, *, stream):
                self.assertTrue(stream)
                emit('llm_delta', {'channel': 'reasoning', 'text': '测试'})
                emit('status', '执行中：1')
                emit('response', ('1', {'status': 'completed', 'report': {
                    'result': 'SUCCESS', 'visual_evaluation': {
                        'status': 'evaluated', 'confidence': 0.95, 'verdict': 'SUCCESS',
                        'images': {'top_view': str(image)}, 'reason': '在目标区域'}}}))
                emit('status', '任务完成')

            with patch('scripts.llm_gui.run_instruction', side_effect=workflow):
                server = CommandWebServer(('127.0.0.1', 0), Path('.env'), 8765)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f'http://127.0.0.1:{server.server_port}'
                try:
                    with request.urlopen(base) as response:
                        self.assertIn('测试预设', response.read().decode())
                    req = request.Request(base + '/run', data=json.dumps({
                        'instruction': '测试', 'threshold': 0.9}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with request.urlopen(req) as response:
                        job_id = json.load(response)['job_id']
                    with request.urlopen(base + '/events/' + job_id) as response:
                        events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                                  if line.startswith('data: ')]
                    self.assertEqual(events[-1]['kind'], 'done')
                    vision = next(e['value'] for e in events if e['kind'] == 'vision')
                    self.assertEqual(vision['acceptance'], '视觉验收通过')
                    with request.urlopen(base + vision['images']['top_view']) as response:
                        self.assertEqual(response.read(), b'fake-png')
                    replay = request.Request(base + '/events/' + job_id,
                                             headers={'Last-Event-ID': str(len(events) - 2)})
                    with request.urlopen(replay) as response:
                        self.assertEqual(response.read().decode().count('data: '), 1)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == '__main__':
    unittest.main()
