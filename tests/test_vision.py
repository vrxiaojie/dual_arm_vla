import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import ssl
import zlib

import numpy as np

from src.config import TaskCommand
from src.llm_command import LLMConfig
from src.vision import encode_png, evaluate_images, inspect_final_state, task_goal, validate_verdict


class VisionTests(unittest.TestCase):
    def test_rgb_png_encoding_preserves_channels_and_rows(self):
        pixels = np.array([[[255, 0, 0], [0, 255, 0]],
                           [[0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
        png = encode_png(pixels)
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        offset, compressed = 8, b''
        while offset < len(png):
            size = int.from_bytes(png[offset:offset+4], 'big')
            if png[offset+4:offset+8] == b'IDAT':
                compressed += png[offset+8:offset+8+size]
            offset += size + 12
        self.assertEqual(zlib.decompress(compressed),
                         b'\0' + pixels[0].tobytes() + b'\0' + pixels[1].tobytes())

    def test_multimodal_http_payload_and_validated_answer(self):
        config = LLMConfig('https://example.test/v1', 'secret', 'vision-model', 'low')
        answer = {'verdict': 'FAILURE', 'confidence': .95, 'reason': '物体仍在蓝区'}
        response = io.BytesIO(json.dumps({'choices': [{'message': {
            'content': json.dumps(answer)}}]}).encode())
        images = {name: encode_png(np.zeros((2, 3, 3), dtype=np.uint8))
                  for name in ('top_view', 'side_view')}
        with patch('src.llm_command.request.urlopen', return_value=response) as send:
            self.assertEqual(evaluate_images(images, 'task1', 1, config), answer)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, config.base_url + '/chat/completions')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'vision-model')
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['reasoning_effort'], 'low')
        content = payload['messages'][1]['content']
        self.assertIn('绿色', content[0]['text'])
        for item, name in zip((content[2], content[4]), images):
            self.assertEqual(base64.b64decode(item['image_url']['url'].split(',')[1]), images[name])
        self.assertNotIn('result', payload)

    def test_rejects_bad_verdicts(self):
        valid = {'verdict': 'SUCCESS', 'confidence': .9, 'reason': '已放下'}
        for changes in ({'verdict': 'yes'}, {'confidence': True}, {'confidence': 1.1},
                        {'confidence': float('nan')}, {'confidence': float('inf')},
                        {'reason': ''}, {'extra': 'command'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_verdict(json.dumps(valid | changes))
        for content in ('```json\n{}\n```', '{}', '[]', 'null'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_verdict(content)
        for verdict in ('SUCCESS', 'FAILURE', 'UNCERTAIN'):
            self.assertEqual(validate_verdict(json.dumps(valid | {'verdict': verdict}))['verdict'], verdict)

    def test_capture_and_api_errors_keep_images_and_never_invent_success(self):
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        scene = Mock(task='task1')
        scene.capture_rgb.return_value = dict(top_view=pixels, side_view=pixels)
        with tempfile.TemporaryDirectory() as folder:
            result = inspect_final_state(scene, TaskCommand(), Path(folder))
            self.assertEqual(result['status'], 'captured')
            self.assertEqual(len(result['images']), 2)
            for path in result['images'].values():
                self.assertTrue(Path(path).is_file())
            with patch('src.vision.load_config'), patch('src.vision.evaluate_images', side_effect=ValueError('unsupported image input')):
                result = inspect_final_state(scene, TaskCommand(), Path(folder), Path('.env'))
            self.assertEqual(result['status'], 'error')
            self.assertEqual(len(result['images']), 2)
            self.assertNotIn('verdict', result)
            scene.capture_rgb.side_effect = RuntimeError('no GL context')
            result = inspect_final_state(scene, TaskCommand(), Path(folder))
            self.assertEqual(result['status'], 'error')
            self.assertIn('no GL context', result['error'])

    def test_task_goals_and_missing_camera(self):
        self.assertIn('蓝色', task_goal('task1', 2))
        self.assertIn('箱子', task_goal('task2', 'start'))
        self.assertIn('交接', task_goal('task3', 'start'))
        with self.assertRaises(ValueError):
            evaluate_images({'top_view': b''}, 'task1', 1, None)
        with self.assertRaises(ValueError):
            task_goal('task2', 1)

    def test_macos_missing_python_ca_uses_verified_system_bundle(self):
        response = io.BytesIO(json.dumps({'choices': [{'message': {'content':
            '{"verdict":"UNCERTAIN","confidence":0.8,"reason":"遮挡"}'}}]}).encode())
        config = LLMConfig('https://example.test/v1', 'secret', 'vision-model')
        verified_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with patch('src.llm_command.sys.platform', 'darwin'), \
             patch.dict('src.llm_command.os.environ', {}, clear=True), \
             patch('src.llm_command.ssl.get_default_verify_paths', return_value=Mock(cafile=None)), \
             patch('src.llm_command.Path.is_file', return_value=True), \
             patch('src.llm_command.ssl.create_default_context', return_value=verified_context) as create_context, \
             patch('src.llm_command.request.urlopen', return_value=response) as send:
            verdict = evaluate_images(dict(top_view=b'png', side_view=b'png'), 'task1', 1, config)
        self.assertEqual(verdict['verdict'], 'UNCERTAIN')
        context = send.call_args.kwargs['context']
        create_context.assert_called_once_with(cafile='/etc/ssl/cert.pem')
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)


if __name__ == '__main__':
    unittest.main()
