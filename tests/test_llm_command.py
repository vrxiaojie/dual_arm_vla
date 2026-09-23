import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import natural_command
from src.llm_command import LLMConfig, load_config, plan_request, validate_plan


class LLMCommandTests(unittest.TestCase):
    def test_load_config_and_missing_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text(
                "# comment\nLLM_BASE_URL=https://example.test/v1/\n"
                "LLM_API_KEY=secret\nLLM_MODEL=test-model\nLLM_REASONING_EFFORT=low\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.base_url, "https://example.test/v1")
            self.assertEqual(config.reasoning_effort, "low")
            path.write_text("LLM_BASE_URL=https://example.test/v1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                load_config(path)

    def test_plan_request_sends_prompt_and_reasoning_effort(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self, *_):
                return json.dumps({"choices": [{"message": {"content":
                    '{"commands":["task1","1"],"message":"搬运横杆"}'}}]}).encode()

        config = LLMConfig("https://example.test/v1", "secret", "test-model", "high")
        with patch("src.llm_command.request.urlopen", return_value=FakeResponse()) as urlopen:
            commands, message = plan_request("在搬运横杆的场景下，将杆子搬到绿色区域", config)
        self.assertEqual(commands, ["task1", "1"])
        self.assertEqual(message, "搬运横杆")
        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, "https://example.test/v1/chat/completions")
        payload = json.loads(http_request.data)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertIn("task3", payload["messages"][0]["content"])

    def test_reject_invalid_or_unsafe_plans(self):
        for content in (
            "not json", '{"commands":["task1","1","reset"],"message":""}',
            '{"commands":["task2","1"],"message":""}',
            '{"commands":["start"],"message":""}',
            '{"commands":["python scripts/send_command.py 1"],"message":""}',
            '{"commands":[1],"message":""}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_plan(content)
        self.assertEqual(validate_plan('{"commands":[],"message":"场景不明确"}')[0], [])

    def test_supported_task_sequences(self):
        for commands in (["task1", "1"], ["task1", "2"],
                         ["task2", "start"], ["task3", "start"], ["reset"]):
            with self.subTest(commands=commands):
                self.assertEqual(validate_plan(json.dumps({
                    "commands": commands, "message": ""
                }))[0], commands)

    def test_execution_stops_if_scene_switch_fails(self):
        config = LLMConfig("https://example.test/v1", "secret", "test-model")
        with patch("scripts.natural_command.load_config", return_value=config), \
             patch("scripts.natural_command.plan_request", return_value=(["task1", "1"], "")), \
             patch("scripts.natural_command.send_command", return_value={"status": "busy"}) as send, \
             patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(natural_command.main(["搬杆到绿色区域"]), 1)
        send.assert_called_once_with("task1", 8765)

    def test_dry_run_does_not_send(self):
        with patch("scripts.natural_command.load_config"), \
             patch("scripts.natural_command.plan_request", return_value=(["task3", "start"], "")), \
             patch("scripts.natural_command.send_command") as send:
            self.assertEqual(natural_command.main(["交接方块", "--dry-run"]), 0)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
