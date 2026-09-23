import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_command import SYSTEM_PROMPT
from src.llm_workflow import run_instruction


class LLMWorkflowTests(unittest.TestCase):
    def test_prompt_names_three_tasks(self):
        for description in ("task1：双臂协同搬运横杆", "task2：双臂搬运箱子",
                            "task3：双臂交接物体"):
            self.assertIn(description, SYSTEM_PROMPT)

    def test_reports_each_command_and_completes(self):
        events = []

        def sender(command, port, on_response):
            result = ({"status": "scene_ready"} if command == "task1" else
                      {"status": "completed", "report": {"result": "SUCCESS"}})
            on_response({"status": "accepted"})
            on_response(result)
            return result

        with patch("src.llm_workflow.load_config"), \
             patch("src.llm_workflow.plan_request", return_value=(["task1", "1"], "搬运横杆")), \
             patch("src.llm_workflow.send_command", side_effect=sender) as send:
            ok = run_instruction("将横杆搬到绿色区域", Path(".env"), 8765,
                                 lambda kind, value: events.append((kind, value)))
        self.assertTrue(ok)
        self.assertEqual([call.args[0] for call in send.call_args_list], ["task1", 1])
        self.assertIn(("plan", ["task1", "1"]), events)
        self.assertEqual(events[-1], ("status", "任务完成"))

    def test_rejects_ambiguous_and_stops_on_failure(self):
        for plan, expected_status in (([], "需要补充信息"), (["task2", "start"], "任务失败")):
            events = []
            with self.subTest(plan=plan), patch("src.llm_workflow.load_config"), \
                 patch("src.llm_workflow.plan_request", return_value=(plan, "请明确场景")), \
                 patch("src.llm_workflow.send_command", return_value={"status": "busy"}) as send:
                ok = run_instruction("开始", Path(".env"), 8765,
                                     lambda kind, value: events.append((kind, value)))
            self.assertFalse(ok)
            self.assertEqual(events[-1], ("status", expected_status))
            self.assertEqual(send.call_count, 0 if not plan else 1)


if __name__ == "__main__":
    unittest.main()
