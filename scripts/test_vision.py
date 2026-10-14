#!/usr/bin/env python3
"""Dedicated real-render test, with optional live positive/negative LLM checks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, TaskConfig, TaskCommand, BoxCommand, HandoverCommand
from src.scene import TaskScene
from src.task import TaskRunner
from src.vision import inspect_final_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('task1', 'task2', 'task3', 'all'), default='all')
    parser.add_argument('--live', action='store_true', help='send real images to the LLM configured in --env')
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/screenshots/vision_test')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for task in (('task1', 'task2', 'task3') if args.task == 'all' else (args.task,)):
        config = TaskConfig(output_dir=args.output / task / 'logs', capture_images=True,
                            vision_env_path=args.env if args.live else None)
        if task != 'task1':
            config = replace(config, carry_height=.25, position_tolerance=.025)
        scene = TaskScene(config.model_path)
        scene.switch_task(task)
        commands = (TaskCommand(1), TaskCommand(2)) if task == 'task1' else (
            BoxCommand() if task == 'task2' else HandoverCommand(),)
        for command in commands:
            # Initial source state is a deliberate negative for this destination.
            name = f'{task}_{command.direction}'
            print(f'Testing {name}: initial negative and completed positive', flush=True)
            before = scene.data.qpos.copy()
            visual = inspect_final_state(scene, command, args.output / name / 'negative',
                                         args.env if args.live else None)
            if not (before == scene.data.qpos).all():
                raise AssertionError('Rendering mutated simulation state')
            cases.append({'case': name + '_initial', 'expected': 'FAILURE', 'visual': visual})
            report = TaskRunner(scene, config, command=command).run()
            cases.append({'case': name + '_completed', 'expected': 'SUCCESS',
                          'motion_result': report['result'], 'visual': report['visual_evaluation']})
    for case in cases:
        visual = case['visual']
        case['passed'] = (visual['status'] == ('evaluated' if args.live else 'captured')
                          and (not args.live or visual.get('verdict') == case['expected'])
                          and case.get('motion_result', 'SUCCESS') == 'SUCCESS')
    summary = {'mode': 'live' if args.live else 'render-only', 'cases': cases,
               'passed': all(case['passed'] for case in cases)}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
