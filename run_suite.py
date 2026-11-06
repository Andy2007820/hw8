# -*- coding: utf-8 -*-
"""顺序执行作业八全部实验，逐命令记录日志、退出码与墙钟耗时。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs_distill"
RUNS.mkdir(exist_ok=True)

COMMANDS = [
    ("validate", [sys.executable, str(ROOT / "validate_experiment.py")]),
    ("quick", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "quick"]),
    ("teacher", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "teacher", "--retrain"]),
    ("all", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "all"]),
    ("ablate", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "ablate"]),
    ("feature", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "all",
                 "--feat", "0.1", "--run-tag", "feat"]),
    ("self_distill", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "self-distill"]),
    ("quantize", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "quantize"]),
    ("teacher_scan", [sys.executable, "-u", str(ROOT / "distill_lm.py"), "--mode", "scan"]),
]


def main():
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"})
    manifest = {"python": sys.executable, "device": "cpu", "runs": []}
    suite_start = time.perf_counter()
    for name, command in COMMANDS:
        print(f"\n===== BEGIN {name} =====\n{' '.join(command)}", flush=True)
        log_path = RUNS / f"{name}.log"
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                       encoding="utf-8", errors="replace",
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        elapsed = time.perf_counter() - start
        record = {"name": name, "command": command, "returncode": code,
                  "elapsed_s": elapsed, "log": log_path.name}
        manifest["runs"].append(record)
        (RUNS / "suite_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"===== END {name}: code={code}, elapsed={elapsed:.2f}s =====", flush=True)
        if code != 0:
            raise SystemExit(code)
    manifest["total_elapsed_s"] = time.perf_counter() - suite_start
    manifest["status"] = "completed"
    (RUNS / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nALL EXPERIMENTS COMPLETED in {manifest['total_elapsed_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
