"""Run the five pre-registered Stage 4B seeds with a two-slot dynamic queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from ..models.elemental_physical_priors.runner import EXPERIMENT_ROOT, FORMAL_SEEDS, MODEL_ROOT
from ..src.data import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = Path("/home/gyf/work/polymer/data/competition_raw/train.csv")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()


def _gpu_sample() -> dict[str, float] | None:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).splitlines()[0]
        utilization, memory = (float(value.strip()) for value in line.split(",", 1))
        return {"gpu_utilization_percent": utilization, "gpu_memory_used_mb": memory}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _require_clean_source() -> None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Formal queue requires a clean source tree; first status lines: {status.splitlines()[:10]}")


def _run_queue(train_csv: Path, logs_dir: Path, concurrency: int) -> dict[str, Any]:
    if concurrency != 2:
        raise ValueError("Stage 4B formal queue is preregistered at concurrency=2")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Frozen benchmark CSV not found: {train_csv}")
    _require_clean_source()
    source_commit = _git("rev-parse", "HEAD")
    logs_dir.mkdir(parents=True, exist_ok=True)
    complete = []
    pending = list(FORMAL_SEEDS)
    active: dict[int, dict[str, Any]] = {}
    run_records: dict[int, dict[str, Any]] = {}
    resource_samples: list[dict[str, Any]] = []
    peak_cpu_percent = 0.0
    peak_ram_used_mb = 0.0
    process = psutil.Process()
    start_monotonic: float | None = None
    started_at: str | None = None
    failed = False
    next_sample = 0.0

    def start_seed(seed: int) -> None:
        nonlocal start_monotonic, started_at
        log_path = logs_dir / f"seed_{seed}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "from_scratch_gnn.models.elemental_physical_priors.train_oof",
            "--train-csv",
            str(train_csv.resolve()),
            "--seed",
            str(seed),
            "--device",
            "cuda",
        ]
        child = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        if start_monotonic is None:
            start_monotonic = time.monotonic()
            started_at = _utc_now()
            process.cpu_percent(interval=None)
        active[seed] = {
            "process": child,
            "log": log,
            "command": command,
            "pid": child.pid,
            "started_monotonic": time.monotonic(),
            "started_at_utc": _utc_now(),
            "log_path": str(log_path),
        }
        run_records[seed] = {
            "seed": seed,
            "pid": child.pid,
            "command": command,
            "started_at_utc": active[seed]["started_at_utc"],
            "log_path": str(log_path),
            "status": "running",
        }
        print(f"started Stage 4B seed {seed} pid={child.pid}", flush=True)

    try:
        for _ in range(min(concurrency, len(pending))):
            start_seed(pending.pop(0))
        while active:
            now = time.monotonic()
            if now >= next_sample:
                try:
                    cpu_percent = float(psutil.cpu_percent(interval=None))
                    ram_used_mb = psutil.virtual_memory().used / (1024.0 * 1024.0)
                    sample: dict[str, Any] = {
                        "elapsed_seconds": now - (start_monotonic or now),
                        "cpu_percent": cpu_percent,
                        "ram_used_mb": ram_used_mb,
                    }
                    peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
                    peak_ram_used_mb = max(peak_ram_used_mb, ram_used_mb)
                    gpu = _gpu_sample()
                    if gpu is not None:
                        sample.update(gpu)
                    resource_samples.append(sample)
                except (OSError, psutil.Error):
                    pass
                next_sample = now + 15.0

            just_finished: list[int] = []
            for seed, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                duration = time.monotonic() - item["started_monotonic"]
                record = run_records[seed]
                record.update(
                    {
                        "return_code": int(code),
                        "completed_at_utc": _utc_now(),
                        "process_wall_seconds": duration,
                        "status": "passed" if code == 0 else "failed",
                    }
                )
                complete.append(seed)
                just_finished.append(seed)
                del active[seed]
                print(f"finished Stage 4B seed {seed} rc={code} wall={duration:.1f}s", flush=True)
                if code != 0:
                    failed = True

            if failed:
                for seed, item in active.items():
                    item["process"].terminate()
                    item["process"].wait(timeout=20)
                    item["log"].close()
                    run_records[seed].update(
                        {"return_code": item["process"].returncode, "status": "cancelled_after_peer_failure"}
                    )
                active.clear()
                break

            for _seed in just_finished:
                if pending:
                    start_seed(pending.pop(0))
            if active:
                time.sleep(1.0)
    finally:
        for item in active.values():
            if item["process"].poll() is None:
                item["process"].terminate()
                item["process"].wait(timeout=20)
            item["log"].close()

    ended_monotonic = time.monotonic()
    formal_wall_seconds = (
        ended_monotonic - start_monotonic if start_monotonic is not None else 0.0
    )
    gpu_util = [sample["gpu_utilization_percent"] for sample in resource_samples if "gpu_utilization_percent" in sample]
    gpu_memory = [sample["gpu_memory_used_mb"] for sample in resource_samples if "gpu_memory_used_mb" in sample]
    notes = {
        "stage": "4B",
        "source_commit": source_commit,
        "branch": _git("branch", "--show-current"),
        "formal_seeds": list(FORMAL_SEEDS),
        "completed_seeds": sorted(complete),
        "concurrency": concurrency,
        "formal_wall_seconds": formal_wall_seconds,
        "summed_process_wall_seconds": float(sum(item.get("process_wall_seconds", 0.0) for item in run_records.values())),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "train_csv_sha256": __import__("hashlib").sha256(train_csv.read_bytes()).hexdigest(),
        "logs_directory": str(logs_dir.resolve()),
        "runs": [run_records[seed] for seed in FORMAL_SEEDS if seed in run_records],
        "resource_monitor": {
            "sample_interval_seconds": 15,
            "sample_count": len(resource_samples),
            "mean_gpu_utilization_percent": float(np.mean(gpu_util)) if gpu_util else None,
            "peak_gpu_utilization_percent": float(max(gpu_util)) if gpu_util else None,
            "peak_nvidia_smi_memory_used_mb": float(max(gpu_memory)) if gpu_memory else None,
            "peak_host_cpu_percent": peak_cpu_percent,
            "peak_host_ram_used_mb": peak_ram_used_mb,
            "samples": resource_samples,
        },
        "engineering_failures": [],
        "status": "passed" if not failed and sorted(complete) == list(FORMAL_SEEDS) else "failed",
    }
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--logs-dir", type=Path, default=Path("/tmp/stage4b_formal_logs"))
    parser.add_argument(
        "--notes-out",
        type=Path,
        default=EXPERIMENT_ROOT / "formal_execution_notes.json",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    notes = _run_queue(args.train_csv.resolve(), args.logs_dir.resolve(), args.concurrency)
    write_json(args.notes_out.resolve(), notes)
    print(json.dumps({key: value for key, value in notes.items() if key != "resource_monitor"}, ensure_ascii=False, indent=2))
    if notes["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
