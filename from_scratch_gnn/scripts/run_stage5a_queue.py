"""Run Stage 5A smoke pilots or the preregistered dynamic formal queue."""
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

from ..models.capacity_scaling.protocol import (
    MODEL_ROOT,
    NEW_WIDTHS,
    SEEDS,
    WIDTHS,
    artifact_dir,
    formal_job_matrix,
)
from ..src.data import sha256_file, write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = Path("/home/gyf/work/polymer/data/competition_raw/train.csv")
DEFAULT_LOGS_ROOT = Path("/tmp/stage5a_runs")
FORMAL_WIDTH_ORDER = (128, 384, 512)


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
    unexpected = [line for line in status.splitlines() if not line.startswith("?? data/")]
    if unexpected:
        raise RuntimeError(
            "Stage 5A CUDA runs require the committed frozen source; "
            f"first status lines: {unexpected[:10]}"
        )


def _job_matrix(pilot: bool) -> list[dict[str, Any]]:
    if pilot:
        return [
            {"phase": "smoke", "hidden_dim": width, "seed": 42, "fold": 0}
            for width in (512, 384)
        ]
    return [
        {"phase": "formal", **job}
        for job in formal_job_matrix()
    ]


def _job_key(job: dict[str, Any]) -> str:
    return f"C{job['hidden_dim']}-seed{job['seed']}" if job["phase"] == "formal" else f"C{job['hidden_dim']}-smoke"


def _command(
    job: dict[str, Any], train_csv: Path, attempt: int
) -> tuple[list[str], Path]:
    width = int(job["hidden_dim"])
    seed = int(job["seed"])
    formal = job["phase"] == "formal"
    output_dir = artifact_dir(
        "formal" if formal else "smoke", width, seed, attempt=attempt
    )
    command = [
        sys.executable,
        "-m",
        "from_scratch_gnn.models.capacity_scaling.runner",
        "--hidden-dim",
        str(width),
        "--train-csv",
        str(train_csv.resolve()),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
    ]
    if formal:
        if width == 256 or seed not in SEEDS or width not in NEW_WIDTHS:
            raise ValueError(f"Invalid preregistered Stage 5A formal job: {job}")
    else:
        command.extend(["--smoke-fold", "0", "--epochs", "5"])
    return command, output_dir


def _run_queue(
    train_csv: Path,
    logs_dir: Path,
    concurrency: int,
    *,
    pilot: bool = False,
    max_retries: int = 1,
) -> dict[str, Any]:
    if concurrency not in (1, 2):
        raise ValueError("Stage 5A queue concurrency must be 1 or 2")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Frozen benchmark CSV not found: {train_csv}")
    _require_clean_source()
    source_commit = _git("rev-parse", "HEAD")
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = [{"job": job, "attempt": 1} for job in _job_matrix(pilot)]
    active: dict[str, dict[str, Any]] = {}
    run_records: dict[str, list[dict[str, Any]]] = {}
    resource_samples: list[dict[str, Any]] = []
    peak_cpu_percent = 0.0
    peak_ram_used_mb = 0.0
    started_at: str | None = None
    queue_started: float | None = None
    process = psutil.Process()
    process.cpu_percent(interval=None)
    max_observed_active = 0
    next_sample = 0.0
    sample_interval = 2.0 if pilot else 15.0

    def start_item(item: dict[str, Any]) -> None:
        nonlocal started_at, queue_started, max_observed_active
        job = item["job"]
        attempt = int(item["attempt"])
        key = _job_key(job)
        command, output_dir = _command(job, train_csv, attempt)
        log_path = logs_dir / f"{key}_attempt_{attempt:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        child = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        if queue_started is None:
            queue_started = time.monotonic()
            started_at = _utc_now()
        record = {
            "job": dict(job),
            "attempt": attempt,
            "pid": child.pid,
            "command": command,
            "output_dir": str(output_dir),
            "started_at_utc": _utc_now(),
            "log_path": str(log_path),
            "status": "running",
        }
        active[key] = {
            "process": child,
            "log": log_handle,
            "record": record,
            "started_monotonic": time.monotonic(),
        }
        run_records.setdefault(key, []).append(record)
        max_observed_active = max(max_observed_active, len(active))
        print(f"started Stage 5A {key} attempt={attempt} pid={child.pid}", flush=True)

    while pending or active:
        while pending and len(active) < concurrency:
            start_item(pending.pop(0))
        now = time.monotonic()
        if now >= next_sample:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=None))
                ram_used_mb = psutil.virtual_memory().used / (1024.0 * 1024.0)
                sample: dict[str, Any] = {
                    "elapsed_seconds": now - (queue_started or now),
                    "cpu_percent": cpu_percent,
                    "ram_used_mb": ram_used_mb,
                    "active_jobs": len(active),
                }
                peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
                peak_ram_used_mb = max(peak_ram_used_mb, ram_used_mb)
                gpu = _gpu_sample()
                if gpu is not None:
                    sample.update(gpu)
                resource_samples.append(sample)
            except (OSError, psutil.Error):
                pass
            next_sample = now + sample_interval

        finished: list[str] = []
        for key, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            duration = time.monotonic() - item["started_monotonic"]
            record = item["record"]
            record.update(
                {
                    "return_code": int(code),
                    "completed_at_utc": _utc_now(),
                    "process_wall_seconds": duration,
                    "status": "passed" if code == 0 else "failed",
                }
            )
            finished.append(key)
            del active[key]
            print(f"finished Stage 5A {key} rc={code} wall={duration:.1f}s", flush=True)
            if code != 0 and int(record["attempt"]) <= max_retries:
                pending.append(
                    {"job": record["job"], "attempt": int(record["attempt"]) + 1}
                )
        if active:
            time.sleep(0.5)

    ended_monotonic = time.monotonic()
    wall_seconds = ended_monotonic - queue_started if queue_started is not None else 0.0
    final_records = [items[-1] for items in run_records.values()]
    failed_records = [record for record in final_records if record["status"] != "passed"]
    gpu_util = [sample["gpu_utilization_percent"] for sample in resource_samples if "gpu_utilization_percent" in sample]
    gpu_memory = [sample["gpu_memory_used_mb"] for sample in resource_samples if "gpu_memory_used_mb" in sample]
    notes = {
        "stage": "5A",
        "mode": "concurrency_2_pilot" if pilot else "formal",
        "source_commit": source_commit,
        "branch": _git("branch", "--show-current"),
        "train_csv_sha256": sha256_file(train_csv),
        "concurrency": concurrency,
        "max_observed_active_jobs": max_observed_active,
        "planned_job_count": len(_job_matrix(pilot)),
        "completed_job_count": sum(record["status"] == "passed" for record in final_records),
        "failed_job_count": len(failed_records),
        "formal_wall_seconds": wall_seconds,
        "summed_process_wall_seconds": float(
            sum(record.get("process_wall_seconds", 0.0) for record in final_records)
        ),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "logs_directory": str(logs_dir.resolve()),
        "runs": [record for key in sorted(run_records) for record in run_records[key]],
        "resource_monitor": {
            "sample_interval_seconds": sample_interval,
            "sample_count": len(resource_samples),
            "mean_gpu_utilization_percent": float(np.mean(gpu_util)) if gpu_util else None,
            "peak_gpu_utilization_percent": float(max(gpu_util)) if gpu_util else None,
            "peak_nvidia_smi_memory_used_mb": float(max(gpu_memory)) if gpu_memory else None,
            "peak_host_cpu_percent": peak_cpu_percent,
            "peak_host_ram_used_mb": peak_ram_used_mb,
            "samples": resource_samples,
        },
        "status": "passed" if not failed_records else "failed",
    }
    if not pilot and len(final_records) != len(formal_job_matrix()):
        notes["status"] = "failed"
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run paired C512/C384 CUDA smokes")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--notes-out", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    label = "pilot" if args.pilot else "formal"
    logs_dir = args.logs_dir or DEFAULT_LOGS_ROOT / label
    notes_out = args.notes_out or MODEL_ROOT.parent.parent / "experiments" / "stage5a" / f"{label}_execution_notes.json"
    notes = _run_queue(
        args.train_csv.resolve(),
        logs_dir.resolve(),
        args.concurrency,
        pilot=args.pilot,
        max_retries=args.max_retries,
    )
    write_json(notes_out.resolve(), notes)
    print(json.dumps({key: value for key, value in notes.items() if key != "resource_monitor"}, ensure_ascii=False, indent=2))
    if notes["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
