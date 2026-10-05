#!/usr/bin/env python3
"""Generate the frozen Kaggle leaderboard probe CSV, figure, and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WARNING = "POST-HOC / LEADERBOARD-PROBED / NOT VALID BLIND PERFORMANCE"
MAIN_SHA = "c9949c088d8822eac775a55ad601d278b747c9f9"
FORMAL_SOURCE_SHA = "72490c1a748ed9395025f4120c9eb04f28268695"
DATASET_SLUG = "safaaa46/clean-own-gnn-c128-5seed-5fold-formal-20261004"

ROWS = [
    {
        "tg_shift_c": 0,
        "submission_ref": "56831294",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-5seed-5fold",
        "kernel_version": 5,
        "submission_message": "Clean Own-GNN C128 5seed x 5fold ensemble",
        "status": "COMPLETE",
        "public_score": 0.06899,
        "private_score": 0.09524,
        "probe_stage": "clean reference",
    },
    {
        "tg_shift_c": 40,
        "submission_ref": "56840734",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-tg-40c-diagnostic",
        "kernel_version": 2,
        "submission_message": "Diagnostic: Clean C128 Tg +40C",
        "status": "COMPLETE",
        "public_score": 0.06417,
        "private_score": 0.08232,
        "probe_stage": "preregistered",
    },
    {
        "tg_shift_c": 70,
        "submission_ref": "56831959",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-posthoc-tg70-diagnostic",
        "kernel_version": 4,
        "submission_message": "Diagnostic: Clean Own-GNN C128 +70C Tg post-processing",
        "status": "COMPLETE",
        "public_score": 0.06493,
        "private_score": 0.07758,
        "probe_stage": "known diagnostic reference",
    },
    {
        "tg_shift_c": 100,
        "submission_ref": "56840820",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-tg-100c-diagnostic",
        "kernel_version": 1,
        "submission_message": "Diagnostic: Clean C128 Tg +100C",
        "status": "COMPLETE",
        "public_score": 0.06911,
        "private_score": 0.07570,
        "probe_stage": "preregistered",
    },
    {
        "tg_shift_c": 115,
        "submission_ref": "56840967",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-tg-115c-diagnostic",
        "kernel_version": 1,
        "submission_message": "Diagnostic: Clean C128 Tg +115C",
        "status": "COMPLETE",
        "public_score": 0.07223,
        "private_score": 0.07597,
        "probe_stage": "single permitted follow-up",
    },
    {
        "tg_shift_c": 130,
        "submission_ref": "56840874",
        "kernel_slug": "safaaa46/clean-own-gnn-c128-tg-130c-diagnostic",
        "kernel_version": 1,
        "submission_message": "Diagnostic: Clean C128 Tg +130C",
        "status": "COMPLETE",
        "public_score": 0.07609,
        "private_score": 0.07657,
        "probe_stage": "preregistered",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative_path: str) -> dict[str, str]:
    path = ROOT / relative_path
    return {"path": relative_path, "sha256": sha256(path)}


def normalized_notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return re.sub(r"(?m)^DELTA = [0-9]+\.0$", "DELTA = __SHIFT__", source)


def write_csv() -> Path:
    output = ROOT / "tg_shift_kaggle_probe.csv"
    fieldnames = [
        "tg_shift_c",
        "submission_ref",
        "kernel_slug",
        "kernel_version",
        "submission_message",
        "status",
        "public_score",
        "private_score",
        "delta_public_vs_clean",
        "delta_private_vs_clean",
        "probe_stage",
    ]
    clean_public = ROWS[0]["public_score"]
    clean_private = ROWS[0]["private_score"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in ROWS:
            writer.writerow(
                {
                    **row,
                    "public_score": f"{row['public_score']:.5f}",
                    "private_score": f"{row['private_score']:.5f}",
                    "delta_public_vs_clean": f"{row['public_score'] - clean_public:+.5f}",
                    "delta_private_vs_clean": f"{row['private_score'] - clean_private:+.5f}",
                }
            )
    with output.open(encoding="utf-8", newline="") as stream:
        parsed = list(csv.DictReader(stream))
    assert len(parsed) == 6
    assert [int(row["tg_shift_c"]) for row in parsed] == [0, 40, 70, 100, 115, 130]
    assert all(row["status"] == "COMPLETE" for row in parsed)
    return output


def write_plot() -> Path:
    output = ROOT / "tg_shift_kaggle_probe.png"
    shifts = [row["tg_shift_c"] for row in ROWS]
    public = [row["public_score"] for row in ROWS]
    private = [row["private_score"] for row in ROWS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    ax.plot(shifts, public, color="#2563EB", marker="o", linewidth=2.2, markersize=6, label="Public")
    ax.plot(shifts, private, color="#D97706", marker="o", linewidth=2.2, markersize=6, label="Private")
    ax.scatter([40], [0.06417], color="#2563EB", edgecolor="white", linewidth=0.8, s=95, zorder=5)
    ax.scatter([100], [0.07570], color="#D97706", edgecolor="white", linewidth=0.8, s=95, zorder=5)
    ax.set_title("Clean Own-GNN C128: Tg shift leaderboard probe\nPOST-HOC / NOT VALID BLIND PERFORMANCE", loc="left", fontsize=12, pad=12)
    ax.set_xlabel("Tg additive shift (°C)")
    ax.set_ylabel("Kaggle score (lower is better)")
    ax.set_xticks(shifts, [str(value) for value in shifts])
    ax.set_ylim(0.060, 0.102)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.annotate("sampled Public min +40", (40, 0.06417), xytext=(18, 12), textcoords="offset points", color="#1D4ED8", fontsize=8)
    ax.annotate("sampled Private min +100", (100, 0.07570), xytext=(12, -17), textcoords="offset points", color="#B45309", fontsize=8)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    assert output.stat().st_size > 10_000
    return output


def write_provenance(csv_path: Path, plot_path: Path) -> Path:
    probes = []
    for shift in (40, 100, 115, 130):
        stem = f"tg-plus{shift}"
        kernel_dir = ROOT / "kernels" / stem
        notebook = next(kernel_dir.glob("*.ipynb"))
        metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        report = ROOT / "runs" / stem / "kernel_run_report.json"
        run = json.loads(report.read_text(encoding="utf-8"))
        row = next(record for record in ROWS if record["tg_shift_c"] == shift)
        assert run["status"] == "passed"
        assert run["models_expected"] == run["models_loaded_and_ensembled"] == 25
        assert run["official_train_only"] is True
        assert run["external_or_supplementary_labels"] is False
        assert run["runtime"]["internet_enabled"] is False
        assert run["runtime"]["accelerator"] == "Tesla T4"
        assert run["clean_submission_comparison"]["non_tg_targets_exactly_equal"] is True
        assert run["clean_submission_comparison"]["id_order_unchanged"] is True
        assert run["clean_submission_comparison"]["rows_unchanged"] is True
        assert abs(run["clean_submission_comparison"]["tg_delta_min"] - shift) <= 1e-10
        assert abs(run["clean_submission_comparison"]["tg_delta_max"] - shift) <= 1e-10
        assert metadata["enable_internet"] is False
        probes.append(
            {
                "tg_shift_c": shift,
                "submission_ref": row["submission_ref"],
                "submission_status": row["status"],
                "public_score": row["public_score"],
                "private_score": row["private_score"],
                "kernel_slug": row["kernel_slug"],
                "kernel_version": row["kernel_version"],
                "kaggle_dataset_slug": DATASET_SLUG,
                "notebook": file_record(str(notebook.relative_to(ROOT))),
                "kernel_metadata": file_record(str((kernel_dir / "kernel-metadata.json").relative_to(ROOT))),
                "successful_runtime_report": file_record(str(report.relative_to(ROOT))),
                "runtime_assertions": {
                    "25_of_25_models_loaded": True,
                    "non_tg_targets_equal_clean": True,
                    "id_order_and_rows_unchanged": True,
                    "finite_predictions": run["finite_predictions"],
                    "only_tg_constant_shift": True,
                    "internet_disabled": True,
                    "accelerator": "Tesla T4",
                },
            }
        )

    code_hashes = {
        hashlib.sha256(normalized_notebook_code(path).encode("utf-8")).hexdigest()
        for probe in probes
        for path in [ROOT / probe["notebook"]["path"]]
    }
    assert len(code_hashes) == 1, "Probe notebook code differs beyond the DELTA literal"
    provenance = {
        "classification": WARNING,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_sha": MAIN_SHA,
        "formal_source_sha": FORMAL_SOURCE_SHA,
        "model_definition": "Stage 5A C128 Own-GNN; keep-dummy; 4-layer GINE; hidden_dim=128; mean|max pooling; seeds 42-46; 5 folds each; arithmetic mean of 25 fold models",
        "model_and_predictions_frozen": True,
        "only_submission_change": "submission['Tg'] += DELTA",
        "prohibited_data_or_methods_used": {
            "released_or_private_labels": False,
            "external_or_supplementary_labels": False,
            "retraining": False,
            "other_targets_changed": False,
            "posthoc_calibration_other_than_registered_tg_offsets": False,
        },
        "preregistered_new_offsets_c": [40, 100, 130],
        "single_followup": {
            "offset_c": 115,
            "reason": "+100 was better than +70 on Private; +130 was worse than +100, so +115 was the one permitted midpoint probe.",
        },
        "no_further_offsets_submitted": True,
        "kaggle_scores_source": {
            "method": "Read-only Kaggle CLI competition submissions listing",
            "competition": "neurips-open-polymer-prediction-2025",
            "queried_date": "2026-10-05",
            "note": "Leaderboard scores are Kaggle-returned aggregate metrics; released labels were not accessed for this probe.",
            "kernel_versions": "Versions are those returned by the successful Kaggle push/run flow during this task; the submissions listing itself does not expose kernel version numbers.",
        },
        "notebook_normalized_code_sha256": next(iter(code_hashes)),
        "probes": probes,
        "reference_submissions": [
            {key: row[key] for key in ("tg_shift_c", "submission_ref", "kernel_slug", "kernel_version", "status", "public_score", "private_score")}
            for row in ROWS
            if row["tg_shift_c"] in (0, 70)
        ],
        "ignored_unscored_submission": {
            "submission_ref": "56831824",
            "status": "COMPLETE",
            "reason": "The Kaggle listing returned blank Public and Private scores, so it is not used as a scored offset point.",
        },
        "known_engineering_attempt": {
            "offset_c": 40,
            "kernel_version": 1,
            "submitted_to_competition": False,
            "outcome": "Failed a post-CSV floating-point equality assertion (maximum parse-roundtrip difference 8.3267e-17); fixed in version 2, which is the sole +40 competition submission.",
        },
        "outputs": {
            "csv": {"path": csv_path.name, "sha256": sha256(csv_path)},
            "figure": {"path": plot_path.name, "sha256": sha256(plot_path)},
        },
    }
    output = ROOT / "probe_provenance.json"
    output.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    csv_path = write_csv()
    plot_path = write_plot()
    provenance_path = write_provenance(csv_path, plot_path)
    print(f"CSV {csv_path} sha256={sha256(csv_path)}")
    print(f"PNG {plot_path} sha256={sha256(plot_path)}")
    print(f"JSON {provenance_path} sha256={sha256(provenance_path)}")


if __name__ == "__main__":
    main()
