"""Audit the pre-registered Stage 4A RDKit and Morgan feature matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..models.global_information_augmentation.features import (
    FeatureGenerationError,
    build_feature_manifest,
    save_feature_cache,
    write_feature_manifest,
)
from ..src.data import load_training_data

TRACK_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"


def audit(train_csv: Path, manifest_path: Path) -> dict:
    from ..src.data import sha256_file

    source_hash = sha256_file(train_csv)
    if source_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            f"Stage 4A requires the frozen training CSV hash {EXPECTED_TRAIN_SHA256}; "
            f"received {source_hash}"
        )
    rows = load_training_data(train_csv)
    if len(rows) != 7973:
        raise ValueError(f"Expected 7,973 raw benchmark rows, received {len(rows)}")
    features = build_feature_manifest(
        smiles_values=[str(row["SMILES"]) for row in rows],
        source_data_path=train_csv,
    )
    if features.manifest["descriptors"]["nonfinite_count"] != 0:
        raise FeatureGenerationError(
            "Descriptor coverage contains NaN/inf; Stage 4A formal training is blocked"
        )
    if features.manifest["source"]["sample_count"] != 7973:
        raise RuntimeError("Feature audit did not cover all frozen rows")
    write_feature_manifest(manifest_path, features.manifest)
    for variant in ("descriptor", "morgan"):
        cache = (
            TRACK_ROOT
            / "models"
            / "global_information_augmentation"
            / variant
            / "artifacts"
            / "cache"
            / "global_feature_matrices.npz"
        )
        save_feature_cache(cache, features)
    return features.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=TRACK_ROOT / "experiments" / "stage4a" / "feature_manifest.json",
    )
    args = parser.parse_args()
    manifest = audit(args.train_csv.resolve(), args.manifest_out.resolve())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
