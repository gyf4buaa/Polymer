"""Create Stage 4B's deterministic periodic-table and observed-element audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ..models.elemental_physical_priors.features import (
    build_element_feature_manifest,
    manifest_json_bytes,
)


def load_smiles_only(train_csv: Path) -> list[str]:
    """Read only the source SMILES column; labels and fold assignments are ignored."""
    with train_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "SMILES" not in reader.fieldnames:
            raise ValueError("Training CSV must contain an SMILES column")
        return [str(row["SMILES"]) for row in reader]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("from_scratch_gnn/experiments/stage4b/element_feature_manifest.json"),
    )
    args = parser.parse_args()
    smiles = load_smiles_only(args.train_csv.resolve())
    manifest = build_element_feature_manifest(
        smiles,
        source_train_sha256=sha256_file(args.train_csv.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(manifest_json_bytes(manifest))
    coverage = manifest["coverage"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_train_sha256": manifest["source"]["source_train_sha256"],
                "rdkit_version": manifest["rdkit_version"],
                "observed_atomic_numbers": coverage["observed_atomic_numbers"],
                "dummy_atom_count": coverage["dummy_atom_count"],
                "real_atom_count": coverage["real_atom_count"],
                "observed_real_element_count": coverage["observed_real_element_count"],
                "all_observed_real_elements_finite_all_five": coverage[
                    "all_observed_real_elements_finite_all_five"
                ],
                "reference_table_sha256": manifest["reference_table_sha256"],
                "observed_element_table_sha256": coverage[
                    "observed_element_table_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
