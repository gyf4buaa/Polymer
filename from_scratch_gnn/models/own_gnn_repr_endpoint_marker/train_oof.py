"""Train Variant B with the frozen Own-GNN v0 training protocol."""
from pathlib import Path

from ..polymer_representation_ablation.runner import main_for_variant
from .graph import GRAPH_SCHEMA, REPRESENTATION, build_polymer_graph


def main() -> None:
    main_for_variant(
        variant_root=Path(__file__).resolve().parent,
        graph_schema=GRAPH_SCHEMA,
        representation=REPRESENTATION,
        graph_builder=build_polymer_graph,
    )


if __name__ == "__main__":
    main()
