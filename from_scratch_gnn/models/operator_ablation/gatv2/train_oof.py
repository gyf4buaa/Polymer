"""Train or smoke-test the frozen Stage 3B GATv2 operator."""
from pathlib import Path

from ..runner import main_for_operator


def main() -> None:
    main_for_operator(Path(__file__).resolve().parent, "gatv2")


if __name__ == "__main__":
    main()
