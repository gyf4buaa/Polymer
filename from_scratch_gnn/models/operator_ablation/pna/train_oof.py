"""Train or smoke-test the frozen Stage 3B PNA operator."""
from pathlib import Path

from ..runner import main_for_operator


def main() -> None:
    main_for_operator(Path(__file__).resolve().parent, "pna")


if __name__ == "__main__":
    main()
