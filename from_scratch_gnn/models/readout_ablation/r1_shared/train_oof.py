"""Train or smoke-test Stage 3C R1."""
from ..runner import main_for_readout


def main() -> None:
    main_for_readout("r1_shared")


if __name__ == "__main__":
    main()
