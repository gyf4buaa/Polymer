"""Train or smoke-test Stage 3C R2."""
from ..runner import main_for_readout


def main() -> None:
    main_for_readout("r2_property")


if __name__ == "__main__":
    main()
