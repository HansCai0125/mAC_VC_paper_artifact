"""IDE entry point: redraw Figures 5 and 6 from existing CSV results."""
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from plot_figures import plot_all

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "outputs" / "vc_v1" / "fig56_dense_20261023_v1"
OUTPUT_DIR = DATA_DIR / "my_replot"
INPUT_FILES = ("fig5_fragmentation_dense.csv", "fig6_canonical_dense.csv")


def main():
    for name in INPUT_FILES:
        if not (DATA_DIR / name).is_file():
            raise FileNotFoundError(f"Required plotting data missing: {DATA_DIR / name}")
    if OUTPUT_DIR.resolve() == DATA_DIR.resolve():
        raise ValueError("Choose an output directory different from the original data directory.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        shutil.copy2(DATA_DIR / name, OUTPUT_DIR / name)
    plot_all(OUTPUT_DIR)
    print(f"Figures 5 and 6 saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
