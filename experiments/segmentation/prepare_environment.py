"""Check installed packages and download SAM weights; no conda environment changes."""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-local-deps", action="store_true", help="Explicit optional installation into third_party, never conda")
    parser.add_argument("--no-weights", action="store_true", help="Only applies to --install-local-deps")
    args = parser.parse_args()
    from downstream.setup_runtime import prepare_runtime, check_runtime_and_weights
    if args.install_local_deps:
        prepare_runtime(ROOT, download_weights=not args.no_weights)
    else:
        check_runtime_and_weights(ROOT, download_weights=True)
