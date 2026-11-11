"""PyCharm entry point. Offline only; source predictions/masks are read-only."""
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / 'experiments/downstream_sam_v1'))
sys.dont_write_bytecode = True
# Avoid nested BLAS threads during bootstrap and small linear solves.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# Normal PyCharm Run executes all core analyses and controlled coarsening.
MODE = 'formal'  # 'preflight', 'smoke', or 'formal'
RUN_NAME = 'formal_v1'  # Never reuse this name after changing the frozen code/protocol.

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'formal'], default=MODE)
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args()
    from diagnostics.pipeline import execute
    execute(ROOT, args.mode, args.run_name or ('smoke_v1' if args.mode == 'smoke' else RUN_NAME))
