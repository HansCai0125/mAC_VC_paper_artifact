"""Small deterministic checks; no model inference or full experiment."""
import run_experiment  # paths/environment only
import unittest
import time
from diagnostics.common import atomic_json

if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(run_experiment.ROOT/'tests'))
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(run_experiment.ROOT/'validation/tests.json',dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start))
    raise SystemExit(0 if result.wasSuccessful() else 1)
