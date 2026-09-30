"""Retired automatic-calibration benchmark.

The last implementation is available in Git at 1d484d43d. Its workload included
built-in calibration and is not comparable with the current input-only runtime.
"""

if __name__ == '__main__':
    raise SystemExit('This benchmark requires the removed automatic GPU calibration path. '
                     'A replacement requires explicit callback support (Stages 2–4). '
                     'Use examples/input_only.py for input-path diagnostics.')
