"""Pytest configuration loaded before test modules import third-party stacks.

Limit BLAS thread pools before NumPy (and pandas/sklearn) initialize. On macOS,
unbounded Accelerate/OpenBLAS threading has been linked to fatal SIGFPE during
NumPy's import-time ``polyfit`` sanity check; single-threaded mode avoids that
class of failures without affecting numerical results in unit tests.
"""

from __future__ import annotations

import os

for _key in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

del _key
