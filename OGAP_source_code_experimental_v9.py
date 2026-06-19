#!/usr/bin/env python3
"""Backward-compatible entrypoint shim for the OGAP package.

The original 16k-line OGAP_source_code_experimental_v9.py was refactored into
the `ogap` package: ogap/legacy.py holds the legacy core verbatim, and
ogap/{models,nas,numerics,longitudinal,ood} hold the clean, tested modules.
This shim preserves the original CLI so every existing sbatch/.sh command runs
unchanged:

    python OGAP_source_code_experimental_v9.py <cmd> ...

New capabilities are available via `python -m ogap <cmd>` (see ogap/cli.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ogap.legacy import main  # noqa: E402

if __name__ == "__main__":
    main()
