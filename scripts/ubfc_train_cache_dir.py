#!/usr/bin/env python3
"""Print TRAIN.DATA.CACHED_PATH (after EXP_DATA_NAME expansion) for a YAML config."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()

    class ConfigArgs:
        config_file = str(args.config_file.resolve())

    # config.py prints "=> Merging ..." to stdout; suppress to keep output machine-parsable.
    with contextlib.redirect_stdout(io.StringIO()):
        config = get_config(ConfigArgs())
    print(config.TRAIN.DATA.CACHED_PATH)


if __name__ == "__main__":
    main()
