#!/usr/bin/env python3
"""Generate local UBFC-rPPG train configs for supported DL models.

Environment variables:
  UBFC_DATA_PATH, UBFC_CACHED_PATH, UBFC_LOG_PATH — dataset / cache / logs
  UBFC_DO_PREPROCESS — true/false (default: false after initial preprocess)
  UBFC_LOW_MEMORY — 1 enables ~2 GB GPU profile (default: 1)
  UBFC_EPOCHS — training epochs for smoke runs (default: 5 when low memory)
  UBFC_FORCE_CUDA — 1 keeps cuda:0 even for heavy models (default: cpu for those)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "train_configs" / "templates"
OUT_DIR = ROOT / "configs" / "train_configs" / "local"

DATA_PATH = os.environ.get(
    "UBFC_DATA_PATH", "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData"
)
CACHED_PATH = os.environ.get(
    "UBFC_CACHED_PATH",
    "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL",
)
LOG_PATH = os.environ.get("UBFC_LOG_PATH", "runs/exp_ubfc_dl")

LOW_MEMORY = os.environ.get("UBFC_LOW_MEMORY", "1").lower() in ("1", "true", "yes")
EPOCHS = int(os.environ.get("UBFC_EPOCHS", "5" if LOW_MEMORY else "30"))
DO_PREPROCESS = os.environ.get("UBFC_DO_PREPROCESS", "false").lower() in (
    "1",
    "true",
    "yes",
)
FORCE_CUDA = os.environ.get("UBFC_FORCE_CUDA", "0").lower() in ("1", "true", "yes")

# Models that usually need >2 GB VRAM at default clip size / resolution.
HEAVY_ON_2GB_GPU = frozenset({"PhysFormer", "RhythmFormer", "PhysMamba", "FactorizePhys"})

LOW_MEMORY_HEADER = """# UBFC-rPPG — local low-memory config (~2 GB GPU, e.g. GTX 960M).
# TRAIN/INFERENCE BATCH_SIZE=1 | EPOCHS={epochs} | DO_PREPROCESS={preprocess}
# CHUNK_LENGTH and RESIZE must match existing Preprocessed_DL cache (do not change here).
# NUM_WORKERS: fixed to 0 in main.py (not a YAML field).
# Heavy models ({heavy}): DEVICE defaults to cpu unless UBFC_FORCE_CUDA=1.
"""

TEMPLATES = [
    ("DeepPhys", "UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml"),
    ("Tscan", "UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml"),
    ("Physnet", "UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml"),
    ("EfficientPhys", "UBFC-rPPG_UBFC-rPPG_PURE_EFFICIENTPHYS.yaml"),
    ("PhysFormer", "UBFC-rPPG_UBFC-rPPG_PURE_PHYSFORMER_BASIC.yaml"),
    ("RhythmFormer", "UBFC-rPPG_UBFC-rPPG_PURE_RHYTHMFORMER_BASIC.yaml"),
    ("PhysMamba", "UBFC-rPPG_UBFC-rPPG_PURE_PHYSMAMBA.yaml"),
    ("FactorizePhys", "UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml"),
]


def patch_common(content: str, model_slug: str) -> str:
    content = re.sub(r"DATA_PATH:.*", f'DATA_PATH: "{DATA_PATH}"', content)
    content = re.sub(r"CACHED_PATH:.*", f'CACHED_PATH: "{CACHED_PATH}"', content)
    content = re.sub(
        r"DO_PREPROCESS: (True|False)",
        f"DO_PREPROCESS: {'True' if DO_PREPROCESS else 'False'}",
        content,
    )
    content = re.sub(r"\n\s+FILE_LIST_PATH:.*", "", content)
    content = re.sub(
        r"MODEL_FILE_NAME:.*",
        f"MODEL_FILE_NAME: UBFC_DL_{model_slug}",
        content,
        count=1,
    )
    content = re.sub(
        r"PATH: runs/exp\s*$",
        f"PATH: {LOG_PATH}",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(r"PATH: runs/exp_ubfc_dl\s*$", f"PATH: {LOG_PATH}", content, flags=re.MULTILINE)
    return content


def patch_low_memory(content: str, model_slug: str) -> str:
    content = re.sub(r"EPOCHS: \d+", f"EPOCHS: {EPOCHS}", content, count=1)
    content = re.sub(r"BATCH_SIZE: \d+", "BATCH_SIZE: 1", content)
    if model_slug in HEAVY_ON_2GB_GPU and not FORCE_CUDA:
        content = re.sub(
            r"DEVICE: cuda:0",
            "DEVICE: cpu  # ~2GB GPU: train on CPU (set UBFC_FORCE_CUDA=1 to try CUDA)",
            content,
        )
    return content


def patch_test_to_ubfc_val(content: str) -> str:
    idx = content.find("TEST:")
    if idx == -1:
        return content
    before, after = content[:idx], content[idx:]
    after = re.sub(r"DATASET: PURE", "DATASET: UBFC-rPPG", after, count=1)
    after = re.sub(
        r"BEGIN: 0\.0\n(\s+)END: 1\.0",
        r"BEGIN: 0.8\n\1END: 1.0",
        after,
        count=1,
    )
    return before + after


def patch_factorizephys_train_split(content: str) -> str:
    idx = content.find("VALID:")
    if idx == -1:
        return content
    train_block = content[:idx]
    train_block = re.sub(
        r"BEGIN: 0\.0\n(\s+)END: 1\.0",
        r"BEGIN: 0.0\n\1END: 0.8",
        train_block,
        count=1,
    )
    return train_block + content[idx:]


def build_header() -> str:
    heavy = ", ".join(sorted(HEAVY_ON_2GB_GPU))
    return LOW_MEMORY_HEADER.format(
        epochs=EPOCHS,
        preprocess="True" if DO_PREPROCESS else "False",
        heavy=heavy,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = build_header() if LOW_MEMORY else ""
    for model_slug, template_name in TEMPLATES:
        src = TEMPLATE_DIR / template_name
        text = src.read_text(encoding="utf-8")
        text = patch_common(text, model_slug)
        text = patch_test_to_ubfc_val(text)
        if model_slug == "FactorizePhys":
            text = patch_factorizephys_train_split(text)
        if LOW_MEMORY:
            text = patch_low_memory(text, model_slug)
        out = OUT_DIR / f"UBFC_DL_{model_slug}.yaml"
        out.write_text(header + text, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
