#!/usr/bin/env python3
"""Génère les YAML UBFC_DL_<Model>_<Roi> pour forehead, cheeks, glabella, nose_upper."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "train_configs" / "local"

ROIS = {
    "forehead": "forehead",
    "cheeks": "cheeks",
    "glabella": "glabella",
    "nose_upper": "nose_upper",
}

NDCHW_HEADER = """# {model_base} + ROI {roi_key}
# Généré par scripts/generate_ubfc_dl_roi_configs.py
# Lancer : bash scripts/run_ubfc_dl_roi.sh {model_base} {roi_key}
"""

PHYSNET_EXTRA = """
MODEL:
  DROP_RATE: 0.2
  NAME: Physnet
  PHYSNET:
    FRAME_NUM: 128
"""

TSCAN_EXTRA = """
MODEL:
  DROP_RATE: 0.2
  NAME: Tscan
  TSCAN:
    FRAME_DEPTH: 10
"""

DEEPPHYS_EXTRA = """
MODEL:
  DROP_RATE: 0.2
  NAME: DeepPhys
"""

NDCHW_BODY = """BASE: ['']
TOOLBOX_MODE: "train_and_test"
TRAIN:
  BATCH_SIZE: 1
  EPOCHS: 5
  LR: 9e-3
  MODEL_FILE_NAME: UBFC_DL_{model}_{roi_slug}
  PLOT_LOSSES_AND_LR: True
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData"
    CACHED_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 0.8
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized','Standardized']
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY : 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72
VALID:
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData"
    CACHED_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL"
    EXP_DATA_NAME: ""
    BEGIN: 0.8
    END: 1.0
    PREPROCESS:
      DATA_TYPE: [ 'DiffNormalized','Standardized' ]
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY : 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
  USE_LAST_EPOCH: True
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/RawData"
    CACHED_PATH: "/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/Preprocessed_DL"
    EXP_DATA_NAME: ""
    BEGIN: 0.8
    END: 1.0
    PREPROCESS:
      DATA_TYPE: [ 'DiffNormalized','Standardized' ]
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY : 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: runs/exp_ubfc_dl
{model_block}
INFERENCE:
  BATCH_SIZE: 1
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH:   ""
"""

NCDHW_BODY = NDCHW_BODY.replace("NDCHW", "NCDHW").replace(
    "['DiffNormalized','Standardized']", "['DiffNormalized']"
).replace(
    "[ 'DiffNormalized','Standardized' ]", "['DiffNormalized']"
).replace("CHUNK_LENGTH: 180", "CHUNK_LENGTH: 128").replace(
    "DYNAMIC_DETECTION_FREQUENCY : 30", "DYNAMIC_DETECTION_FREQUENCY : 32"
)


def roi_slug(roi_key: str) -> str:
    """forehead -> Forehead, nose_upper -> Nose_upper"""
    return "_".join(p[:1].upper() + p[1:] for p in roi_key.split("_"))


def write_config(model_base: str, roi_key: str) -> Path:
    slug = roi_slug(roi_key)
    if model_base == "Physnet":
        body = NCDHW_BODY
        extra = PHYSNET_EXTRA
    elif model_base == "Tscan":
        body = NDCHW_BODY
        extra = TSCAN_EXTRA
    else:
        body = NDCHW_BODY
        extra = DEEPPHYS_EXTRA

    text = NDCHW_HEADER.format(model_base=model_base, roi_key=roi_key)
    text += body.format(model=model_base, roi_slug=slug, model_block=extra)

    out = OUT / f"UBFC_DL_{model_base}_{slug}.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for roi_key in ROIS:
        for model in ("DeepPhys", "Tscan", "Physnet"):
            path = OUT / f"UBFC_DL_{model}_{roi_slug(roi_key)}.yaml"
            if path.exists():
                print(f"skip (exists): {path.relative_to(ROOT)}")
                continue
            p = write_config(model, roi_key)
            print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
