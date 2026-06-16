"""ROI à la volée pour l'entraînement deep learning (NDCHW / NCDHW)."""

import os

import cv2
import numpy as np
import torch

from tools.Selection_ROI import apply_roi_to_frames
from tools.Selection_ROI import apply_roi_to_ndchw_batch
from tools.Selection_ROI import resolve_roi_name


def init_dl_roi(config, model_label: str):
    """Lit RPPG_ROI / taille de patch ; affiche un message si ROI actif."""
    roi_name = resolve_roi_name(os.environ.get("RPPG_ROI", "full_face"))
    roi_patch_size = int(os.environ.get("RPPG_ROI_PATCH_SIZE", "64"))
    if config.TOOLBOX_MODE == "train_and_test":
        resize = config.TRAIN.DATA.PREPROCESS.RESIZE
    else:
        resize = config.TEST.DATA.PREPROCESS.RESIZE
    roi_target_size = int(resize.H)
    if roi_name != "full_face":
        print(
            f"{model_label} ROI: {roi_name} "
            f"(patch={roi_patch_size}px -> {roi_target_size}x{roi_target_size})"
        )
    return roi_name, roi_patch_size, roi_target_size


def maybe_apply_roi_ndchw(data, roi_name, roi_patch_size, roi_target_size):
    if roi_name == "full_face":
        return data
    batch = apply_roi_to_ndchw_batch(
        data.cpu().numpy(),
        roi_name,
        out_size=roi_patch_size,
        target_size=roi_target_size,
    )
    return torch.from_numpy(batch).to(data.device, dtype=data.dtype)


def apply_roi_to_ncdhw_clip(clip, roi_name, out_size=64, target_size=72):
    """clip: (C, D, H, W) — format PhysNet."""
    roi_name = resolve_roi_name(roi_name)
    if roi_name == "full_face":
        return clip
    frames = np.transpose(clip, (1, 2, 3, 0))
    roi_frames = apply_roi_to_frames(frames, roi_name, out_size=out_size)
    resized = np.stack(
        [
            cv2.resize(
                frame,
                (target_size, target_size),
                interpolation=cv2.INTER_AREA,
            )
            for frame in roi_frames
        ]
    )
    return np.transpose(resized, (3, 0, 1, 2))


def apply_roi_to_ncdhw_batch(batch, roi_name, out_size=64, target_size=72):
    """batch: (N, C, D, H, W)."""
    roi_name = resolve_roi_name(roi_name)
    if roi_name == "full_face":
        return batch
    out = np.empty_like(batch)
    for n in range(batch.shape[0]):
        out[n] = apply_roi_to_ncdhw_clip(
            batch[n], roi_name, out_size=out_size, target_size=target_size
        )
    return out


def maybe_apply_roi_ncdhw(data, roi_name, roi_patch_size, roi_target_size):
    if roi_name == "full_face":
        return data
    batch = apply_roi_to_ncdhw_batch(
        data.cpu().numpy(),
        roi_name,
        out_size=roi_patch_size,
        target_size=roi_target_size,
    )
    return torch.from_numpy(batch).to(data.device, dtype=data.dtype)
