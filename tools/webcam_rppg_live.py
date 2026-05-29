#!/usr/bin/env python3
"""
Prototype rPPG en temps réel depuis la webcam.

Pipeline :
    Webcam -> détection visage -> ROI (défaut Zhao 2024) -> méthode rPPG -> HR (FFT) -> affichage + CSV

Usage :
    python tools/webcam_rppg_live.py --method pos
    python tools/webcam_rppg_live.py --method pos --roi zhao2024_motion_robust
    python tools/webcam_rppg_live.py --method pos --garmin
    python tools/webcam_rppg_live.py --method pos --garmin --garmin-calibrate
    python tools/webcam_rppg_live.py --method chrom
    python tools/webcam_rppg_live.py --method green

Touches :
    q ou Echap : quitter immédiatement
    r : reset du buffer
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Racine du dépôt pour importer rPPG-Toolbox sans installation editable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Modules rPPG-Toolbox chargés à la demande (évite un import lourd si --help).
_TOOLBOX_IMPORT_ERROR: Optional[Exception] = None


def _ensure_toolbox_imports() -> None:
    """Charge les méthodes non supervisées et l'estimation FFT HR du toolbox."""
    global _TOOLBOX_IMPORT_ERROR
    if _TOOLBOX_IMPORT_ERROR is not None:
        raise _TOOLBOX_IMPORT_ERROR
    try:
        from evaluation.post_process import _calculate_fft_hr  # noqa: F401
        from unsupervised_methods.methods.CHROME_DEHAAN import (  # noqa: F401
            CHROME_DEHAAN,
        )
        from unsupervised_methods.methods.GREEN import GREEN  # noqa: F401
        from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: F401
    except ImportError as exc:
        _TOOLBOX_IMPORT_ERROR = exc
        print(
            "Erreur d'import rPPG-Toolbox. Vérifiez l'environnement (scipy, scikit-image, ...)\n"
            "et lancez depuis la racine du dépôt :\n"
            "  python tools/webcam_rppg_live.py --method pos\n"
            f"Détail : {exc}",
            file=sys.stderr,
        )
        raise


# ---------------------------------------------------------------------------
# Méthodes rPPG (extensible)
# ---------------------------------------------------------------------------

# Alias CLI -> identifiant interne
METHOD_ALIASES = {
    "pos": "pos",
    "chrom": "chrom",
    "chrome": "chrom",
    "green": "green",
    "deep_model": "deep_model",
}

# Méthodes réellement disponibles dans ce prototype
SUPPORTED_METHODS = {"pos", "chrom", "green"}


class MethodNotAvailableError(ValueError):
    """Levée quand la méthode demandée n'est pas implémentée ou inconnue."""


def normalize_method_name(method_name: str) -> str:
    key = method_name.strip().lower()
    if key not in METHOD_ALIASES:
        known = ", ".join(sorted(METHOD_ALIASES))
        raise MethodNotAvailableError(
            f"Méthode inconnue '{method_name}'. Méthodes reconnues : {known}"
        )
    return METHOD_ALIASES[key]


def min_frames_required(fs: float, method_name: str) -> int:
    """Nombre minimal de frames avant une estimation HR fiable."""
    method = normalize_method_name(method_name)
    if method in ("pos", "chrom"):
        # Fenêtre glissante ~1.6 s dans POS_WANG / CHROME_DEHAAN
        return int(math.ceil(1.6 * fs)) + 10
    if method == "green":
        return max(int(fs * 2), 30)
    return max(int(fs * 2), 30)


def compute_method_output(
    frames: np.ndarray, fs: float, method_name: str
) -> np.ndarray:
    """
    Applique une méthode rPPG non supervisée sur une séquence de frames RGB.

    Args:
        frames: (T, H, W, 3) float ou uint8, RGB
        fs: fréquence d'échantillonnage estimée (Hz)
        method_name: alias CLI (pos, chrom, green, deep_model, ...)

    Returns:
        Signal BVP 1D de longueur T (ou proche).
    """
    method = normalize_method_name(method_name)

    if method == "deep_model":
        raise MethodNotAvailableError(
            "La méthode 'deep_model' n'est pas encore branchée dans ce prototype. "
            "Utilisez pos, chrom ou green pour l'instant."
        )

    if method not in SUPPORTED_METHODS:
        raise MethodNotAvailableError(
            f"Méthode '{method_name}' non disponible dans ce prototype."
        )

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames doit avoir la forme (T, H, W, 3), reçu {frames.shape}"
        )

    if len(frames) < 9:
        raise ValueError(
            f"Pas assez de frames pour la méthode ({len(frames)} < 9)."
        )

    frames = np.asarray(frames, dtype=np.float64)
    _ensure_toolbox_imports()
    from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
    from unsupervised_methods.methods.GREEN import GREEN
    from unsupervised_methods.methods.POS_WANG import POS_WANG

    try:
        if method == "pos":
            return np.asarray(POS_WANG(frames, fs), dtype=np.float64).reshape(-1)
        if method == "chrom":
            return np.asarray(CHROME_DEHAAN(frames, fs), dtype=np.float64).reshape(-1)
        if method == "green":
            return np.asarray(GREEN(frames), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            f"Échec d'exécution de la méthode '{method}' : {exc}"
        ) from exc

    raise MethodNotAvailableError(f"Méthode '{method_name}' non gérée.")


# Taille de chaque sous-région (aligné sur tools/Selection_ROI.apply_roi_to_frames).
ROI_PATCH_SIZE = 64

# ROI par défaut : meilleurs résultats rapportés avec POS + Zhao 2024.
DEFAULT_ROI = "zhao2024_motion_robust"

ROI_CLI_ALIASES = {
    "zhao2024": "zhao2024_motion_robust",
    "zhao": "zhao2024_motion_robust",
    "full": "full_face",
}

# Bande FFT recommandée dans evaluation/post_process.py (papier NeurIPS 2023).
FFT_LOW_PASS_HZ = 0.75
FFT_HIGH_PASS_HZ = 2.5


def preprocess_bvp_for_hr(bvp: np.ndarray, fs: float) -> np.ndarray:
    """
    Post-traitement BVP avant FFT HR (aligné sur calculate_metric_per_video).
    Detrend + passe-bande [0.75, 2.5] Hz.
    """
    import scipy.signal
    from evaluation.post_process import _detrend

    signal_1d = np.asarray(bvp, dtype=np.float64).reshape(-1)
    if len(signal_1d) < 9:
        raise ValueError("Signal BVP trop court pour le post-traitement.")
    signal_1d = np.asarray(_detrend(signal_1d, 100), dtype=np.float64).reshape(-1)
    nyq = fs / 2.0
    low = FFT_LOW_PASS_HZ / nyq
    high = FFT_HIGH_PASS_HZ / nyq
    if low <= 0 or high >= 1 or low >= high:
        return signal_1d
    b, a = scipy.signal.butter(1, [low, high], btype="bandpass")
    return scipy.signal.filtfilt(b, a, signal_1d)


def estimate_hr_from_bvp(
    bvp: np.ndarray,
    fs: float,
    use_antiharmonic: bool = True,
) -> float:
    """HR en bpm via FFT (option anti-harmonique pour éviter demi/double FC)."""
    processed = preprocess_bvp_for_hr(bvp, fs)
    if use_antiharmonic:
        hr = estimate_hr_fft_antiharmonic(
            processed, fs, FFT_LOW_PASS_HZ, FFT_HIGH_PASS_HZ
        )
    else:
        _ensure_toolbox_imports()
        from evaluation.post_process import _calculate_fft_hr

        hr = float(
            _calculate_fft_hr(
                processed,
                fs=fs,
                low_pass=FFT_LOW_PASS_HZ,
                high_pass=FFT_HIGH_PASS_HZ,
            )
        )
    if not np.isfinite(hr) or hr <= 0:
        raise ValueError("HR FFT invalide.")
    return float(hr)


def estimate_hr_fft_antiharmonic(
    signal_1d: np.ndarray,
    fs: float,
    low_pass: float,
    high_pass: float,
) -> float:
    """
    FFT HR avec correction harmonique : évite de choisir la demi-FC (ex. 51 au lieu de 90)
    ou le double (ex. 100 au lieu de 50).
    """
    import scipy.signal
    from evaluation.post_process import _next_power_of_2

    sig = np.asarray(signal_1d, dtype=np.float64).reshape(-1)
    n = len(sig)
    if n < 9:
        raise ValueError("Signal trop court pour FFT anti-harmonique.")

    nfft = _next_power_of_2(n)
    freqs, pxx = scipy.signal.periodogram(sig, fs=fs, nfft=nfft, detrend=False)
    pxx = np.asarray(pxx).reshape(-1)

    band = (freqs >= low_pass) & (freqs <= high_pass)
    f_band = freqs[band]
    p_band = pxx[band]
    if len(f_band) == 0:
        raise ValueError("Bande FFT vide.")

    peak_indices = np.argsort(p_band)[::-1][:6]
    candidates: List[Tuple[float, float, float]] = []
    for idx in peak_indices:
        freq_hz = float(f_band[idx])
        hr_bpm = freq_hz * 60.0
        power = float(p_band[idx])
        if 40.0 <= hr_bpm <= 200.0:
            candidates.append((hr_bpm, freq_hz, power))

    if not candidates:
        raise ValueError("Aucun pic FFT plausible.")

    best_hr, best_f, best_power = candidates[0]

    for hr_bpm, freq_hz, power in candidates[1:]:
        # Pic dominant = harmonique (2x) d'un autre pic plus fondamental
        if abs(best_hr - 2.0 * hr_bpm) < 10.0 and power > 0.2 * best_power:
            best_hr, best_f, best_power = hr_bpm, freq_hz, power

    for hr_bpm, freq_hz, power in candidates:
        # Pic dominant = sous-harmonique (moitié) d'un pic plus fort à 2x
        if abs(hr_bpm - 2.0 * best_hr) < 10.0 and power > 0.35 * best_power:
            best_hr, best_f, best_power = hr_bpm, freq_hz, power

    return best_hr


def resolve_processing_fs(
    measured_fps: Optional[float],
    target_fps: float,
    min_ratio: float = 0.75,
) -> Tuple[float, str]:
    """
    fs unique pour POS/CHROM et FFT : mesuré si fiable, sinon cible.
    Retourne (fs, message statut).
    """
    if measured_fps is None or measured_fps <= 0:
        return target_fps, f"fs={target_fps:.0f} Hz (cible, buffer court)"

    if measured_fps < target_fps * min_ratio:
        return (
            measured_fps,
            f"ATTENTION fps={measured_fps:.1f} — fs reel utilise pour POS/FFT",
        )

    if measured_fps > target_fps * 1.35:
        return (
            measured_fps,
            f"fs={measured_fps:.1f} Hz (mesure > cible {target_fps:.0f})",
        )

    return measured_fps, f"fs={measured_fps:.1f} Hz (mesure)"


class HREstimator:
    """Lissage + rejet des sauts aberrants (HR figée ou spike)."""

    def __init__(self, window: int = 7, max_jump_bpm: float = 12.0):
        self._window = max(1, window)
        self._max_jump = max(3.0, max_jump_bpm)
        self._history: Deque[float] = deque(maxlen=self._window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, hr: float) -> float:
        if self._history:
            ref = float(np.median(list(self._history)))
            if abs(hr - ref) > self._max_jump:
                return ref
        self._history.append(hr)
        return float(np.median(list(self._history)))

    @property
    def ready(self) -> bool:
        return len(self._history) > 0


class GarminBiasCalibrator:
    """Correction adaptative : rPPG + biais median(Garmin - rPPG)."""

    def __init__(self, window: int = 40, min_samples: int = 8):
        self._window = max(5, window)
        self._min_samples = max(3, min_samples)
        self._offsets: Deque[float] = deque(maxlen=self._window)
        self.bias_bpm: float = 0.0

    def reset(self) -> None:
        self._offsets.clear()
        self.bias_bpm = 0.0

    def update(self, rppg_hr: float, garmin_hr: float) -> None:
        self._offsets.append(float(garmin_hr) - float(rppg_hr))
        if len(self._offsets) >= self._min_samples:
            self.bias_bpm = float(np.median(list(self._offsets)))

    def apply(self, rppg_hr: Optional[float]) -> Optional[float]:
        if rppg_hr is None:
            return None
        if len(self._offsets) < self._min_samples:
            return rppg_hr
        return float(rppg_hr) + self.bias_bpm

    @property
    def is_ready(self) -> bool:
        return len(self._offsets) >= self._min_samples


# ---------------------------------------------------------------------------
# Visage / ROI
# ---------------------------------------------------------------------------


def normalize_roi_name(roi_name: str) -> str:
    key = roi_name.strip().lower()
    key = ROI_CLI_ALIASES.get(key, key)
    from tools.Selection_ROI import ROI_BOXES

    if key not in ROI_BOXES:
        raise ValueError(
            f"ROI inconnue '{roi_name}'. Choix : {', '.join(sorted(ROI_BOXES))}"
        )
    return key


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    """Détection Haar Cascade + lissage de la boîte (moins de jitter ROI)."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        smooth_alpha: float = 0.25,
    ):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(
                f"Impossible de charger le classificateur Haar : {cascade_path}"
            )
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._smooth_alpha = smooth_alpha
        self._smooth_box: Optional[Tuple[float, float, float, float]] = None

    def reset(self) -> None:
        self._smooth_box = None

    def _smooth(self, face: FaceDetection) -> FaceDetection:
        box = (float(face.x), float(face.y), float(face.w), float(face.h))
        if self._smooth_box is None:
            self._smooth_box = box
        else:
            a = self._smooth_alpha
            self._smooth_box = tuple(
                (1.0 - a) * old + a * new for old, new in zip(self._smooth_box, box)
            )
        x, y, w, h = self._smooth_box
        return FaceDetection(int(x), int(y), int(w), int(h))

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return self._smooth(FaceDetection(int(x), int(y), int(w), int(h)))


def crop_face_rgb(frame_bgr: np.ndarray, face: FaceDetection) -> np.ndarray:
    """Crop RGB du visage détecté (repère local pour Selection_ROI)."""
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(face.x, 0)
    y1 = max(face.y, 0)
    x2 = min(face.x + face.w, w_img)
    y2 = min(face.y + face.h, h_img)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Boîte visage invalide.")
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)


def extract_method_roi(
    frame_bgr: np.ndarray,
    face: FaceDetection,
    roi_name: str,
    patch_size: int = ROI_PATCH_SIZE,
) -> np.ndarray:
    """
    Extrait la ROI pour la méthode rPPG (même logique que le toolbox offline).

    Utilise tools/Selection_ROI.apply_roi_to_frames sur le crop visage :
    - full_face : visage entier redimensionné
    - zhao2024_motion_robust : 7 sous-régions concaténées horizontalement
    """
    from tools.Selection_ROI import apply_roi_to_frames

    face_rgb = crop_face_rgb(frame_bgr, face)
    roi_name = normalize_roi_name(roi_name)

    if roi_name == "full_face":
        return cv2.resize(
            face_rgb,
            (patch_size, patch_size),
            interpolation=cv2.INTER_AREA,
        )

    # (1, H, W, 3) -> (H, W', 3) avec W' = n_patches * patch_size
    batch = face_rgb[np.newaxis, ...]
    processed = apply_roi_to_frames(batch, roi_name, out_size=patch_size)
    return processed[0]


def face_subregion_rects(
    face: FaceDetection, roi_name: str
) -> List[Tuple[int, int, int, int]]:
    """Rectangles (x, y, w, h) image entière pour l'overlay webcam."""
    from tools.Selection_ROI import ROI_BOXES

    roi_name = normalize_roi_name(roi_name)
    boxes = ROI_BOXES.get(roi_name)
    if boxes is None:
        return [(face.x, face.y, face.w, face.h)]

    rects = []
    for x1, y1, x2, y2 in boxes:
        rx = face.x + int(x1 * face.w)
        ry = face.y + int(y1 * face.h)
        rw = int((x2 - x1) * face.w)
        rh = int((y2 - y1) * face.h)
        rects.append((rx, ry, max(rw, 1), max(rh, 1)))
    return rects


# ---------------------------------------------------------------------------
# Buffer temporel
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Buffer glissant de frames RGB + timestamps."""

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._frames: Deque[np.ndarray] = deque()
        self._timestamps: Deque[float] = deque()

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._timestamps.clear()

    def add(self, frame_rgb: np.ndarray, timestamp: float) -> None:
        self._frames.append(frame_rgb)
        self._timestamps.append(timestamp)
        self._trim(timestamp)

    def _trim(self, now: float) -> None:
        while self._timestamps and (now - self._timestamps[0]) > self.max_seconds:
            self._frames.popleft()
            self._timestamps.popleft()

    def as_array(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0,), dtype=np.float64)
        return np.stack(list(self._frames), axis=0)

    def estimated_fps(self) -> Optional[float]:
        if len(self._timestamps) < 2:
            return None
        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return None
        return (len(self._timestamps) - 1) / duration

    def buffer_seconds(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]


# ---------------------------------------------------------------------------
# Webcam (Windows : privilégier DirectShow)
# ---------------------------------------------------------------------------


def _camera_backend_candidates() -> List[Tuple[str, int]]:
    if sys.platform == "win32":
        backends: List[Tuple[str, int]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", cv2.CAP_MSMF))
        backends.append(("ANY", cv2.CAP_ANY))
        return backends
    return [("ANY", cv2.CAP_ANY)]


def configure_camera_capture(
    cap: cv2.VideoCapture,
    target_fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> None:
    """Demande 30 fps et résolution fixes (le driver peut ignorer partiellement)."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def measure_capture_fps(cap: cv2.VideoCapture, n_frames: int = 30) -> Optional[float]:
    """Mesure le FPS réel juste après ouverture."""
    t0 = time.perf_counter()
    count = 0
    for _ in range(n_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            count += 1
    elapsed = time.perf_counter() - t0
    if count < 2 or elapsed <= 0:
        return None
    return (count - 1) / elapsed


def _try_open_capture(
    index: int,
    backend_name: str,
    backend: int,
    target_fps: float,
    width: int,
    height: int,
) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    configure_camera_capture(cap, target_fps=target_fps, width=width, height=height)
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            reported = cap.get(cv2.CAP_PROP_FPS)
            measured = measure_capture_fps(cap, n_frames=20)
            fps_info = f"cible={target_fps:.0f}"
            if measured is not None:
                fps_info += f", mesure={measured:.1f}"
            if reported and reported > 0:
                fps_info += f", driver={reported:.1f}"
            print(
                f"Webcam ouverte : index={index}, backend={backend_name}, "
                f"{w}x{h}, {fps_info}"
            )
            return cap
    cap.release()
    return None


def open_webcam(
    camera_index: int,
    auto_fallback: bool = True,
    target_fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> cv2.VideoCapture:
    """
    Ouvre la webcam. Sur Windows teste DSHOW puis MSMF (souvent requis).
    Si auto_fallback, essaie aussi les index 0..5.
    """
    indices: List[int] = [camera_index]
    if auto_fallback:
        indices.extend(i for i in range(6) if i not in indices)

    last_error = ""
    for idx in indices:
        for backend_name, backend in _camera_backend_candidates():
            cap = _try_open_capture(
                idx, backend_name, backend, target_fps, width, height
            )
            if cap is not None:
                return cap
            last_error = f"index {idx} / {backend_name}"

    raise RuntimeError(
        "Impossible d'ouvrir une webcam.\n"
        f"Dernier essai : {last_error}\n"
        "Essayez :\n"
        "  python tools/webcam_rppg_live.py --list-cameras\n"
        "  python tools/webcam_rppg_live.py --camera 1\n"
        "Fermez Zoom/Teams, vérifiez les autorisations caméra Windows."
    )


def list_available_cameras(max_index: int = 6) -> List[int]:
    """Liste les index de webcam qui renvoient au moins une image."""
    print("Recherche des webcams disponibles...")
    found: List[int] = []
    for idx in range(max_index):
        for backend_name, backend in _camera_backend_candidates():
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None and frame.size > 0:
                print(f"  [OK] index {idx} (backend {backend_name})")
                found.append(idx)
                break
        else:
            print(f"  [--] index {idx}")
    return found


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------


def draw_face_overlay(
    frame_bgr: np.ndarray,
    face: Optional[FaceDetection],
    roi_name: str,
    status: str,
) -> np.ndarray:
    display = frame_bgr.copy()
    if face is not None:
        cv2.rectangle(
            display,
            (face.x, face.y),
            (face.x + face.w, face.y + face.h),
            (0, 255, 0),
            1,
        )
        for rx, ry, rw, rh in face_subregion_rects(face, roi_name):
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)
        label = normalize_roi_name(roi_name)
        if label == "zhao2024_motion_robust":
            label = "Zhao 2024"
        cv2.putText(
            display,
            f"ROI: {label}",
            (face.x, max(face.y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            display,
            "Aucun visage detecte",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        status,
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def compute_hr_difference(
    rppg_hr: Optional[float], garmin_hr: Optional[float]
) -> Optional[float]:
    if rppg_hr is None or garmin_hr is None:
        return None
    return rppg_hr - garmin_hr


def render_stats_panel(
    width: int,
    height: int,
    method: str,
    roi_name: str,
    rppg_hr: Optional[float],
    garmin_hr: Optional[float],
    fps: Optional[float],
    buffer_sec: float,
    message: str,
    garmin_status: Optional[str] = None,
    rppg_hr_raw: Optional[float] = None,
    calib_bias: Optional[float] = None,
    processing_fs: Optional[float] = None,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    roi_label = normalize_roi_name(roi_name)
    if roi_label == "zhao2024_motion_robust":
        roi_label = "Zhao 2024 (7 patches)"
    diff = compute_hr_difference(rppg_hr, garmin_hr)
    lines = [
        "rPPG live",
        f"Method: {method}",
        f"ROI: {roi_label}",
        (
            f"HR rPPG: {rppg_hr:.1f} bpm"
            if rppg_hr is not None
            else "HR rPPG: --"
        ),
        (
            f"  (brut {rppg_hr_raw:.0f})"
            if rppg_hr_raw is not None and rppg_hr is not None
            and abs(rppg_hr_raw - rppg_hr) > 0.5
            else ""
        ),
        (
            f"  calib +{calib_bias:.1f}"
            if calib_bias is not None and abs(calib_bias) > 0.1
            else ""
        ),
        f"HR Garmin: {garmin_hr:.0f} bpm" if garmin_hr is not None else "HR Garmin: --",
        (
            f"Diff (rPPG-Garmin): {diff:+.1f} bpm"
            if diff is not None
            else "Diff (rPPG-Garmin): --"
        ),
        f"FPS buf.: {fps:.2f}" if fps is not None else "FPS buf.: --",
        (
            f"fs trait.: {processing_fs:.1f} Hz"
            if processing_fs is not None
            else ""
        ),
        f"Buffer: {buffer_sec:.1f} s",
        "",
        message,
    ]
    lines = [line for line in lines if line]
    if garmin_status:
        lines.append(garmin_status[:60])
    lines.extend(["", "q / Echap = quitter", "r = reset buffer"])
    y = 28
    for line in lines:
        cv2.putText(
            panel,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 28
    return panel


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "timestamp",
    "method",
    "roi",
    "rppg_hr_bpm",
    "rppg_hr_raw_bpm",
    "calibration_offset_bpm",
    "garmin_hr_bpm",
    "difference_bpm",
    "processing_fs_hz",
    "estimated_fps",
    "buffer_seconds",
]


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def _csv_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{decimals}f}"


def append_csv_row(
    path: Path,
    method: str,
    roi_name: str,
    rppg_hr: Optional[float],
    rppg_hr_raw: Optional[float],
    calib_offset: Optional[float],
    garmin_hr: Optional[float],
    processing_fs: Optional[float],
    fps: Optional[float],
    buffer_sec: float,
) -> None:
    diff = compute_hr_difference(rppg_hr, garmin_hr)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "method": method,
                "roi": normalize_roi_name(roi_name),
                "rppg_hr_bpm": _csv_number(rppg_hr),
                "rppg_hr_raw_bpm": _csv_number(rppg_hr_raw),
                "calibration_offset_bpm": _csv_number(calib_offset),
                "garmin_hr_bpm": _csv_number(garmin_hr, decimals=0),
                "difference_bpm": _csv_number(diff),
                "processing_fs_hz": _csv_number(processing_fs),
                "estimated_fps": "" if fps is None else f"{fps:.3f}",
                "buffer_seconds": f"{buffer_sec:.3f}",
            }
        )


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype rPPG webcam en temps réel (rPPG-Toolbox)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pos",
        help="Méthode rPPG : pos (défaut), chrom, green, deep_model (non implémenté)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=DEFAULT_ROI,
        help=(
            "ROI sur le visage (défaut zhao2024_motion_robust). "
            "Alias : zhao2024, full_face, forehead, ..."
        ),
    )
    parser.add_argument(
        "--roi-patch-size",
        type=int,
        default=ROI_PATCH_SIZE,
        help="Taille de chaque sous-région en pixels (défaut 64, comme Selection_ROI)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Index de la webcam")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lister les webcams disponibles puis quitter",
    )
    parser.add_argument(
        "--no-camera-fallback",
        action="store_true",
        help="Ne pas essayer d'autres index si --camera echoue",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=15.0,
        help="Durée du buffer glissant (secondes). Plus long = FFT plus stable (défaut 15).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="FPS cible webcam + reference POS/FFT (defaut 30)",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=None,
        help="Deprecated: utiliser --target-fps (defaut 30)",
    )
    parser.add_argument(
        "--max-hr-jump",
        type=float,
        default=12.0,
        help="Rejet des sauts HR rPPG > N bpm entre deux estimations",
    )
    parser.add_argument(
        "--no-antiharmonic",
        action="store_true",
        help="Desactiver la correction FFT anti-harmonique",
    )
    parser.add_argument(
        "--hr-smooth-window",
        type=int,
        default=7,
        help="Nombre d'estimations HR pour la médiane affichée (lissage temporel)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="",
        help="Chemin CSV (défaut : outputs/webcam_rppg_live_<method>_<timestamp>.csv)",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Intervalle minimum entre deux lignes CSV (secondes)",
    )
    parser.add_argument(
        "--hr-update-interval",
        type=float,
        default=1.0,
        help="Intervalle entre deux recalculs HR (secondes, défaut 1.0)",
    )
    parser.add_argument(
        "--garmin",
        action="store_true",
        help="Lire la HR Garmin Forerunner en BLE (mode diffuser la FC)",
    )
    parser.add_argument(
        "--garmin-scan-timeout",
        type=float,
        default=25.0,
        help="Timeout scan BLE Garmin (secondes, defaut 25)",
    )
    parser.add_argument(
        "--garmin-address",
        type=str,
        default="",
        help="Adresse MAC Bluetooth de la montre (si le scan auto echoue)",
    )
    parser.add_argument(
        "--garmin-debug",
        action="store_true",
        help="Afficher les peripheriques BLE detectes dans le terminal",
    )
    parser.add_argument(
        "--garmin-scan-devices",
        action="store_true",
        help="Scanner les appareils BLE puis quitter (depannage Garmin)",
    )
    parser.add_argument(
        "--garmin-calibrate",
        action="store_true",
        help="Corriger le biais rPPG avec Garmin (necessite --garmin)",
    )
    parser.add_argument(
        "--calibrate-window",
        type=int,
        default=40,
        help="Fenetre (echantillons) pour la calibration biais Garmin",
    )
    return parser.parse_args()


def default_csv_path(method: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"webcam_rppg_live_{method}_{stamp}.csv"


def run() -> int:
    args = parse_args()

    if args.list_cameras:
        found = list_available_cameras()
        if not found:
            print("\nAucune webcam detectee.")
            return 1
        print(f"\nUtilisez par ex. : --camera {found[0]}")
        return 0

    if args.garmin_scan_devices:
        try:
            from tools.garmin_ble_reader import print_ble_scan_report
        except ImportError:
            print(
                f"bleak absent dans {sys.executable}\n"
                "  conda activate rppg-toolbox   OU   pip install bleak",
                file=sys.stderr,
            )
            return 1
        return print_ble_scan_report(timeout=args.garmin_scan_timeout)

    try:
        method = normalize_method_name(args.method)
    except MethodNotAvailableError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    if method == "deep_model":
        print(
            "Erreur : la méthode 'deep_model' n'est pas encore disponible dans ce prototype.",
            file=sys.stderr,
        )
        return 1

    try:
        roi_name = normalize_roi_name(args.roi)
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    target_fps = float(args.target_fps)
    if args.fallback_fps is not None:
        target_fps = float(args.fallback_fps)

    use_garmin_calibrate = bool(args.garmin_calibrate)
    if use_garmin_calibrate and not args.garmin:
        print(
            "Avertissement : --garmin-calibrate necessite --garmin. Option ignoree.",
            file=sys.stderr,
        )
        use_garmin_calibrate = False

    use_antiharmonic = not args.no_antiharmonic

    csv_path = Path(args.csv_output) if args.csv_output else default_csv_path(method)
    init_csv(csv_path)
    print(f"CSV : {csv_path}")

    garmin_reader = None
    if args.garmin:
        try:
            from tools.garmin_ble_reader import GarminBLEReader

            print(
                "Etape 1/2 : Garmin BLE (comme test_garmin_ble_hr.py, SANS webcam).\n"
                "  Montre : Diffuser la frequence cardiaque | BLE telephone OFF"
            )
            garmin_reader = GarminBLEReader(
                scan_timeout=args.garmin_scan_timeout,
                device_address=args.garmin_address or None,
                debug=args.garmin_debug,
                on_status=lambda msg: print(f"[Garmin] {msg}"),
            )
            garmin_reader.start()
            wait_until = time.perf_counter() + args.garmin_scan_timeout
            while time.perf_counter() < wait_until:
                if garmin_reader.is_connected:
                    print(f"[Garmin] Connectee : {garmin_reader.device_name}")
                    break
                time.sleep(0.4)
            else:
                print(
                    "[Garmin] Pas encore connectee — la webcam demarre quand meme "
                    "(reconnexion en arriere-plan)."
                )
        except ImportError:
            print(
                "\n*** ERREUR : module 'bleak' absent dans cet environnement Python ***\n"
                f"  Python utilise : {sys.executable}\n"
                "  Vous etes probablement dans (base) au lieu de (rppg-toolbox).\n\n"
                "  Solution A (recommandee) :\n"
                "    conda activate rppg-toolbox\n"
                "    python tools/webcam_rppg_live.py --method pos --garmin\n\n"
                "  Solution B (installer bleak ici) :\n"
                "    pip install bleak\n",
                file=sys.stderr,
            )
            return 1

    print(f"Etape 2/2 : Ouverture webcam (cible {target_fps:.0f} fps)...")
    try:
        cap = open_webcam(
            args.camera,
            auto_fallback=not args.no_camera_fallback,
            target_fps=target_fps,
        )
    except RuntimeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    # Imports rPPG lourds seulement au premier calcul HR (pas avant la webcam).
    print("Webcam prete. Imports rPPG au premier calcul de HR.")

    try:
        detector = FaceDetector()
    except RuntimeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        cap.release()
        return 1

    buffer = FrameBuffer(max_seconds=args.buffer_seconds)
    hr_estimator = HREstimator(
        window=args.hr_smooth_window,
        max_jump_bpm=args.max_hr_jump,
    )
    garmin_calibrator = (
        GarminBiasCalibrator(window=args.calibrate_window)
        if use_garmin_calibrate
        else None
    )
    min_buffer_fill_ratio = 0.85

    last_face: Optional[FaceDetection] = None
    last_hr: Optional[float] = None
    last_hr_raw: Optional[float] = None
    processing_fs: Optional[float] = None
    fs_status = ""
    status_msg = "Initialisation..."
    last_hr_compute_t = 0.0
    last_csv_log_t = 0.0

    print(
        "Conseil précision : restez immobile, visage bien éclairé, "
        f"attendez ~{args.buffer_seconds:.0f} s après le démarrage."
    )

    win_cam = "Webcam rPPG"
    win_stats = "rPPG Stats"
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_stats, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cam, 960, 540)

    patch_size = max(16, int(args.roi_patch_size))

    print(
        f"Méthode : {method} | ROI : {roi_name} | Buffer : {args.buffer_seconds}s | "
        f"FPS cible : {target_fps:.0f} | Anti-harm. : {'oui' if use_antiharmonic else 'non'} | "
        f"Garmin : {'oui' if args.garmin else 'non'} | "
        f"Calib. : {'oui' if use_garmin_calibrate else 'non'} | "
        f"Touches : q ou Echap=quitter, r=reset buffer"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Parfois une frame rate sur Windows : réessayer une fois
                ok, frame = cap.read()
            if not ok or frame is None:
                status_msg = "Erreur lecture webcam"
                break

            now = time.perf_counter()
            face = detector.detect(frame)
            if face is not None:
                last_face = face
                try:
                    roi_rgb = extract_method_roi(
                        frame, face, roi_name, patch_size=patch_size
                    )
                    buffer.add(roi_rgb, now)
                    status_msg = f"Buffer: {len(buffer)} frames"
                except ValueError:
                    status_msg = "ROI invalide"
            else:
                status_msg = "Aucun visage — buffer non mis a jour"

            fps_est = buffer.estimated_fps()
            processing_fs, fs_status = resolve_processing_fs(fps_est, target_fps)
            buf_sec = buffer.buffer_seconds()
            min_frames = max(
                min_frames_required(processing_fs, method),
                int(processing_fs * args.buffer_seconds * min_buffer_fill_ratio),
            )

            # Recalcul HR périodique
            if (
                face is not None
                and len(buffer) >= min_frames
                and (now - last_hr_compute_t) >= args.hr_update_interval
            ):
                last_hr_compute_t = now
                try:
                    frames = buffer.as_array()
                    bvp = compute_method_output(frames, processing_fs, method)
                    n = min(len(bvp), len(frames))
                    if n < 9:
                        raise ValueError("Signal trop court après méthode.")
                    raw_hr = estimate_hr_from_bvp(
                        bvp[-n:], processing_fs, use_antiharmonic=use_antiharmonic
                    )
                    last_hr_raw = raw_hr
                    smoothed_hr = hr_estimator.update(raw_hr)
                    garmin_hr_now = (
                        garmin_reader.get_hr() if garmin_reader is not None else None
                    )
                    if garmin_calibrator is not None and garmin_hr_now is not None:
                        garmin_calibrator.update(smoothed_hr, garmin_hr_now)
                    last_hr = (
                        garmin_calibrator.apply(smoothed_hr)
                        if garmin_calibrator is not None
                        else smoothed_hr
                    )
                    parts = [f"HR brut {raw_hr:.0f}", f"lisse {smoothed_hr:.0f}"]
                    if garmin_calibrator is not None and garmin_calibrator.is_ready:
                        parts.append(f"calib {last_hr:.0f}")
                    status_msg = " -> ".join(parts) + f" | {fs_status}"
                except (MethodNotAvailableError, ValueError, RuntimeError) as exc:
                    last_hr = None
                    last_hr_raw = None
                    status_msg = str(exc)[:80]
                except Exception as exc:
                    last_hr = None
                    last_hr_raw = None
                    status_msg = f"Erreur HR: {exc}"[:80]
            elif len(buffer) < min_frames:
                last_hr = None
                last_hr_raw = None
                if face is not None:
                    status_msg = (
                        f"Buffer insuffisant ({len(buffer)}/{min_frames} frames)"
                    )

            garmin_hr = garmin_reader.get_hr() if garmin_reader is not None else None
            garmin_status = (
                garmin_reader.get_status() if garmin_reader is not None else None
            )

            # Log CSV
            if (now - last_csv_log_t) >= args.log_interval:
                last_csv_log_t = now
                calib_off = (
                    garmin_calibrator.bias_bpm if garmin_calibrator is not None else None
                )
                append_csv_row(
                    csv_path,
                    method,
                    roi_name,
                    last_hr,
                    last_hr_raw,
                    calib_off if use_garmin_calibrate else None,
                    garmin_hr,
                    processing_fs,
                    fps_est,
                    buf_sec,
                )

            cam_view = draw_face_overlay(
                frame,
                last_face if face is None else face,
                roi_name,
                status_msg,
            )
            stats_view = render_stats_panel(
                width=480,
                height=400,
                method=method,
                roi_name=roi_name,
                rppg_hr=last_hr,
                garmin_hr=garmin_hr,
                fps=fps_est,
                buffer_sec=buf_sec,
                message=status_msg,
                garmin_status=garmin_status,
                rppg_hr_raw=last_hr_raw,
                calib_bias=(
                    garmin_calibrator.bias_bpm if garmin_calibrator is not None else None
                ),
                processing_fs=processing_fs,
            )

            cv2.imshow(win_cam, cam_view)
            cv2.imshow(win_stats, stats_view)

            key = cv2.waitKey(1) & 0xFF
            # q ou Echap (27) : arrêt immédiat
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                buffer.clear()
                hr_estimator.reset()
                if garmin_calibrator is not None:
                    garmin_calibrator.reset()
                detector.reset()
                last_hr = None
                last_hr_raw = None
                status_msg = "Buffer reinitialise"

    except KeyboardInterrupt:
        print("\nInterruption clavier.")
    finally:
        if garmin_reader is not None:
            print("Arret Garmin BLE...")
            garmin_reader.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam fermee proprement.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
