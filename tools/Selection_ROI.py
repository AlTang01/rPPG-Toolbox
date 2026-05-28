import os
import cv2
import numpy as np


ROI_BOXES = {
    # Baseline : visage complet déjà prétraité par rPPG-Toolbox
    "full_face": None,

    # Régions générales fréquemment utilisées en rPPG.
    # Inspiré de la revue systématique :
    # Bondarenko, Menon & Elgendi (2025),
    # "The role of face regions in remote photoplethysmography for contactless heart rate monitoring"
    "forehead": [
        (0.20, 0.05, 0.80, 0.30),
    ],

    "cheeks": [
        (0.16, 0.38, 0.43, 0.68),
        (0.57, 0.38, 0.84, 0.68),
    ],

    "glabella": [
        (0.42, 0.22, 0.58, 0.36),
    ],

    "nose_upper": [
        (0.42, 0.32, 0.58, 0.55),
    ],

    # ROI inspirée de Zhao et al. (2024)
    # "Toward Motion Robustness: A Masked Attention Regularization Framework
    # in Remote Photoplethysmography"
    #
    # Idée adaptée aux méthodes classiques :
    # - utiliser plusieurs zones symétriques du visage ;
    # - garder les régions stables : front, glabella, joues/malars ;
    # - éviter les zones plus bruitées : yeux, bouche, menton ;
    # - rendre la sélection plus robuste aux mouvements légers.
    #
    # Attention : ce n'est pas une reproduction complète de MAR-rPPG,
    # car l'article utilise un modèle deep learning avec attention maps.
    "zhao2024_motion_robust": [
        (0.34, 0.06, 0.66, 0.22),  # front médial
        (0.14, 0.08, 0.34, 0.26),  # front latéral gauche
        (0.66, 0.08, 0.86, 0.26),  # front latéral droit
        (0.42, 0.22, 0.58, 0.36),  # glabella
        (0.16, 0.38, 0.43, 0.68),  # joue / malar gauche
        (0.57, 0.38, 0.84, 0.68),  # joue / malar droit
        (0.42, 0.32, 0.58, 0.54),  # haut du nez
    ],

    # TOP-5 regions rapportées dans :
    # Bondarenko, Menon & Elgendi (2025),
    # "The role of face regions in remote photoplethysmography for contactless heart rate monitoring"
    #
    # Le tableau 2 de l'article synthétise les régions suivantes :
    # upper medial forehead, lower medial forehead, glabella,
    # right malar et left malar.
    "bondarenko2025_top5": [
        (0.36, 0.05, 0.64, 0.16),  # upper medial forehead
        (0.36, 0.16, 0.64, 0.28),  # lower medial forehead
        (0.42, 0.22, 0.58, 0.36),  # glabella
        (0.16, 0.38, 0.43, 0.68),  # left malar / cheek
        (0.57, 0.38, 0.84, 0.68),  # right malar / cheek
    ],

    # Régions inspirées de :
    # Li, Elgendi & Menon (2024),
    # "Optimal facial regions for remote heart rate measurement during physical and cognitive activities"
    #
    # Régions principales : glabella, medial forehead,
    # left/right lateral forehead, left/right malar regions,
    # upper nasal dorsum.
    "li2024_extended": [
        (0.36, 0.05, 0.64, 0.28),  # medial forehead
        (0.14, 0.07, 0.34, 0.26),  # left lateral forehead
        (0.66, 0.07, 0.86, 0.26),  # right lateral forehead
        (0.42, 0.22, 0.58, 0.36),  # glabella
        (0.16, 0.38, 0.43, 0.68),  # left malar
        (0.57, 0.38, 0.84, 0.68),  # right malar
        (0.42, 0.32, 0.58, 0.55),  # upper nasal dorsum
    ],
}


def apply_roi_to_frames(frames, roi_name="full_face", out_size=64):
    """
    frames shape attendue : (T, H, W, C)

    Sélectionne une ou plusieurs ROI dans les frames déjà prétraitées.
    Si plusieurs ROI sont demandées, elles sont redimensionnées puis concaténées.
    """

    if roi_name == "full_face" or roi_name is None:
        return frames

    if roi_name not in ROI_BOXES:
        raise ValueError(
            f"ROI inconnue : {roi_name}. "
            f"Choix possibles : {list(ROI_BOXES.keys())}"
        )

    boxes = ROI_BOXES[roi_name]

    if frames.ndim != 4:
        raise ValueError(f"Format inattendu : {frames.shape}. Attendu : (T, H, W, C)")

    T, H, W, C = frames.shape
    roi_clips = []

    for x1, y1, x2, y2 in boxes:
        x1 = int(x1 * W)
        x2 = int(x2 * W)
        y1 = int(y1 * H)
        y2 = int(y2 * H)

        x1 = max(0, min(x1, W - 1))
        x2 = max(x1 + 1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(y1 + 1, min(y2, H))

        crop = frames[:, y1:y2, x1:x2, :]

        resized = np.stack([
            cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_AREA)
            for frame in crop
        ])

        roi_clips.append(resized)

    # Plusieurs ROI : concaténation horizontale des régions sélectionnées.
    return np.concatenate(roi_clips, axis=2)