# Prototype webcam + Garmin (mai 2026)

Petit outil à part du reste du projet : filmer le visage avec la webcam, estimer la fréquence cardiaque (méthode **POS** par défaut), et comparer avec une **Garmin** si elle est en mode « diffuser la fréquence cardiaque ».

## Fichiers

- `webcam_rppg_live.py` — script principal
- Option `--garmin` : nécessite `garmin_ble_reader.py` (non inclus dans la version allégée du dépôt)

## Utilisation

```bash
conda activate rppg-toolbox
python tools/webcam_rppg_live.py --method pos --garmin --garmin-calibrate
```

## Ce qu’on a constaté

- Sur les **vidéos UBFC** (dataset), POS + ROI Zhao donne de bons résultats.
- En **live webcam**, c’était beaucoup moins bon au début (montre ~90 bpm, rPPG souvent ~60–70).
- Après quelques réglages + **calibration Garmin**, l’écart moyen est tombé autour de **5 bpm** (au lieu de ~16).
- La webcam tourne souvent à **~15 images/s** au lieu de 30 : ça limite encore la précision.

## Pas encore fait

- Modèles **deep learning** du toolbox sur UBFC (prévu après ce commit).
- Les fichiers CSV dans `outputs/` sont des notes de test locales — inutiles à mettre sur Git.
