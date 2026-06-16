# Usage local (UBFC-rPPG)

Ce dépôt a été allégé pour un usage centré sur **UBFC-rPPG** :

- **Non supervisé** : `configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml` (+ variante ROI)
- **Deep learning** : `configs/train_configs/local/UBFC_DL_*.yaml`
- **ROI** : `tools/Selection_ROI.py` + variable `RPPG_ROI` (ex. `zhao2024`, `li2024`)
- **Scripts** : `scripts/run_ubfc_dl_one.sh`, `scripts/run_deepphys_zhao2024.sh`, `scripts/run_deepphys_li2024.sh`

Les configs d’origine pour PURE, SCAMPS, MMPD, etc. ont été retirées.
Les gabarits pour régénérer les YAML locaux sont dans `configs/train_configs/templates/`.

Données (hors dépôt) : `/media/simeon/rPPP-Data/Datasets/UBFC-rPPG/`

## Graphiques et comparaison des méthodes

### À chaque run (traçabilité automatique)

Les scripts `run_ubfc_dl_one.sh` et `run_with_results.sh` définissent automatiquement :

- **`RPPG_CONFIG_FILE`** — YAML utilisé (chemin relatif au repo)
- **`RPPG_RUN_NAME`** — nom unique avec timestamp (évite d’écraser les JSON)
- **`RPPG_ROI`** — ROI active (défaut `full_face`)

Exemple DL :

```bash
bash scripts/run_tscan_zhao2024.sh
# → run_name du type UBFC_DL_Tscan_Zhao2024_zhao2024_20260602_143022
```

Exemple non supervisé :

```bash
RPPG_ROI=zhao2024 bash scripts/run_with_results.sh POS_zhao2024 configs/infer_configs/UBFC-rPPG_UNSUPERVISED_POS.yaml
```

Vérifier le CSV après plusieurs runs :

```bash
python scripts/verify_run_traceability.py
```

Colonnes dans `runs/comparison/results.csv` : `timestamp`, `run_name`, `method`, `roi`, `config_file`, métriques (`mae`, `rmse`, …), `cached_path`, `results_dir`, `model_file_name`.

2. **Graphiques produits automatiquement**

| Type | Où | Condition |
|------|-----|-----------|
| Bland-Altman (scatter + différence) | `runs/exp_ubfc_dl/<EXP_DATA_NAME>/bland_altman_plots/` | `'BA'` dans `METRICS` |
| Courbes train/valid (DL) | `.../plots/*_losses.pdf` | `PLOT_LOSSES_AND_LR: True` |
| **Tableau comparatif** | `runs/comparison/results.csv` | toujours (nouveau) |
| HR par fenêtre (JSON) | `runs/comparison/runs/<RPPG_RUN_NAME>.json` | toujours |

3. **Comparer toutes les méthodes**

```bash
python scripts/compare_runs.py
# -> runs/comparison/plots/compare_mae.png, compare_rmse.png, compare_pearson.png, compare_scatter_all.png
```

### Exemples de noms de runs

```bash
RPPG_RUN_NAME=POS_fullface ...
RPPG_RUN_NAME=POS_zhao2024 RPPG_ROI=zhao2024 ...
RPPG_RUN_NAME=DeepPhys_fullface ...
RPPG_RUN_NAME=DeepPhys_Zhao2024 RPPG_ROI=zhao2024 ...
RPPG_RUN_NAME=Tscan_Zhao2024 RPPG_ROI=zhao2024 ...   # bash scripts/run_tscan_zhao2024.sh
RPPG_RUN_NAME=Physnet_Zhao2024 RPPG_ROI=zhao2024 ... # bash scripts/run_physnet_zhao2024.sh
RPPG_RUN_NAME=DeepPhys_Li2024 RPPG_ROI=li2024 ...    # bash scripts/run_deepphys_li2024.sh
RPPG_RUN_NAME=Tscan_Li2024 RPPG_ROI=li2024 ...       # bash scripts/run_tscan_li2024.sh
RPPG_RUN_NAME=Physnet_Li2024 RPPG_ROI=li2024 ...     # bash scripts/run_physnet_li2024.sh
RPPG_RUN_NAME=Tscan_fullface ...
```

### Config : métriques et plots

Dans les YAML, garder :

```yaml
METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']   # BA = Bland-Altman PDF
```

Pour le deep learning :

```yaml
TRAIN:
  PLOT_LOSSES_AND_LR: True
```
