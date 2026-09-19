# Separation of Shallow and Deep Groundwater Storage Variations
## Beijing–Tianjin–Hebei · GRACE + 3D-CNN

Open-source companion code for the manuscript:

> **Separation of Shallow and Deep Groundwater Storage Variations in the Beijing–Tianjin–Hebei Region Based on GRACE and 3D-CNN**

This repository provides the **proposed M2 model** (2-layer 3D-CNN encoder + last-frame temporal aggregation + Persistence residual head), training/evaluation utilities, baselines, ablation helpers, and deep–shallow residual separation scripts.

**Public repository:** https://github.com/gaick/Separation-of-Shallow-and-Deep-Groundwater-Storage-Variations-in-the-Beijing-Tianjin-Hebei-Region

---

## Computer Code Availability (journal checklist)

| Requirement | Status in this repo |
|-------------|---------------------|
| Open public repository | Yes (GitHub, anonymous clone) |
| Individual source files (not a single `.zip`) | Yes |
| `README` with purpose + usage | This file |
| At least one quick-test / example | `examples/quick_test.py` |
| Instructions to run the test | See below |
| License | MIT (`LICENSE`) |

---

## Requirements

- Python 3.10+
- Dependencies: see `requirements.txt`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick test (run this first)

No GRACE / monitoring-well data are required. The script builds synthetic fields with **high lag-1 autocorrelation** (similar regime to shallow GWS in the paper), instantiates the **paper M2** class, and checks:

1. Tensor layout `(B, T=12, C=6, H, W)`
2. Persistence residual identity \(\hat Y(t)=Y(t-1)+\Delta\)
3. Short training (loss decreases)
4. Same metrics API as the paper (`R²`, `MAE`, `RMSE`)
5. Water-balance residual definition \(\Delta\mathrm{GWS}_{deep}=\Delta\mathrm{GWS}_{total}-\Delta\mathrm{GWS}_{shallow}\)

```bash
python examples/quick_test.py
```

Expected final line:

```text
Quick test OK — architecture/residual/training/metrics consistent with paper.
```

> **Important:** This smoke test does **not** reproduce the BTH independent-test scores (e.g. R²≈0.961). Those require the licensed research data package.

---

## Proposed model (paper M2)

| Item | Setting |
|------|---------|
| Class | `Simple_ST_Net_Ablation2Enc` in `gws_model_variants.py` |
| Model id | `baseline_2enc_res` |
| Encoder | 2 × Conv3D–BN–GELU–Dropout3d (channels 24 → 48 by default) |
| Temporal agg. | **Last frame** (not temporal mean) |
| Head | Persistence residual: \(\hat Y(t)=Y(t-1)+\Delta\) |
| Input channels | 6 = `[pre, tmp, pet, lst, ndvi, gws_history]` |
| Window | 12 months |

---

## Repository layout

```text
.
├── examples/quick_test.py          # required runnable demo
├── gws_model_variants.py           # M1–M5 / M2 definition
├── gws_st_models.py                # ConvLSTM / ConvGRU / STConv baselines
├── gws_train.py                    # training loop + metrics
├── gws_io.py / gws_paths.py / gws_splits.py
├── experiments/
│   ├── AutoTrain.py                # Optuna + final training
│   ├── baseline_persistence.py     # Persistence / climatology
│   ├── st_model_compare.py         # spatiotemporal horizontal comparison
│   └── sy_sensitivity.py           # specific-yield ±10%/±20%
├── plotting/                       # result & deep-GWS residual figures
├── requirements.txt
└── LICENSE
```

---

## Running with your own aligned data

Place NumPy arrays under `data/npy/` (or edit paths in `gws_paths.py`):

| File | Shape | Notes |
|------|-------|-------|
| `X_drivers.npy` | `(T, H, W, 5)` | standardized `[pre,tmp,pet,lst,ndvi]` |
| `Y_shallow_gws.npy` | `(T, H, W)` | shallow GWS change (cm) |
| `lat_coords.npy` / `lon_coords.npy` / `time_coords.npy` | 1-D | coordinates |

Paper temporal split (target months):

- Train: 2019-01 – 2023-09  
- Validation: 2023-10 – 2024-04  
- Test: 2024-05 – 2024-12  

Example:

```bash
PYTHONPATH=. python experiments/AutoTrain.py
PYTHONPATH=. python experiments/baseline_persistence.py
PYTHONPATH=. python plotting/plot_results.py --lang en
```

---

## Data availability

GRACE/GLDAS products and monitoring-well observations follow their original licenses and are **not fully redistributed** here. Processed research packages may be shared by the corresponding author on reasonable request when permitted.

---

## Citation

Please cite the published article (DOI to be added after acceptance).

## License

MIT — see `LICENSE`.

## Contact

Corresponding author: Yangrui Yang — yangyangrui@ncwu.edu.cn  
Code maintainer: Mingyang Liu (repository owner: [gaick](https://github.com/gaick))
