"""
表13 给水度(Sy)参数扰动敏感性分析 — 浅层标签扰动对深层 GWS 残差的影响（区域平均，不分区）。

用法:
  PYTHONPATH=. .venv/bin/python experiments/sy_sensitivity.py
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import csv
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

from gws_io import DATA_DIR, load_coords
from gws_paths import SY_SENSITIVITY_JSON_DIR, SY_SENSITIVITY_TABLE_DIR, ensure_parent
from plotting.plot_deep_gws import _monthly_aggregate, masked_regional_mean

SY_ROOT = "./data_sy_sensitivity"
NC_PATH = "./data/nc/jingjinji_groundwater_storage_final.nc"
SCENARIOS = [
    ("Sy -20%", "Sy-20%"),
    ("Sy -10%", "Sy-10%"),
    ("基准 Sy", "baseline"),
    ("Sy +10%", "Sy+10%"),
    ("Sy +20%", "Sy+20%"),
]


def _load_y(scenario_key: str) -> np.ndarray:
    if scenario_key == "baseline":
        return np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    return np.load(os.path.join(SY_ROOT, scenario_key, "Y_shallow_gws.npy"))


def _spatial_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.zeros(arrays[0].shape[1:], dtype=bool)
    for arr in arrays:
        mask |= np.any(np.isfinite(arr) & (np.abs(arr) > 1e-6), axis=0)
    return mask


def _time_mean_spatial_field(grid: np.ndarray) -> np.ndarray:
    return np.nanmean(grid, axis=0)


def _spatial_correlation(field_a: np.ndarray, field_b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(field_a) & np.isfinite(field_b)
    if valid.sum() < 3:
        return float("nan")
    r, _ = pearsonr(field_a[valid].ravel(), field_b[valid].ravel())
    return float(r)


def _conclusion(spatial_r: float, regional_mean_cm: float, baseline_mean_cm: float) -> str:
    if np.isnan(spatial_r):
        return "有效格点不足"
    rel_change = abs(regional_mean_cm - baseline_mean_cm) / (abs(baseline_mean_cm) + 1e-6)
    if spatial_r >= 0.95 and rel_change < 0.05:
        return "空间格局与区域均值对 Sy 扰动不敏感，趋势保持稳定"
    if spatial_r >= 0.85:
        return "空间格局基本一致，区域均值略有偏移"
    return "Sy 扰动导致空间格局与区域均值发生明显变化"


def _build_deep_grids(
    shallow_y: np.ndarray,
    spatial_mask: np.ndarray,
    ref_times: pd.DatetimeIndex,
    nc_times: pd.DatetimeIndex,
    gws_total: np.ndarray,
) -> np.ndarray:
    shallow = shallow_y.copy().astype(float)
    shallow[:, ~spatial_mask] = np.nan
    deep = np.full_like(shallow, np.nan)
    for i, t in enumerate(ref_times):
        idx = np.where(nc_times == t)[0]
        j = int(idx[0]) if len(idx) else int(np.argmin(np.abs(nc_times - t)))
        if j < gws_total.shape[0]:
            deep[i] = gws_total[j] - shallow[i]
    return deep


def run_sy_sensitivity() -> dict:
    lat, lon, all_time = load_coords()
    ref_times = pd.to_datetime(all_time) if len(all_time) else pd.date_range("2018-01-01", periods=84, freq="MS")

    ds = xr.open_dataset(NC_PATH)
    nc_times, gws_total = _monthly_aggregate(ds, "delta_gws", lat, lon)
    ds.close()

    baseline_y = _load_y("baseline")
    spatial_mask = _spatial_mask(baseline_y, *[_load_y(k) for _, k in SCENARIOS if k != "baseline"])

    baseline_deep = _build_deep_grids(baseline_y, spatial_mask, ref_times, nc_times, gws_total)
    baseline_field = _time_mean_spatial_field(baseline_deep)
    baseline_regional = float(np.nanmean(masked_regional_mean(baseline_deep, spatial_mask)))

    rows = []
    for label, key in SCENARIOS:
        y = baseline_y if key == "baseline" else _load_y(key)
        deep = _build_deep_grids(y, spatial_mask, ref_times, nc_times, gws_total)
        regional = float(np.nanmean(masked_regional_mean(deep, spatial_mask)))
        spatial_r = 1.0 if key == "baseline" else _spatial_correlation(
            _time_mean_spatial_field(deep), baseline_field, spatial_mask,
        )
        rows.append({
            "scenario": label,
            "scenario_key": key,
            "regional_mean_deep_gws_cm": round(regional, 1),
            "spatial_correlation_with_baseline": round(spatial_r, 3) if not np.isnan(spatial_r) else None,
            "conclusion": "基准情景" if key == "baseline" else _conclusion(spatial_r, regional, baseline_regional),
        })

    return {
        "title": "表13 给水度参数扰动敏感性分析结果",
        "region": "京津冀",
        "metric": "2018–2024 全时段区域平均深层 GWS 残差（GWS_total − GWS_shallow，cm）",
        "spatial_correlation_method": "全时段时间平均深层 GWS 空间场与基准 Sy 的 Pearson 相关系数",
        "baseline_regional_mean_deep_gws_cm": baseline_regional,
        "rows": rows,
    }


def save_results(summary: dict) -> None:
    os.makedirs(SY_SENSITIVITY_TABLE_DIR, exist_ok=True)
    os.makedirs(SY_SENSITIVITY_JSON_DIR, exist_ok=True)

    json_path = os.path.join(SY_SENSITIVITY_JSON_DIR, "table_13_sy_sensitivity.json")
    csv_path = os.path.join(SY_SENSITIVITY_TABLE_DIR, "table_13_sy_sensitivity.csv")

    ensure_parent(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "扰动情景",
            "区域平均深层 GWS (cm)",
            "与基准结果空间相关",
            "结论",
        ])
        for row in summary["rows"]:
            writer.writerow([
                row["scenario"],
                row["regional_mean_deep_gws_cm"],
                row["spatial_correlation_with_baseline"] if row["spatial_correlation_with_baseline"] is not None else "",
                row["conclusion"],
            ])

    print("\n✅ 表13 结果已保存:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    save_results(run_sy_sensitivity())


if __name__ == "__main__":
    main()
