"""results/ 目录布局与路径工具。"""

from __future__ import annotations

import os

RESULTS_ROOT = "./results"
MODELS_ROOT = os.path.join(RESULTS_ROOT, "models")
JSON_ROOT = os.path.join(RESULTS_ROOT, "json")
FIGURES_ROOT = os.path.join(RESULTS_ROOT, "figures")
LOGS_ROOT = os.path.join(RESULTS_ROOT, "logs")
TABLES_ROOT = os.path.join(RESULTS_ROOT, "tables")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
ARCHIVE_ROOT = os.path.join(RESULTS_ROOT, "archive")

RESULTS_DIR = os.path.join(MODELS_ROOT, "shallow_gws")
# 本文最终模型：M2（2 层 Encoder，24→48，末帧 + Persistence 残差）
RESULTS_PROPOSED = RESULTS_DIR
RESULTS_M4_LEGACY = os.path.join(MODELS_ROOT, "shallow_gws_plus1enc_reslastt")
RESULTS_BASELINE = RESULTS_DIR
BASELINE_PERSISTENCE_DIR = os.path.join(MODELS_ROOT, "baseline_persistence")
LEGACY_AUTOTRAIN_DIR = os.path.join(MODELS_ROOT, "shallow_gws_autotrain_final")
PLUS1ENC_DIR = os.path.join(MODELS_ROOT, "shallow_gws_plus1enc")

COMPARE_JSON_DIR = os.path.join(JSON_ROOT, "compare")
COMPARE_FIG_DIR = os.path.join(FIGURES_ROOT, "compare")
COMPARE_TABLE_DIR = os.path.join(TABLES_ROOT, "compare")
COMPARE_LOG_DIR = os.path.join(LOGS_ROOT, "compare")

ST_COMPARE_JSON_DIR = os.path.join(JSON_ROOT, "st_compare")
ST_COMPARE_FIG_DIR = os.path.join(FIGURES_ROOT, "st_compare")
ST_COMPARE_LOG_DIR = os.path.join(LOGS_ROOT, "st_compare")
ST_COMPARE_MODELS_ROOT = os.path.join(MODELS_ROOT, "st_compare")

DEEP_GWS_JSON_DIR = os.path.join(JSON_ROOT, "deep_gws")
DEEP_GWS_FIG_DIR = os.path.join(FIGURES_ROOT, "deep_gws")

SY_SENSITIVITY_TABLE_DIR = os.path.join(TABLES_ROOT, "sy_sensitivity")
SY_SENSITIVITY_JSON_DIR = os.path.join(JSON_ROOT, "sy_sensitivity")
RESULTS_HEXI = os.path.join(MODELS_ROOT, "hexi_shallow_gws")
RESULTS_HUANGHUAI = os.path.join(MODELS_ROOT, "huanghuai_shallow_gws")
BASELINE_PERSISTENCE_HUANGHUAI_DIR = os.path.join(MODELS_ROOT, "huanghuai_baseline_persistence")

RESULTS_COMPARE = COMPARE_JSON_DIR


def rel_from_results(path: str) -> str:
    norm = os.path.normpath(path)
    root = os.path.normpath(RESULTS_ROOT)
    if norm == root:
        return "."
    if norm.startswith(root + os.sep):
        return os.path.relpath(norm, root)
    return path.lstrip("./")


def json_file(parent_dir: str, filename: str) -> str:
    rel = rel_from_results(os.path.normpath(parent_dir))
    return os.path.join(JSON_ROOT, rel, filename)


def models_root(root: str = MODELS_ROOT) -> str:
    return root.rstrip("/")


def results_dir_for_subdir(subdir: str, root: str = MODELS_ROOT) -> str:
    return os.path.join(root.rstrip("/"), subdir)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


_LEGACY_DIR_MAP = {
    "./results/shallow_gws": RESULTS_DIR,
    "./results/shallow_gws_plus1enc": PLUS1ENC_DIR,
    "./results/shallow_gws_plus1enc_reslastt": RESULTS_M4_LEGACY,
    "./results/shallow_gws_autotrain_final": LEGACY_AUTOTRAIN_DIR,
    "./results/baseline_persistence": BASELINE_PERSISTENCE_DIR,
    "./results/ablation": os.path.join(MODELS_ROOT, "ablation"),
    "./results/st_compare": ST_COMPARE_MODELS_ROOT,
}


def resolve_model_dir(path: str) -> str:
    """将旧版 results/ 路径解析为 models/ 下的新路径。"""
    if os.path.exists(path):
        return path
    mapped = _LEGACY_DIR_MAP.get(path.replace("\\", "/"))
    if mapped and os.path.exists(mapped):
        return mapped
    if path.startswith("./results/") and "/models/" not in path:
        candidate = path.replace("./results/", "./results/models/", 1)
        if os.path.exists(candidate):
            return candidate
    return path
