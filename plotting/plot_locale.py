"""Plot label strings for Chinese (zh) and English (en) figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotLocale:
    lang: str
    proposed_model_label: str
    # axes
    latitude: str
    longitude: str
    month: str
    time: str
    time_monthly: str
    epoch: str
    # datasets
    train: str
    val: str
    test: str
    dataset_col: str
    # spatial maps
    observed: str
    predicted: str
    error: str
    error_cm: str
    gws_cm: str
    ideal_line: str
    # hints
    higher_better: str
    lower_better: str
    # plot_results titles
    spatial_suptitle_full: str
    spatial_suptitle_part: str
    timeseries_title: str
    scatter_title: str
    rmse_title: str
    generalization_title: str
    learning_curves_title: str
    grid_metrics: str
    points_suffix: str
    observed_regional: str
    predicted_regional: str
    shallow_gws_change: str
    figure_guide_name: str
    # st compare
    st_suptitle: str
    st_legend_dl: str
    st_legend_paper: str
    st_legend_persistence: str
    st_params_xlabel: str
    st_params_title: str
    persistence_baseline: str
    params_ratio_suffix: str
    # shallow-deep trend
    shallow_series: str
    deep_series: str
    train_end_label: str
    val_test_label: str
    regional_mean_dgws: str
    shallow_deep_title: str
    shallow_deep_footnote: str
    # gws paper figs
    gws_spatial_title: str
    shallow_gws: str
    deep_gws: str
    total_gws: str
    gws_components_title: str
    total_gws_grace: str
    # technical route
    tech_grace: str
    tech_mixed: str
    tech_risk: str
    tech_wells: str
    tech_drivers: str
    tech_cnn: str
    tech_shallow: str
    tech_gldas: str
    tech_balance: str
    tech_deep: str
    tech_footnote: str


_ZH = PlotLocale(
    lang="zh",
    proposed_model_label="3DCNN(M2·2层·残差+末时刻·本文)",
    latitude="纬度 (°N)",
    longitude="经度 (°E)",
    month="月份",
    time="时间",
    time_monthly="时间（月）",
    epoch="Epoch",
    train="训练集",
    val="验证集",
    test="测试集",
    dataset_col="数据集",
    observed="观测值（真值）",
    predicted="预测值（{model}）",
    error="误差（±{lim:.1f} cm）",
    error_cm="误差 (cm)",
    gws_cm="GWS (cm)",
    ideal_line="理想线 y=x",
    higher_better="越高越好",
    lower_better="越低越好",
    spatial_suptitle_full="测试集全格网时空预测对比（8个月）\n左=真值  中={model}  右=误差（白=准确）",
    spatial_suptitle_part="测试集时空预测对比（{part}）{start} ~ {end}",
    timeseries_title="图2  区域平均时间序列（测试集8个月）",
    scatter_title="图3  散点图：每个点 = 一个格点在某月的值",
    rmse_title="图4  8个月平均空间 RMSE（黄=准，红=差）",
    generalization_title="{model} 训练/验证/测试三分集精度对比",
    learning_curves_title="图6  学习曲线：训练集 vs 验证集",
    grid_metrics="全格网指标",
    points_suffix="点",
    observed_regional="观测值（区域平均）",
    predicted_regional="预测值（区域平均）",
    shallow_gws_change="浅层地下水储量变化 (cm)",
    figure_guide_name="00_图例说明.txt",
    st_suptitle="时空模型 vs Persistence 基线 — 测试集指标（2024-05~12）",
    st_legend_dl="DL 横向对比",
    st_legend_paper="本文最终模型",
    st_legend_persistence="Persistence 基线",
    st_params_xlabel="参数量（Persistence 取 0 作示意）",
    st_params_title="参数量 vs 测试集 R²",
    persistence_baseline="Persistence(基线)",
    params_ratio_suffix="参",
    shallow_series="浅层 GWS（井网格网化观测）",
    deep_series="深层 GWS（总量 − 浅层观测）",
    train_end_label="训练目标期末",
    val_test_label="验证→测试",
    regional_mean_dgws="区域平均 ΔGWS (cm)",
    shallow_deep_title="京津冀浅层 / 深层地下水储量变化长期趋势",
    shallow_deep_footnote="数据：浅层=观测格网；深层=GRACE−GLDAS 总量残差减浅层观测 | 时段 {t0}—{t1}（月尺度，n={n}）",
    gws_spatial_title="京津冀GWS数据空间分布（0.25° 栅格）",
    shallow_gws="浅层GWS",
    deep_gws="深层GWS",
    total_gws="总GWS",
    gws_components_title="京津冀地区地下水储量变化分量",
    total_gws_grace="总GWS (GRACE)",
    tech_grace="GRACE 反演\n地下水总储量变化",
    tech_mixed="混合信号\n浅层季节波动 + 深层长期亏损",
    tech_risk="仅看总量：可能误判风险\n例：浅层 $+3$ cm，深层 $-4$ cm\n总量仅表现为 $-1$ cm",
    tech_wells="浅层监测井",
    tech_drivers="气象/遥感驱动因子\n降水、蒸散发、LST、NDVI",
    tech_cnn="3D-CNN\n浅层 GWS 时空预测",
    tech_shallow="浅层 GWS\n连续格网化表达",
    tech_gldas="GLDAS 扣除非地下水分量\n获得 $\\Delta\\mathrm{GWS}_{\\mathrm{total}}$",
    tech_balance="水量平衡残差分离\n$\\Delta\\mathrm{GWS}_{\\mathrm{deep}}=\\Delta\\mathrm{GWS}_{\\mathrm{total}}-\\Delta\\mathrm{GWS}_{\\mathrm{shallow}}$",
    tech_deep="深层 GWS 残差估计\n识别长期亏损与深层消耗风险",
    tech_footnote=(
        "Note: 0.25° shallow details mainly come from well labels and multi-source drivers; "
        "deep results are residual estimates under total-storage constraint, not direct GRACE deep observations."
    ),
)

_EN = PlotLocale(
    lang="en",
    proposed_model_label="3DCNN (M2, proposed)",
    latitude="Latitude (°N)",
    longitude="Longitude (°E)",
    month="Month",
    time="Time",
    time_monthly="Time (monthly)",
    epoch="Epoch",
    train="Training",
    val="Validation",
    test="Test",
    dataset_col="Dataset",
    observed="Observed (ground truth)",
    predicted="Predicted ({model})",
    error="Error (±{lim:.1f} cm)",
    error_cm="Error (cm)",
    gws_cm="GWS (cm)",
    ideal_line="Ideal line y = x",
    higher_better="higher is better",
    lower_better="lower is better",
    spatial_suptitle_full=(
        "Test-set spatiotemporal prediction (8 months)\n"
        "Left = truth  |  Center = {model}  |  Right = error (white = accurate)"
    ),
    spatial_suptitle_part="Test-set spatiotemporal comparison ({part}) {start} ~ {end}",
    timeseries_title="Regional mean time series (8-month test set)",
    scatter_title="Scatter plot: each point = one grid cell in one month",
    rmse_title="Mean spatial RMSE over 8 months (yellow = good, red = poor)",
    generalization_title="{model}: training / validation / test accuracy",
    learning_curves_title="Learning curves: training vs validation",
    grid_metrics="Full-grid metrics",
    points_suffix="pts",
    observed_regional="Observed (regional mean)",
    predicted_regional="Predicted (regional mean)",
    shallow_gws_change="Shallow groundwater storage change (cm)",
    figure_guide_name="00_figure_guide.txt",
    st_suptitle="Spatiotemporal models vs Persistence — test metrics (2024-05 to 12)",
    st_legend_dl="DL comparison",
    st_legend_paper="Proposed model",
    st_legend_persistence="Persistence baseline",
    st_params_xlabel="Parameter count (Persistence shown at 0)",
    st_params_title="Parameter count vs test R²",
    persistence_baseline="Persistence (baseline)",
    params_ratio_suffix="×params",
    shallow_series="Shallow GWS (gridded well observations)",
    deep_series="Deep GWS (total − shallow observations)",
    train_end_label="End of training targets",
    val_test_label="Val → test",
    regional_mean_dgws="Regional mean ΔGWS (cm)",
    shallow_deep_title="Long-term shallow / deep GWS trends in Beijing–Tianjin–Hebei",
    shallow_deep_footnote=(
        "Shallow = observed grid; deep = GRACE−GLDAS total residual minus shallow | "
        "Period {t0}–{t1} (monthly, n={n})"
    ),
    gws_spatial_title="Spatial distribution of GWS in BTH (0.25° grid)",
    shallow_gws="Shallow GWS",
    deep_gws="Deep GWS",
    total_gws="Total GWS",
    gws_components_title="GWS change components in Beijing–Tianjin–Hebei",
    total_gws_grace="Total GWS (GRACE)",
    tech_grace="GRACE inversion\nTotal groundwater storage change",
    tech_mixed="Mixed signal\nShallow seasonal fluctuation + deep long-term depletion",
    tech_risk=(
        "Total-only view: misinterpretation risk\n"
        "e.g. shallow +3 cm, deep −4 cm\nappears as only −1 cm total"
    ),
    tech_wells="Shallow monitoring wells",
    tech_drivers="Meteorological / RS drivers\nPrecipitation, PET, LST, NDVI",
    tech_cnn="3D-CNN\nShallow GWS spatiotemporal prediction",
    tech_shallow="Shallow GWS\nContinuous gridded field",
    tech_gldas="GLDAS non-GW removal\n$\\Delta\\mathrm{GWS}_{\\mathrm{total}}$",
    tech_balance=(
        "Water-balance residual separation\n"
        "$\\Delta\\mathrm{GWS}_{\\mathrm{deep}}=\\Delta\\mathrm{GWS}_{\\mathrm{total}}"
        "-\\Delta\\mathrm{GWS}_{\\mathrm{shallow}}$"
    ),
    tech_deep="Deep GWS residual estimate\nLong-term depletion & deep-use risk",
    tech_footnote=(
        "Note: 0.25° shallow details mainly come from well labels and multi-source drivers; "
        "deep results are residual estimates under total-storage constraint, not direct GRACE deep observations."
    ),
)

_LOCALES = {"zh": _ZH, "en": _EN}


def get_locale(lang: str = "zh") -> PlotLocale:
    key = (lang or "zh").lower()
    if key not in _LOCALES:
        raise ValueError(f"Unsupported language: {lang!r} (use 'zh' or 'en')")
    return _LOCALES[key]


def setup_matplotlib(lang: str = "zh") -> None:
    import matplotlib.pyplot as plt

    if get_locale(lang).lang == "en":
        plt.rcParams.update({
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        })
    else:
        plt.rcParams.update({
            "font.sans-serif": ["SimHei", "Arial Unicode MS", "Arial"],
            "axes.unicode_minus": False,
        })


def english_figures_root() -> str:
    import os
    from gws_paths import FIGURES_ROOT

    return os.path.join(FIGURES_ROOT, "en")
