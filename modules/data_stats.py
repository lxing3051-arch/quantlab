"""
模块 1：数据清洗与描述性统计
展示维度、类型、缺失值、描述统计，并绘制相关热力图与分布直方图。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import get_numeric_columns, validate_dataframe
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """计算各列缺失值数量与占比。"""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "列名": df.columns,
            "数据类型": [str(t) for t in df.dtypes],
            "缺失数量": missing_count.values,
            "缺失占比(%)": missing_pct.values,
            "非空数量": df.notna().sum().values,
        }
    )
    return report.sort_values("缺失占比(%)", ascending=False).reset_index(drop=True)


def _correlation_heatmap(num_df: pd.DataFrame) -> go.Figure:
    """绘制 Plotly 相关系数热力图。"""
    corr = num_df.corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[
                [0.0, "#be123c"],
                [0.5, "#f8fafc"],
                [1.0, "#0f766e"],
            ],
            zmin=-1,
            zmax=1,
            colorbar={"title": "相关系数"},
            hovertemplate="X: %{x}<br>Y: %{y}<br>ρ: %{z:.3f}<extra></extra>",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#0f172a"},
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="变量相关系数热力图 (Pearson)",
        height=520,
        xaxis={"side": "bottom", "tickangle": -35},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _distribution_histograms(num_df: pd.DataFrame, selected_cols: list) -> go.Figure:
    """为选定数值列绘制分布直方图（可叠加多列）。"""
    if len(selected_cols) == 1:
        fig = px.histogram(
            num_df,
            x=selected_cols[0],
            nbins=40,
            marginal="box",
            opacity=0.85,
            color_discrete_sequence=["#0f766e"],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"「{selected_cols[0]}」分布直方图",
            height=420,
            bargap=0.05,
        )
    else:
        # 多列：长表形式叠加
        long_df = num_df[selected_cols].melt(var_name="变量", value_name="取值")
        fig = px.histogram(
            long_df,
            x="取值",
            color="变量",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="多变量分布对比直方图",
            height=420,
            bargap=0.05,
            legend_title_text="变量",
        )
    return fig


def render_data_stats(df: pd.DataFrame) -> None:
    """渲染模块 1 完整页面。"""
    st.markdown('<div class="hero-title">📊 数据清洗与描述性统计</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">快速洞察数据维度、类型、缺失情况与变量分布，为后续建模打好基础。</div>',
        unsafe_allow_html=True,
    )

    if not validate_dataframe(df):
        return

    try:
        n_rows, n_cols = df.shape
        n_numeric = len(get_numeric_columns(df))
        n_missing_cells = int(df.isna().sum().sum())
        missing_ratio = n_missing_cells / (n_rows * n_cols) * 100 if n_rows * n_cols else 0

        render_metric_cards(
            [
                ("样本行数", f"{n_rows:,}"),
                ("特征列数", f"{n_cols}"),
                ("数值型列", f"{n_numeric}"),
                ("整体缺失率", f"{missing_ratio:.2f}%"),
            ]
        )

        st.markdown("<br>", unsafe_allow_html=True)

        tab_overview, tab_desc, tab_corr, tab_dist = st.tabs(
            ["📋 数据概览", "📐 描述性统计", "🔥 相关热力图", "📉 分布直方图"]
        )

        with tab_overview:
            st.markdown('<div class="section-title">原始数据预览</div>', unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True, height=320)

            st.markdown('<div class="section-title">字段类型与缺失值</div>', unsafe_allow_html=True)
            report = _missing_report(df)
            st.dataframe(report, use_container_width=True, height=300)

            # 简易清洗建议
            high_missing = report[report["缺失占比(%)"] > 20]["列名"].tolist()
            if high_missing:
                st.warning(
                    f"以下列缺失占比超过 20%，建议重点检查或剔除："
                    f" {', '.join(map(str, high_missing))}"
                )
            else:
                st.success("各列缺失占比均低于 20%，数据完整度良好。")

        with tab_desc:
            num_cols = get_numeric_columns(df)
            if not num_cols:
                st.warning("当前数据中没有数值型列，无法计算描述性统计。")
            else:
                st.markdown(
                    '<div class="section-title">数值变量描述性统计摘要</div>',
                    unsafe_allow_html=True,
                )
                desc = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
                desc = desc.rename(
                    columns={
                        "count": "计数",
                        "mean": "均值",
                        "std": "标准差",
                        "min": "最小值",
                        "5%": "5%分位",
                        "25%": "25%分位",
                        "50%": "中位数",
                        "75%": "75%分位",
                        "95%": "95%分位",
                        "max": "最大值",
                    }
                )
                # 额外附加偏度与峰度
                desc["偏度"] = df[num_cols].skew()
                desc["峰度"] = df[num_cols].kurtosis()
                st.dataframe(desc.round(4), use_container_width=True, height=380)

                # 非数值列简要统计
                cat_cols = [c for c in df.columns if c not in num_cols]
                if cat_cols:
                    st.markdown(
                        '<div class="section-title">非数值列简要统计</div>',
                        unsafe_allow_html=True,
                    )
                    cat_summary = pd.DataFrame(
                        {
                            "列名": cat_cols,
                            "唯一值数量": [df[c].nunique(dropna=True) for c in cat_cols],
                            "众数": [
                                str(df[c].mode(dropna=True).iloc[0])
                                if not df[c].mode(dropna=True).empty
                                else "-"
                                for c in cat_cols
                            ],
                        }
                    )
                    st.dataframe(cat_summary, use_container_width=True)

        with tab_corr:
            num_cols = get_numeric_columns(df)
            if len(num_cols) < 2:
                st.warning("至少需要 2 个数值型变量才能绘制相关热力图。")
            else:
                default_sel = num_cols[: min(12, len(num_cols))]
                selected = st.multiselect(
                    "选择参与相关分析的变量",
                    options=num_cols,
                    default=default_sel,
                    key="corr_vars",
                )
                if len(selected) < 2:
                    st.info("请至少选择 2 个变量。")
                else:
                    # 删除含缺失的行后再计算相关，避免偏差
                    clean = df[selected].dropna()
                    if len(clean) < 3:
                        st.warning("有效样本过少（删除缺失后不足 3 行），无法计算相关矩阵。")
                    else:
                        fig = _correlation_heatmap(clean)
                        st.plotly_chart(fig, use_container_width=True)

        with tab_dist:
            num_cols = get_numeric_columns(df)
            if not num_cols:
                st.warning("没有数值型列可供绘制分布图。")
            else:
                selected_dist = st.multiselect(
                    "选择要查看分布的变量（可多选对比）",
                    options=num_cols,
                    default=[num_cols[0]],
                    key="dist_vars",
                )
                if not selected_dist:
                    st.info("请选择至少一个变量。")
                else:
                    clean = df[selected_dist].dropna()
                    if clean.empty:
                        st.warning("所选变量在删除缺失值后无有效数据。")
                    else:
                        fig = _distribution_histograms(clean, selected_dist)
                        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:  # noqa: BLE001
        st.error(f"数据统计分析过程中发生错误: {exc}")
        st.exception(exc)
