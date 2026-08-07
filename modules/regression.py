"""
模块 2：统计建模与 OLS 回归分析
使用 statsmodels 运行多元线性回归，输出完整 summary 与诊断图。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from utils.data_loader import get_numeric_columns, validate_dataframe
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def _fit_ols(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    拟合 OLS 模型。
    自动添加常数项，并在拟合前删除含缺失的行。

    Returns
    -------
    (results, data) : 回归结果对象与清洗后的建模数据
    """
    data = df[[y_col] + x_cols].dropna()
    if len(data) < len(x_cols) + 2:
        raise ValueError(
            f"有效样本量 ({len(data)}) 不足以估计含 {len(x_cols)} 个自变量的模型。"
        )

    y = data[y_col]
    x = sm.add_constant(data[x_cols], has_constant="add")

    # 检查自变量是否存在完全共线 / 零方差
    if x.shape[1] >= 2:
        variances = x.var()
        zero_var = variances[variances < 1e-14].index.tolist()
        if zero_var:
            raise ValueError(f"以下变量方差过小或为常数，无法进入回归: {zero_var}")

    model = sm.OLS(y, x)
    results = model.fit()
    return results, data


def _fit_line_figure(data: pd.DataFrame, y_col: str, x_col: str, results) -> go.Figure:
    """单自变量时绘制散点 + 拟合直线。"""
    x_vals = data[x_col].values
    y_vals = data[y_col].values
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    # 使用模型参数预测：const + beta * x
    const = float(results.params.get("const", 0.0))
    beta = float(results.params.get(x_col, 0.0))
    y_line = const + beta * x_line

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name="观测值",
            marker={
                "size": 8,
                "color": "rgba(15, 118, 110, 0.55)",
                "line": {"width": 0.5, "color": "#0f766e"},
            },
            hovertemplate=f"{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="OLS 拟合线",
            line={"color": "#b45309", "width": 2.5},
            hovertemplate="拟合值: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"OLS 拟合直线：{y_col} ~ {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=460,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _actual_vs_pred_figure(y_true: np.ndarray, y_pred: np.ndarray, y_col: str) -> go.Figure:
    """多元回归时绘制实际值 vs 预测值散点图。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="样本点",
            marker={"size": 8, "color": "rgba(15, 118, 110, 0.55)"},
            hovertemplate="实际: %{x:.4f}<br>预测: %{y:.4f}<extra></extra>",
        )
    )
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="理想拟合 (y=x)",
            line={"color": "#b45309", "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"实际值 vs 预测值（因变量: {y_col}）",
        xaxis_title="实际值",
        yaxis_title="预测值",
        height=460,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _residual_plots(y_pred: np.ndarray, residuals: np.ndarray) -> tuple:
    """绘制残差散点图与残差分布直方图。"""
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="残差",
            marker={"size": 8, "color": "rgba(3, 105, 161, 0.55)"},
            hovertemplate="拟合值: %{x:.4f}<br>残差: %{y:.4f}<extra></extra>",
        )
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#be123c", line_width=1.5)
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title="残差图（拟合值 vs 残差）",
        xaxis_title="拟合值",
        yaxis_title="残差",
        height=400,
    )

    fig_hist = px.histogram(
        x=residuals,
        nbins=40,
        opacity=0.85,
        color_discrete_sequence=["#0369a1"],
        labels={"x": "残差"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#be123c", line_width=1.5)
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title="残差分布直方图",
        xaxis_title="残差",
        yaxis_title="频数",
        height=400,
        showlegend=False,
    )
    return fig_scatter, fig_hist


def render_regression(df: pd.DataFrame) -> None:
    """渲染模块 2 完整页面。"""
    st.markdown('<div class="hero-title">📈 统计建模与 OLS 回归分析</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">自由选择因变量与自变量，运行 statsmodels OLS，'
        "查看完整回归报告与诊断图表。</div>",
        unsafe_allow_html=True,
    )

    if not validate_dataframe(df, min_rows=5):
        return

    try:
        num_cols = get_numeric_columns(df)
        if len(num_cols) < 2:
            st.warning("OLS 回归至少需要 2 个数值型变量（1 个因变量 + 1 个自变量）。")
            return

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ 模型设定</div>', unsafe_allow_html=True)

        col_y, col_x = st.columns([1, 2])
        with col_y:
            # 默认优先选择常见因变量名
            default_y_candidates = [
                c
                for c in num_cols
                if any(
                    k in c.lower()
                    for k in ("ret", "y", "target", "market", "excess", "close")
                )
            ]
            default_y = default_y_candidates[0] if default_y_candidates else num_cols[0]
            y_col = st.selectbox("因变量 Y", options=num_cols, index=num_cols.index(default_y))

        with col_x:
            x_candidates = [c for c in num_cols if c != y_col]
            default_x = x_candidates[: min(3, len(x_candidates))]
            x_cols = st.multiselect(
                "自变量 X（可多选）",
                options=x_candidates,
                default=default_x,
            )

        run_btn = st.button("🚀 运行 OLS 回归", type="primary", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if not x_cols:
            st.info("请至少选择一个自变量。")
            return

        if not run_btn and "ols_cache" not in st.session_state:
            st.info("设定好变量后，点击「运行 OLS 回归」开始估计。")
            return

        # 点击运行或使用缓存结果
        if run_btn:
            results, data = _fit_ols(df, y_col, x_cols)
            st.session_state["ols_cache"] = {
                "results": results,
                "data": data,
                "y_col": y_col,
                "x_cols": x_cols,
            }

        cache = st.session_state.get("ols_cache")
        if not cache:
            return

        results = cache["results"]
        data = cache["data"]
        y_col = cache["y_col"]
        x_cols = cache["x_cols"]

        # 关键指标卡片
        render_metric_cards(
            [
                ("R²", f"{results.rsquared:.4f}"),
                ("Adj. R²", f"{results.rsquared_adj:.4f}"),
                ("F 统计量", f"{results.fvalue:.3f}"),
                ("样本量 n", f"{int(results.nobs)}"),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_sum, tab_fit, tab_res = st.tabs(
            ["📄 回归摘要", "📐 拟合图", "🧪 残差诊断"]
        )

        with tab_sum:
            st.markdown(
                '<div class="section-title">statsmodels OLS Regression Summary</div>',
                unsafe_allow_html=True,
            )
            # 完整文本摘要
            st.code(results.summary().as_text(), language=None)

            # 系数表便于阅读
            coef_df = pd.DataFrame(
                {
                    "系数": results.params,
                    "标准误": results.bse,
                    "t 值": results.tvalues,
                    "P>|t|": results.pvalues,
                    "置信下界(0.025)": results.conf_int()[0],
                    "置信上界(0.975)": results.conf_int()[1],
                }
            )
            st.dataframe(coef_df.round(6), use_container_width=True)

            # 显著性提示
            sig = coef_df[coef_df["P>|t|"] < 0.05].index.tolist()
            sig = [s for s in sig if s != "const"]
            if sig:
                st.success(f"在 5% 显著性水平下显著的自变量: {', '.join(map(str, sig))}")
            else:
                st.warning("在 5% 显著性水平下，没有自变量系数显著异于零。")

        with tab_fit:
            y_pred = results.fittedvalues.values
            y_true = data[y_col].values
            if len(x_cols) == 1:
                fig = _fit_line_figure(data, y_col, x_cols[0], results)
            else:
                fig = _actual_vs_pred_figure(y_true, y_pred, y_col)
            st.plotly_chart(fig, use_container_width=True)

        with tab_res:
            residuals = results.resid.values
            y_pred = results.fittedvalues.values
            fig_scatter, fig_hist = _residual_plots(y_pred, residuals)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_scatter, use_container_width=True)
            with c2:
                st.plotly_chart(fig_hist, use_container_width=True)

            # 简单诊断指标
            dw = sm.stats.stattools.durbin_watson(residuals)
            jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(residuals)
            st.markdown(
                f"""
                **残差诊断速览**
                - Durbin-Watson 统计量: `{dw:.4f}`（接近 2 表示残差自相关较弱）
                - Jarque-Bera 正态性检验: 统计量 `{jb_stat:.4f}`，p 值 `{jb_p:.4g}`
                - 残差偏度: `{skew:.4f}`，峰度: `{kurt:.4f}`
                """
            )

    except Exception as exc:  # noqa: BLE001
        st.error(f"回归分析过程中发生错误: {exc}")
        st.exception(exc)
