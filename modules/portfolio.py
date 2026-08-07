"""
模块 4：马克维茨资产组合优化 (Markowitz Portfolio)
通过随机采样权重绘制有效前沿，并标注最大夏普与最小方差组合。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import get_numeric_columns, validate_dataframe
from utils.report_export import register_report_section
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def _prepare_returns(df: pd.DataFrame, asset_cols: list) -> pd.DataFrame:
    """
    提取并清洗收益率矩阵。
    若数值看起来像价格（均值 > 1 且几乎全为正），则自动转为对数收益率。
    """
    data = df[asset_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("所选资产列在删除缺失值后无有效数据。")

    # 启发式：像价格序列则转为日收益率
    looks_like_price = (data.mean().abs() > 1.0).all() and (data > 0).all().all()
    if looks_like_price:
        st.info("检测到所选列更像价格序列，已自动转换为对数收益率。")
        data = np.log(data / data.shift(1)).dropna()

    if len(data) < 30:
        raise ValueError(f"有效收益率样本过少（{len(data)}），建议至少 30 个交易日。")

    return data


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray, rf: float) -> tuple:
    """
    计算组合年化收益、年化波动与夏普比率。
    假设输入为日度收益，年化因子 252。
    """
    port_ret = float(np.dot(weights, mean_returns) * 252)
    port_vol = float(np.sqrt(weights.T @ cov @ weights) * np.sqrt(252))
    sharpe = (port_ret - rf) / port_vol if port_vol > 1e-12 else np.nan
    return port_ret, port_vol, sharpe


def _random_portfolios(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int,
    rf: float,
    seed: int = 42,
) -> pd.DataFrame:
    """随机采样权重组合，返回收益/风险/夏普及权重明细。"""
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)
    # Dirichlet 分布天然满足权重非负且和为 1
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    rets = weights @ mean_returns * 252
    # 批量计算波动：diag(W Cov W.T)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights)) * np.sqrt(252)
    sharpes = (rets - rf) / np.where(vols > 1e-12, vols, np.nan)

    result = pd.DataFrame(
        {
            "return": rets,
            "volatility": vols,
            "sharpe": sharpes,
        }
    )
    for i in range(n_assets):
        result[f"w{i}"] = weights[:, i]
    return result


def _efficient_frontier_figure(
    portfolios: pd.DataFrame,
    max_sharpe_row: pd.Series,
    min_var_row: pd.Series,
) -> go.Figure:
    """绘制马克维茨散点图并标注关键组合。"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolios["volatility"],
            y=portfolios["return"],
            mode="markers",
            name="随机组合",
            marker={
                "size": 6,
                "color": portfolios["sharpe"],
                "colorscale": [
                    [0.0, "#be123c"],
                    [0.35, "#f59e0b"],
                    [0.7, "#0d9488"],
                    [1.0, "#0f766e"],
                ],
                "showscale": True,
                "colorbar": {"title": "夏普比率"},
                "opacity": 0.75,
                "line": {"width": 0},
            },
            hovertemplate=(
                "波动率: %{x:.2%}<br>"
                "预期收益: %{y:.2%}<br>"
                "夏普: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )

    # Max Sharpe
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_row["volatility"]],
            y=[max_sharpe_row["return"]],
            mode="markers+text",
            name="最大夏普 (Max Sharpe)",
            marker={
                "size": 16,
                "color": "#b45309",
                "symbol": "star",
                "line": {"width": 1.5, "color": "#fffbeb"},
            },
            text=["Max Sharpe"],
            textposition="top center",
            textfont={"size": 12, "color": "#b45309", "family": "Segoe UI"},
            hovertemplate=(
                "<b>最大夏普组合</b><br>"
                "波动率: %{x:.2%}<br>"
                "预期收益: %{y:.2%}<br>"
                f"夏普: {max_sharpe_row['sharpe']:.3f}<extra></extra>"
            ),
        )
    )

    # Min Variance
    fig.add_trace(
        go.Scatter(
            x=[min_var_row["volatility"]],
            y=[min_var_row["return"]],
            mode="markers+text",
            name="最小方差 (Min Variance)",
            marker={
                "size": 15,
                "color": "#0369a1",
                "symbol": "diamond",
                "line": {"width": 1.5, "color": "#e0f2fe"},
            },
            text=["Min Variance"],
            textposition="bottom center",
            textfont={"size": 12, "color": "#0369a1", "family": "Segoe UI"},
            hovertemplate=(
                "<b>最小方差组合</b><br>"
                "波动率: %{x:.2%}<br>"
                "预期收益: %{y:.2%}<br>"
                f"夏普: {min_var_row['sharpe']:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="马克维茨有效前沿（随机采样近似）",
        xaxis_title="年化波动率 σ",
        yaxis_title="年化预期收益 μ",
        height=560,
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _weights_bar_figure(weights: np.ndarray, asset_names: list, title: str, color: str) -> go.Figure:
    """组合权重柱状图。"""
    order = np.argsort(weights)[::-1]
    names = [asset_names[i] for i in order]
    vals = weights[order]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=color,
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        yaxis_title="权重",
        yaxis_tickformat=".0%",
        height=380,
        showlegend=False,
    )
    return fig


def render_portfolio(df: pd.DataFrame) -> None:
    """渲染模块 4 完整页面。"""
    st.markdown(
        '<div class="hero-title">💼 马克维茨资产组合优化</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">基于均值-协方差矩阵随机采样组合权重，'
        "可视化有效前沿并定位最大夏普与最小方差组合。</div>",
        unsafe_allow_html=True,
    )

    if not validate_dataframe(df, min_rows=30):
        return

    try:
        num_cols = get_numeric_columns(df)
        if len(num_cols) < 2:
            st.warning("组合优化至少需要 2 个数值型资产列。")
            return

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ 参数设置</div>', unsafe_allow_html=True)

        # 默认排除明显非资产列
        exclude_keywords = ("date", "time", "period", "volume", "index", "unnamed")
        default_assets = [
            c for c in num_cols if not any(k in str(c).lower() for k in exclude_keywords)
        ]
        if len(default_assets) < 2:
            default_assets = num_cols[: min(5, len(num_cols))]
        else:
            default_assets = default_assets[: min(8, len(default_assets))]

        asset_cols = st.multiselect(
            "选择资产收益率（或价格）列",
            options=num_cols,
            default=default_assets,
            key="pf_assets",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            n_portfolios = st.slider(
                "随机组合采样次数",
                min_value=500,
                max_value=20000,
                value=5000,
                step=500,
                key="pf_n",
            )
        with c2:
            rf = st.number_input(
                "无风险利率（年化）",
                min_value=0.0,
                max_value=0.2,
                value=0.02,
                step=0.005,
                format="%.4f",
                key="pf_rf",
            )
        with c3:
            seed = st.number_input("随机种子", min_value=0, value=42, step=1, key="pf_seed")

        run_btn = st.button("🚀 生成有效前沿", type="primary", key="pf_run")
        st.markdown("</div>", unsafe_allow_html=True)

        if len(asset_cols) < 2:
            st.info("请至少选择 2 个资产。")
            return

        if not run_btn and "pf_cache" not in st.session_state:
            st.info("选择资产并设置参数后，点击「生成有效前沿」。建议使用内置「多资产日收益率」示例。")
            return

        if run_btn:
            with st.spinner("正在计算均值-协方差并采样组合..."):
                returns = _prepare_returns(df, asset_cols)
                mean_returns = returns.mean().values
                cov = returns.cov().values

                # 协方差矩阵正定性检查与微调
                eigvals = np.linalg.eigvalsh(cov)
                if eigvals.min() < 1e-12:
                    cov = cov + np.eye(len(asset_cols)) * 1e-8

                portfolios = _random_portfolios(
                    mean_returns=mean_returns,
                    cov=cov,
                    n_portfolios=int(n_portfolios),
                    rf=float(rf),
                    seed=int(seed),
                )

                max_sharpe_idx = portfolios["sharpe"].idxmax()
                min_var_idx = portfolios["volatility"].idxmin()
                max_sharpe_row = portfolios.loc[max_sharpe_idx]
                min_var_row = portfolios.loc[min_var_idx]

                st.session_state["pf_cache"] = {
                    "returns": returns,
                    "mean_returns": mean_returns,
                    "cov": cov,
                    "portfolios": portfolios,
                    "max_sharpe_row": max_sharpe_row,
                    "min_var_row": min_var_row,
                    "asset_cols": asset_cols,
                    "rf": float(rf),
                }

        cache = st.session_state.get("pf_cache")
        if not cache:
            return

        returns = cache["returns"]
        mean_returns = cache["mean_returns"]
        cov = cache["cov"]
        portfolios = cache["portfolios"]
        max_sharpe_row = cache["max_sharpe_row"]
        min_var_row = cache["min_var_row"]
        asset_cols = cache["asset_cols"]
        rf = cache["rf"]

        render_metric_cards(
            [
                ("资产数量", f"{len(asset_cols)}"),
                ("样本交易日", f"{len(returns):,}"),
                ("最大夏普", f"{max_sharpe_row['sharpe']:.3f}"),
                ("最小波动", f"{min_var_row['volatility']:.2%}"),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        fig = _efficient_frontier_figure(portfolios, max_sharpe_row, min_var_row)

        tab_frontier, tab_stats, tab_weights = st.tabs(
            ["🌌 有效前沿", "📊 均值协方差", "⚖️ 最优权重"]
        )

        with tab_frontier:
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**⭐ 最大夏普组合**")
                st.write(
                    {
                        "年化收益": f"{max_sharpe_row['return']:.2%}",
                        "年化波动": f"{max_sharpe_row['volatility']:.2%}",
                        "夏普比率": f"{max_sharpe_row['sharpe']:.4f}",
                        "无风险利率": f"{rf:.2%}",
                    }
                )
            with c2:
                st.markdown("**💠 最小方差组合**")
                st.write(
                    {
                        "年化收益": f"{min_var_row['return']:.2%}",
                        "年化波动": f"{min_var_row['volatility']:.2%}",
                        "夏普比率": f"{min_var_row['sharpe']:.4f}",
                        "无风险利率": f"{rf:.2%}",
                    }
                )

        with tab_stats:
            ann_mean = pd.Series(mean_returns * 252, index=asset_cols, name="年化均值收益")
            ann_vol = pd.Series(
                np.sqrt(np.diag(cov)) * np.sqrt(252),
                index=asset_cols,
                name="年化波动率",
            )
            summary = pd.concat([ann_mean, ann_vol], axis=1)
            summary["夏普(单资产)"] = (summary["年化均值收益"] - rf) / summary["年化波动率"]
            st.markdown(
                '<div class="section-title">各资产年化收益与风险</div>',
                unsafe_allow_html=True,
            )
            display_summary = summary.copy()
            display_summary["年化均值收益"] = display_summary["年化均值收益"].map(
                lambda v: f"{v:.2%}"
            )
            display_summary["年化波动率"] = display_summary["年化波动率"].map(
                lambda v: f"{v:.2%}"
            )
            display_summary["夏普(单资产)"] = display_summary["夏普(单资产)"].map(
                lambda v: f"{v:.3f}"
            )
            st.dataframe(display_summary, use_container_width=True)

            st.markdown(
                '<div class="section-title">协方差矩阵（日度）</div>',
                unsafe_allow_html=True,
            )
            cov_df = pd.DataFrame(cov, index=asset_cols, columns=asset_cols)
            st.dataframe(cov_df.round(6), use_container_width=True)

            # 相关矩阵热力（用 plotly）
            corr = returns.corr()
            heat = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=asset_cols,
                    y=asset_cols,
                    colorscale=[
                        [0.0, "#be123c"],
                        [0.5, "#f8fafc"],
                        [1.0, "#0f766e"],
                    ],
                    zmin=-1,
                    zmax=1,
                    colorbar={"title": "ρ"},
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}",
                )
            )
            heat.update_layout(
                **PLOTLY_LAYOUT,
                title="资产收益相关系数矩阵",
                height=480,
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(heat, use_container_width=True)

        with tab_weights:
            n = len(asset_cols)
            w_sharpe = np.array([max_sharpe_row[f"w{i}"] for i in range(n)])
            w_minvar = np.array([min_var_row[f"w{i}"] for i in range(n)])

            fig1 = _weights_bar_figure(
                w_sharpe, asset_cols, "最大夏普组合权重分配", "#b45309"
            )
            fig2 = _weights_bar_figure(
                w_minvar, asset_cols, "最小方差组合权重分配", "#0369a1"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                st.plotly_chart(fig2, use_container_width=True)

            weight_table = pd.DataFrame(
                {
                    "资产": asset_cols,
                    "最大夏普权重": w_sharpe,
                    "最小方差权重": w_minvar,
                }
            )
            display_w = weight_table.copy()
            display_w["最大夏普权重"] = display_w["最大夏普权重"].map(lambda v: f"{v:.2%}")
            display_w["最小方差权重"] = display_w["最小方差权重"].map(lambda v: f"{v:.2%}")
            st.dataframe(display_w, use_container_width=True)

            # 等权组合对照
            eq_w = np.ones(n) / n
            eq_ret, eq_vol, eq_sharpe = _portfolio_stats(eq_w, mean_returns, cov, rf)
            st.caption(
                f"等权组合对照：年化收益 {eq_ret:.2%}，年化波动 {eq_vol:.2%}，夏普 {eq_sharpe:.3f}"
            )

        register_report_section(
            module="资产组合优化",
            title="马克维茨有效前沿",
            metrics={
                "资产数": str(len(asset_cols)),
                "最大夏普": f"{max_sharpe_row['sharpe']:.4f}",
                "Max Sharpe 收益": f"{max_sharpe_row['return']:.2%}",
                "Max Sharpe 波动": f"{max_sharpe_row['volatility']:.2%}",
                "最小波动": f"{min_var_row['volatility']:.2%}",
                "Min Var 收益": f"{min_var_row['return']:.2%}",
            },
            figures=[("有效前沿", fig)],
            notes="随机采样权重近似有效前沿，标注最大夏普与最小方差组合。",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"资产组合优化过程中发生错误: {exc}")
        st.exception(exc)
