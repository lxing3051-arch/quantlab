"""
风险度量与回测工具
计算 VaR / CVaR（历史模拟、正态参数、蒙特卡洛）以及最大回撤，并可视化。
供金融量化模块作为子页签调用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

from utils.data_loader import get_numeric_columns, validate_dataframe
from utils.report_export import register_report_section
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def extract_return_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    从选定列提取收益率序列。
    若数值像价格（均值较大且全为正），自动转为日收益率。
    """
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise ValueError("所选列无有效数值。")

    looks_like_price = s.mean() > 1.0 and (s > 0).all() and s.std() / abs(s.mean()) < 0.5
    # 更稳健：若列名含 price/close，或相邻比值接近 1
    name_hint = any(k in col.lower() for k in ("close", "price", "adj"))
    if name_hint or (looks_like_price and s.min() > 1):
        rets = s.pct_change().dropna()
        st.info(f"列「{col}」已按价格序列转换为日收益率。")
    else:
        rets = s
        # 若绝对值普遍 > 0.5，可能是百分比单位，尝试 /100
        if rets.abs().median() > 0.5:
            rets = rets / 100.0
            st.info("检测到收益率可能以百分比记录，已自动除以 100。")

    if len(rets) < 30:
        raise ValueError(f"有效收益率样本过少（{len(rets)}），建议至少 30 个。")
    return rets.astype(float)


def compute_var_cvar(
    returns: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict:
    """
    计算三种方法下的 VaR 与 CVaR。

    约定：返回值为「损失为正」的比例（例如 0.03 表示 3% 损失）。
    alpha=0.95 表示 95% 置信度，关注左尾 5%。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        raise ValueError("收益率样本不足，无法计算风险指标。")

    tail_q = 1.0 - alpha  # 例如 0.05

    # ---- 1. 历史模拟法 ----
    hist_var = -float(np.quantile(r, tail_q))
    hist_tail = r[r <= -hist_var]
    hist_cvar = -float(hist_tail.mean()) if len(hist_tail) else hist_var

    # ---- 2. 正态分布法（参数法）----
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = float(norm.ppf(tail_q))
    norm_var = -(mu + sigma * z)
    # 正态 CVaR = -(mu - sigma * phi(z) / tail_q)
    norm_cvar = -(mu - sigma * float(norm.pdf(z)) / tail_q)

    # ---- 3. 蒙特卡洛法（基于正态假设模拟）----
    rng = np.random.default_rng(seed)
    sims = rng.normal(mu, sigma, size=n_sims)
    mc_var = -float(np.quantile(sims, tail_q))
    mc_tail = sims[sims <= -mc_var]
    mc_cvar = -float(mc_tail.mean()) if len(mc_tail) else mc_var

    return {
        "alpha": alpha,
        "mu": mu,
        "sigma": sigma,
        "historical": {"VaR": hist_var, "CVaR": hist_cvar},
        "normal": {"VaR": norm_var, "CVaR": norm_cvar},
        "monte_carlo": {"VaR": mc_var, "CVaR": mc_cvar, "n_sims": n_sims},
    }


def compute_max_drawdown(price_or_wealth: pd.Series) -> dict:
    """
    计算最大回撤及起止区间。

    Parameters
    ----------
    price_or_wealth : 价格或净值序列（索引为时间）
    """
    s = price_or_wealth.astype(float).dropna()
    if s.empty:
        raise ValueError("净值序列为空。")

    running_max = s.cummax()
    drawdown = s / running_max - 1.0
    min_dd_idx = drawdown.idxmin()
    max_dd = float(drawdown.loc[min_dd_idx])

    # 回撤起点：回撤谷底之前的峰值
    peak_idx = s.loc[:min_dd_idx].idxmax()
    recovery_candidates = s.loc[min_dd_idx:]
    recovered = recovery_candidates[recovery_candidates >= s.loc[peak_idx]]
    recovery_idx = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_idx,
        "trough_date": min_dd_idx,
        "recovery_date": recovery_idx,
        "drawdown_series": drawdown,
        "wealth": s,
        "peak_value": float(s.loc[peak_idx]),
        "trough_value": float(s.loc[min_dd_idx]),
    }


def build_drawdown_figure(mdd: dict, title: str = "净值曲线与最大回撤") -> go.Figure:
    """绘制净值曲线，并用红色半透明阴影高亮最大回撤区间。"""
    wealth = mdd["wealth"]
    dd = mdd["drawdown_series"]
    peak = mdd["peak_date"]
    trough = mdd["trough_date"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=(title, "回撤序列"),
    )

    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth.values,
            mode="lines",
            name="净值 / 价格",
            line={"color": "#0f766e", "width": 2},
        ),
        row=1,
        col=1,
    )

    # 红线描边最大回撤区间路径 + vrect 红色半透明阴影
    mask = (wealth.index >= peak) & (wealth.index <= trough)
    fig.add_trace(
        go.Scatter(
            x=wealth.index[mask],
            y=wealth.values[mask],
            mode="lines",
            name="最大回撤区间",
            line={"color": "#be123c", "width": 2.8},
        ),
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=peak,
        x1=trough,
        fillcolor="rgba(190, 18, 60, 0.18)",
        line_width=0,
        row=1,
        col=1,
        annotation_text="Max DD",
        annotation_position="top left",
    )

    fig.add_trace(
        go.Scatter(
            x=[peak],
            y=[mdd["peak_value"]],
            mode="markers+text",
            name="峰值",
            marker={"size": 11, "color": "#b45309", "symbol": "triangle-up"},
            text=["Peak"],
            textposition="top center",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[trough],
            y=[mdd["trough_value"]],
            mode="markers+text",
            name="谷底",
            marker={"size": 11, "color": "#be123c", "symbol": "triangle-down"},
            text=["Trough"],
            textposition="bottom center",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="回撤",
            line={"color": "#be123c", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(190, 18, 60, 0.2)",
        ),
        row=2,
        col=1,
    )
    fig.add_vrect(
        x0=peak,
        x1=trough,
        fillcolor="rgba(190, 18, 60, 0.18)",
        line_width=0,
        row=2,
        col=1,
    )

    layout_kwargs = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **layout_kwargs,
        height=620,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        yaxis2_tickformat=".1%",
    )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1)
    return fig


def build_var_distribution_figure(returns: np.ndarray, risk: dict) -> go.Figure:
    """收益率分布直方图 + 各方法 VaR 竖线。"""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="收益率分布",
            marker_color="rgba(15, 118, 110, 0.55)",
            opacity=0.85,
        )
    )

    colors = {
        "历史模拟 VaR": "#be123c",
        "正态 VaR": "#b45309",
        "蒙特卡洛 VaR": "#0369a1",
    }
    values = {
        "历史模拟 VaR": -risk["historical"]["VaR"],
        "正态 VaR": -risk["normal"]["VaR"],
        "蒙特卡洛 VaR": -risk["monte_carlo"]["VaR"],
    }
    for name, x in values.items():
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color=colors[name],
            annotation_text=name,
            annotation_position="top",
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"收益率分布与 VaR（置信度 {risk['alpha']:.0%}）",
        xaxis_title="日收益率",
        yaxis_title="频数",
        height=420,
        showlegend=False,
    )
    return fig


def render_risk_tools(df: pd.DataFrame) -> None:
    """渲染风险度量与回测工具 UI（金融模块子页签）。"""
    st.markdown(
        '<div class="section-title">⚠️ 风险度量与回测工具</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "基于历史收益率计算 VaR / CVaR（历史模拟、正态分布、蒙特卡洛），"
        "并识别最大回撤区间（红色阴影高亮）。"
    )

    if not validate_dataframe(df, min_rows=30):
        st.warning("请先在侧边栏加载含收益率或价格序列的数据集。")
        return

    try:
        num_cols = get_numeric_columns(df)
        if not num_cols:
            st.warning("当前数据无数值列。")
            return

        # 智能默认列
        default_candidates = [
            c
            for c in num_cols
            if any(k in c.lower() for k in ("ret", "close", "price", "pnl"))
        ]
        default_col = default_candidates[0] if default_candidates else num_cols[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            col = st.selectbox(
                "选择收益率 / 价格列",
                options=num_cols,
                index=num_cols.index(default_col),
                key="risk_col",
            )
        with c2:
            alpha = st.slider(
                "置信度 α",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="risk_alpha",
            )
        with c3:
            n_sims = st.number_input(
                "蒙特卡洛模拟次数",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                key="risk_sims",
            )

        run = st.button("▶ 计算风险指标", type="primary", key="risk_run")

        if not run and "risk_cache" not in st.session_state:
            st.info("选择列与置信度后点击「计算风险指标」。")
            return

        if run:
            returns = extract_return_series(df, col)
            risk = compute_var_cvar(
                returns.values, alpha=float(alpha), n_sims=int(n_sims)
            )

            # 构建净值序列用于回撤
            wealth = (1 + returns).cumprod()
            wealth.name = "wealth"
            if isinstance(returns.index, pd.DatetimeIndex) or returns.index.dtype == "object":
                wealth.index = returns.index
            mdd = compute_max_drawdown(wealth)

            st.session_state["risk_cache"] = {
                "returns": returns,
                "risk": risk,
                "mdd": mdd,
                "col": col,
            }

        cache = st.session_state["risk_cache"]
        returns = cache["returns"]
        risk = cache["risk"]
        mdd = cache["mdd"]

        render_metric_cards(
            [
                ("历史 VaR", f"{risk['historical']['VaR']:.2%}"),
                ("历史 CVaR", f"{risk['historical']['CVaR']:.2%}"),
                ("最大回撤", f"{mdd['max_drawdown']:.2%}"),
                ("日均收益", f"{risk['mu']:.3%}"),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 三种方法对比表
        cmp_df = pd.DataFrame(
            {
                "方法": ["历史模拟法", "正态分布法", "蒙特卡洛法"],
                "VaR（损失）": [
                    risk["historical"]["VaR"],
                    risk["normal"]["VaR"],
                    risk["monte_carlo"]["VaR"],
                ],
                "CVaR（损失）": [
                    risk["historical"]["CVaR"],
                    risk["normal"]["CVaR"],
                    risk["monte_carlo"]["CVaR"],
                ],
            }
        )
        display_cmp = cmp_df.copy()
        display_cmp["VaR（损失）"] = display_cmp["VaR（损失）"].map(lambda v: f"{v:.4%}")
        display_cmp["CVaR（损失）"] = display_cmp["CVaR（损失）"].map(lambda v: f"{v:.4%}")
        st.dataframe(display_cmp, use_container_width=True, hide_index=True)

        st.caption(
            f"说明：VaR/CVaR 以「正损失比例」表示，置信度 {risk['alpha']:.0%}。"
            f" 日波动率 σ ≈ {risk['sigma']:.4%}；年化波动 ≈ {risk['sigma'] * np.sqrt(252):.2%}。"
        )

        fig_var = build_var_distribution_figure(returns.values, risk)
        st.plotly_chart(fig_var, use_container_width=True)

        fig_dd = build_drawdown_figure(
            mdd, title=f"基于「{cache['col']}」的净值曲线与最大回撤"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        peak_s = str(mdd["peak_date"])
        trough_s = str(mdd["trough_date"])
        rec_s = str(mdd["recovery_date"]) if mdd["recovery_date"] is not None else "尚未恢复"
        st.markdown(
            f"""
            **最大回撤详情**
            - 峰值日期: `{peak_s}`（净值 {mdd['peak_value']:.4f}）
            - 谷底日期: `{trough_s}`（净值 {mdd['trough_value']:.4f}）
            - 恢复日期: `{rec_s}`
            - 最大回撤幅度: `{mdd['max_drawdown']:.2%}`
            """
        )

        register_report_section(
            module="风险度量与回测",
            title="VaR / CVaR / 最大回撤分析",
            metrics={
                "分析列": cache["col"],
                "置信度": f"{risk['alpha']:.0%}",
                "历史VaR": f"{risk['historical']['VaR']:.4%}",
                "历史CVaR": f"{risk['historical']['CVaR']:.4%}",
                "正态VaR": f"{risk['normal']['VaR']:.4%}",
                "蒙特卡洛VaR": f"{risk['monte_carlo']['VaR']:.4%}",
                "最大回撤": f"{mdd['max_drawdown']:.2%}",
                "回撤峰值": peak_s,
                "回撤谷底": trough_s,
            },
            tables=[("VaR/CVaR 方法对比", cmp_df)],
            figures=[
                ("收益率分布与 VaR", fig_var),
                ("净值与最大回撤", fig_dd),
            ],
            notes="VaR/CVaR 含历史模拟、正态分布与蒙特卡洛三种方法；最大回撤区间以红色阴影标注。",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"风险度量计算失败: {exc}")
        st.exception(exc)
