"""
模块 3：金融量化与期权定价
包含 Black-Scholes 定价 / 希腊值 与 几何布朗运动蒙特卡洛模拟。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

from utils.guides import render_beginner_guide
from utils.report_export import register_report_section
from utils.styles import PLOTLY_LAYOUT, render_metric_cards

from .risk_metrics import render_risk_tools


# ---------------------------------------------------------------------------
# Black-Scholes 核心公式
# ---------------------------------------------------------------------------

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    计算欧式期权 Black-Scholes 价格。

    Parameters
    ----------
    S : 标的现价
    K : 行权价
    T : 到期时间（年）
    r : 无风险利率（连续复利）
    sigma : 波动率
    option_type : 'call' 或 'put'
    """
    if T <= 0:
        # 到期时内在价值
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0:
        # 零波动时退化为折现内在价值
        forward = S * np.exp(r * T)
        if option_type == "call":
            return np.exp(-r * T) * max(forward - K, 0.0)
        return np.exp(-r * T) * max(K - forward, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    计算 Call / Put 的主要希腊值。
    Vega 以「波动率变动 1 个百分点」为单位（即原始 Vega / 100）。
    Theta 以「每日」为单位（原始年化 Theta / 365）。
    """
    if T <= 0 or sigma <= 0:
        # 边界情形给出安全默认值
        call_price = black_scholes_price(S, K, max(T, 1e-8), r, max(sigma, 1e-8), "call")
        put_price = black_scholes_price(S, K, max(T, 1e-8), r, max(sigma, 1e-8), "put")
        return {
            "call_price": call_price,
            "put_price": put_price,
            "delta_call": 1.0 if S > K else 0.0,
            "delta_put": -1.0 if S < K else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta_call": 0.0,
            "theta_put": 0.0,
            "rho_call": 0.0,
            "rho_put": 0.0,
        }

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = norm.pdf(d1)

    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    put_price = black_scholes_price(S, K, T, r, sigma, "put")

    delta_call = float(norm.cdf(d1))
    delta_put = float(delta_call - 1.0)
    gamma = float(pdf_d1 / (S * sigma * sqrt_t))
    vega = float(S * pdf_d1 * sqrt_t / 100.0)  # 每 1% 波动率

    theta_call_annual = (
        -S * pdf_d1 * sigma / (2 * sqrt_t)
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    theta_put_annual = (
        -S * pdf_d1 * sigma / (2 * sqrt_t)
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )
    theta_call = float(theta_call_annual / 365.0)
    theta_put = float(theta_put_annual / 365.0)

    rho_call = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0)
    rho_put = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0)

    return {
        "call_price": call_price,
        "put_price": put_price,
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


def _option_price_curve(K: float, T: float, r: float, sigma: float, S_center: float) -> go.Figure:
    """绘制 Call/Put 价格随标的价格 S 变化的曲线。"""
    s_min = max(S_center * 0.5, 0.01)
    s_max = S_center * 1.5
    s_grid = np.linspace(s_min, s_max, 120)
    calls = [black_scholes_price(s, K, T, r, sigma, "call") for s in s_grid]
    puts = [black_scholes_price(s, K, T, r, sigma, "put") for s in s_grid]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_grid,
            y=calls,
            mode="lines",
            name="Call 价格",
            line={"color": "#0f766e", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s_grid,
            y=puts,
            mode="lines",
            name="Put 价格",
            line={"color": "#b45309", "width": 2.5},
        )
    )
    fig.add_vline(
        x=S_center,
        line_dash="dot",
        line_color="#64748b",
        annotation_text=f"当前 S={S_center:.2f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=K,
        line_dash="dash",
        line_color="#be123c",
        annotation_text=f"行权价 K={K:.2f}",
        annotation_position="bottom",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="期权价格随标的资产价格变化",
        xaxis_title="标的资产价格 S",
        yaxis_title="期权价格",
        height=460,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


# ---------------------------------------------------------------------------
# 蒙特卡洛 GBM 模拟
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    days: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    """
    基于几何布朗运动模拟资产价格路径。

    离散化: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns
    -------
    paths : ndarray, shape (n_sims, days+1)，含初始价格列
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    z = rng.standard_normal((n_sims, days))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)
    # 在最前方插入初始价格
    paths = np.concatenate([np.full((n_sims, 1), S0), paths], axis=1)
    return paths


def _paths_figure(paths: np.ndarray, max_display: int = 80) -> go.Figure:
    """绘制蒙特卡洛全路径（抽样展示以避免卡顿）。"""
    n_sims, n_steps = paths.shape
    days = np.arange(n_steps)
    fig = go.Figure()

    # 抽样路径用于展示
    display_n = min(n_sims, max_display)
    idx = np.linspace(0, n_sims - 1, display_n, dtype=int)
    for i in idx:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=paths[i],
                mode="lines",
                line={"width": 1, "color": "rgba(15, 118, 110, 0.18)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 均值路径与分位数带
    mean_path = paths.mean(axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig.add_trace(
        go.Scatter(
            x=days,
            y=p95,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p5,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(180, 83, 9, 0.15)",
            line={"width": 0},
            name="5%–95% 分位带",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean_path,
            mode="lines",
            name="均值路径",
            line={"color": "#b45309", "width": 2.8},
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"GBM 蒙特卡洛路径模拟（展示 {display_n}/{n_sims} 条）",
        xaxis_title="交易日",
        yaxis_title="资产价格",
        height=480,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _terminal_hist(paths: np.ndarray) -> go.Figure:
    """终值分布直方图。"""
    terminal = paths[:, -1]
    fig = px.histogram(
        x=terminal,
        nbins=50,
        opacity=0.85,
        color_discrete_sequence=["#0f766e"],
        labels={"x": "终值价格"},
    )
    fig.add_vline(
        x=float(np.mean(terminal)),
        line_dash="dash",
        line_color="#b45309",
        annotation_text=f"均值 {np.mean(terminal):.2f}",
    )
    fig.add_vline(
        x=float(np.median(terminal)),
        line_dash="dot",
        line_color="#0369a1",
        annotation_text=f"中位数 {np.median(terminal):.2f}",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="模拟终值分布直方图",
        xaxis_title="终值价格 S_T",
        yaxis_title="频数",
        height=420,
        showlegend=False,
    )
    return fig


def _render_black_scholes() -> None:
    """Black-Scholes 定价子模块 UI。"""
    st.markdown(
        '<div class="section-title">🖤 Black-Scholes 欧式期权定价</div>',
        unsafe_allow_html=True,
    )
    st.caption("输入参数后即时计算 Call/Put 价格与希腊值，并绘制价格曲线。")

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("标的资产价格 S", min_value=0.01, value=100.0, step=1.0, key="bs_s")
        K = st.number_input("行权价 K", min_value=0.01, value=100.0, step=1.0, key="bs_k")
    with c2:
        T = st.number_input("到期时间 T（年）", min_value=0.001, value=1.0, step=0.05, format="%.3f", key="bs_t")
        r = st.number_input("无风险利率 r", min_value=-0.5, value=0.05, step=0.005, format="%.4f", key="bs_r")
    with c3:
        sigma = st.number_input("波动率 σ", min_value=0.001, value=0.20, step=0.01, format="%.4f", key="bs_sig")
        st.markdown(
            f"<div style='margin-top:1.8rem;color:#64748b;font-size:0.9rem;'>"
            f"隐含状态：{'平值 ATM' if abs(S-K)<1e-6 else ('实值 ITM(Call)' if S>K else '虚值 OTM(Call)')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            st.error("请确保 S、K、T、σ 均为正数。")
            return

        greeks = black_scholes_greeks(S, K, T, r, sigma)

        render_metric_cards(
            [
                ("Call 价格", f"{greeks['call_price']:.4f}"),
                ("Put 价格", f"{greeks['put_price']:.4f}"),
                ("Delta (Call)", f"{greeks['delta_call']:.4f}"),
                ("Gamma", f"{greeks['gamma']:.6f}"),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Vega (每1%σ)", f"{greeks['vega']:.4f}")
        g2.metric("Theta Call (每日)", f"{greeks['theta_call']:.4f}")
        g3.metric("Theta Put (每日)", f"{greeks['theta_put']:.4f}")
        g4.metric("Delta (Put)", f"{greeks['delta_put']:.4f}")

        # 希腊值明细表
        greek_table = pd.DataFrame(
            {
                "指标": [
                    "Call Price",
                    "Put Price",
                    "Delta Call",
                    "Delta Put",
                    "Gamma",
                    "Vega (/1%)",
                    "Theta Call (/日)",
                    "Theta Put (/日)",
                    "Rho Call (/1%)",
                    "Rho Put (/1%)",
                ],
                "数值": [
                    greeks["call_price"],
                    greeks["put_price"],
                    greeks["delta_call"],
                    greeks["delta_put"],
                    greeks["gamma"],
                    greeks["vega"],
                    greeks["theta_call"],
                    greeks["theta_put"],
                    greeks["rho_call"],
                    greeks["rho_put"],
                ],
            }
        )
        with st.expander("查看完整希腊值表", expanded=False):
            display_greeks = greek_table.copy()
            display_greeks["数值"] = display_greeks["数值"].map(lambda v: f"{v:.6f}")
            st.dataframe(display_greeks, use_container_width=True)

        fig = _option_price_curve(K, T, r, sigma, S)
        st.plotly_chart(fig, use_container_width=True)

        # 平价关系校验
        parity_lhs = greeks["call_price"] - greeks["put_price"]
        parity_rhs = S - K * np.exp(-r * T)
        st.caption(
            f"欧式期权平价检验：C − P = {parity_lhs:.6f}，"
            f"S − K·e^(−rT) = {parity_rhs:.6f}，"
            f"误差 = {abs(parity_lhs - parity_rhs):.2e}"
        )

        register_report_section(
            module="金融量化",
            title="Black-Scholes 期权定价",
            metrics={
                "S": f"{S:.4f}",
                "K": f"{K:.4f}",
                "T": f"{T:.4f}",
                "r": f"{r:.4f}",
                "σ": f"{sigma:.4f}",
                "Call": f"{greeks['call_price']:.4f}",
                "Put": f"{greeks['put_price']:.4f}",
                "Delta Call": f"{greeks['delta_call']:.4f}",
                "Gamma": f"{greeks['gamma']:.6f}",
                "Vega": f"{greeks['vega']:.4f}",
            },
            figures=[("期权价格曲线", fig)],
            notes="欧式期权 Black-Scholes 定价与希腊值。",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"Black-Scholes 计算失败: {exc}")


def _render_monte_carlo() -> None:
    """蒙特卡洛 GBM 模拟子模块 UI。"""
    st.markdown(
        '<div class="section-title">🎲 蒙特卡洛资产价格模拟（几何布朗运动）</div>',
        unsafe_allow_html=True,
    )
    st.caption("设定初始价格、漂移、波动率与模拟规模，生成未来价格路径及终值分布。")

    c1, c2, c3 = st.columns(3)
    with c1:
        S0 = st.number_input("初始价格 S₀", min_value=0.01, value=100.0, step=1.0, key="mc_s0")
        mu = st.number_input("年化漂移 μ", value=0.10, step=0.01, format="%.4f", key="mc_mu")
    with c2:
        sigma = st.number_input("年化波动率 σ", min_value=0.001, value=0.25, step=0.01, format="%.4f", key="mc_sig")
        days = st.number_input("未来交易日数", min_value=1, max_value=1260, value=252, step=1, key="mc_days")
    with c3:
        n_sims = st.number_input("模拟路径数", min_value=10, max_value=5000, value=500, step=50, key="mc_n")
        seed = st.number_input("随机种子", min_value=0, value=42, step=1, key="mc_seed")

    run = st.button("▶ 开始蒙特卡洛模拟", type="primary", key="mc_run")

    if not run and "mc_paths" not in st.session_state:
        st.info("调整参数后点击「开始蒙特卡洛模拟」。")
        return

    try:
        if run:
            with st.spinner("正在生成随机路径..."):
                paths = simulate_gbm_paths(
                    S0=float(S0),
                    mu=float(mu),
                    sigma=float(sigma),
                    days=int(days),
                    n_sims=int(n_sims),
                    seed=int(seed),
                )
                st.session_state["mc_paths"] = paths

        paths = st.session_state["mc_paths"]
        terminal = paths[:, -1]

        render_metric_cards(
            [
                ("终值均值", f"{terminal.mean():.2f}"),
                ("终值中位数", f"{np.median(terminal):.2f}"),
                ("终值标准差", f"{terminal.std():.2f}"),
                ("5% VaR 分位", f"{np.percentile(terminal, 5):.2f}"),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        fig_paths = _paths_figure(paths)
        st.plotly_chart(fig_paths, use_container_width=True)

        fig_hist = _terminal_hist(paths)
        st.plotly_chart(fig_hist, use_container_width=True)

        # 终值分位数表
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        q_df = pd.DataFrame(
            {
                "分位数": [f"{q}%" for q in qs],
                "终值": [np.percentile(terminal, q) for q in qs],
            }
        )
        with st.expander("终值分位数明细", expanded=False):
            display_q = q_df.copy()
            display_q["终值"] = display_q["终值"].map(lambda v: f"{v:.4f}")
            st.dataframe(display_q, use_container_width=True)

        register_report_section(
            module="金融量化",
            title="蒙特卡洛 GBM 模拟",
            metrics={
                "终值均值": f"{terminal.mean():.2f}",
                "终值中位数": f"{np.median(terminal):.2f}",
                "终值标准差": f"{terminal.std():.2f}",
                "5%分位": f"{np.percentile(terminal, 5):.2f}",
                "路径数": str(paths.shape[0]),
                "交易日": str(paths.shape[1] - 1),
            },
            tables=[("终值分位数", q_df)],
            figures=[("GBM 路径", fig_paths), ("终值分布", fig_hist)],
            notes="几何布朗运动蒙特卡洛资产价格路径模拟。",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"蒙特卡洛模拟失败: {exc}")
        st.exception(exc)


def render_finance(df: pd.DataFrame = None) -> None:
    """
    渲染模块 3 完整页面。
    本模块以参数计算器为主，数据集可选用于提示历史波动率。
    """
    st.markdown('<div class="hero-title">💰 金融量化、期权定价与风险度量</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Black-Scholes 定价、GBM 蒙特卡洛模拟，以及 VaR / CVaR / 最大回撤风险工具。</div>',
        unsafe_allow_html=True,
    )
    render_beginner_guide("finance", expanded=False)

    # 若用户加载了含 close/returns 的数据，给出历史波动率参考
    if df is not None and not df.empty:
        try:
            from utils.data_loader import get_numeric_columns

            num_cols = get_numeric_columns(df)
            ret_candidates = [c for c in num_cols if "ret" in c.lower()]
            price_candidates = [
                c for c in num_cols if c.lower() in ("close", "adj_close", "price", "adj close")
            ]
            hist_vol = None
            if ret_candidates:
                hist_vol = float(df[ret_candidates[0]].dropna().std() * np.sqrt(252))
            elif price_candidates:
                prices = df[price_candidates[0]].dropna().astype(float)
                if len(prices) > 2:
                    log_ret = np.diff(np.log(prices.values))
                    hist_vol = float(np.std(log_ret, ddof=1) * np.sqrt(252))
            if hist_vol is not None and np.isfinite(hist_vol):
                st.info(f"💡 根据当前数据集估算的历史年化波动率约为 **{hist_vol:.2%}**，可作 σ 参考。")
        except Exception:  # noqa: BLE001
            pass

    tab_bs, tab_mc, tab_risk = st.tabs(
        ["🖤 Black-Scholes 定价", "🎲 蒙特卡洛 GBM 模拟", "⚠️ 风险度量与回测"]
    )
    with tab_bs:
        _render_black_scholes()
    with tab_mc:
        _render_monte_carlo()
    with tab_risk:
        render_risk_tools(df if df is not None else pd.DataFrame())
