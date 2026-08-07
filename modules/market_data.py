"""
模块 5：实时/历史市场数据接驳与技术指标计算器
使用 yfinance 拉取行情，计算 MA / MACD / RSI，并用 Plotly 绘制交互式量化图表。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.report_export import register_report_section
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def fetch_market_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    通过 yfinance 拉取历史行情数据。

    Parameters
    ----------
    ticker : 股票代码，如 AAPL、TSLA、600519.SS
    period : 时间跨度，默认 1y
    interval : K 线周期，默认 1d
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("股票代码不能为空。")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)

    if df is None or df.empty:
        # 回退：用 download 接口再试一次
        end = datetime.now()
        start = end - timedelta(days=400)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

    if df is None or df.empty:
        raise ValueError(
            f"未能获取「{ticker}」的行情数据。请检查代码是否正确，或网络是否可访问 Yahoo Finance。"
        )

    # 统一列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"行情数据缺少必要字段: {missing}")

    df = df.dropna(subset=required).copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA20/MA60、MACD、RSI(14)。"""
    out = df.copy()
    close = out["close"].astype(float)

    out["ma20"] = close.rolling(window=20, min_periods=1).mean()
    out["ma60"] = close.rolling(window=60, min_periods=1).mean()

    # MACD: EMA12 - EMA26，信号线 EMA9，柱状图
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI(14) —— Wilder 平滑
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi"] = out["rsi"].fillna(50.0)

    out["returns"] = close.pct_change()
    return out


def build_quant_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    构建含 K 线、MA20/MA60、成交量、MACD、RSI 的多行 Plotly 图表。
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.42, 0.16, 0.22, 0.20],
        subplot_titles=(
            f"{ticker} · K线 / MA20 / MA60",
            "成交量",
            "MACD (12,26,9)",
            "RSI (14)",
        ),
    )

    # K 线
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#0f766e",
            increasing_fillcolor="#0f766e",
            decreasing_line_color="#be123c",
            decreasing_fillcolor="#be123c",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma20"],
            mode="lines",
            name="MA20",
            line={"color": "#b45309", "width": 1.6},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma60"],
            mode="lines",
            name="MA60",
            line={"color": "#0369a1", "width": 1.6},
        ),
        row=1,
        col=1,
    )

    # 成交量
    if "volume" in df.columns:
        colors = np.where(df["close"] >= df["open"], "rgba(15,118,110,0.65)", "rgba(190,18,60,0.65)")
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="成交量",
                marker_color=colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # MACD
    hist_colors = np.where(df["macd_hist"] >= 0, "rgba(15,118,110,0.7)", "rgba(190,18,60,0.7)")
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["macd_hist"],
            name="MACD Hist",
            marker_color=hist_colors,
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["macd"],
            mode="lines",
            name="MACD",
            line={"color": "#0f766e", "width": 1.5},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["macd_signal"],
            mode="lines",
            name="Signal",
            line={"color": "#b45309", "width": 1.5},
        ),
        row=3,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi"],
            mode="lines",
            name="RSI",
            line={"color": "#7c3aed", "width": 1.6},
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(190,18,60,0.55)", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(15,118,110,0.55)", row=4, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(15,118,110,0.04)", line_width=0, row=4, col=1)

    layout_kwargs = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **layout_kwargs,
        height=920,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    fig.update_xaxes(title_text="日期", row=4, col=1)
    return fig


def render_market_data(df: pd.DataFrame = None) -> None:
    """渲染市场数据与技术指标模块页面。"""
    st.markdown(
        '<div class="hero-title">📡 市场数据接驳与技术指标</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">输入任意股票代码，自动拉取近一年行情，'
        "并绘制 K 线、均线、MACD 与 RSI 交互式量化图表。</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔎 行情拉取设置</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        ticker = st.text_input(
            "股票代码",
            value=st.session_state.get("market_ticker", "AAPL"),
            placeholder="例如 AAPL / TSLA / 600519.SS / 0700.HK",
            help="美股直接代码；A 股加 .SS/.SZ；港股加 .HK",
        )
    with c2:
        period = st.selectbox(
            "时间跨度",
            options=["1y", "6mo", "3mo", "2y", "5y"],
            index=0,
            key="market_period",
        )
    with c3:
        interval = st.selectbox(
            "K线周期",
            options=["1d", "1wk", "1h"],
            index=0,
            key="market_interval",
        )
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("⬇ 拉取并计算", type="primary", use_container_width=True)

    quick = st.columns(5)
    for i, code in enumerate(["AAPL", "TSLA", "MSFT", "600519.SS", "0700.HK"]):
        if quick[i].button(code, key=f"quick_{code}", use_container_width=True):
            ticker = code
            st.session_state["market_ticker"] = code
            fetch_btn = True

    st.markdown("</div>", unsafe_allow_html=True)

    if fetch_btn or "market_df" in st.session_state:
        try:
            if fetch_btn:
                with st.spinner(f"正在从 Yahoo Finance 拉取 {ticker} ..."):
                    raw = fetch_market_data(ticker, period=period, interval=interval)
                    enriched = compute_technical_indicators(raw)
                    st.session_state["market_df"] = enriched
                    st.session_state["market_ticker"] = ticker.strip().upper()
                    st.session_state["market_meta"] = {
                        "period": period,
                        "interval": interval,
                    }

            data = st.session_state["market_df"]
            tk = st.session_state.get("market_ticker", ticker)
            last = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else last
            chg = (last["close"] / prev["close"] - 1) if prev["close"] else 0.0

            render_metric_cards(
                [
                    ("最新收盘", f"{last['close']:.2f}"),
                    ("日涨跌", f"{chg:+.2%}"),
                    ("RSI(14)", f"{last['rsi']:.1f}"),
                    ("MACD", f"{last['macd']:.3f}"),
                ]
            )
            st.markdown("<br>", unsafe_allow_html=True)

            fig = build_quant_chart(data, tk)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("查看指标数据表", expanded=False):
                show_cols = [
                    c
                    for c in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "ma20",
                        "ma60",
                        "macd",
                        "macd_signal",
                        "macd_hist",
                        "rsi",
                        "returns",
                    ]
                    if c in data.columns
                ]
                st.dataframe(data[show_cols].tail(120).round(4), use_container_width=True)

            # 注册到报告
            register_report_section(
                module="市场数据与技术指标",
                title=f"{tk} 技术分析",
                metrics={
                    "代码": tk,
                    "最新收盘": f"{last['close']:.2f}",
                    "日涨跌": f"{chg:+.2%}",
                    "RSI(14)": f"{last['rsi']:.1f}",
                    "MA20": f"{last['ma20']:.2f}",
                    "MA60": f"{last['ma60']:.2f}",
                    "样本数": str(len(data)),
                },
                figures=[("量化技术图表", fig)],
                notes="数据来源：Yahoo Finance（yfinance）。指标含 MA20/MA60、MACD、RSI(14)。",
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"市场数据模块错误: {exc}")
            st.exception(exc)
    else:
        st.info("输入股票代码后点击「拉取并计算」，或使用快捷按钮。")
