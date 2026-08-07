"""
模块 7：看盘助手 —— 自选股、条件选股、单票一页纸。
基于 yfinance 免费数据；股票池为内置常见标的，非真正全市场。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from modules.market_data import (
    build_quant_chart,
    compute_technical_indicators,
    fetch_market_data,
)
from modules.risk_metrics import compute_max_drawdown, compute_var_cvar
from utils.guides import render_beginner_guide
from utils.report_export import register_report_section
from utils.styles import render_metric_cards


# 内置可筛选股票池（免费教学用，覆盖美股/A股/港股常见标的）
STOCK_UNIVERSE: Dict[str, str] = {
    "AAPL": "苹果",
    "MSFT": "微软",
    "NVDA": "英伟达",
    "TSLA": "特斯拉",
    "AMZN": "亚马逊",
    "GOOGL": "谷歌",
    "META": "Meta",
    "JPM": "摩根大通",
    "XOM": "埃克森美孚",
    "JNJ": "强生",
    "600519.SS": "贵州茅台",
    "000858.SZ": "五粮液",
    "601318.SS": "中国平安",
    "600036.SS": "招商银行",
    "000001.SZ": "平安银行",
    "300750.SZ": "宁德时代",
    "002594.SZ": "比亚迪",
    "601012.SS": "隆基绿能",
    "600276.SS": "恒瑞医药",
    "0700.HK": "腾讯控股",
    "9988.HK": "阿里巴巴",
    "3690.HK": "美团",
    "1810.HK": "小米集团",
    "9618.HK": "京东集团",
}


DEFAULT_WATCHLIST = ["AAPL", "TSLA", "600519.SS", "0700.HK", "NVDA"]


def _ensure_watchlist() -> List[str]:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = list(DEFAULT_WATCHLIST)
    return st.session_state["watchlist"]


@st.cache_data(ttl=1800, show_spinner=False)
def _analyze_ticker(ticker: str, period: str = "6mo") -> dict:
    """拉取并计算单票关键指标（带缓存）。"""
    df = fetch_market_data(ticker, period=period, interval="1d")
    ind = compute_technical_indicators(df)
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else last
    close = float(last["close"])
    prev_close = float(prev["close"])
    chg = close / prev_close - 1.0 if prev_close else 0.0

    # 近 20 日收益
    if len(ind) >= 21:
        ret_20 = float(ind["close"].iloc[-1] / ind["close"].iloc[-21] - 1.0)
    else:
        ret_20 = float("nan")

    ma20 = float(last["ma20"])
    ma60 = float(last["ma60"])
    rsi = float(last["rsi"])
    above_ma20 = close >= ma20
    above_ma60 = close >= ma60

    rets = ind["close"].pct_change().dropna()
    vol_20 = float(rets.tail(20).std() * np.sqrt(252)) if len(rets) >= 5 else float("nan")

    return {
        "ticker": ticker,
        "name": STOCK_UNIVERSE.get(ticker, ticker),
        "close": close,
        "change": chg,
        "ret_20d": ret_20,
        "rsi": rsi,
        "ma20": ma20,
        "ma60": ma60,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "vol_ann": vol_20,
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "macd_hist": float(last["macd_hist"]),
        "asof": str(ind.index[-1].date()) if hasattr(ind.index[-1], "date") else str(ind.index[-1]),
        "history": ind,
    }


def _safe_analyze(ticker: str, period: str = "6mo") -> dict | None:
    try:
        return _analyze_ticker(ticker.strip().upper(), period=period)
    except Exception:  # noqa: BLE001
        return None


def _render_watchlist_tab() -> None:
    st.markdown('<div class="section-title">⭐ 自选股看板</div>', unsafe_allow_html=True)
    st.caption("把你想跟踪的股票加进来，定期刷新看涨跌与技术状态。数据来自 Yahoo Finance，可能有延迟。")

    watchlist = _ensure_watchlist()

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_ticker = st.text_input(
            "添加股票代码",
            placeholder="如 AAPL / 600519.SS / 0700.HK",
            key="wd_add_ticker",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ 加入自选", use_container_width=True):
            t = (new_ticker or "").strip().upper()
            if not t:
                st.warning("请输入代码。")
            elif t in watchlist:
                st.info(f"{t} 已在自选中。")
            else:
                watchlist.append(t)
                st.session_state["watchlist"] = watchlist
                st.success(f"已添加 {t}")
                st.rerun()
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 刷新行情", type="primary", use_container_width=True)

    # 快捷加入宇宙中的票
    st.markdown("**快捷添加：**")
    quick_cols = st.columns(5)
    popular = ["AAPL", "NVDA", "600519.SS", "300750.SZ", "0700.HK"]
    for i, code in enumerate(popular):
        label = f"{code}"
        if quick_cols[i].button(label, key=f"wd_quick_{code}", use_container_width=True):
            if code not in watchlist:
                watchlist.append(code)
                st.session_state["watchlist"] = watchlist
            st.rerun()

    if not watchlist:
        st.info("自选股为空，请先添加。")
        return

    if refresh or "watchlist_snapshot" not in st.session_state:
        rows = []
        progress = st.progress(0, text="正在拉取自选股...")
        for i, t in enumerate(watchlist):
            info = _safe_analyze(t, period="6mo")
            if info is None:
                rows.append(
                    {
                        "代码": t,
                        "名称": STOCK_UNIVERSE.get(t, "-"),
                        "现价": np.nan,
                        "日涨跌": np.nan,
                        "20日涨跌": np.nan,
                        "RSI": np.nan,
                        "站上MA20": "-",
                        "站上MA60": "-",
                        "状态": "拉取失败",
                    }
                )
            else:
                rows.append(
                    {
                        "代码": info["ticker"],
                        "名称": info["name"],
                        "现价": info["close"],
                        "日涨跌": info["change"],
                        "20日涨跌": info["ret_20d"],
                        "RSI": info["rsi"],
                        "站上MA20": "是" if info["above_ma20"] else "否",
                        "站上MA60": "是" if info["above_ma60"] else "否",
                        "状态": info["asof"],
                    }
                )
            progress.progress((i + 1) / len(watchlist), text=f"已更新 {i + 1}/{len(watchlist)}")
        progress.empty()
        st.session_state["watchlist_snapshot"] = pd.DataFrame(rows)

    snap = st.session_state["watchlist_snapshot"]
    show = snap.copy()
    for col in ["现价", "RSI"]:
        show[col] = show[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    for col in ["日涨跌", "20日涨跌"]:
        show[col] = show[col].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
    st.dataframe(show, use_container_width=True, hide_index=True)

    # 删除自选
    remove = st.multiselect("移除自选股", options=watchlist, key="wd_remove")
    if st.button("删除所选") and remove:
        st.session_state["watchlist"] = [t for t in watchlist if t not in remove]
        st.session_state.pop("watchlist_snapshot", None)
        st.rerun()

    # 跳转一页纸
    pick = st.selectbox("查看单票一页纸", options=watchlist, key="wd_pick_one")
    if st.button("📄 打开该票一页纸", key="wd_open_one"):
        st.session_state["onepager_ticker"] = pick
        st.session_state["watchdesk_goto"] = "单票一页纸"
        st.rerun()


def _render_screener_tab() -> None:
    st.markdown('<div class="section-title">🔎 条件选股（内置股票池）</div>', unsafe_allow_html=True)
    st.warning(
        "免费版在「内置常见股票池」中筛选（约 "
        f"{len(STOCK_UNIVERSE)} 只），不是交易所全市场扫描。结果仅供学习，不构成投资建议。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        rsi_min, rsi_max = st.slider("RSI 区间", 0, 100, (0, 100), key="sc_rsi")
        require_ma20 = st.checkbox("必须站上 MA20", value=False, key="sc_ma20")
    with c2:
        ret_min = st.number_input("20日涨跌下限", value=-1.0, step=0.05, format="%.2f", key="sc_ret_min")
        ret_max = st.number_input("20日涨跌上限", value=1.0, step=0.05, format="%.2f", key="sc_ret_max")
        st.caption("例如 -0.10 表示近20日至少跌超10% 才入选（配合超卖思路）")
    with c3:
        require_ma60 = st.checkbox("必须站上 MA60", value=False, key="sc_ma60")
        macd_pos = st.checkbox("MACD 柱 > 0（动能偏多）", value=False, key="sc_macd")
        universe_keys = list(STOCK_UNIVERSE.keys())

    run = st.button("▶ 开始筛选", type="primary", key="sc_run")

    if not run and "screener_result" not in st.session_state:
        st.info("设好条件后点「开始筛选」。首次可能需要 1–2 分钟拉取股票池。")
        return

    if run:
        results = []
        bar = st.progress(0, text="扫描股票池中...")
        for i, t in enumerate(universe_keys):
            info = _safe_analyze(t, period="6mo")
            bar.progress((i + 1) / len(universe_keys), text=f"扫描 {t} ({i + 1}/{len(universe_keys)})")
            if info is None:
                continue
            if not (rsi_min <= info["rsi"] <= rsi_max):
                continue
            if require_ma20 and not info["above_ma20"]:
                continue
            if require_ma60 and not info["above_ma60"]:
                continue
            if macd_pos and not (info["macd_hist"] > 0):
                continue
            r20 = info["ret_20d"]
            if pd.isna(r20) or not (ret_min <= r20 <= ret_max):
                continue
            results.append(info)
        bar.empty()
        st.session_state["screener_result"] = results

    results = st.session_state.get("screener_result", [])
    if not results:
        st.warning("没有股票满足当前条件，试试放宽 RSI 或涨跌区间。")
        return

    table = pd.DataFrame(
        [
            {
                "代码": r["ticker"],
                "名称": r["name"],
                "现价": r["close"],
                "日涨跌": r["change"],
                "20日涨跌": r["ret_20d"],
                "RSI": r["rsi"],
                "MA20之上": "是" if r["above_ma20"] else "否",
                "MACD柱": r["macd_hist"],
            }
            for r in results
        ]
    ).sort_values("RSI")

    show = table.copy()
    show["现价"] = show["现价"].map(lambda v: f"{v:.2f}")
    show["日涨跌"] = show["日涨跌"].map(lambda v: f"{v:+.2%}")
    show["20日涨跌"] = show["20日涨跌"].map(lambda v: f"{v:+.2%}")
    show["RSI"] = show["RSI"].map(lambda v: f"{v:.1f}")
    show["MACD柱"] = show["MACD柱"].map(lambda v: f"{v:.3f}")
    st.success(f"共筛出 {len(table)} 只候选")
    st.dataframe(show, use_container_width=True, hide_index=True)

    # 一键加入自选
    add_codes = st.multiselect(
        "将候选加入自选股",
        options=table["代码"].tolist(),
        key="sc_add_sel",
    )
    if st.button("加入自选") and add_codes:
        wl = _ensure_watchlist()
        for c in add_codes:
            if c not in wl:
                wl.append(c)
        st.session_state["watchlist"] = wl
        st.session_state.pop("watchlist_snapshot", None)
        st.success(f"已加入 {len(add_codes)} 只")

    register_report_section(
        module="看盘助手",
        title="条件选股结果",
        metrics={"候选数量": str(len(table)), "股票池大小": str(len(STOCK_UNIVERSE))},
        tables=[("筛选结果", table)],
        notes="内置股票池条件筛选，仅供学习。",
    )


def _render_onepager_tab() -> None:
    st.markdown('<div class="section-title">📄 单票一页纸</div>', unsafe_allow_html=True)
    st.caption("一张图看完：价格位置、技术指标、简单风险。适合复盘笔记。")

    default = st.session_state.get("onepager_ticker", "AAPL")
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("股票代码", value=default, key="op_ticker")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("生成一页纸", type="primary", use_container_width=True)

    if not go_btn and "onepager_cache" not in st.session_state:
        st.info("输入代码后点「生成一页纸」。")
        return

    if go_btn:
        with st.spinner(f"正在生成 {ticker} 一页纸..."):
            info = _safe_analyze(ticker.strip().upper(), period="1y")
            if info is None:
                st.error("拉取失败，请检查代码或稍后重试。")
                return
            st.session_state["onepager_cache"] = info
            st.session_state["onepager_ticker"] = info["ticker"]

    info = st.session_state["onepager_cache"]
    hist = info["history"]

    render_metric_cards(
        [
            ("代码", info["ticker"]),
            ("现价", f"{info['close']:.2f}"),
            ("日涨跌", f"{info['change']:+.2%}"),
            ("RSI", f"{info['rsi']:.1f}"),
        ]
    )
    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("20日涨跌", f"{info['ret_20d']:+.2%}" if pd.notna(info["ret_20d"]) else "N/A")
    g2.metric("站上MA20", "是" if info["above_ma20"] else "否")
    g3.metric("站上MA60", "是" if info["above_ma60"] else "否")
    g4.metric("年化波动(近20日)", f"{info['vol_ann']:.1%}" if pd.notna(info["vol_ann"]) else "N/A")

    # 解读小抄
    notes = []
    if info["rsi"] >= 70:
        notes.append("RSI 偏高（常见解读：超买区），注意回调风险。")
    elif info["rsi"] <= 30:
        notes.append("RSI 偏低（常见解读：超卖区），可能存在反弹，但仍需确认趋势。")
    else:
        notes.append("RSI 处于中性区间。")
    if info["above_ma20"] and info["above_ma60"]:
        notes.append("价格同时位于 MA20 与 MA60 上方，短中期均线结构偏强。")
    elif (not info["above_ma20"]) and (not info["above_ma60"]):
        notes.append("价格同时位于 MA20 与 MA60 下方，短中期均线结构偏弱。")
    else:
        notes.append("价格与均线关系出现交叉状态，建议结合模块 5 的完整图观察。")
    if info["macd_hist"] > 0:
        notes.append("MACD 柱为正，动能偏多。")
    else:
        notes.append("MACD 柱为负，动能偏空。")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**🤖 自动解读（教学口径，非投顾建议）**")
    for n in notes:
        st.markdown(f"- {n}")
    st.markdown("</div>", unsafe_allow_html=True)

    fig = build_quant_chart(hist, info["ticker"])
    st.plotly_chart(fig, use_container_width=True)

    # 简单风险
    try:
        rets = hist["close"].pct_change().dropna()
        risk = compute_var_cvar(rets.values, alpha=0.95, n_sims=5000)
        wealth = (1 + rets).cumprod()
        mdd = compute_max_drawdown(wealth)
        r1, r2, r3 = st.columns(3)
        r1.metric("历史VaR(95%)", f"{risk['historical']['VaR']:.2%}")
        r2.metric("历史CVaR(95%)", f"{risk['historical']['CVaR']:.2%}")
        r3.metric("样本期最大回撤", f"{mdd['max_drawdown']:.2%}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"风险指标暂不可用: {exc}")

    # 复盘笔记
    note_key = f"trade_note_{info['ticker']}"
    note = st.text_area(
        "我的复盘笔记（仅保存在当前浏览器会话）",
        value=st.session_state.get(note_key, ""),
        height=120,
        key=f"note_area_{info['ticker']}",
    )
    if st.button("保存笔记"):
        st.session_state[note_key] = note
        st.success("已保存到当前会话。")

    if st.button("⭐ 加入自选股"):
        wl = _ensure_watchlist()
        if info["ticker"] not in wl:
            wl.append(info["ticker"])
            st.session_state["watchlist"] = wl
            st.session_state.pop("watchlist_snapshot", None)
        st.success(f"{info['ticker']} 已在自选中")

    register_report_section(
        module="看盘助手",
        title=f"{info['ticker']} 单票一页纸",
        metrics={
            "现价": f"{info['close']:.2f}",
            "日涨跌": f"{info['change']:+.2%}",
            "RSI": f"{info['rsi']:.1f}",
            "20日涨跌": f"{info['ret_20d']:+.2%}" if pd.notna(info["ret_20d"]) else "N/A",
        },
        figures=[("技术图表", fig)],
        notes="；".join(notes),
    )


def render_watchdesk(df: pd.DataFrame = None) -> None:
    """渲染模块 7 完整页面。"""
    st.markdown(
        '<div class="hero-title">🧭 看盘助手：自选股 · 条件选股 · 一页纸</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">把学习用的技术指标，变成每天能打开看一眼的观察台。'
        "免费版使用内置股票池，不做真下单。</div>",
        unsafe_allow_html=True,
    )

    render_beginner_guide("watchdesk", expanded=True)

    # 允许从自选跳转
    tab_names = ["⭐ 自选股", "🔎 条件选股", "📄 单票一页纸"]
    goto = st.session_state.pop("watchdesk_goto", None)
    if goto == "单票一页纸":
        # 用 radio 模拟默认页：Streamlit tabs 无法直接设默认，改为 session 标记
        st.session_state["wd_tab_radio"] = "📄 单票一页纸"

    choice = st.radio(
        "功能页",
        options=tab_names,
        horizontal=True,
        key="wd_tab_radio",
        label_visibility="collapsed",
    )

    if choice.startswith("⭐"):
        _render_watchlist_tab()
    elif choice.startswith("🔎"):
        _render_screener_tab()
    else:
        _render_onepager_tab()
