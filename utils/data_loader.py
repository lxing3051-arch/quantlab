"""
数据加载与内置示例数据集生成模块。
支持 CSV / Excel 上传，以及多种金融与统计场景的示例数据。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# 侧边栏可选的示例数据集名称
SAMPLE_DATASET_OPTIONS = {
    "多资产日收益率（组合优化）": "asset_returns",
    "模拟股票 OHLCV K 线": "ohlcv",
    "宏观经济与因子面板": "macro_panel",
    "单资产收益与因子（回归）": "factor_returns",
}


def _generate_asset_returns(n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """
    生成多资产历史日收益率（约 2 年交易日）。
    使用多元正态相关结构，贴近真实资产联动特征。
    """
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "GLD"]
    n_assets = len(tickers)

    # 年化波动率与日均收益（简化设定）
    annual_vol = np.array([0.28, 0.24, 0.26, 0.32, 0.45, 0.22, 0.27, 0.15])
    annual_ret = np.array([0.14, 0.12, 0.11, 0.13, 0.22, 0.09, 0.08, 0.05])
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = annual_ret / 252

    # 构造相关矩阵（对角为 1，行业相近资产相关性更高）
    corr = np.full((n_assets, n_assets), 0.35)
    np.fill_diagonal(corr, 1.0)
    # 科技股簇
    for i in range(5):
        for j in range(5):
            if i != j:
                corr[i, j] = 0.62
    # 黄金与股市低相关
    corr[-1, :-1] = 0.08
    corr[:-1, -1] = 0.08

    # 保证正定
    corr = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    cov = np.outer(daily_vol, daily_vol) * corr
    returns = rng.multivariate_normal(daily_mu, cov, size=n_days)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(returns, index=dates, columns=tickers)
    df.index.name = "date"
    df = df.reset_index()
    return df


def _generate_ohlcv(n_days: int = 252, seed: int = 7) -> pd.DataFrame:
    """
    基于几何布朗运动生成模拟股票 OHLCV K 线数据。
    """
    rng = np.random.default_rng(seed)
    mu, sigma, s0 = 0.12, 0.28, 100.0
    dt = 1 / 252

    # 生成收盘价路径
    shocks = rng.normal(0, 1, n_days)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    close = s0 * np.exp(np.cumsum(log_returns))

    # 由收盘价派生 OHLC
    open_ = np.roll(close, 1)
    open_[0] = s0
    # 日内振幅
    intra = np.abs(rng.normal(0, 0.008, n_days))
    high = np.maximum(open_, close) * (1 + intra)
    low = np.minimum(open_, close) * (1 - intra)
    volume = rng.integers(1_000_000, 8_000_000, n_days).astype(float)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
            "returns": np.round(np.concatenate([[0.0], np.diff(np.log(close))]), 6),
        }
    )
    return df


def _generate_macro_panel(n: int = 180, seed: int = 21) -> pd.DataFrame:
    """
    生成宏观经济与因子面板数据（含少量缺失值，便于清洗演示）。
    """
    rng = np.random.default_rng(seed)
    gdp_growth = rng.normal(2.5, 1.2, n)
    inflation = rng.normal(2.0, 0.8, n)
    unemployment = np.clip(rng.normal(5.0, 1.0, n), 2.5, 12)
    interest_rate = np.clip(rng.normal(3.0, 1.5, n), 0.1, 8)
    market_return = (
        0.4 * gdp_growth
        - 0.3 * inflation
        - 0.2 * unemployment
        + 0.15 * interest_rate
        + rng.normal(0, 2.5, n)
    )
    consumer_confidence = 100 + 3 * gdp_growth - 2 * unemployment + rng.normal(0, 5, n)

    df = pd.DataFrame(
        {
            "period": pd.period_range("2010-01", periods=n, freq="M").astype(str),
            "gdp_growth": np.round(gdp_growth, 3),
            "inflation": np.round(inflation, 3),
            "unemployment": np.round(unemployment, 3),
            "interest_rate": np.round(interest_rate, 3),
            "consumer_confidence": np.round(consumer_confidence, 2),
            "market_return": np.round(market_return, 3),
        }
    )

    # 人为注入约 3% 缺失值，便于展示缺失值统计
    for col in ["inflation", "consumer_confidence", "market_return"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def _generate_factor_returns(n: int = 360, seed: int = 99) -> pd.DataFrame:
    """
    生成单资产超额收益与多因子数据，适合 OLS 回归演示。
    """
    rng = np.random.default_rng(seed)
    mkt_rf = rng.normal(0.0008, 0.012, n)
    smb = rng.normal(0.0002, 0.008, n)
    hml = rng.normal(0.0001, 0.007, n)
    rmw = rng.normal(0.00015, 0.006, n)
    cma = rng.normal(0.00005, 0.005, n)
    # 真实 beta 关系 + 噪声
    alpha = 0.0003
    asset_excess = (
        alpha
        + 1.15 * mkt_rf
        + 0.35 * smb
        - 0.20 * hml
        + 0.10 * rmw
        + 0.05 * cma
        + rng.normal(0, 0.01, n)
    )

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    df = pd.DataFrame(
        {
            "date": dates,
            "asset_excess_ret": np.round(asset_excess, 6),
            "mkt_rf": np.round(mkt_rf, 6),
            "smb": np.round(smb, 6),
            "hml": np.round(hml, 6),
            "rmw": np.round(rmw, 6),
            "cma": np.round(cma, 6),
            "rf": np.round(rng.normal(0.00008, 0.00002, n), 6),
        }
    )
    return df


def get_sample_dataset(key: str) -> Tuple[pd.DataFrame, str]:
    """
    根据键名返回示例数据集与说明文字。

    Parameters
    ----------
    key : str
        SAMPLE_DATASET_OPTIONS 中的内部键，或中文显示名。

    Returns
    -------
    (DataFrame, description)
    """
    # 允许传入中文显示名或内部键
    internal = SAMPLE_DATASET_OPTIONS.get(key, key)
    generators = {
        "asset_returns": (
            _generate_asset_returns,
            "8 只资产约 2 年日收益率，含科技/金融/能源/黄金，适合组合优化与相关分析。",
        ),
        "ohlcv": (
            _generate_ohlcv,
            "模拟股票一年 OHLCV K 线及对数收益，适合描述统计与分布可视化。",
        ),
        "macro_panel": (
            _generate_macro_panel,
            "宏观经济月度面板（含少量缺失值），适合清洗与多元回归演示。",
        ),
        "factor_returns": (
            _generate_factor_returns,
            "单资产超额收益与 Fama-French 风格因子，适合 OLS 回归分析。",
        ),
    }
    if internal not in generators:
        raise ValueError(f"未知的示例数据集: {key}")

    func, desc = generators[internal]
    return func(), desc


def load_user_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    读取用户上传的 CSV 或 Excel 文件。
    自动尝试常见编码，并进行基础校验。

    Returns
    -------
    DataFrame 或 None（失败时）
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # 依次尝试常见编码
            raw = uploaded_file.getvalue()
            last_err = None
            for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
                try:
                    from io import BytesIO

                    df = pd.read_csv(BytesIO(raw), encoding=encoding)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    df = None
            if df is None:
                raise ValueError(f"CSV 解码失败: {last_err}")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("仅支持 CSV、XLSX、XLS 格式文件。")
            return None

        if df.empty:
            st.error("文件内容为空，请检查后重新上传。")
            return None

        # 去除全空行/列
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df

    except Exception as exc:  # noqa: BLE001
        st.error(f"读取文件失败: {exc}")
        return None


def get_numeric_columns(df: pd.DataFrame) -> list:
    """返回数值型列名列表。"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def validate_dataframe(df: Optional[pd.DataFrame], min_rows: int = 2) -> bool:
    """校验 DataFrame 是否可用于分析。"""
    if df is None:
        st.warning("暂无可用数据，请上传文件或选择内置示例。")
        return False
    if df.empty:
        st.warning("数据为空，无法进行分析。")
        return False
    if len(df) < min_rows:
        st.warning(f"样本量过少（当前 {len(df)} 行），至少需要 {min_rows} 行。")
        return False
    return True
