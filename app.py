"""
本地数据分析与金融量化计算工具箱
================================
运行方式:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# 必须在任何其它 Streamlit 调用 / 含 @st.cache_* 的模块导入之前设置
st.set_page_config(
    page_title="QuantLab · 数据分析与金融量化工具箱",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.data_stats import render_data_stats
from modules.finance import render_finance
from modules.market_data import render_market_data
from modules.ml_classify import render_ml_classify
from modules.portfolio import render_portfolio
from modules.regression import render_regression
from modules.watchdesk import render_watchdesk
from utils.data_loader import (
    SAMPLE_DATASET_OPTIONS,
    get_sample_dataset,
    load_user_file,
)
from utils.guides import render_learning_path_sidebar
from utils.report_export import render_report_download_panel
from utils.styles import inject_custom_css


MODULE_OPTIONS = [
    "📊 1. 数据清洗与描述性统计",
    "📈 2. 统计建模与 OLS 回归分析",
    "💰 3. 金融量化与风险度量",
    "💼 4. 马克维茨资产组合优化",
    "📡 5. 市场数据与技术指标",
    "🤖 6. 机器学习分类与预测",
    "🧭 7. 看盘助手与条件选股",
]


def _init_session_state() -> None:
    """初始化会话状态默认值。"""
    defaults = {
        "data_source": "使用内置示例",
        "sample_name": list(SAMPLE_DATASET_OPTIONS.keys())[0],
        "dataframe": None,
        "dataset_desc": "",
        "report_sections": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_active_dataframe():
    """根据侧边栏选择加载当前活动数据集。"""
    source = st.session_state.get("data_source", "使用内置示例")

    if source == "使用内置示例":
        sample_name = st.session_state.get(
            "sample_name", list(SAMPLE_DATASET_OPTIONS.keys())[0]
        )
        try:
            df, desc = get_sample_dataset(sample_name)
            st.session_state["dataframe"] = df
            st.session_state["dataset_desc"] = desc
            return df, f"内置示例 · {sample_name}"
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"加载示例数据失败: {exc}")
            return None, "加载失败"

    uploaded = st.session_state.get("uploaded_file_obj")
    if uploaded is None:
        return None, "等待上传文件"
    df = load_user_file(uploaded)
    if df is not None:
        st.session_state["dataframe"] = df
        st.session_state["dataset_desc"] = f"用户上传: {uploaded.name}"
        return df, f"自定义文件 · {uploaded.name}"
    return None, "文件读取失败"


def render_sidebar() -> str:
    """渲染侧边栏：品牌、模块导航、数据源、报告导出。"""
    with st.sidebar:
        st.markdown('<div class="brand-mark">QuantLab</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-caption">本地数据分析 · 金融量化 · 机器学习</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        render_learning_path_sidebar()

        st.markdown("#### 📂 模块导航")
        module = st.radio(
            "选择功能模块",
            options=MODULE_OPTIONS,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### 🗂️ 数据源")

        if module.startswith("📡") or module.startswith("🧭"):
            st.caption("本模块通过在线行情工作，侧边栏数据集可选（不强制）。")

        data_source = st.radio(
            "数据来源",
            options=["使用内置示例", "上传自定义 CSV/Excel"],
            key="data_source",
            label_visibility="collapsed",
        )

        if data_source == "使用内置示例":
            module_default_sample = {
                "📊 1. 数据清洗与描述性统计": "模拟股票 OHLCV K 线",
                "📈 2. 统计建模与 OLS 回归分析": "单资产收益与因子（回归）",
                "💰 3. 金融量化与风险度量": "模拟股票 OHLCV K 线",
                "💼 4. 马克维茨资产组合优化": "多资产日收益率（组合优化）",
                "📡 5. 市场数据与技术指标": "模拟股票 OHLCV K 线",
                "🤖 6. 机器学习分类与预测": "宏观经济与因子面板",
                "🧭 7. 看盘助手与条件选股": "模拟股票 OHLCV K 线",
            }
            preferred = module_default_sample.get(
                module, list(SAMPLE_DATASET_OPTIONS.keys())[0]
            )
            options = list(SAMPLE_DATASET_OPTIONS.keys())

            last_module = st.session_state.get("_last_module_for_sample")
            if last_module != module:
                st.session_state["sample_name"] = preferred
                st.session_state["_last_module_for_sample"] = module

            current = st.session_state.get("sample_name", preferred)
            if current not in options:
                current = preferred
                st.session_state["sample_name"] = current

            sample_name = st.selectbox(
                "选择示例数据集",
                options=options,
                index=options.index(current),
                help="切换模块时会推荐更匹配的示例，您仍可手动更改。",
            )
            st.session_state["sample_name"] = sample_name

            try:
                _, desc = get_sample_dataset(sample_name)
                st.caption(desc)
            except Exception:  # noqa: BLE001
                pass

        else:
            uploaded = st.file_uploader(
                "上传 CSV / Excel",
                type=["csv", "xlsx", "xls"],
                help="支持 UTF-8 / GBK 编码的 CSV，以及 .xlsx / .xls",
                key="file_uploader",
            )
            st.session_state["uploaded_file_obj"] = uploaded
            if uploaded is None:
                st.caption("请上传文件以开始分析。")

        st.markdown("---")
        render_report_download_panel()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem; opacity:0.65; line-height:1.5;">
            🛠 Tech Stack<br>
            Streamlit · Plotly · yfinance<br>
            Statsmodels · Scikit-Learn · SciPy
            </div>
            """,
            unsafe_allow_html=True,
        )

    return module


def render_data_banner(source_label: str, df, module: str) -> None:
    """在主区顶部展示当前数据源状态条。"""
    if module.startswith("📡") or module.startswith("🧭"):
        return

    if df is None:
        if module.startswith("🤖"):
            st.markdown(
                """
                <div style="
                    background: rgba(180, 83, 9, 0.08);
                    border: 1px solid rgba(180, 83, 9, 0.2);
                    border-radius: 12px;
                    padding: 0.75rem 1rem;
                    margin-bottom: 1rem;
                    color: #92400e;
                    font-size: 0.92rem;
                ">
                    💡 侧边栏暂无数据，可在本模块勾选内置二分类示例，或先上传 CSV。
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f"""
            <div style="
                background: rgba(190, 18, 60, 0.08);
                border: 1px solid rgba(190, 18, 60, 0.2);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                margin-bottom: 1rem;
                color: #9f1239;
                font-size: 0.92rem;
            ">
                ⚠️ 当前无可用数据（{source_label}）。请在侧边栏选择内置示例或上传文件。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    n_rows, n_cols = df.shape
    desc = st.session_state.get("dataset_desc", "")
    st.markdown(
        f"""
        <div style="
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 14px;
            padding: 0.8rem 1.1rem;
            margin-bottom: 1rem;
            color: #134e4a;
            font-size: 0.92rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1.5rem;
            align-items: center;
            box-shadow: 0 2px 12px rgba(15, 118, 110, 0.06);
        ">
            <span><span class="badge">DATA</span></span>
            <span><b>来源</b>：{source_label}</span>
            <span><b>规模</b>：{n_rows:,} 行 × {n_cols} 列</span>
            <span style="opacity:0.85;">{desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """应用主入口。"""
    inject_custom_css()
    _init_session_state()

    module = render_sidebar()
    df, source_label = _load_active_dataframe()
    render_data_banner(source_label, df, module)

    try:
        if module.startswith("📊"):
            if df is None:
                st.stop()
            render_data_stats(df)

        elif module.startswith("📈"):
            if df is None:
                st.stop()
            render_regression(df)

        elif module.startswith("💰"):
            render_finance(df)

        elif module.startswith("💼"):
            if df is None:
                st.stop()
            render_portfolio(df)

        elif module.startswith("📡"):
            render_market_data(df)

        elif module.startswith("🤖"):
            render_ml_classify(df)

        elif module.startswith("🧭"):
            render_watchdesk(df)

        else:
            st.error("未知模块，请从侧边栏重新选择。")

    except Exception as exc:  # noqa: BLE001
        st.error(f"页面渲染发生未捕获异常: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
