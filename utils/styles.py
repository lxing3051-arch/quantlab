"""
全局样式与 Plotly 主题配置。
提供高级感、响应式的视觉风格（青绿 + 深石板色）。
"""

import streamlit as st


# Plotly 统一布局模板（各模块图表复用）
PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "font": {
        "family": "DM Sans, Noto Sans SC, Segoe UI, Microsoft YaHei, sans-serif",
        "size": 13,
        "color": "#1e293b",
    },
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(248,250,252,0.65)",
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "colorway": [
        "#0f766e",
        "#b45309",
        "#0369a1",
        "#be123c",
        "#4f46e5",
        "#059669",
        "#c2410c",
        "#7c3aed",
    ],
    "hoverlabel": {
        "bgcolor": "#0f172a",
        "font_size": 12,
        "font_family": "DM Sans, Microsoft YaHei, sans-serif",
        "font_color": "#f8fafc",
    },
}


def inject_custom_css() -> None:
    """向 Streamlit 页面注入自定义 CSS，重构 Card / Metric / 字体排版。"""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

        :root {
            --ql-teal: #0f766e;
            --ql-teal-bright: #0d9488;
            --ql-ink: #0f172a;
            --ql-muted: #64748b;
            --ql-amber: #b45309;
            --ql-rose: #be123c;
            --ql-card: rgba(255,255,255,0.88);
            --ql-line: rgba(15, 23, 42, 0.08);
            --ql-shadow: 0 8px 30px rgba(15, 23, 42, 0.07);
            --ql-radius: 16px;
        }

        html, body, [class*="css"],
        .stMarkdown, .stText, .stSelectbox, .stMultiSelect {
            font-family: 'DM Sans', 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
        }

        .stApp {
            background:
                radial-gradient(ellipse 80% 50% at 8% -8%, rgba(15, 118, 110, 0.14), transparent 52%),
                radial-gradient(ellipse 55% 40% at 92% 0%, rgba(180, 83, 9, 0.09), transparent 48%),
                radial-gradient(ellipse 40% 30% at 50% 100%, rgba(3, 105, 161, 0.06), transparent 50%),
                linear-gradient(180deg, #f1f5f9 0%, #e8eef5 100%);
            color: var(--ql-ink);
        }

        .main .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3.5rem;
            max-width: 1320px;
        }

        /* ========== 侧边栏 ========== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(185deg, #0b1220 0%, #134e4a 52%, #0f766e 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.6rem 0.8rem;
            border-radius: 12px;
            margin-bottom: 0.28rem;
            border: 1px solid transparent;
            transition: background 0.2s ease, transform 0.15s ease, border-color 0.2s ease;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.08);
            border-color: rgba(255,255,255,0.08);
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.12) !important;
        }

        /* ========== 标题排版 ========== */
        .hero-title {
            font-size: clamp(1.55rem, 2.8vw, 2.1rem);
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--ql-ink);
            margin-bottom: 0.3rem;
            line-height: 1.22;
        }

        .hero-sub {
            color: var(--ql-muted);
            font-size: 1.01rem;
            margin-bottom: 1.2rem;
            line-height: 1.6;
            max-width: 52rem;
        }

        /* ========== 自定义卡片 ========== */
        .metric-card {
            background: var(--ql-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--ql-line);
            border-radius: var(--ql-radius);
            padding: 1.15rem 1.3rem;
            box-shadow: var(--ql-shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, var(--ql-teal), var(--ql-amber));
            border-radius: 3px 0 0 3px;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.12);
            border-color: rgba(15, 118, 110, 0.22);
        }

        .metric-card .label {
            font-size: 0.8rem;
            color: var(--ql-muted);
            font-weight: 560;
            margin-bottom: 0.4rem;
            letter-spacing: 0.01em;
        }

        .metric-card .value {
            font-size: 1.48rem;
            font-weight: 700;
            color: var(--ql-teal);
            letter-spacing: -0.025em;
            font-variant-numeric: tabular-nums;
        }

        .section-card {
            background: var(--ql-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--ql-line);
            border-radius: 18px;
            padding: 1.4rem 1.55rem;
            margin-bottom: 1.15rem;
            box-shadow: var(--ql-shadow);
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 650;
            color: var(--ql-ink);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            letter-spacing: -0.01em;
        }

        .badge {
            display: inline-block;
            background: linear-gradient(135deg, var(--ql-teal), var(--ql-teal-bright));
            color: white !important;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .brand-mark {
            font-size: 1.45rem;
            font-weight: 700;
            background: linear-gradient(90deg, #5eead4, #fcd34d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }

        .sidebar-caption {
            font-size: 0.78rem !important;
            opacity: 0.75;
            margin-bottom: 1rem !important;
            line-height: 1.45 !important;
        }

        /* ========== st.metric 强化 ========== */
        div[data-testid="stMetric"] {
            background: var(--ql-card);
            border: 1px solid var(--ql-line);
            border-radius: 14px;
            padding: 0.95rem 1.1rem 0.85rem;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.09);
        }

        div[data-testid="stMetric"] label {
            color: var(--ql-muted) !important;
            font-weight: 560 !important;
            font-size: 0.82rem !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ql-teal) !important;
            font-weight: 700 !important;
            font-size: 1.45rem !important;
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
        }

        /* ========== 按钮 / 输入 / Tabs ========== */
        .stButton > button {
            background: linear-gradient(135deg, var(--ql-teal), var(--ql-teal-bright));
            color: white !important;
            border: none;
            border-radius: 11px;
            font-weight: 650;
            padding: 0.5rem 1.15rem;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(15, 118, 110, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, var(--ql-teal-bright), #14b8a6);
            box-shadow: 0 8px 22px rgba(15, 118, 110, 0.38);
            transform: translateY(-1px);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.25);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border-color: var(--ql-line) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: transparent;
            border-bottom: 1px solid var(--ql-line);
            padding-bottom: 0.15rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 1.05rem;
            font-weight: 650;
            color: var(--ql-muted);
            border: 1px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(15, 118, 110, 0.1) !important;
            color: var(--ql-teal) !important;
            border-color: rgba(15, 118, 110, 0.15) !important;
        }

        /* 数据框 / Expander */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--ql-line);
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stExpander"] {
            background: var(--ql-card);
            border: 1px solid var(--ql-line);
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        }

        /* Alert 圆角 */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--ql-line);
        }

        /* Plotly 容器轻微卡片感 */
        div[data-testid="stPlotlyChart"] {
            background: var(--ql-card);
            border: 1px solid var(--ql-line);
            border-radius: 16px;
            padding: 0.4rem 0.35rem 0.15rem;
            box-shadow: var(--ql-shadow);
            margin-bottom: 0.6rem;
        }

        /* Download 按钮区分色 */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #0f172a, #134e4a) !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25) !important;
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }
            .hero-title { font-size: 1.4rem; }
            .metric-card .value { font-size: 1.2rem; }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
        }

        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: rgba(241, 245, 249, 0.65);
            backdrop-filter: blur(8px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(items: list) -> None:
    """
    渲染一组指标卡片。
    items: [(label, value), ...]
    """
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
