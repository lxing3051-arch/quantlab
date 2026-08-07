"""
模块 6：机器学习分类与预测
支持 StandardScaler 标准化，训练逻辑回归与随机森林，
输出混淆矩阵、ROC 曲线与特征重要性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import get_numeric_columns, validate_dataframe
from utils.report_export import register_report_section
from utils.styles import PLOTLY_LAYOUT, render_metric_cards


def _prepare_classification_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
):
    """清洗数据、编码目标变量，返回 X, y, 特征名, 类别标签。"""
    from sklearn.preprocessing import LabelEncoder

    data = df[feature_cols + [target_col]].copy()
    data = data.dropna()
    if len(data) < 20:
        raise ValueError(f"删除缺失值后样本量过少（{len(data)}），至少需要 20 行。")

    # 特征必须为数值
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"以下特征含无法转为数值的内容: {bad}")

    y_raw = data[target_col]
    # 若目标是连续数值且唯一值很多，提示用户
    n_unique = y_raw.nunique()
    if n_unique < 2:
        raise ValueError("目标变量类别数少于 2，无法进行分类。")
    if n_unique > 20 and pd.api.types.is_numeric_dtype(y_raw):
        raise ValueError(
            f"目标变量有 {n_unique} 个唯一值，更像回归问题。"
            "请选择类别数较少的分类变量，或先对目标做分箱。"
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in encoder.classes_]
    return X.values, y, feature_cols, class_names, encoder


def _train_models(X, y, test_size: float = 0.25, seed: int = 42):
    """标准化后训练逻辑回归与随机森林。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 逻辑回归
    logit = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    logit.fit(X_train_s, y_train)

    # 随机森林（在标准化后的特征上训练，保持流程一致）
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    results = {}
    for name, model in [("逻辑回归", logit), ("随机森林", rf)]:
        y_pred = model.predict(X_test_s)
        # 概率：二分类取正类，多分类用 one-vs-rest AUC
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_s)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        roc_data = None
        auc_val = None
        try:
            if proba is not None:
                n_classes = proba.shape[1]
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                    auc_val = roc_auc_score(y_test, proba[:, 1])
                    roc_data = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
                else:
                    # 宏平均 OvR
                    from sklearn.preprocessing import label_binarize

                    y_bin = label_binarize(y_test, classes=np.arange(n_classes))
                    auc_val = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                    # 为每个类计算 ROC，供绘图
                    curves = []
                    for i in range(n_classes):
                        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], proba[:, i])
                        curves.append((fpr_i, tpr_i, i))
                    roc_data = {"multi": curves, "auc": auc_val}
        except Exception:  # noqa: BLE001
            roc_data = None
            auc_val = None

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "cm": cm,
            "accuracy": acc,
            "f1": f1,
            "auc": auc_val,
            "roc": roc_data,
            "report": report,
        }

    # 特征重要性：随机森林；逻辑回归用 |coef| 均值
    rf_importance = rf.feature_importances_
    if logit.coef_.ndim == 1:
        logit_importance = np.abs(logit.coef_)
    else:
        logit_importance = np.abs(logit.coef_).mean(axis=0)

    return {
        "results": results,
        "scaler": scaler,
        "rf_importance": rf_importance,
        "logit_importance": logit_importance,
        "X_test": X_test_s,
        "y_test": y_test,
    }


def _confusion_matrix_figure(cm: np.ndarray, class_names: list, title: str) -> go.Figure:
    """绘制混淆矩阵热力图。"""
    labels = class_names if len(class_names) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.5, "#99f6e4"],
                [1.0, "#0f766e"],
            ],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#0f172a"},
            hovertemplate="真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>",
            colorbar={"title": "样本数"},
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="预测类别",
        yaxis_title="真实类别",
        yaxis={"autorange": "reversed"},
        height=420,
    )
    return fig


def _roc_figure(roc_data: dict, title: str, class_names: list) -> go.Figure:
    """绘制 ROC 曲线。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="随机猜测",
            line={"dash": "dash", "color": "#94a3b8", "width": 1.5},
        )
    )

    if roc_data is None:
        fig.update_layout(**PLOTLY_LAYOUT, title=title + "（无法计算）", height=420)
        return fig

    if "fpr" in roc_data:
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"AUC = {roc_data['auc']:.3f}",
                line={"color": "#0f766e", "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.12)",
            )
        )
    elif "multi" in roc_data:
        palette = ["#0f766e", "#b45309", "#0369a1", "#be123c", "#7c3aed", "#059669"]
        for idx, (fpr, tpr, cls_i) in enumerate(roc_data["multi"]):
            name = class_names[cls_i] if cls_i < len(class_names) else f"Class {cls_i}"
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=str(name),
                    line={"color": palette[idx % len(palette)], "width": 2},
                )
            )
        fig.update_layout(
            annotations=[
                {
                    "text": f"Macro AUC = {roc_data['auc']:.3f}",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.98,
                    "y": 0.05,
                    "showarrow": False,
                    "font": {"size": 12, "color": "#0f172a"},
                }
            ]
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="假阳性率 (FPR)",
        yaxis_title="真阳性率 (TPR)",
        height=420,
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.05]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _feature_importance_figure(
    feature_names: list,
    rf_imp: np.ndarray,
    logit_imp: np.ndarray,
) -> go.Figure:
    """特征重要性对比条形图（按随机森林排序）。"""
    order = np.argsort(rf_imp)
    names = [feature_names[i] for i in order]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=rf_imp[order],
            orientation="h",
            name="随机森林",
            marker_color="#0f766e",
        )
    )
    # 归一化逻辑回归系数重要性以便同图对比
    logit_norm = logit_imp / (logit_imp.sum() + 1e-12)
    fig.add_trace(
        go.Bar(
            y=names,
            x=logit_norm[order],
            orientation="h",
            name="逻辑回归 |coef|（归一化）",
            marker_color="#b45309",
            opacity=0.85,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="特征重要性排序",
        xaxis_title="重要性",
        barmode="group",
        height=max(380, 28 * len(names) + 120),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _generate_binary_sample(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """生成二分类示例数据，便于无上传时演示。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.uniform(-2, 2, n)
    logits = 0.8 * x1 - 0.6 * x2 + 0.4 * x3 + 0.3 * x4
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "feature_a": np.round(x1, 4),
            "feature_b": np.round(x2, 4),
            "feature_c": np.round(x3, 4),
            "feature_d": np.round(x4, 4),
            "target_class": y,
        }
    )


def render_ml_classify(df: pd.DataFrame = None) -> None:
    """渲染机器学习分类模块页面。"""
    st.markdown(
        '<div class="hero-title">🤖 机器学习分类与预测</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">标准化特征后训练逻辑回归与随机森林，'
        "输出混淆矩阵、ROC 曲线与特征重要性。</div>",
        unsafe_allow_html=True,
    )

    use_sample = False
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning("侧边栏暂无数据。可使用内置二分类示例，或上传 CSV。")
        if st.checkbox("使用内置二分类示例数据", value=True, key="ml_use_sample"):
            df = _generate_binary_sample()
            use_sample = True
        else:
            return

    if not validate_dataframe(df, min_rows=20):
        return

    try:
        all_cols = df.columns.tolist()
        num_cols = get_numeric_columns(df)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ 模型配置</div>', unsafe_allow_html=True)

        # 默认目标：含 class/target/label 的列，或最后一列
        target_candidates = [
            c
            for c in all_cols
            if any(k in str(c).lower() for k in ("class", "target", "label", "y"))
        ]
        default_target = target_candidates[0] if target_candidates else all_cols[-1]

        c1, c2 = st.columns([1, 2])
        with c1:
            target_col = st.selectbox(
                "目标分类变量 Y",
                options=all_cols,
                index=all_cols.index(default_target),
                key="ml_target",
            )
        with c2:
            feat_opts = [c for c in num_cols if c != target_col]
            # 若目标本身是数值且在 num_cols 中，特征排除它
            if not feat_opts:
                feat_opts = [c for c in all_cols if c != target_col]
                feat_opts = [
                    c
                    for c in feat_opts
                    if pd.api.types.is_numeric_dtype(df[c])
                    or df[c].dtype == object
                ]
                # 仅保留可数值化的
                feat_opts = get_numeric_columns(df)
                feat_opts = [c for c in feat_opts if c != target_col]

            default_feats = feat_opts[: min(8, len(feat_opts))]
            feature_cols = st.multiselect(
                "特征变量 X（数值型）",
                options=feat_opts,
                default=default_feats,
                key="ml_features",
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            test_size = st.slider("测试集比例", 0.15, 0.4, 0.25, 0.05, key="ml_test")
        with c4:
            seed = st.number_input("随机种子", 0, 9999, 42, key="ml_seed")
        with c5:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 训练模型", type="primary", key="ml_run")

        if use_sample:
            st.caption("当前使用内置示例：四特征二分类数据。")
        st.markdown("</div>", unsafe_allow_html=True)

        if not feature_cols:
            st.info("请至少选择一个特征变量。")
            return

        if not run_btn and "ml_cache" not in st.session_state:
            st.info("配置完成后点击「训练模型」。")
            return

        if run_btn:
            with st.spinner("正在标准化并训练逻辑回归 / 随机森林..."):
                X, y, feat_names, class_names, encoder = _prepare_classification_data(
                    df, target_col, feature_cols
                )
                # 分层划分需要每类至少 2 个样本
                counts = np.bincount(y)
                if counts.min() < 2:
                    raise ValueError(
                        f"某些类别样本过少（最少 {counts.min()} 个），无法分层划分测试集。"
                    )
                trained = _train_models(X, y, test_size=float(test_size), seed=int(seed))
                st.session_state["ml_cache"] = {
                    "trained": trained,
                    "feat_names": feat_names,
                    "class_names": class_names,
                    "target_col": target_col,
                }

        cache = st.session_state["ml_cache"]
        trained = cache["trained"]
        feat_names = cache["feat_names"]
        class_names = cache["class_names"]
        results = trained["results"]

        # 指标总览
        logit_res = results["逻辑回归"]
        rf_res = results["随机森林"]
        render_metric_cards(
            [
                ("逻辑回归 Accuracy", f"{logit_res['accuracy']:.3f}"),
                ("随机森林 Accuracy", f"{rf_res['accuracy']:.3f}"),
                (
                    "逻辑回归 AUC",
                    f"{logit_res['auc']:.3f}" if logit_res["auc"] is not None else "N/A",
                ),
                (
                    "随机森林 AUC",
                    f"{rf_res['auc']:.3f}" if rf_res["auc"] is not None else "N/A",
                ),
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_cm, tab_roc, tab_imp, tab_rpt = st.tabs(
            ["🧩 混淆矩阵", "📉 ROC 曲线", "⭐ 特征重要性", "📄 分类报告"]
        )

        figs_for_report = []

        with tab_cm:
            c1, c2 = st.columns(2)
            with c1:
                fig_cm_l = _confusion_matrix_figure(
                    logit_res["cm"], class_names, "逻辑回归 · 混淆矩阵"
                )
                st.plotly_chart(fig_cm_l, use_container_width=True)
                figs_for_report.append(("逻辑回归混淆矩阵", fig_cm_l))
            with c2:
                fig_cm_r = _confusion_matrix_figure(
                    rf_res["cm"], class_names, "随机森林 · 混淆矩阵"
                )
                st.plotly_chart(fig_cm_r, use_container_width=True)
                figs_for_report.append(("随机森林混淆矩阵", fig_cm_r))

        with tab_roc:
            c1, c2 = st.columns(2)
            with c1:
                fig_roc_l = _roc_figure(
                    logit_res["roc"], "逻辑回归 · ROC", class_names
                )
                st.plotly_chart(fig_roc_l, use_container_width=True)
                figs_for_report.append(("逻辑回归 ROC", fig_roc_l))
            with c2:
                fig_roc_r = _roc_figure(
                    rf_res["roc"], "随机森林 · ROC", class_names
                )
                st.plotly_chart(fig_roc_r, use_container_width=True)
                figs_for_report.append(("随机森林 ROC", fig_roc_r))

        with tab_imp:
            fig_imp = _feature_importance_figure(
                feat_names,
                trained["rf_importance"],
                trained["logit_importance"],
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            figs_for_report.append(("特征重要性", fig_imp))

            imp_df = pd.DataFrame(
                {
                    "特征": feat_names,
                    "随机森林重要性": trained["rf_importance"],
                    "逻辑回归|coef|归一化": trained["logit_importance"]
                    / (trained["logit_importance"].sum() + 1e-12),
                }
            ).sort_values("随机森林重要性", ascending=False)
            st.dataframe(
                imp_df.style.format(
                    {
                        "随机森林重要性": "{:.4f}",
                        "逻辑回归|coef|归一化": "{:.4f}",
                    }
                )
                if False
                else imp_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

        with tab_rpt:
            for name in ["逻辑回归", "随机森林"]:
                st.markdown(f"**{name}**")
                rep = results[name]["report"]
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.round(4), use_container_width=True)

        register_report_section(
            module="机器学习分类",
            title=f"分类预测 · 目标「{cache['target_col']}」",
            metrics={
                "目标变量": cache["target_col"],
                "特征数": str(len(feat_names)),
                "类别": ", ".join(class_names),
                "逻辑回归 Accuracy": f"{logit_res['accuracy']:.4f}",
                "随机森林 Accuracy": f"{rf_res['accuracy']:.4f}",
                "逻辑回归 F1": f"{logit_res['f1']:.4f}",
                "随机森林 F1": f"{rf_res['f1']:.4f}",
            },
            figures=figs_for_report,
            notes="特征经 StandardScaler 标准化；模型：Logistic Regression 与 Random Forest。",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(f"机器学习模块错误: {exc}")
        st.exception(exc)
