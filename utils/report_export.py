"""
分析报告导出工具
将各模块注册的指标、表格与 Plotly 图表汇总为精美 HTML，并尝试生成 PDF。
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def _ensure_store() -> List[dict]:
    """获取 / 初始化报告片段存储。"""
    if "report_sections" not in st.session_state:
        st.session_state["report_sections"] = []
    return st.session_state["report_sections"]


def clear_report_sections() -> None:
    """清空已注册的报告内容。"""
    st.session_state["report_sections"] = []


def register_report_section(
    module: str,
    title: str,
    metrics: Optional[Dict[str, Any]] = None,
    tables: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    figures: Optional[List[Tuple[str, Any]]] = None,
    notes: str = "",
) -> None:
    """
    注册一个分析片段到会话，供后续汇总导出。
    同名 title 会被覆盖，避免重复堆叠。
    """
    store = _ensure_store()
    # 移除同模块同标题旧条目
    store[:] = [
        s for s in store if not (s.get("module") == module and s.get("title") == title)
    ]

    # 表格转 records，避免 DataFrame 引用失效
    table_payload = []
    if tables:
        for t_name, t_df in tables:
            if t_df is None:
                continue
            table_payload.append(
                {
                    "name": t_name,
                    "html": t_df.round(6).to_html(
                        index=False,
                        classes="report-table",
                        border=0,
                    ),
                }
            )

    fig_payload = []
    if figures:
        for f_name, fig in figures:
            if fig is None:
                continue
            fig_payload.append({"name": f_name, "fig": fig})

    store.append(
        {
            "module": module,
            "title": title,
            "metrics": metrics or {},
            "tables": table_payload,
            "figures": fig_payload,
            "notes": notes,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
    )


def _fig_to_html_div(fig, include_js: bool = False) -> str:
    """将 Plotly Figure 转为可嵌入的 HTML 片段。"""
    try:
        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
            config={"displayModeBar": False, "responsive": True},
        )
    except Exception as exc:  # noqa: BLE001
        return f"<p class='warn'>图表渲染失败: {exc}</p>"


def _fig_to_png_b64(fig, width: int = 900, height: int = 520) -> Optional[str]:
    """尝试用 kaleido 导出 PNG base64，失败则返回 None。"""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return base64.b64encode(img_bytes).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def build_html_report(
    sections: Optional[List[dict]] = None,
    report_title: str = "QuantLab 分析报告",
) -> bytes:
    """
    生成格式精美的 HTML 报告（内嵌 Plotly CDN 交互图）。

    Returns
    -------
    UTF-8 编码的 HTML 字节
    """
    sections = sections if sections is not None else list(_ensure_store())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body_parts: List[str] = []
    first_fig = True

    if not sections:
        body_parts.append(
            "<div class='empty'>暂无已注册的分析结果。请先在各模块完成分析后再导出。</div>"
        )
    else:
        for idx, sec in enumerate(sections, start=1):
            metric_items = []
            for k, v in (sec.get("metrics") or {}).items():
                metric_items.append(
                    f"<div class='mcard'><div class='mlabel'>{k}</div>"
                    f"<div class='mval'>{v}</div></div>"
                )
            metrics_html = (
                f"<div class='metrics'>{''.join(metric_items)}</div>" if metric_items else ""
            )

            tables_html = ""
            for t in sec.get("tables") or []:
                tables_html += f"<h4>{t['name']}</h4>{t['html']}"

            figs_html = ""
            for f in sec.get("figures") or []:
                # 兼容 dict 载荷与 (name, fig) 元组
                if isinstance(f, (tuple, list)) and len(f) >= 2:
                    f_name, f_fig = f[0], f[1]
                else:
                    f_name, f_fig = f.get("name", "图表"), f.get("fig")
                figs_html += f"<h4>{f_name}</h4>"
                figs_html += _fig_to_html_div(f_fig, include_js=first_fig)
                first_fig = False

            notes = sec.get("notes") or ""
            notes_html = f"<p class='notes'>{notes}</p>" if notes else ""

            body_parts.append(
                f"""
                <section class='section'>
                  <div class='sec-head'>
                    <span class='badge'>{sec.get('module', '')}</span>
                    <h2>{idx}. {sec.get('title', '')}</h2>
                    <div class='ts'>生成于会话 · {sec.get('ts', '')}</div>
                  </div>
                  {metrics_html}
                  {tables_html}
                  {figs_html}
                  {notes_html}
                </section>
                """
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{report_title}</title>
<style>
  :root {{
    --teal: #0f766e;
    --ink: #0f172a;
    --muted: #64748b;
    --line: rgba(15, 23, 42, 0.08);
    --bg: #f8fafc;
    --card: #ffffff;
    --amber: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    color: var(--ink);
    background:
      radial-gradient(ellipse 70% 40% at 0% 0%, rgba(15,118,110,0.10), transparent 50%),
      radial-gradient(ellipse 50% 30% at 100% 0%, rgba(180,83,9,0.08), transparent 45%),
      var(--bg);
    line-height: 1.6;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 28px 80px; }}
  header.report-hero {{
    background: linear-gradient(135deg, #0f172a 0%, #134e4a 55%, #0f766e 100%);
    color: #f8fafc;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.18);
  }}
  header.report-hero h1 {{
    margin: 0 0 8px;
    font-size: 1.85rem;
    letter-spacing: -0.02em;
  }}
  header.report-hero .sub {{ opacity: 0.85; font-size: 0.98rem; }}
  .section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
  }}
  .sec-head h2 {{
    margin: 8px 0 4px;
    font-size: 1.25rem;
  }}
  .badge {{
    display: inline-block;
    background: linear-gradient(135deg, var(--teal), #0d9488);
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
  }}
  .ts {{ color: var(--muted); font-size: 0.82rem; }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0;
  }}
  .mcard {{
    background: rgba(15, 118, 110, 0.06);
    border: 1px solid rgba(15, 118, 110, 0.12);
    border-radius: 12px;
    padding: 12px 14px;
  }}
  .mlabel {{ font-size: 0.78rem; color: var(--muted); }}
  .mval {{ font-size: 1.05rem; font-weight: 700; color: var(--teal); margin-top: 2px; }}
  table.report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 18px;
    font-size: 0.9rem;
  }}
  table.report-table th {{
    background: #ecfdf5;
    color: #134e4a;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
  }}
  table.report-table td {{
    padding: 9px 12px;
    border-bottom: 1px solid var(--line);
  }}
  table.report-table tr:nth-child(even) td {{ background: #f8fafc; }}
  h4 {{ margin: 18px 0 8px; color: #134e4a; font-size: 1rem; }}
  .notes {{
    margin-top: 14px;
    padding: 12px 14px;
    background: #fffbeb;
    border-left: 3px solid var(--amber);
    border-radius: 0 10px 10px 0;
    color: #78350f;
    font-size: 0.9rem;
  }}
  .empty {{
    background: white;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    color: var(--muted);
    border: 1px dashed var(--line);
  }}
  footer {{
    margin-top: 28px;
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
  }}
  @media print {{
    body {{ background: white; }}
    header.report-hero {{ box-shadow: none; }}
    .section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="report-hero">
      <div style="font-size:0.85rem;opacity:0.75;margin-bottom:6px;">QUANTLAB REPORT</div>
      <h1>{report_title}</h1>
      <div class="sub">导出时间：{now} · 共 {len(sections)} 个分析片段</div>
    </header>
    {''.join(body_parts)}
    <footer>
      由 QuantLab · 本地数据分析与金融量化计算工具箱 自动生成<br/>
      图表基于 Plotly · 可在浏览器中交互缩放
    </footer>
  </div>
</body>
</html>
"""
    return html.encode("utf-8")


def build_pdf_report(
    sections: Optional[List[dict]] = None,
    report_title: str = "QuantLab 分析报告",
) -> bytes:
    """
    生成 PDF 报告。
    优先：将图表导出为 PNG 后用 xhtml2pdf 排版；
    若依赖不可用，则回退为「简化 HTML → PDF」。
    """
    sections = sections if sections is not None else list(_ensure_store())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: List[str] = []
    parts.append(
        f"""
        <h1 style="color:#0f766e;">{report_title}</h1>
        <p style="color:#64748b;">导出时间：{now}</p>
        <hr/>
        """
    )

    if not sections:
        parts.append("<p>暂无分析结果可供导出。</p>")
    else:
        for idx, sec in enumerate(sections, start=1):
            parts.append(
                f"<h2 style='color:#134e4a;'>{idx}. [{sec.get('module','')}] "
                f"{sec.get('title','')}</h2>"
            )
            if sec.get("metrics"):
                parts.append("<ul>")
                for k, v in sec["metrics"].items():
                    parts.append(f"<li><b>{k}</b>: {v}</li>")
                parts.append("</ul>")

            for t in sec.get("tables") or []:
                parts.append(f"<h3>{t['name']}</h3>")
                parts.append(t["html"])

            for f in sec.get("figures") or []:
                if isinstance(f, (tuple, list)) and len(f) >= 2:
                    f_name, f_fig = f[0], f[1]
                else:
                    f_name, f_fig = f.get("name", "图表"), f.get("fig")
                parts.append(f"<h3>{f_name}</h3>")
                b64 = _fig_to_png_b64(f_fig)
                if b64:
                    parts.append(
                        f'<img src="data:image/png;base64,{b64}" width="680"/>'
                    )
                else:
                    parts.append(
                        "<p style='color:#b45309;'>（图表静态导出失败，"
                        "请改用 HTML 报告查看交互图。需安装 kaleido。）</p>"
                    )

            if sec.get("notes"):
                parts.append(f"<p><i>{sec['notes']}</i></p>")
            parts.append("<hr/>")

    html_src = f"""
    <html><head><meta charset="utf-8"/>
    <style>
      body {{ font-family: Helvetica, 'Microsoft YaHei', sans-serif; font-size: 11pt; color: #0f172a; }}
      h1 {{ font-size: 20pt; }} h2 {{ font-size: 14pt; }} h3 {{ font-size: 12pt; color: #0f766e; }}
      table.report-table {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
      table.report-table th, table.report-table td {{
        border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left;
      }}
      table.report-table th {{ background: #ecfdf5; }}
    </style></head>
    <body>{''.join(parts)}</body></html>
    """

    # 尝试 xhtml2pdf
    try:
        from xhtml2pdf import pisa

        buffer = io.BytesIO()
        status = pisa.CreatePDF(src=html_src, dest=buffer, encoding="utf-8")
        if not status.err:
            return buffer.getvalue()
    except Exception:  # noqa: BLE001
        pass

    # 回退：尝试 weasyprint
    try:
        from weasyprint import HTML

        return HTML(string=html_src).write_pdf()
    except Exception:  # noqa: BLE001
        pass

    # 最后回退：返回说明性「伪 PDF」不可行，改为抛出清晰错误
    # 调用方可捕获后仅提供 HTML
    raise RuntimeError(
        "PDF 引擎不可用。请安装 xhtml2pdf（或 weasyprint）与 kaleido 后重试；"
        "亦可先下载 HTML 报告。"
    )


def render_report_download_panel() -> None:
    """在侧边栏或主区渲染报告导出按钮与下载控件。"""
    sections = _ensure_store()
    n = len(sections)

    st.markdown("#### 📥 导出分析报告")
    st.caption(f"当前已缓存 **{n}** 个分析片段（切换模块并完成分析后自动累积）。")

    c1, c2 = st.columns(2)
    with c1:
        clear = st.button("清空缓存", key="rpt_clear", use_container_width=True)
    with c2:
        gen = st.button("生成报告", type="primary", key="rpt_gen", use_container_width=True)

    if clear:
        clear_report_sections()
        st.success("报告缓存已清空。")
        st.rerun()

    if gen or st.session_state.get("report_html_bytes"):
        try:
            if gen:
                with st.spinner("正在汇总图表与指标，生成报告..."):
                    html_bytes = build_html_report()
                    st.session_state["report_html_bytes"] = html_bytes
                    pdf_bytes = None
                    pdf_error = None
                    try:
                        pdf_bytes = build_pdf_report()
                    except Exception as exc:  # noqa: BLE001
                        pdf_error = str(exc)
                    st.session_state["report_pdf_bytes"] = pdf_bytes
                    st.session_state["report_pdf_error"] = pdf_error

            html_bytes = st.session_state.get("report_html_bytes")
            pdf_bytes = st.session_state.get("report_pdf_bytes")
            pdf_error = st.session_state.get("report_pdf_error")

            if html_bytes:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇ 下载 HTML 报告",
                    data=html_bytes,
                    file_name=f"QuantLab_Report_{stamp}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_html",
                )
                if pdf_bytes:
                    st.download_button(
                        label="⬇ 下载 PDF 报告",
                        data=pdf_bytes,
                        file_name=f"QuantLab_Report_{stamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_pdf",
                    )
                elif pdf_error:
                    st.warning(f"PDF 未能生成：{pdf_error}")

        except Exception as exc:  # noqa: BLE001
            st.error(f"报告生成失败: {exc}")
