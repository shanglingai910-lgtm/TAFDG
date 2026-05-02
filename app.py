from __future__ import annotations

from pathlib import Path
import html
import subprocess
import sys
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from demo_app.data_loader import (
    DATASET_INFO,
    DATASET_ORDER,
    nodes_for_dataset,
    scenarios_for_dataset,
    scenario_by_name,
    dataset_summary_dict,
    load_metrics,
)
from demo_app.report import build_markdown_report

st.set_page_config(
    page_title="TAFDG-ICV 车路云协同交通感知演示平台",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.main-title {font-size: 2.05rem; font-weight: 780; color: #f3fbff; letter-spacing: 0.5px;}
.sub-title {font-size: 1rem; color: #a9cfe2; margin-top: -0.4rem;}
.icv-card {border: 1px solid rgba(66,184,236,.35); border-radius: 18px; padding: 18px 18px;
  background: linear-gradient(145deg, rgba(10,40,70,.95), rgba(8,28,52,.90)); box-shadow: 0 10px 26px rgba(0,0,0,.18); color: #eaf7ff; min-height: 120px;}
.icv-card h3 {margin: 0 0 8px 0; font-size: 1.05rem; color: #fff;}
.icv-card p {margin: 2px 0; color: #b9d7e6; font-size: .92rem;}
.metric-big {font-size: 1.8rem; font-weight: 780; color: #24d3ff; margin: .15rem 0;}
.tag {display: inline-block; padding: 4px 9px; border-radius: 999px; background: rgba(36,211,255,.12); color: #77e7ff; border: 1px solid rgba(36,211,255,.35); font-size: .78rem; margin-right: 6px; margin-top: 4px;}
.green {color: #38d996 !important;} .orange {color: #ffad4d !important;} .red {color: #ff6b6b !important;}
.flow-step {border-left: 4px solid #24d3ff; padding: 9px 12px; margin: 8px 0; background: rgba(36,211,255,.08); border-radius: 10px;}
.note-box {background: rgba(255,173,77,.10); border: 1px solid rgba(255,173,77,.35); padding: 12px 14px; border-radius: 14px; color: #ffddb0;}
.truth-box {background: rgba(56,217,150,.10); border: 1px solid rgba(56,217,150,.35); padding: 12px 14px; border-radius: 14px; color: #d8fff0;}
.svg-wrap {border-radius: 18px; overflow: hidden; border: 1px solid rgba(66,184,236,.35);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

metrics, summary_df = load_metrics()
summary = dataset_summary_dict(metrics)


def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def metric_card(title: str, value: str, text: str, color_class: str = ""):
    st.markdown(f"""
    <div class="icv-card"><h3>{esc(title)}</h3><div class="metric-big {color_class}">{esc(value)}</div><p>{esc(text)}</p></div>
    """, unsafe_allow_html=True)


def small_card(title: str, body: str, tags: list[str] | None = None):
    tags = tags or []
    tag_html = "".join(f"<span class='tag'>{esc(t)}</span>" for t in tags)
    st.markdown(f"""
    <div class="icv-card"><h3>{esc(title)}</h3><p>{esc(body)}</p><div>{tag_html}</div></div>
    """, unsafe_allow_html=True)


def line_chart(df_map: Dict[str, pd.DataFrame], column: str, title: str, y_title: str):
    fig = go.Figure()
    colors = {"GTSRB": "#24d3ff", "TT100K": "#ffad4d", "MIO-TCD": "#38d996"}
    for name, df in df_map.items():
        if column in df.columns:
            fig.add_trace(go.Scatter(x=df["round_idx"], y=df[column], mode="lines", name=name, line=dict(width=3, color=colors.get(name))))
    fig.update_layout(title=title, xaxis_title="Communication Round", yaxis_title=y_title, height=360,
                      margin=dict(l=20, r=20, t=60, b=40), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)




def render_svg(svg_html: str, height: int = 560):
    """Render custom SVG/HTML in an iframe to avoid Markdown treating indented SVG tags as code."""
    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
            .svg-wrap {{
              border-radius: 18px;
              overflow: hidden;
              border: 1px solid rgba(66,184,236,.35);
              box-sizing: border-box;
              width: 100%;
            }}
            svg {{ display: block; width: 100%; height: auto; }}
          </style>
        </head>
        <body>{svg_html}</body>
        </html>
        """,
        height=height,
        scrolling=False,
    )

def page_header(title: str, subtitle: str):
    st.markdown(f"<div class='main-title'>{esc(title)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{esc(subtitle)}</div>", unsafe_allow_html=True)
    st.write("")


def dataset_scene_select(dataset: str, key: str = "scene_select") -> dict:
    scenes = scenarios_for_dataset(dataset)
    names = [s["name"] for s in scenes]
    chosen = st.selectbox("选择当前数据集下的应用场景", names, key=f"{key}_{dataset}")
    return scenario_by_name(dataset, chosen)


def dataset_metric_cards(dataset: str):
    s = summary[dataset]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Final Test Acc", fmt_pct(s["final_test_acc"]), DATASET_INFO[dataset]["role"], "green")
    with c2: metric_card("Peak Test Acc", fmt_pct(s["peak_test_acc"]), f"第 {s['peak_round']} 轮峰值", "")
    with c3: metric_card("Final Test Loss", f"{s['final_test_loss']:.3f}", "当前数据集最终损失", "orange")
    with c4: metric_card("通信开销", f"{s['communication_mb']:.2f} MB", "单轮关键更新上传量", "")


def svg_dashboard(dataset: str) -> str:
    info = DATASET_INFO[dataset]
    nodes = nodes_for_dataset(dataset)
    theme = info["theme"]
    positions = [(110, 150), (110, 425), (1010, 150), (1010, 425)]
    node_svg = []
    for n, (x, y) in zip(nodes, positions):
        color = {"green": "#38d996", "orange": "#ffad4d", "red": "#ff6b6b"}.get(n["color"], "#38d996")
        node_svg.append(f'''
        <g>
          <rect x="{x}" y="{y}" width="300" height="150" rx="20" fill="#092d4f" stroke="#4098c8" stroke-width="2"/>
          <text x="{x+24}" y="{y+38}" fill="#fff" font-size="24" font-weight="700">{esc(n['name'])}</text>
          <text x="{x+24}" y="{y+72}" fill="{theme}" font-size="16">域：{esc(n['domain'])}</text>
          <text x="{x+24}" y="{y+102}" fill="#c7e3f0" font-size="15">任务：{esc(n['task'])}</text>
          <text x="{x+24}" y="{y+130}" fill="{color}" font-size="15">状态：{esc(n['state'])}｜风险：{esc(n['risk'])}</text>
          <circle cx="{x+150}" cy="{y+75}" r="8" fill="{theme}"/>
          <line x1="{x+150}" y1="{y+75}" x2="700" y2="340" stroke="{theme}" stroke-width="3" opacity="0.75"/>
        </g>''')
    badges = ["原始图像本地保留", "Local Align 筛选", "Top-K 关键上传", "DP 加噪保护", "Global Align 聚合"]
    badge_svg = []
    x = 70
    for b in badges:
        badge_svg.append(f'<rect x="{x}" y="670" width="240" height="48" rx="16" fill="#0b3c60" stroke="{theme}"/><text x="{x+20}" y="701" fill="#eaf7ff" font-size="16">{esc(b)}</text>')
        x += 255
    svg = f'''
    <div class="svg-wrap"><svg viewBox="0 0 1400 760" width="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="1400" height="760" fill="#071b33"/>
      <defs><pattern id="grid" width="70" height="70" patternUnits="userSpaceOnUse"><path d="M70 0 L0 0 0 70" fill="none" stroke="#0a2a4a" stroke-width="1"/></pattern></defs>
      <rect width="1400" height="760" fill="url(#grid)"/>
      <text x="40" y="48" fill="#f0fcff" font-size="30" font-weight="800">TAFDG-ICV｜{esc(dataset)} {esc(info['task'])}</text>
      <text x="42" y="84" fill="#acd2e6" font-size="16">四个交通节点随数据集变化：本地训练、关键更新上传、云端对齐聚合、全局模型下发</text>
      <rect x="555" y="280" width="290" height="170" rx="24" fill="#0d385d" stroke="{theme}" stroke-width="3"/>
      <text x="603" y="322" fill="#fff" font-size="26" font-weight="700">云端协同中心</text>
      <text x="595" y="365" fill="{theme}" font-size="20">Global Align 聚合</text>
      <text x="615" y="405" fill="#b8e8f6" font-size="18">Model v2 下发</text>
      {''.join(node_svg)}
      {''.join(badge_svg)}
    </svg></div>
    '''
    return svg


def svg_pipeline(dataset: str) -> str:
    info = DATASET_INFO[dataset]
    theme = info["theme"]
    steps = [("本地图像", "不出节点"), ("Local Align", "更新筛选"), ("Top-K", "关键维度"), ("DP", "裁剪加噪"), ("Global Align", "云端聚合")]
    blocks = []
    x = 60
    for i, (a, b) in enumerate(steps):
        blocks.append(f'<rect x="{x}" y="165" width="220" height="170" rx="22" fill="#0c3656" stroke="{theme}" stroke-width="3"/><text x="{x+38}" y="225" fill="#fff" font-size="24" font-weight="700">{esc(a)}</text><text x="{x+48}" y="270" fill="#c4e9f5" font-size="18">{esc(b)}</text>')
        if i < len(steps) - 1:
            blocks.append(f'<line x1="{x+225}" y1="250" x2="{x+285}" y2="250" stroke="{theme}" stroke-width="5"/><polygon points="{x+285},250 {x+268},238 {x+268},262" fill="{theme}"/>')
        x += 270
    return f'''
    <div class="svg-wrap"><svg viewBox="0 0 1400 520" width="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="1400" height="520" fill="#071b33"/>
      <text x="40" y="52" fill="#f0fcff" font-size="30" font-weight="800">{esc(dataset)}｜{esc(info['task'])} 协同链路</text>
      <text x="42" y="88" fill="#acd2e6" font-size="16">当前数据集切换后，节点、场景图像、识别结果、协同指标与报告均联动变化。</text>
      {''.join(blocks)}
      <text x="60" y="430" fill="#dceff8" font-size="18">链路含义：用车联网应用动作展示算法机制，而不是把曲线硬包装成动态效果。</text>
    </svg></div>
    '''


def svg_comparison(dataset: str, scenario: dict) -> str:
    info = DATASET_INFO[dataset]
    theme = info["theme"]
    visual = scenario["visual"]
    b = scenario["baseline"]
    t = scenario["tafdg"]
    is_vehicle = visual == "vehicle"
    sign_baseline = "?" if dataset != "TT100K" else "!"
    sign_tafdg = "60" if dataset == "GTSRB" else "STOP"
    object_svg_left = """
      <circle cx="390" cy="285" r="50" fill="#d9d9d9" stroke="#b06666" stroke-width="8"/>
      <text x="370" y="297" fill="#b06666" font-size="28" font-weight="800">{}</text>
      <rect x="330" y="225" width="125" height="125" fill="none" stroke="#ff6b6b" stroke-width="5"/>
    """.format(sign_baseline)
    object_svg_right = """
      <circle cx="1090" cy="285" r="50" fill="#f7f7f7" stroke="{}" stroke-width="8"/>
      <text x="1052" y="297" fill="{}" font-size="24" font-weight="800">{}</text>
      <rect x="1028" y="223" width="130" height="130" fill="none" stroke="{}" stroke-width="5"/>
    """.format(theme, theme, sign_tafdg, theme)
    if is_vehicle:
        object_svg_left = '''
          <rect x="250" y="300" width="90" height="42" rx="8" fill="#3c96e6"/><rect x="375" y="318" width="90" height="42" rx="8" fill="#999"/><rect x="500" y="295" width="90" height="42" rx="8" fill="#d49840"/>
          <rect x="230" y="260" width="390" height="130" fill="none" stroke="#ff6b6b" stroke-width="5"/>
        '''
        object_svg_right = f'''
          <rect x="950" y="300" width="90" height="42" rx="8" fill="#3c96e6"/><rect x="1075" y="318" width="90" height="42" rx="8" fill="#e6b24a"/><rect x="1200" y="295" width="90" height="42" rx="8" fill="#a8d35a"/>
          <rect x="930" y="260" width="390" height="130" fill="none" stroke="{theme}" stroke-width="5"/>
        '''
    fog_overlay = '<rect x="80" y="180" width="560" height="330" fill="#dfe7e8" opacity="0.25"/><rect x="780" y="180" width="560" height="330" fill="#dfe7e8" opacity="0.12"/>' if ("fog" in scenario["domain"] or "rain" in scenario["domain"]) else ""
    night_overlay = '<rect x="80" y="180" width="560" height="330" fill="#061225" opacity="0.45"/><rect x="780" y="180" width="560" height="330" fill="#061225" opacity="0.25"/>' if "night" in scenario["domain"] else ""
    return f'''
    <div class="svg-wrap"><svg viewBox="0 0 1420 760" width="100%" xmlns="http://www.w3.org/2000/svg">
      <rect width="1420" height="760" fill="#071b33"/>
      <text x="38" y="48" fill="#f0fcff" font-size="30" font-weight="800">{esc(dataset)}｜{esc(scenario['name'])}</text>
      <text x="40" y="84" fill="#acd2e6" font-size="16">应用场景示意：图像用于演示，数值和曲线来自当前数据集 100 轮结果；切换数据集后此画面与结果同步变化。</text>
      <rect x="50" y="120" width="635" height="580" rx="24" fill="#122b3f" stroke="#777" stroke-width="2"/>
      <rect x="735" y="120" width="635" height="580" rx="24" fill="#0e393e" stroke="{theme}" stroke-width="3"/>
      <text x="82" y="158" fill="#fff" font-size="24" font-weight="700">常规处理方式</text>
      <text x="767" y="158" fill="#fff" font-size="24" font-weight="700">TAFDG-ICV 协同模型</text>
      <polygon points="95,510 640,510 520,215 215,215" fill="#303438"/><polygon points="795,510 1340,510 1220,215 915,215" fill="#303438"/>
      <line x1="370" y1="510" x2="400" y2="215" stroke="#e1d15c" stroke-width="4" stroke-dasharray="20 18"/><line x1="1070" y1="510" x2="1100" y2="215" stroke="#e1d15c" stroke-width="4" stroke-dasharray="20 18"/>
      {fog_overlay}{night_overlay}
      {object_svg_left}
      {object_svg_right}
      <rect x="330" y="180" width="310" height="38" rx="8" fill="#ff6b6b"/><text x="345" y="205" fill="#fff" font-size="17">{esc(b['label'])}  {b['confidence']:.1f}%</text>
      <rect x="1028" y="178" width="330" height="38" rx="8" fill="{theme}"/><text x="1043" y="203" fill="#fff" font-size="17">{esc(t['label'])}  {t['confidence']:.1f}%</text>
      <text x="92" y="560" fill="#ffb8b8" font-size="18" font-weight="700">问题：{esc(b['risk'])}</text>
      <text x="782" y="560" fill="#d7fff0" font-size="18" font-weight="700">优势：{esc(t['risk'])}</text>
      <rect x="92" y="615" width="300" height="42" rx="14" fill="#1b405e" stroke="#408ec8"/><text x="112" y="642" fill="#eaf7ff" font-size="16">场景域：{esc(scenario['domain'])}</text>
      <rect x="782" y="615" width="350" height="42" rx="14" fill="#17483f" stroke="{theme}"/><text x="802" y="642" fill="#eaf7ff" font-size="16">数据策略：原始图像本地保留</text>
    </svg></div>
    '''

with st.sidebar:
    st.markdown("### 🚦 TAFDG-ICV")
    st.caption("车路云协同交通感知演示平台")
    page = st.radio("展示模块", ["车路云协同驾驶舱", "复杂交通场景识别效果", "隐私保护与低通信协同", "多路口模型协同更新", "系统效果与报告导出"])
    st.divider()
    selected_dataset = st.selectbox("当前数据集", DATASET_ORDER, index=0)
    info = DATASET_INFO[selected_dataset]
    current = summary[selected_dataset]
    st.markdown(f"**当前任务：** {info['task']}")
    st.caption(f"代理域：{info['domains']}")
    st.metric("Final Test Acc", fmt_pct(current["final_test_acc"]))
    st.metric("Peak Test Acc", fmt_pct(current["peak_test_acc"]), f"Round {current['peak_round']}")
    st.caption(f"数据性质：{info['nature']}")

if page == "车路云协同驾驶舱":
    page_header("TAFDG-ICV 车路云协同交通感知演示平台", "数据集切换会同步改变路口节点、应用任务、场景图像、识别结果和指标摘要。")
    st.markdown("<div class='truth-box'><b>联动说明：</b>当前页面不是固定展示图。选择 GTSRB、TT100K 或 MIO-TCD 后，驾驶舱节点、交通域、任务名称和后续识别场景都会随数据集变化；实验指标来自 demo_results 中对应数据集的 100 轮结果文件。</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        render_svg(svg_dashboard(selected_dataset), height=560)
    with c2:
        small_card("当前数据集", f"{selected_dataset}：{info['description']}", [info['task'], info['nature']])
        st.write("")
        small_card("应用目标", "展示多路口、多天气、多设备交通视觉任务中的安全协同训练和跨域鲁棒感知能力。", ["人-车-路-云", "车联网智能计算", "隐私保护"])
        st.write("")
        small_card("云端协同中心", "只接收经过 Top-K 压缩与差分隐私保护的模型更新，不接收原始交通图像。", ["Global Align", "Model v2", "Secure Update"])
    st.write("")
    dataset_metric_cards(selected_dataset)
    st.write("")
    st.markdown("#### 当前数据集的四个交通节点")
    cols = st.columns(4)
    color_map = {"green": "#38d996", "orange": "#ffad4d", "red": "#ff6b6b"}
    for col, n in zip(cols, nodes_for_dataset(selected_dataset)):
        with col:
            st.markdown(f"""
            <div class='icv-card'><h3>{esc(n['name'])}</h3><p><b>交通域：</b>{esc(n['domain'])}</p><p><b>任务：</b>{esc(n['task'])}</p>
            <p><b>节点状态：</b><span style='color:{color_map.get(n['color'], '#38d996')}'>{esc(n['state'])}</span></p><p><b>风险级别：</b>{esc(n['risk'])}</p></div>
            """, unsafe_allow_html=True)

elif page == "复杂交通场景识别效果":
    page_header("复杂交通场景识别效果", "当前页面只展示所选数据集对应的场景，不再用同一张固定图替代不同任务。")
    scenario = dataset_scene_select(selected_dataset, "effect")
    s = summary[selected_dataset]
    top1, top2, top3 = st.columns([1.2, 1, 1])
    with top1:
        st.markdown(f"#### {esc(scenario['name'])}")
        st.markdown(f"<span class='tag'>{esc(selected_dataset)}</span><span class='tag'>{esc(scenario['domain'])}</span><span class='tag'>{esc(DATASET_INFO[selected_dataset]['task'])}</span>", unsafe_allow_html=True)
        st.write(scenario["story"])
    with top2:
        metric_card("TAFDG-ICV Final Test Acc", fmt_pct(s["final_test_acc"]), DATASET_INFO[selected_dataset]["role"], "green")
    with top3:
        metric_card("Peak Test Acc", fmt_pct(s["peak_test_acc"]), f"第 {s['peak_round']} 轮达到峰值", "")
    render_svg(svg_comparison(selected_dataset, scenario), height=500)
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("#### 常规处理方式的表现")
        b = scenario["baseline"]
        st.markdown(f"<div class='note-box'><b>预测结果：</b>{esc(b['label'])}<br><b>置信度：</b>{b['confidence']:.1f}%<br><b>问题：</b>{esc(b['risk'])}</div>", unsafe_allow_html=True)
    with rcol:
        st.markdown("#### TAFDG-ICV 协同后的表现")
        t = scenario["tafdg"]
        st.markdown(f"<div class='note-box'><b>预测结果：</b>{esc(t['label'])}<br><b>置信度：</b>{t['confidence']:.1f}%<br><b>优势：</b>{esc(t['risk'])}</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='truth-box'><b>真实性边界：</b>此处交通画面是应用场景示意素材，目的是展示不同数据集/场景与算法机制的联动；数值指标和曲线读取自对应数据集 100 轮结果。后续接入真实图片目录后，只需替换场景素材与预测结果 JSON，页面会继续按数据集联动。</div>", unsafe_allow_html=True)
    with st.expander("查看当前数据集的 100 轮效果曲线"):
        line_chart({selected_dataset: metrics[selected_dataset]}, "test_accuracy", f"{selected_dataset} Test Accuracy", "Accuracy (%)")
        line_chart({selected_dataset: metrics[selected_dataset]}, "test_loss", f"{selected_dataset} Test Loss", "Loss")

elif page == "隐私保护与低通信协同":
    page_header("隐私保护与低通信协同过程", "同一套 TAFDG 机制在不同交通任务中对应不同节点与指标状态。")
    render_svg(svg_pipeline(selected_dataset), height=430)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 协同训练链路")
        steps = [("1. 原始交通图像本地保留", "车端或路侧摄像头采集的图像不上传云端，避免车辆、位置和道路环境敏感信息集中暴露。"), ("2. Local Align 抑制本地学偏", "当某一路口因强雾、夜间弱光或摄像头抖动产生异常更新时，系统通过更新方向一致性筛选降低其影响。"), ("3. Top-K 压缩关键更新", "只上传幅值贡献最大的关键更新维度，适配路侧设备到云端的低带宽链路。"), ("4. 差分隐私保护", "上传前执行裁剪与噪声扰动，使模型更新不直接暴露单个客户端的敏感特征。"), ("5. Global Align 云端聚合", "云端根据各节点更新方向一致性分配权重，而不是简单平均所有客户端。")]
        for title, text in steps:
            st.markdown(f"<div class='flow-step'><b>{esc(title)}</b><br>{esc(text)}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("#### 当前数据集协同指标")
        s = summary[selected_dataset]
        m1, m2 = st.columns(2)
        with m1: metric_card("ε", f"{s['epsilon']:.2f}", "轮次感知隐私预算", "")
        with m2: metric_card("σ", f"{s['sigma']:.3f}", "高斯噪声尺度", "")
        m3, m4 = st.columns(2)
        with m3: metric_card("Local Cos", f"{s['local_cos']:.3f}", "客户端更新一致性", "green")
        with m4: metric_card("Server Cos", f"{s['server_cos']:.3f}", "服务端聚合一致性", "green")
        m5, m6 = st.columns(2)
        with m5: metric_card("Keep Ratio", f"{s['kept_batch_ratio']:.3f}", "Local Align 后保留比例", "")
        with m6: metric_card("Noisy Norm", f"{s['noisy_norm']:.3f}", "加噪后更新范数", "")
        line_chart({selected_dataset: metrics[selected_dataset]}, "mean_epsilon_t", f"{selected_dataset} Privacy Budget Schedule", "epsilon_t")

elif page == "多路口模型协同更新":
    page_header("多路口模型协同更新", "根据当前数据集展示不同路口/路侧节点的协同状态，避免固定节点模板。")
    stage_key = f"sync_stage_{selected_dataset}"
    if stage_key not in st.session_state:
        st.session_state[stage_key] = 0
    b1, b2, b3 = st.columns([1, 1, 5])
    with b1:
        if st.button("开始协同更新", type="primary"):
            st.session_state[stage_key] = min(5, st.session_state[stage_key] + 1)
    with b2:
        if st.button("重置流程"):
            st.session_state[stage_key] = 0
    stages = ["待开始", "本地训练", "Local Align", "Top-K + DP", "云端对齐聚合", "全局模型下发"]
    stage = st.session_state[stage_key]
    st.progress(stage / 5)
    st.markdown(f"### 当前阶段：{esc(stages[stage])}｜{esc(selected_dataset)} {esc(DATASET_INFO[selected_dataset]['task'])}")
    cols = st.columns(4)
    color_map = {"green": "#38d996", "orange": "#ffad4d", "red": "#ff6b6b"}
    for col, n in zip(cols, nodes_for_dataset(selected_dataset)):
        with col:
            st.markdown(f"""
            <div class='icv-card'><h3>{esc(n['name'])}</h3><p><b>交通域：</b>{esc(n['domain'])}</p><p><b>任务：</b>{esc(n['task'])}</p>
            <p><b>节点状态：</b><span style='color:{color_map.get(n['color'], '#38d996')}'>{esc(n['state'])}</span></p>
            <p><b>数据策略：</b>原始图像本地保留</p><p><b>上传内容：</b>压缩加噪模型更新</p></div>""", unsafe_allow_html=True)
    st.write("")
    hardest = max(nodes_for_dataset(selected_dataset), key=lambda x: {"低": 1, "中": 2, "高": 3}.get(x["risk"], 1))
    if stage >= 1: st.info(f"本地训练：{selected_dataset} 的各节点基于自身交通域数据更新本地模型，不共享原始图像。")
    if stage >= 2: st.warning(f"Local Align：检测到 {hardest['name']} 的 {hardest['domain']} 更新更容易偏离全局趋势，系统按更新方向一致性降低其干扰。")
    if stage >= 3: st.info(f"Top-K + DP：针对 {DATASET_INFO[selected_dataset]['task']}，仅上传关键更新维度，并在上传前执行裁剪与高斯噪声扰动。")
    if stage >= 4: st.success(f"Global Align：云端根据 {selected_dataset} 各节点更新方向一致性进行加权聚合，生成全局模型 v2。")
    if stage >= 5: st.success(f"模型下发完成：{selected_dataset} 多节点同步接收新的全局模型，当前任务识别稳定性提升。")
    with st.expander("可选：从界面启动原始 quickstart 训练脚本"):
        st.caption("该入口用于证明代码主干可运行。比赛录屏时建议先展示应用效果，再根据机器性能运行 quickstart。")
        if st.button("运行 python main.py（当前 main.py 配置）"):
            with st.spinner("训练脚本运行中，请等待终端输出..."):
                try:
                    proc = subprocess.run([sys.executable, "main.py"], cwd=Path(__file__).resolve().parent, text=True, capture_output=True, timeout=300)
                    st.code(proc.stdout[-5000:] or proc.stderr[-5000:])
                except Exception as e:
                    st.error(f"训练脚本未成功完成：{e}")

elif page == "系统效果与报告导出":
    page_header("系统效果总结与报告导出", "报告内容会根据当前数据集与当前场景生成，而不是固定总览文本。")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.write("")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        line_chart({selected_dataset: metrics[selected_dataset]}, "test_accuracy", f"{selected_dataset} Test Accuracy 100 轮趋势", "Accuracy (%)")
        line_chart({selected_dataset: metrics[selected_dataset]}, "test_loss", f"{selected_dataset} Test Loss 100 轮趋势", "Loss")
    with c2:
        st.markdown("#### 当前数据集价值归纳")
        small_card("应用价值", DATASET_INFO[selected_dataset]["description"], [DATASET_INFO[selected_dataset]["task"], DATASET_INFO[selected_dataset]["role"]])
        st.write("")
        small_card("技术先进性", "双端对齐、Top-K 压缩和轮次感知差分隐私不是简单拼接，而是分别对应本地训练、上传通信和云端聚合三个误差来源。", ["Local Align", "Top-K", "DP", "Global Align"])
        st.write("")
        small_card("第二赛段展示能力", "前端能够按数据集联动展示应用场景、运行流程、实验效果和报告导出，适合配合源代码、系统运行视频和在线演示链接提交。", ["运行演示", "报告导出", "设计展示"])
    scenes = scenarios_for_dataset(selected_dataset)
    scenario_name = st.selectbox("报告中的默认场景", [s["name"] for s in scenes], key=f"report_scenario_{selected_dataset}")
    report = build_markdown_report(summary, selected_dataset, scenario_name)
    st.download_button("下载 Markdown 运行报告", report.encode("utf-8"), file_name=f"TAFDG-ICV_{selected_dataset}_demo_report.md", mime="text/markdown")
    current_df = metrics[selected_dataset].copy()
    st.download_button("下载当前数据集 100 轮 CSV", current_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"TAFDG-ICV_{selected_dataset}_round_metrics.csv", mime="text/csv")
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载三数据集摘要 CSV", csv_bytes, file_name="TAFDG-ICV_summary.csv", mime="text/csv")
