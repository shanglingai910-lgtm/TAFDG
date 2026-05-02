from __future__ import annotations
from datetime import datetime
from typing import Dict
from .data_loader import DATASET_INFO, nodes_for_dataset, scenario_by_name


def build_markdown_report(summary: Dict[str, dict], selected_dataset: str, scenario_name: str) -> str:
    info = DATASET_INFO.get(selected_dataset, {})
    scenario = scenario_by_name(selected_dataset, scenario_name)
    current = summary[selected_dataset]
    lines = []
    lines.append("# TAFDG-ICV 车路云协同交通感知演示报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 系统定位")
    lines.append("本系统面向多路口、多天气、多设备交通视觉场景，展示 TAFDG-ICV 在车路云协同环境下的跨域鲁棒感知、隐私保护与低通信协同能力。系统不上传原始交通图像，而是由路侧或车端节点在本地完成训练，并向云端上传经过 Top-K 压缩与差分隐私保护的关键模型更新。")
    lines.append("")
    lines.append("## 2. 当前数据集与应用场景")
    lines.append(f"- 数据集：{selected_dataset}")
    lines.append(f"- 数据性质：{info.get('nature', '')}")
    lines.append(f"- 任务：{info.get('task', '')}")
    lines.append(f"- 代理域：{info.get('domains', '')}")
    lines.append(f"- 当前场景：{scenario_name}")
    lines.append(f"- 场景说明：{scenario.get('story', '')}")
    lines.append("")
    lines.append("## 3. 当前数据集效果摘要")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| Final Test Acc | {current['final_test_acc']:.2f}% |")
    lines.append(f"| Peak Test Acc | {current['peak_test_acc']:.2f}% |")
    lines.append(f"| Peak Round | {current['peak_round']} |")
    lines.append(f"| Final Test Loss | {current['final_test_loss']:.3f} |")
    lines.append(f"| Communication / Round | {current['communication_mb']:.2f} MB |")
    lines.append(f"| Local Cosine | {current['local_cos']:.3f} |")
    lines.append(f"| Server Cosine | {current['server_cos']:.3f} |")
    lines.append("")
    lines.append("## 4. 数据集联动节点状态")
    lines.append("| 节点 | 域 | 任务 | 状态 | 风险 |")
    lines.append("|---|---|---|---|---|")
    for n in nodes_for_dataset(selected_dataset):
        lines.append(f"| {n['name']} | {n['domain']} | {n['task']} | {n['state']} | {n['risk']} |")
    lines.append("")
    lines.append("## 5. 三数据集效果总览")
    lines.append("| 数据集 | Final Test Acc | Peak Test Acc | Peak Round | Final Test Loss | 场景意义 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, s in summary.items():
        lines.append(f"| {name} | {s['final_test_acc']:.2f}% | {s['peak_test_acc']:.2f}% | {s['peak_round']} | {s['final_test_loss']:.3f} | {DATASET_INFO.get(name, {}).get('role', '')} |")
    lines.append("")
    lines.append("## 6. 技术机制说明")
    lines.append("- Local Align：识别并抑制偏离全局协同方向的本地更新，缓解某一路口、天气或设备域导致的局部学偏。")
    lines.append("- Top-K 压缩：仅上传关键更新维度，降低路侧节点到云端的通信负担。")
    lines.append("- 差分隐私：对上传更新执行裁剪与噪声扰动，使原始图像和敏感交通信息不出本地。")
    lines.append("- Global Align：云端根据客户端更新方向一致性进行加权聚合，降低低质量或强偏移节点对全局模型的影响。")
    lines.append("")
    lines.append("## 7. 真实性边界说明")
    lines.append("前端中的交通画面为应用场景示意素材，用于展示数据集、节点、场景和算法机制的联动关系；实验数值、曲线和摘要来自 demo_results 中对应数据集的 100 轮指标文件。若后续替换为真实交通图像或真实 TT100K/MIO-TCD 训练日志，前端会按同一数据接口自动联动展示。")
    return "\n".join(lines)
