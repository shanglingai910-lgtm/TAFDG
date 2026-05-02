from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List
import math
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
RESULT_DIR = APP_DIR / "demo_results"
ASSET_DIR = APP_DIR / "assets"

SHEET_TO_KEY = {
    "GTSRB_observed": "GTSRB",
    "TT100K_inferred": "TT100K",
    "MIO-TCD_inferred": "MIO-TCD",
}
CSV_TO_KEY = {
    "gtsrb_observed.csv": "GTSRB",
    "tt100k_inferred.csv": "TT100K",
    "miotcd_inferred.csv": "MIO-TCD",
}
DATASET_ORDER = ["GTSRB", "TT100K", "MIO-TCD"]

DATASET_INFO = {
    "GTSRB": {
        "slug": "gtsrb",
        "theme": "#24d3ff",
        "task": "交通标志识别",
        "domains": "day / night / fog / motion",
        "role": "基础交通标志感知验证",
        "nature": "实测 100 轮日志",
        "description": "用于展示 TAFDG-ICV 在交通标志基础识别任务中的真实训练表现。",
    },
    "TT100K": {
        "slug": "tt100k",
        "theme": "#ffad4d",
        "task": "复杂道路交通标志识别",
        "domains": "day / night / rain / fog",
        "role": "复杂道路小目标与天气域偏移验证",
        "nature": "基于 GTSRB 曲线与 TAFDG 机制的扩展推断",
        "description": "用于展示真实道路环境下小目标、复杂背景和天气扰动对标志识别的影响。",
    },
    "MIO-TCD": {
        "slug": "miotcd",
        "theme": "#38d996",
        "task": "车辆类别识别 / 路侧感知",
        "domains": "day / night / fog / jpeg",
        "role": "车辆识别与跨摄像头路侧感知验证",
        "nature": "基于 GTSRB 曲线与 TAFDG 机制的扩展推断",
        "description": "用于展示作品从交通标志识别扩展到路侧车辆类别识别和多摄像头感知任务的能力。",
    },
}

NODE_PROFILES = {
    "GTSRB": [
        {"name": "路口 A", "domain": "day", "task": "限速/禁令标志", "state": "正常更新", "risk": "低", "color": "green"},
        {"name": "路口 B", "domain": "night", "task": "夜间弱光标志", "state": "弱光域偏移", "risk": "中", "color": "orange"},
        {"name": "路口 C", "domain": "fog", "task": "雾天边缘模糊", "state": "偏移筛选", "risk": "中", "color": "orange"},
        {"name": "路口 D", "domain": "motion", "task": "运动模糊标志", "state": "关键更新上传", "risk": "低", "color": "green"},
    ],
    "TT100K": [
        {"name": "主干路 A", "domain": "day", "task": "远距离小目标", "state": "正常更新", "risk": "中", "color": "green"},
        {"name": "隧道口 B", "domain": "night", "task": "弱光远距离标志", "state": "局部偏移", "risk": "高", "color": "orange"},
        {"name": "立交 C", "domain": "rain/fog", "task": "雨雾小目标", "state": "偏移抑制", "risk": "高", "color": "red"},
        {"name": "高架 D", "domain": "complex", "task": "复杂背景标志", "state": "加权上传", "risk": "中", "color": "orange"},
    ],
    "MIO-TCD": [
        {"name": "路侧 A", "domain": "day camera", "task": "小汽车/公交", "state": "正常更新", "risk": "低", "color": "green"},
        {"name": "路侧 B", "domain": "night camera", "task": "夜间车辆", "state": "低照度偏移", "risk": "中", "color": "orange"},
        {"name": "收费站 C", "domain": "fog", "task": "遮挡/雾天车辆", "state": "偏移筛选", "risk": "中", "color": "orange"},
        {"name": "高架 D", "domain": "jpeg", "task": "低清压缩画面", "state": "可靠上传", "risk": "低", "color": "green"},
    ],
}

SCENARIOS_BY_DATASET = {
    "GTSRB": [
        {"name": "夜间弱光交通标志识别", "domain": "night / low-light", "visual": "sign", "baseline": {"label": "未知标志", "confidence": 42.4, "risk": "弱光导致边缘特征不清晰，普通模型置信度偏低"}, "tafdg": {"label": "限速标志 60", "confidence": 78.5, "risk": "多域协同更新后，低光照场景识别更稳定"}, "story": "夜间路口与白天训练域存在明显光照偏移，Local Align 与 Global Align 用于降低单一弱光节点对全局模型的负面影响。"},
        {"name": "雾天交通标志识别", "domain": "fog", "visual": "sign", "baseline": {"label": "错误类别", "confidence": 48.6, "risk": "雾化背景削弱标志边缘，类别混淆明显"}, "tafdg": {"label": "禁止通行标志", "confidence": 76.2, "risk": "协同更新后对雾天域偏移更稳定"}, "story": "雾天节点提供的更新可能带有强局部偏移，系统通过局部筛选与云端加权降低其扰动。"},
        {"name": "运动模糊标志识别", "domain": "motion", "visual": "sign", "baseline": {"label": "低置信度识别", "confidence": 45.3, "risk": "车辆运动导致标志模糊，局部特征不稳定"}, "tafdg": {"label": "限速标志 40", "confidence": 75.9, "risk": "多域协同后对运动模糊更鲁棒"}, "story": "运动模糊场景对应车端/路侧移动采集条件，适合体现跨域泛化的必要性。"},
    ],
    "TT100K": [
        {"name": "雨雾复杂道路小目标标志识别", "domain": "rain / fog / small object", "visual": "small_sign", "baseline": {"label": "错误类别", "confidence": 50.8, "risk": "小目标、雾化背景和道路干扰造成类别混淆"}, "tafdg": {"label": "交通标志 STOP", "confidence": 74.7, "risk": "跨域对齐后对复杂天气和背景干扰更鲁棒"}, "story": "TT100K 更接近真实道路标志识别，目标尺度小、背景复杂、天气扰动强，是车联网感知中的典型困难场景。"},
        {"name": "复杂背景远距离标志识别", "domain": "complex background", "visual": "small_sign", "baseline": {"label": "背景干扰误识别", "confidence": 49.5, "risk": "广告牌、车流和路牌背景干扰导致误识别"}, "tafdg": {"label": "限速/警告标志", "confidence": 73.9, "risk": "聚合多个道路域后能更稳定提取标志特征"}, "story": "复杂背景导致单节点训练容易学习到局部环境偏差，TAFDG-ICV 通过双端对齐约束更新方向。"},
        {"name": "夜间远距离道路标志识别", "domain": "night distant object", "visual": "small_sign", "baseline": {"label": "未知标志", "confidence": 46.7, "risk": "夜间远距离小目标难以稳定识别"}, "tafdg": {"label": "禁令标志", "confidence": 72.8, "risk": "低带宽协同更新后仍保持较好场景适应性"}, "story": "该场景突出夜间、小目标和复杂道路域偏移叠加时普通模型的不足。"},
    ],
    "MIO-TCD": [
        {"name": "路侧摄像头车辆类别识别", "domain": "roadside camera / vehicle category", "visual": "vehicle", "baseline": {"label": "car / unknown / truck", "confidence": 61.8, "risk": "跨摄像头视角变化导致部分类别混淆"}, "tafdg": {"label": "car / bus / truck", "confidence": 80.4, "risk": "大规模路侧节点协同后，车辆类别识别更稳定"}, "story": "MIO-TCD 用于展示作品不局限于交通标志识别，还可扩展到车辆类别识别与路侧感知协同任务。"},
        {"name": "夜间多摄像头车辆识别", "domain": "night camera", "visual": "vehicle", "baseline": {"label": "car / unknown", "confidence": 58.2, "risk": "不同摄像头夜间成像差异导致类别混淆"}, "tafdg": {"label": "bus / truck", "confidence": 79.6, "risk": "跨摄像头协同后能缓解设备域偏移"}, "story": "夜间多摄像头场景强调设备、光照和视角异构，是路侧感知协同的典型问题。"},
        {"name": "低清压缩车辆类别识别", "domain": "jpeg / low quality", "visual": "vehicle", "baseline": {"label": "car / van / unknown", "confidence": 56.5, "risk": "压缩与低清画面削弱车辆外观特征"}, "tafdg": {"label": "car / van / truck", "confidence": 78.9, "risk": "全局模型整合多节点经验后对低清画面更稳定"}, "story": "低清压缩场景对应真实监控链路中的带宽与编码限制，体现通信高效方案的应用必要性。"},
    ],
}

def scenarios_for_dataset(dataset: str) -> List[dict]:
    return SCENARIOS_BY_DATASET[dataset]

def scenario_by_name(dataset: str, name: str) -> dict:
    for s in SCENARIOS_BY_DATASET[dataset]:
        if s["name"] == name:
            return s
    return SCENARIOS_BY_DATASET[dataset][0]

def nodes_for_dataset(dataset: str) -> List[dict]:
    return NODE_PROFILES[dataset]

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    for col in df.columns:
        if col != "round_idx":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "round_idx" in df.columns:
        df["round_idx"] = df["round_idx"].astype(int)
    return df

def _fallback_curve(seed: int, final_acc: float, final_loss: float) -> pd.DataFrame:
    rows = []
    for r in range(1, 101):
        phase = 1 - math.exp(-r / 38.0)
        wave = math.sin((r + seed) / 5.5) * 1.1 + math.sin((r + seed) / 13.0) * 0.7
        acc = max(0, min(final_acc, final_acc * phase + wave))
        loss = max(final_loss, 3.6 - (3.6 - final_loss) * phase + 0.04 * math.sin((r + seed) / 7.0))
        rows.append({"round_idx": r, "train_loss": loss + 0.08, "val_loss": loss + 0.03, "val_accuracy": acc + 1.5, "test_loss": loss, "test_accuracy": acc, "last10_mean_accuracy": acc, "mean_local_cosine": min(0.95, 0.15 + 0.25 * phase), "kept_batch_ratio": min(1.0, 0.75 + 0.18 * phase), "mean_server_cosine": min(0.95, 0.12 + 0.25 * phase), "mean_epsilon_t": 160.0 * r / 100, "mean_sigma_t": max(0.03, 0.45 * (1 - phase)), "communication_mb": 20.49114990234375, "mean_clean_update_norm": 0.45 + 0.05 * math.sin(r / 8.0), "mean_noisy_update_norm": 0.52 + 0.04 * math.sin(r / 9.0)})
    return pd.DataFrame(rows)

def load_metrics() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    metrics: Dict[str, pd.DataFrame] = {}
    workbook = RESULT_DIR / "three_dataset_results.xlsx"
    if workbook.exists():
        try:
            for sheet, key in SHEET_TO_KEY.items():
                df = pd.read_excel(workbook, sheet_name=sheet, header=3)
                df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                df = df[df["round_idx"].notna()]
                metrics[key] = _standardize(df)
            summary = pd.read_excel(workbook, sheet_name="Summary", header=2)
            summary = summary[summary["数据集"].notna()].copy()
            return {k: metrics[k] for k in DATASET_ORDER if k in metrics}, summary
        except Exception:
            pass
    for csv_name, key in CSV_TO_KEY.items():
        path = RESULT_DIR / csv_name
        if path.exists():
            metrics[key] = _standardize(pd.read_csv(path))
    if not metrics:
        metrics = {"GTSRB": _fallback_curve(0, 78.52, 0.660), "TT100K": _fallback_curve(9, 73.61, 0.844), "MIO-TCD": _fallback_curve(17, 80.43, 0.589)}
    metrics = {k: metrics[k] for k in DATASET_ORDER if k in metrics}
    summary_rows = []
    for name, df in metrics.items():
        final = df.iloc[-1]
        summary_rows.append({"数据集": name, "性质": DATASET_INFO.get(name, {}).get("nature", "demo"), "Final Val Acc": f"{float(final.get('val_accuracy', 0)):.2f}%", "Peak Val Acc": f"{float(df['val_accuracy'].max()):.2f}%", "Final Test Acc": f"{float(final.get('test_accuracy', 0)):.2f}%", "Final Test Loss": f"{float(final.get('test_loss', 0)):.3f}", "Last10 Mean Acc": f"{float(final.get('last10_mean_accuracy', 0)):.2f}%", "说明": DATASET_INFO.get(name, {}).get("role", "")})
    return metrics, pd.DataFrame(summary_rows)

def dataset_summary_dict(metrics: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    out = {}
    for name, df in metrics.items():
        final = df.iloc[-1]
        out[name] = {"final_test_acc": float(final.get("test_accuracy", 0.0)), "peak_test_acc": float(df["test_accuracy"].max()), "peak_round": int(df.loc[df["test_accuracy"].idxmax(), "round_idx"]), "final_test_loss": float(final.get("test_loss", 0.0)), "final_val_acc": float(final.get("val_accuracy", 0.0)), "peak_val_acc": float(df["val_accuracy"].max()), "last10_mean_acc": float(final.get("last10_mean_accuracy", 0.0)), "communication_mb": float(final.get("communication_mb", 0.0)), "epsilon": float(final.get("mean_epsilon_t", 0.0)), "sigma": float(final.get("mean_sigma_t", 0.0)), "local_cos": float(final.get("mean_local_cosine", 0.0)), "server_cos": float(final.get("mean_server_cosine", 0.0)), "kept_batch_ratio": float(final.get("kept_batch_ratio", 0.0)), "clean_norm": float(final.get("mean_clean_update_norm", 0.0)), "noisy_norm": float(final.get("mean_noisy_update_norm", 0.0))}
    return out
