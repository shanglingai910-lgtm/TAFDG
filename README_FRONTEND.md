# TAFDG-ICV 车路云协同交通感知演示平台

## 运行方式

```bash
cd rewrite_fast
pip install -r requirements.txt
streamlit run app.py
```

## 本次前端修正重点

本版本已将“固定展示图”改为“数据集联动展示”：

- 选择 **GTSRB**：展示交通标志识别、day/night/fog/motion 四个路口域、GTSRB 对应三类场景和 100 轮实测指标。
- 选择 **TT100K**：展示复杂道路小目标标志识别、day/night/rain/fog 代理域、TT100K 对应场景和 100 轮扩展指标。
- 选择 **MIO-TCD**：展示车辆类别识别/路侧感知、day/night/fog/jpeg 代理域、MIO-TCD 对应场景和 100 轮扩展指标。

数据集切换后，以下内容都会同步变化：

1. 车路云驾驶舱图；
2. 四个交通节点的名称、任务、域和状态；
3. Before / After 场景画面；
4. 普通模型与 TAFDG-ICV 的预测文本与置信度；
5. 隐私、通信、对齐等指标；
6. 报告导出的数据集和场景内容；
7. 当前数据集 100 轮 CSV 导出内容。

## 真实性边界

前端中的交通画面是应用场景示意素材，用于展示数据集、节点、场景和算法机制的联动关系；实验数值、曲线和摘要来自 `demo_app/demo_results/` 中对应数据集的 100 轮指标文件。

如果后续接入真实交通图片或真实 TT100K/MIO-TCD 实验日志，只需要替换：

- `demo_app/demo_results/` 中的 100 轮 CSV/XLSX；
- `demo_app/data_loader.py` 中的场景说明或预测结果文本；
- 或将 `svg_comparison()` 中的示意图逻辑改为读取真实图片路径。

页面会按相同接口继续联动展示。
