# TAFDG Clean Reproduction Code (Revised)



## 1. 现在推荐的运行方式

### 第一步：打开 `main.py`

你只需要改这一行：

```python
PROFILE = "easy_real"
```

可选值：

- `quickstart`
- `easy_real`
- `paper_imagefolder`

### 第二步：直接运行

```bash
python main.py
```

---

## 2. 三种 profile 的用途

### 2.1 `quickstart`

最快冒烟测试。

特点：

- 数据集：`synthetic`
- 模型：`tinycnn`
- 客户端数量少
- 通信轮数少

适合：

- 先验证整条 TAFDG 流程能不能跑通
- 查环境问题
- 改代码后快速回归

### 2.2 `easy_real`

真实数据上的轻量实验档位。

特点：

- 数据集：`gtsrb`
- 支持自动下载
- 仍然保留 TAFDG 的双端对齐 + Top-K + DP
- 但把规模缩到更容易运行

适合：

- 普通机器先做真实数据实验
- 先看曲线趋势和结果格式
- 再决定是否扩到更大的论文设置

### 2.3 `paper_imagefolder`

更贴近论文实验设置。

特点：

- 数据集：`imagefolder`
- 适合 TerraInc / DomainNet 这类天然多域目录
- 模型：`resnet18`
- `num_clients=100`
- `clients_per_round=1.0`
- `rounds=100`
- `batch_size=128`

你只需要把：

```python
data_root="/path/to/TerraInc_or_DomainNet"
domains=[...]
holdout_domain="..."
num_classes=...
```

改成你的真实目录即可。

---

## 3. 目前默认 profile 摘要

默认 `easy_real` 采用：

```python
num_clients=20
clients_per_round=0.2
rounds=20
batch_size=64
model="tinycnn"
dp_mode="exact"
align_warmup_rounds=1
```



---

## 4. 想切到论文档位时怎么改

在 `main.py` 里把：

```python
PROFILE = "paper_imagefolder"
```




详细说明见 `README_FRONTEND.md`。
