# Character-level GPT Knowledge Distillation on CPU

一个不依赖预训练权重或外部语料的字符级 GPT 知识蒸馏实验。项目实现了教师模型训练、学生从零训练、软标签蒸馏、16 组温度/权重消融，以及特征蒸馏、自蒸馏、动态量化和教师规模扫描。

## 主要特性

- 字符级 Pre-Norm GPT，完全在 CPU 上运行。
- 标准蒸馏目标：
  `L = (1 - alpha) * CE + alpha * T^2 * KL(teacher || student)`。
- 正确处理 `[B, S, V]` 张量展平、`batchmean` 归约和 KL 方向。
- 学生从零训练与蒸馏训练共享初始化、数据批顺序和训练预算。
- 完整的 `T={1,2,4,8} × alpha={0.3,0.5,0.7,0.9}` 消融网格。
- 生成训练曲线、消融热力图、量化指标与教师规模扫描结果。
- 独立的数值验证脚本，检查参数量、因果掩码、KL 公式、梯度及公平对照。

## 已测结果

以下数值来自 Windows 11、Python 3.14.8、PyTorch 2.11.0 的 CPU 实测；验证指标为固定抽样批估计。

| 模型 | 参数量 | 训练步数 | 留出 loss | PPL | 与教师一致率 |
|---|---:|---:|---:|---:|---:|
| 教师（C=160） | 737,280 | 200 | 0.7772 | 2.1754 | — |
| 学生·从零训练（C=64） | 147,456 | 120 | 0.8054 | 2.2376 | 0.7389 |
| 学生·蒸馏（T=4，alpha=0.5） | 147,456 | 120 | 0.7755 | 2.1717 | 0.7577 |

蒸馏相对从零训练的留出 loss 下降 **3.71%**。消融最优网格点为 **T=2、alpha=0.5，val_loss=0.8628**；相同 80 步日程的硬标签基线为 **0.9135**。

## 环境

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 快速开始

运行短流程自检：

```bash
python distill_lm.py --mode quick
```

数值与公平性验证：

```bash
python validate_experiment.py
```

依次完成正式实验：

```bash
python distill_lm.py --mode teacher --retrain
python distill_lm.py --mode all
python distill_lm.py --mode ablate
```

## 一键运行全部实验

```bash
python run_suite.py
```

脚本会顺序执行验证、quick、教师训练、主对照、16 组消融及拓展实验，并在 `runs_distill/` 中保存每条命令的日志、退出码和实际耗时。正式教师使用 200 步，学生使用 120 步，消融每组使用 80 步。

也可以单独运行拓展：

```bash
# A：特征蒸馏
python distill_lm.py --mode all --feat 0.1 --run-tag feat

# C：自蒸馏
python distill_lm.py --mode self-distill

# D：动态量化
python distill_lm.py --mode quantize

# E：教师规模扫描 C={96,160,256}
python distill_lm.py --mode scan
```

拓展 B（温度/权重热力图）随 `--mode ablate` 自动生成。真实语料拓展未包含在本项目中。

## 输出文件

| 文件 | 内容 |
|---|---|
| `runs_distill/results_teacher.json` | 教师训练曲线和指标 |
| `runs_distill/results_all.json` | 主对照指标与三段生成样例 |
| `runs_distill/results_ablate.json` | 16 个消融点及同日程 scratch 基线 |
| `runs_distill/results_feat.json` | 特征蒸馏结果 |
| `runs_distill/results_self_distill.json` | 自蒸馏结果 |
| `runs_distill/results_quantize.json` | FP32/INT8 文件大小、loss 与延迟 |
| `runs_distill/results_teacher_scan.json` | 教师规模扫描 |
| `runs_distill/curves_all.png` | 统一硬标签 CE 口径的训练曲线 |
| `runs_distill/ablation_heatmap.png` | 4×4 消融热力图 |
| `runs_distill/suite_manifest.json` | 命令、退出码和墙钟耗时 |

模型 checkpoint 和运行产物均已加入 `.gitignore`，不会默认提交到 GitHub。

## 实验口径与注意事项

- 训练文本由代码现场生成，不下载外部语料。
- 训练集 49,457 字符，留出集 4,901 字符，词表大小 322。
- 留出集移除了训练文本中的固定诗歌和重复行，实测共享行及 OOV 字符均为 0。
- `evaluate()` 使用固定 20 批抽样估计，`agreement()` 使用固定 10 批估计，并非完整留出集遍历。
- KD 训练曲线另存硬标签 CE，绘图时不把 CE+KL 总目标与纯 CE 直接比较。
- 默认通过 `CUDA_VISIBLE_DEVICES=""` 禁用 GPU，模型和数据均使用 CPU 张量。
- 动态量化只量化 `nn.Linear`；小模型上的文件体积下降不保证推理加速。

## 文件说明

- `distill_lm.py`：数据、模型、训练、评估、生成、绘图、消融和拓展实验。
- `validate_experiment.py`：数值正确性、公平对照及环境验证。
- `run_suite.py`：完整实验的顺序执行与日志记录。
