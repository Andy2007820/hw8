# -*- coding: utf-8 -*-
"""作业八的最小正确性验证与实验环境记录。"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import winreg
from pathlib import Path

import distill_lm as d
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs_distill"
OUT.mkdir(exist_ok=True)
checks = {}


def close(a, b, tol=1e-8):
    return abs(float(a) - float(b)) <= tol


# 数据、参数量与近似手算误差。
tok, train_ids, val_ids = d.make_data()
assert tok.vocab_size == 322, tok.vocab_size
assert len(train_ids) == 49457, len(train_ids)
assert len(val_ids) == 4901, len(val_ids)
train_text = d.build_corpus(3000, seed=0, include_poems=True)
val_candidate = d.build_corpus(300, seed=999, include_poems=False)
train_lines = set(train_text.splitlines())
val_lines = [line for line in val_candidate.splitlines() if line and line not in train_lines]
val_text = "".join(val_lines) + "\n"
overlap = set(val_lines) & train_lines
oov_count = len(val_text) - len(tok.encode(val_text))
assert not overlap and oov_count == 0
checks["data"] = {"vocab": tok.vocab_size, "train_chars": len(train_ids),
                  "val_chars": len(val_ids), "shared_val_lines": len(overlap),
                  "val_oov_characters": oov_count}
param_rows = {}
for name, cfg in (("teacher", d.TEACHER_CFG), ("student", d.STUDENT_CFG)):
    model = d.GPT(tok.vocab_size, cfg)
    C, L, T, V = cfg["n_embd"], cfg["n_layer"], cfg["block_size"], tok.vocab_size
    approximate = 2 * V * C + T * C + L * 12 * C * C
    relative_error = abs(model.n_param - approximate) / model.n_param
    assert relative_error < 0.02, (name, model.n_param, approximate, relative_error)
    param_rows[name] = {"actual": model.n_param, "approximate": approximate,
                        "relative_error_pct": relative_error * 100}
checks["parameter_count"] = param_rows

# 两个学生必须使用完全一致的初始参数，两个数据生成器必须给出完全一致的批次。
torch.manual_seed(0)
s0 = d.new_student(tok.vocab_size, 0)
s1 = d.new_student(tok.vocab_size, 0)
assert all(torch.equal(a, b) for (ka, a), (kb, b) in
           zip(s0.state_dict().items(), s1.state_dict().items()) if ka == kb)
g0 = torch.Generator().manual_seed(1)
g1 = torch.Generator().manual_seed(1)
for _ in range(120):
    x0, y0 = d.get_batch(train_ids, 24, 96, g0)
    x1, y1 = d.get_batch(train_ids, 24, 96, g1)
    assert torch.equal(x0, x1) and torch.equal(y0, y1)
checks["fair_control"] = "identical initialization and all 120 training batches: PASS"

# 修改最后一个 token 不得影响任何过去位置的 logits。
cfg = dict(n_layer=1, n_head=2, n_embd=16, block_size=8, dropout=0.0)
small = d.GPT(50, cfg).eval()
idx = torch.randint(0, 50, (1, 8))
changed = idx.clone()
changed[0, -1] = (changed[0, -1] + 1) % 50
with torch.no_grad():
    a = small(idx)[0]
    b = small(changed)[0]
max_difference = (a[:, :-1] - b[:, :-1]).abs().max().item()
assert max_difference < 1e-6, max_difference
checks["causal_mask"] = {"max_past_logit_difference": max_difference}

# 验证 KL 方向、reshape 后的 batchmean 与手工公式完全一致。
torch.manual_seed(9)
student_logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
teacher_logits = torch.randn(2, 3, 5, dtype=torch.float64)
targets = torch.randint(0, 5, (2, 3))
T = 4.0
student_logp = F.log_softmax(student_logits.reshape(-1, 5) / T, dim=-1)
teacher_p = F.softmax(teacher_logits.reshape(-1, 5) / T, dim=-1)
actual = F.kl_div(student_logp, teacher_p, reduction="batchmean") * T * T
manual = ((teacher_p * (teacher_p.log() - student_logp)).sum(dim=-1).mean()) * T * T
assert torch.allclose(actual, manual, atol=1e-12), (actual, manual)
checks["kl_direction_and_reduction"] = {"torch": actual.item(), "manual": manual.item()}
loss_alpha0, ce_alpha0, _ = d.distillation_loss(student_logits, teacher_logits, targets, T, 0.0)
assert torch.equal(loss_alpha0, ce_alpha0)
assert torch.isfinite(loss_alpha0) and loss_alpha0.item() > 0
checks["alpha_zero"] = "loss exactly equals hard-label CE: PASS"

# 对第一个 logit 做中心差分，核对 T^2*KL 的反向梯度。
analytic = torch.autograd.grad(actual, student_logits, retain_graph=False)[0][0, 0, 0].item()
base = student_logits.detach().clone()
eps = 1e-5
base[0, 0, 0] += eps
plus = F.kl_div(F.log_softmax(base.reshape(-1, 5) / T, dim=-1), teacher_p,
                reduction="batchmean") * T * T
base[0, 0, 0] -= 2 * eps
minus = F.kl_div(F.log_softmax(base.reshape(-1, 5) / T, dim=-1), teacher_p,
                 reduction="batchmean") * T * T
finite_difference = ((plus - minus) / (2 * eps)).item()
assert abs(analytic - finite_difference) < 1e-6, (analytic, finite_difference)
checks["kd_gradient"] = {"analytic": analytic, "finite_difference": finite_difference}

# 温度升高应提高软标签熵；相同分布的 KL 应为零。
logits = torch.tensor([[1.0, 0.5, -0.2, -1.0]], dtype=torch.float64)
entropy1 = -(F.softmax(logits / 1, -1) * F.log_softmax(logits / 1, -1)).sum().item()
entropy4 = -(F.softmax(logits / 4, -1) * F.log_softmax(logits / 4, -1)).sum().item()
assert entropy4 > entropy1
zero_kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(logits, -1), reduction="batchmean")
assert close(zero_kl, 0.0, 1e-12)
checks["temperature"] = {"entropy_T1": entropy1, "entropy_T4": entropy4,
                         "identical_distribution_kl": float(zero_kl)}

# 环境信息：注册表读取 CPU 名称，GlobalMemoryStatusEx 读取物理内存。
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
    cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


memory = MEMORYSTATUSEX()
memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
environment = {
    "os": platform.system(),
    "os_release": platform.release(),
    "os_version": platform.version(),
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "device": "cpu",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_available_after_import": torch.cuda.is_available(),
    "cpu": cpu_name,
    "logical_cpus": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "physical_memory_bytes": memory.ullTotalPhys,
    "physical_memory_gib": memory.ullTotalPhys / 1024 ** 3,
}
checks["environment"] = environment
(OUT / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(checks, ensure_ascii=False, indent=2))
print("ALL VALIDATION CHECKS PASSED")
