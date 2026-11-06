# -*- coding: utf-8 -*-
"""自制语言模型的知识蒸馏——纯 CPU、零外部数据实验。

规定流程：
    python distill_lm.py --mode quick
    python distill_lm.py --mode teacher --retrain
    python distill_lm.py --mode all
    python distill_lm.py --mode ablate

拓展流程：
    python distill_lm.py --mode all --feat 0.1 --run-tag feat
    python distill_lm.py --mode self-distill
    python distill_lm.py --mode quantize
    python distill_lm.py --mode scan

教师和学生均随机初始化；训练与验证语料由同一文法、不同种子生成。
所有张量计算均在 CPU 上进行，不下载模型或语料。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

# 必须在导入 torch 前关闭 GPU 可见性，保证实验只使用 CPU。
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs_distill"


# ---------------------------------------------------------------------------
# 1. 可复现合成语料
# ---------------------------------------------------------------------------
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0, include_poems=True):
    """同文法、不同种子生成文本；是否追加固定唐诗由调用方显式决定。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append(f"{s}{p}{v}{o}，{m}。")
        elif i % 3 == 1:
            out.append(f"{s}在{p}{o}了一夜{rng.choice(['看', '听', '守', '忆'])}，{m}。")
        else:
            out.append(f"{s}{v}道：“{o}{rng.choice(['正好', '可怜', '可惜', '依旧'])}。”{m}。")
    # 与指南字符数对齐：全部生成句之后只追加一个换行符。
    out.append("\n")
    if include_poems:
        out.extend(POEMS)
    return "".join(out)


class Tokenizer:
    def __init__(self, text):
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 2. 与作业三同构的字符级 Pre-Norm GPT
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, need_hidden=False):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.9, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data():
    train_text = build_corpus(3000, seed=0, include_poems=True)
    # 指南参考代码把相同 POEMS 同时追加到 train/val，会造成泄漏。这里训练集
    # 保留它们以维持 V=322 与教师/学生参数量，但留出集只用独立生成文本。
    val_candidate = build_corpus(300, seed=999, include_poems=False)
    train_lines = set(train_text.splitlines())
    val_lines = [line for line in val_candidate.splitlines()
                 if line and line not in train_lines]
    val_text = "".join(val_lines) + "\n"
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    assert len(val_ids) == len(tok.encode(val_text))
    return tok, train_ids, val_ids


def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = total = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        student_top1 = student(x)[0].argmax(-1)
        teacher_top1 = teacher(x)[0].argmax(-1)
        hit += (student_top1 == teacher_top1).sum().item()
        total += teacher_top1.numel()
    student.train()
    teacher.train()
    return hit / total


def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
    """L=(1-a)CE+a*T^2*KL(p_teacher^T || p_student^T)。

    先 reshape 为 (B*seq_len,V)，使 batchmean 同时平均 batch 与序列位置。
    F.kl_div 的 input 是学生 log 概率，target 是教师概率，方向不能写反。
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    V = student_logits.size(-1)
    ce = F.cross_entropy(student_logits.reshape(-1, V), targets.reshape(-1))
    student_logp = F.log_softmax(student_logits.reshape(-1, V) / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.reshape(-1, V) / temperature, dim=-1)
    kd = F.kl_div(student_logp, teacher_p, reduction="batchmean") * temperature ** 2
    loss = (1.0 - alpha) * ce + alpha * kd
    return loss, ce.detach(), kd.detach()


def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag=""):
    """alpha=0 为硬标签对照；alpha>0 使用教师软标签及可选特征蒸馏。"""
    projection = None
    if teacher is not None and feat_weight > 0:
        projection = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(projection.parameters()) if projection else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    history = {"loss": [], "ce": [], "kd": [], "feature": []}
    if teacher is not None:
        teacher.eval()
    model.train()
    start = time.perf_counter()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        need_hidden = projection is not None
        student_out = model(x, need_hidden=need_hidden)
        logits = student_out[0]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        kd_value = torch.zeros((), device=logits.device)
        feat_value = torch.zeros((), device=logits.device)
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                teacher_out = teacher(x, need_hidden=need_hidden)
                teacher_logits = teacher_out[0]
            loss, ce, kd_value = distillation_loss(logits, teacher_logits, y, T, alpha)
        if projection is not None:
            feat_value = F.mse_loss(projection(student_out[2]), teacher_out[2])
            loss = loss + feat_weight * feat_value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        history["loss"].append(float(loss.item()))
        history["ce"].append(float(ce.item()))
        history["kd"].append(float(kd_value.item()))
        history["feature"].append(float(feat_value.item()))
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = f" ce {ce.item():.4f} kd(T^2) {kd_value.item():.4f}"
            if projection is not None:
                extra += f" feat {feat_value.item():.4f}"
            elapsed = time.perf_counter() - start
            print(f"  [{tag or 'train'}] iter {it + 1:4d}/{iters} "
                  f"loss {loss.item():.4f}{extra} ({elapsed:.1f}s)", flush=True)
    return history, time.perf_counter() - start


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def make_row(model, history, elapsed, train_ids, val_ids, bs, block,
             teacher=None, **extra):
    val_loss = evaluate(model, val_ids, bs, block)
    train_loss = evaluate(model, train_ids, bs, block, seed=1235)
    row = {
        "params": model.n_param,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "ppl": math.exp(val_loss),
        "time_s": elapsed,
        "curve": history["loss"],
        "ce_curve": history["ce"],
        "kd_curve": history["kd"],
        "feature_curve": history["feature"],
        "final_train_batch_loss": history["loss"][-1],
        "final_train_batch_ce": history["ce"][-1],
    }
    if teacher is not None:
        row["agree"] = agreement(model, teacher, val_ids, bs, block)
    row.update(extra)
    return row


def print_row(name, row):
    agree = f" agree={row['agree']:.3f}" if "agree" in row else ""
    print(f"[{name}] train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
          f"ppl={row['ppl']:.2f}{agree} {row['time_s']:.1f}s", flush=True)


def new_student(vocab, seed):
    torch.manual_seed(seed + 7)
    return GPT(vocab, STUDENT_CFG)


def generate_sample(model, tok, prompt="月", seed=2025):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    generated = model.generate(ids, 60, temperature=0.9, top_k=20)
    return tok.decode(generated[0].tolist())


def load_teacher(path, vocab, cfg=TEACHER_CFG):
    teacher = GPT(vocab, cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state)
    teacher.eval()
    return teacher


# ---------------------------------------------------------------------------
# 3. 绘图
# ---------------------------------------------------------------------------
def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_training(res, path):
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for key, label, color in (("teacher", "Teacher (C=160)", "#1f77b4"),
                              ("scratch", "Student from scratch", "#ff7f0e"),
                              ("kd", "Student with KD", "#2ca02c")):
        curve = res.get(key, {}).get("ce_curve") or res.get(key, {}).get("curve")
        if curve:
            axes[0].plot(range(1, len(curve) + 1), curve,
                         label=label, linewidth=1.2, color=color)
    axes[0].set(xlabel="Iteration", ylabel="Hard-label cross-entropy",
                title="Training curves (same CE metric)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    if "scratch" in res and "kd" in res:
        keys = ["teacher", "scratch", "kd"]
        vals = [res[k]["val_loss"] for k in keys if k in res]
        labels = [k for k in keys if k in res]
        bars = axes[1].bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(vals)])
        axes[1].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axes[1].set_ylim(0, max(vals) * 1.18)
    axes[1].set(ylabel="Validation loss", title="Held-out performance")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_ablation(rows, path):
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for T in sorted({r["T"] for r in rows}):
        subset = sorted((r for r in rows if r["T"] == T), key=lambda r: r["alpha"])
        ax.plot([r["alpha"] for r in subset], [r["val_loss"] for r in subset],
                marker="o", linewidth=1.4, label=f"T={T:g}")
    best = min(rows, key=lambda r: r["val_loss"])
    ax.scatter([best["alpha"]], [best["val_loss"]], s=140, facecolors="none",
               edgecolors="red", linewidths=2, label="Best")
    ax.set(xlabel="Distillation weight alpha", ylabel="Validation loss",
           title="Temperature / weight ablation")
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows, path):
    import numpy as np
    plt = get_pyplot()
    temps = sorted({r["T"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})
    matrix = np.array([[next(r["val_loss"] for r in rows if r["T"] == T and r["alpha"] == a)
                        for a in alphas] for T in temps])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(alphas)), [f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(temps)), [f"{T:g}" for T in temps])
    ax.set(xlabel="Distillation weight alpha", ylabel="Temperature T",
           title="Validation loss: 4 x 4 KD grid")
    threshold = (matrix.min() + matrix.max()) / 2
    for i in range(len(temps)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    color="white" if matrix[i, j] < threshold else "black", fontsize=10)
    best_index = np.unravel_index(np.argmin(matrix), matrix.shape)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (best_index[1] - 0.48, best_index[0] - 0.48), 0.96, 0.96,
        fill=False, edgecolor="red", linewidth=2.5))
    fig.colorbar(image, ax=ax, label="Validation loss")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def update_combined_curves(out_dir):
    all_path = Path(out_dir) / "results_all.json"
    ablate_path = Path(out_dir) / "results_ablate.json"
    if not all_path.exists() or not ablate_path.exists():
        return
    base = load_json(all_path)
    rows = load_json(ablate_path)["ablation"]
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for key, label in (("teacher", "Teacher"), ("scratch", "Student-scratch"),
                       ("kd", "Student-KD")):
        curve = base[key].get("ce_curve") or base[key]["curve"]
        axes[0].plot(range(1, len(curve) + 1), curve, label=label, linewidth=1.2)
    axes[0].set(xlabel="Iteration", ylabel="Hard-label cross-entropy",
                title="Training loss (same CE metric)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for T in sorted({r["T"] for r in rows}):
        subset = sorted((r for r in rows if r["T"] == T), key=lambda r: r["alpha"])
        axes[1].plot([r["alpha"] for r in subset], [r["val_loss"] for r in subset],
                     marker="o", label=f"T={T:g}", linewidth=1.2)
    axes[1].set(xlabel="Alpha", ylabel="Validation loss", title="T / alpha ablation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "curves.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 主实验与拓展
# ---------------------------------------------------------------------------
def base_result(tok, train_ids, val_ids, args):
    return {
        "vocab": tok.vocab_size,
        "block": args.block_size if hasattr(args, "block_size") else TEACHER_CFG["block_size"],
        "batch": args.batch_size,
        "train_chars": int(len(train_ids)),
        "val_chars": int(len(val_ids)),
        "train_ids_sha256": tensor_sha256(train_ids),
        "val_ids_sha256": tensor_sha256(val_ids),
        "validation_protocol": "fixed 20 sampled batches; agreement uses fixed 10 batches",
        "data_isolation": "fixed POEMS and duplicate lines removed from validation; OOV=0",
        "seed": args.seed,
        "learning_rate": args.lr,
        "teacher_iters": args.teacher_iters,
        "student_iters": args.student_iters,
        "torch_version": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def train_teacher(args, tok, train_ids, val_ids, res, cfg=TEACHER_CFG, path=None, tag="teacher"):
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, cfg)
    print(f"[{tag}] params = {teacher.n_param / 1e6:.6f} M ({teacher.n_param:,})", flush=True)
    history, elapsed = train(teacher, train_ids, args.teacher_iters, args.batch_size,
                             cfg["block_size"], lr=args.lr, seed=args.seed,
                             log_every=args.log_every, tag=tag)
    row = make_row(teacher, history, elapsed, train_ids, val_ids,
                   args.batch_size, cfg["block_size"], n_layer=cfg["n_layer"],
                   n_head=cfg["n_head"], n_embd=cfg["n_embd"])
    print_row(tag, row)
    path = Path(path) if path else Path(OUT_DIR) / "teacher.pt"
    torch.save(teacher.state_dict(), path)
    res[tag] = row
    return teacher


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data()
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print(f"[data] vocab={tok.vocab_size} train_chars={len(train_ids)} "
          f"val_chars={len(val_ids)} block={block} batch={bs}", flush=True)
    print(f"[env] torch={torch.__version__} device=cpu threads={torch.get_num_threads()}", flush=True)
    res = base_result(tok, train_ids, val_ids, args)
    teacher_path = Path(OUT_DIR) / "teacher.pt"

    if args.mode == "teacher" or not teacher_path.exists() or args.retrain:
        teacher = train_teacher(args, tok, train_ids, val_ids, res)
        res["teacher_checkpoint_sha256"] = file_sha256(teacher_path)
        if args.mode == "teacher":
            save_json(res, Path(OUT_DIR) / "results_teacher.json")
            return
    else:
        teacher = load_teacher(teacher_path, tok.vocab_size)
        value = evaluate(teacher, val_ids, bs, block)
        train_value = evaluate(teacher, train_ids, bs, block, seed=1235)
        res["teacher"] = {"params": teacher.n_param, "train_loss": train_value,
                          "val_loss": value, "ppl": math.exp(value), "time_s": 0.0,
                          "curve": [], "cached": True, "n_embd": 160}
        teacher_metrics_path = Path(OUT_DIR) / "results_teacher.json"
        if teacher_metrics_path.exists():
            trained_row = load_json(teacher_metrics_path)["teacher"]
            res["teacher"]["curve"] = trained_row.get("curve", [])
            res["teacher"]["ce_curve"] = trained_row.get("ce_curve", [])
            res["teacher"]["training_time_s"] = trained_row.get("time_s", 0.0)
            res["teacher"]["final_train_batch_ce"] = trained_row.get("final_train_batch_ce")
        res["teacher_checkpoint_sha256"] = file_sha256(teacher_path)
        print(f"[teacher] reuse teacher.pt val_loss={value:.4f} ppl={math.exp(value):.2f}", flush=True)

    if args.mode == "ablate":
        run_ablation(args, tok, train_ids, val_ids, teacher, res)
        return
    if args.mode == "quantize":
        run_quantization(args, tok, train_ids, val_ids, teacher, res)
        return
    if args.mode == "self-distill":
        run_self_distillation(args, tok, train_ids, val_ids, teacher, res)
        return
    if args.mode == "scan":
        run_teacher_scan(args, tok, train_ids, val_ids, teacher, res)
        return

    # 必做：同初始化、同数据顺序的从零学生与蒸馏学生。
    probe = new_student(tok.vocab_size, args.seed)
    print(f"[student] params={probe.n_param / 1e6:.6f} M "
          f"compression={teacher.n_param / probe.n_param:.3f}x", flush=True)
    del probe

    scratch = new_student(tok.vocab_size, args.seed)
    h0, t0 = train(scratch, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, log_every=args.log_every, tag="scratch")
    res["scratch"] = make_row(scratch, h0, t0, train_ids, val_ids, bs, block,
                              teacher=teacher, T=None, alpha=0.0)
    print_row("scratch", res["scratch"])

    kd_student = new_student(tok.vocab_size, args.seed)
    h1, t1 = train(kd_student, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                   feat_weight=args.feat, log_every=args.log_every, tag="kd")
    res["kd"] = make_row(kd_student, h1, t1, train_ids, val_ids, bs, block,
                         teacher=teacher, T=args.T, alpha=args.alpha, feat=args.feat)
    print_row("kd", res["kd"])
    improvement = 100 * (res["scratch"]["val_loss"] - res["kd"]["val_loss"]) / res["scratch"]["val_loss"]
    res["improvement_pct"] = improvement
    print(f"[compare] scratch {res['scratch']['val_loss']:.4f} -> KD {res['kd']['val_loss']:.4f}; "
          f"relative change {improvement:.2f}%", flush=True)

    res["sample_teacher"] = generate_sample(teacher, tok)
    res["sample_scratch"] = generate_sample(scratch, tok)
    res["sample_kd"] = generate_sample(kd_student, tok)
    res["sample"] = res["sample_kd"]
    for name in ("teacher", "scratch", "kd"):
        print(f"[generate:{name}] {res['sample_' + name].replace(chr(10), ' ')}", flush=True)

    tag = args.run_tag or args.mode
    torch.save(scratch.state_dict(), Path(OUT_DIR) / f"student_scratch_{tag}.pt")
    torch.save(kd_student.state_dict(), Path(OUT_DIR) / f"student_kd_{tag}.pt")
    if tag == "all":
        # 兼容拓展命令默认读取的文件名。
        torch.save(scratch.state_dict(), Path(OUT_DIR) / "student_scratch.pt")
        torch.save(kd_student.state_dict(), Path(OUT_DIR) / "student_kd.pt")
    save_json(res, Path(OUT_DIR) / f"results_{tag}.json")
    plot_training(res, Path(OUT_DIR) / f"curves_{tag}.png")
    if tag == "all":
        plot_training(res, Path(OUT_DIR) / "curves.png")
    update_combined_curves(OUT_DIR)
    print(f"[done] results saved to {OUT_DIR}", flush=True)


def run_ablation(args, tok, train_ids, val_ids, teacher, res):
    """完整 4x4 网格；每组复用相同 32 批数据及相同初始化。"""
    grid_T = [1, 2, 4, 8]
    grid_alpha = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    # 同一 80 步、同一 32 批循环的硬标签基线，避免把消融格与主实验直接混比。
    scratch = new_student(V, args.seed)
    scratch_opt = torch.optim.AdamW(scratch.parameters(), lr=args.lr, weight_decay=0.01)
    scratch.train()
    start = time.perf_counter()
    for it in range(args.abl_iters):
        x, y, _ = cache[it % len(cache)]
        logits = scratch(x)[0]
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        scratch_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scratch.parameters(), 1.0)
        scratch_opt.step()
    scratch_val = evaluate(scratch, val_ids, bs, block)
    res["ablation_scratch"] = {"val_loss": scratch_val, "ppl": math.exp(scratch_val),
                               "iters": args.abl_iters, "data_seed": 2024}
    print(f"[ablate-scratch] val={scratch_val:.4f} ppl={math.exp(scratch_val):.3f}", flush=True)

    rows = []
    for T in grid_T:
        for alpha in grid_alpha:
            student = new_student(V, args.seed)
            optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
            student.train()
            history = []
            for it in range(args.abl_iters):
                x, y, teacher_logits = cache[it % len(cache)]
                logits = student(x)[0]
                loss, _, _ = distillation_loss(logits, teacher_logits, y, T, alpha)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                history.append(loss.item())
            val_loss = evaluate(student, val_ids, bs, block)
            row = {"T": T, "alpha": alpha, "val_loss": val_loss,
                   "ppl": math.exp(val_loss), "train_loss": evaluate(
                       student, train_ids, bs, block, seed=1235),
                   "final_objective": history[-1]}
            rows.append(row)
            print(f"[ablate] T={T} alpha={alpha:.1f} val={val_loss:.4f} "
                  f"ppl={row['ppl']:.3f} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    res["ablation"] = rows
    res["abl_iters"] = args.abl_iters
    res["ablation_time_s"] = time.perf_counter() - start
    res["ablation_data_seed"] = 2024
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    res["ablation_best"] = best
    res["ablation_worst"] = worst
    print(f"[ablate] best T={best['T']} alpha={best['alpha']} val={best['val_loss']:.4f}; "
          f"worst T={worst['T']} alpha={worst['alpha']} val={worst['val_loss']:.4f}", flush=True)
    save_json(res, Path(OUT_DIR) / "results_ablate.json")
    plot_ablation(rows, Path(OUT_DIR) / "ablation_curves.png")
    plot_heatmap(rows, Path(OUT_DIR) / "ablation_heatmap.png")
    update_combined_curves(OUT_DIR)


def run_self_distillation(args, tok, train_ids, val_ids, teacher, res):
    """同结构、更长训练的学生充当教师，再教一个同初始化的新学生。"""
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    self_teacher = new_student(tok.vocab_size, args.seed)
    history, elapsed = train(self_teacher, train_ids, args.self_teacher_iters, bs, block,
                             lr=args.lr, seed=args.seed, log_every=args.log_every,
                             tag="self-teacher")
    self_row = make_row(self_teacher, history, elapsed, train_ids, val_ids, bs, block,
                        n_embd=STUDENT_CFG["n_embd"], iters=args.self_teacher_iters)
    print_row("self-teacher", self_row)
    torch.save(self_teacher.state_dict(), Path(OUT_DIR) / "self_teacher.pt")

    student = new_student(tok.vocab_size, args.seed)
    h, elapsed = train(student, train_ids, args.student_iters, bs, block, lr=args.lr,
                       seed=args.seed, teacher=self_teacher, T=args.T, alpha=args.alpha,
                       log_every=args.log_every, tag="self-kd")
    kd_row = make_row(student, h, elapsed, train_ids, val_ids, bs, block,
                      teacher=self_teacher, T=args.T, alpha=args.alpha)
    print_row("self-kd", kd_row)
    baseline_path = Path(OUT_DIR) / "results_all.json"
    baseline = load_json(baseline_path) if baseline_path.exists() else None
    scratch_row = baseline["scratch"] if baseline else None
    result = dict(res)
    result.update({"self_teacher": self_row, "self_kd": kd_row,
                   "scratch_baseline": scratch_row,
                   "teacher_kind": "same student architecture, longer hard-label training"})
    if scratch_row:
        result["improvement_pct"] = 100 * (scratch_row["val_loss"] - kd_row["val_loss"]) / scratch_row["val_loss"]
    save_json(result, Path(OUT_DIR) / "results_self_distill.json")
    torch.save(student.state_dict(), Path(OUT_DIR) / "student_self_kd.pt")


def measure_latency(model, x, repeat=20, warmup=3):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(repeat):
            model(x)
        elapsed = time.perf_counter() - start
    return elapsed * 1000 / repeat


def run_quantization(args, tok, train_ids, val_ids, teacher, res):
    from torch.ao import quantization as tq
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    student = new_student(tok.vocab_size, args.seed)
    student.load_state_dict(torch.load(Path(OUT_DIR) / "student_kd.pt",
                                       map_location="cpu", weights_only=True))
    student.eval()
    fp32_loss = evaluate(student, val_ids, bs, block)
    fp32_path = Path(OUT_DIR) / "student_kd_fp32.pt"
    torch.save(student.state_dict(), fp32_path)
    quantized = tq.quantize_dynamic(copy.deepcopy(student), {nn.Linear}, dtype=torch.qint8)
    int8_path = Path(OUT_DIR) / "student_kd_int8.pt"
    torch.save(quantized.state_dict(), int8_path)
    int8_loss = evaluate(quantized, val_ids, bs, block)
    gen = torch.Generator().manual_seed(77)
    x, _ = get_batch(val_ids, 1, block, gen)
    fp32_ms = measure_latency(student, x)
    int8_ms = measure_latency(quantized, x)
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    result = dict(res)
    result["quantization"] = {
        "fp32_loss": fp32_loss, "int8_loss": int8_loss,
        "loss_change": int8_loss - fp32_loss,
        "fp32_ppl": math.exp(fp32_loss), "int8_ppl": math.exp(int8_loss),
        "fp32_bytes": fp32_size, "int8_bytes": int8_size,
        "size_ratio_fp32_over_int8": fp32_size / int8_size,
        "size_reduction_pct": 100 * (1 - int8_size / fp32_size),
        "fp32_latency_ms": fp32_ms, "int8_latency_ms": int8_ms,
        "speedup_fp32_over_int8": fp32_ms / int8_ms,
        "latency_batch": 1, "latency_block": block, "latency_repeats": 20,
        "int8_agree_with_fp32_teacher": agreement(quantized, teacher, val_ids, bs, block),
    }
    save_json(result, Path(OUT_DIR) / "results_quantize.json")
    print(json.dumps(result["quantization"], ensure_ascii=False, indent=2), flush=True)


def run_teacher_scan(args, tok, train_ids, val_ids, teacher160, res):
    """教师 C∈{96,160,256}，均训练同样步数；学生初始化及数据保持一致。"""
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    baseline_path = Path(OUT_DIR) / "results_all.json"
    baseline = load_json(baseline_path)
    scratch_loss = baseline["scratch"]["val_loss"]
    rows = []
    for C in (96, 160, 256):
        cfg = dict(TEACHER_CFG, n_embd=C)
        path = Path(OUT_DIR) / f"teacher_C{C}.pt"
        if C == 160:
            teacher = teacher160
            teacher_row = dict(baseline["teacher"])
            # 训练耗时以单独 teacher 运行记录为准。
            teacher_metrics_path = Path(OUT_DIR) / "results_teacher.json"
            if teacher_metrics_path.exists():
                teacher_row = load_json(teacher_metrics_path)["teacher"]
        elif path.exists() and not args.retrain:
            teacher = load_teacher(path, tok.vocab_size, cfg)
            value = evaluate(teacher, val_ids, bs, block)
            teacher_row = {"params": teacher.n_param, "val_loss": value,
                           "ppl": math.exp(value), "cached": True}
        else:
            temp = dict(res)
            teacher = train_teacher(args, tok, train_ids, val_ids, temp,
                                    cfg=cfg, path=path, tag=f"teacher_C{C}")
            teacher_row = temp[f"teacher_C{C}"]
        student = new_student(tok.vocab_size, args.seed)
        history, elapsed = train(student, train_ids, args.student_iters, bs, block,
                                 lr=args.lr, seed=args.seed, teacher=teacher,
                                 T=args.T, alpha=args.alpha, log_every=args.log_every,
                                 tag=f"kd_C{C}")
        student_row = make_row(student, history, elapsed, train_ids, val_ids,
                               bs, block, teacher=teacher, T=args.T, alpha=args.alpha)
        row = {"n_embd": C, "teacher": teacher_row, "student_kd": student_row,
               "scratch_val_loss": scratch_loss,
               "improvement_pct": 100 * (scratch_loss - student_row["val_loss"]) / scratch_loss}
        rows.append(row)
        print(f"[scan] C={C} teacher_val={teacher_row['val_loss']:.4f} "
              f"student_val={student_row['val_loss']:.4f} gain={row['improvement_pct']:.2f}%", flush=True)
        save_json(dict(res, teacher_scan=rows), Path(OUT_DIR) / "results_teacher_scan.json")
    result = dict(res, teacher_scan=rows)
    save_json(result, Path(OUT_DIR) / "results_teacher_scan.json")
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    Cs = [r["n_embd"] for r in rows]
    ax.plot(Cs, [r["teacher"]["val_loss"] for r in rows], marker="o", label="Teacher")
    ax.plot(Cs, [r["student_kd"]["val_loss"] for r in rows], marker="o", label="Student KD")
    ax.axhline(scratch_loss, color="gray", linestyle="--", label="Student scratch")
    ax.set(xlabel="Teacher embedding dimension C", ylabel="Validation loss",
           title="Teacher scale scan")
    ax.set_xticks(Cs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(OUT_DIR) / "teacher_scale_scan.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["quick", "teacher", "all", "ablate", "self-distill",
                                 "quantize", "scan"])
    parser.add_argument("--teacher-iters", type=int, default=200)
    parser.add_argument("--student-iters", type=int, default=120)
    parser.add_argument("--abl-iters", type=int, default=80)
    parser.add_argument("--self-teacher-iters", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--feat", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--run-tag", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "quick":
        global OUT_DIR
        OUT_DIR = OUT_DIR / "quick"
        args.teacher_iters, args.student_iters, args.abl_iters = 30, 20, 10
        args.batch_size, args.log_every = 8, 10
        args.mode = "all"
        args.run_tag = "quick"
    run(args)


if __name__ == "__main__":
    main()
