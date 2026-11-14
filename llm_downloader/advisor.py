"""硬件自适应的模型推荐。

为什么需要这一层：下载器能多源下载，但“该下哪个”才是真正的决策成本。
面对镜像站上一个仓库里几十个量化变体（q2_k / q4_k_m / q5_k_m / q8_0 …），
用户得自己算显存、估内存、判断要不要 mmproj —— 一次选错就是几个 GB 白下。

这里做两件事：

1. **精选目录**（CATALOG）：按用途分类的少数几个“确实好用”的模型，
   每个都带确定的仓库引用与量化档，不让人在镜像站里盲翻。
2. **适配打分**：按本机**可用**内存 + 显存实测算出预算，给每条打
   good / tight / too_big 三档，并给出人话理由；已下载过的直接标“已装”。

另外每条候选带 refs（多个等价仓库），下载时按顺序回落到第一个能解析出
文件清单的仓库 —— 国内镜像站的同名仓库经常缺文件或改名，写死单一引用迟早踩空。
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from llm_downloader.util import (
    cpu_count,
    decode_bytes,
    default_threads,
    has_nvidia_gpu,
    human_size,
    total_ram_gb,
)


# 用途分类（顺序即界面展示顺序）
ROLES: dict[str, str] = {
    "tools": "工具调用",
    "chat": "通用对话",
    "coding": "代码",
    "vision": "看图（多模态）",
    "image": "文生图",
}

FIT_LABEL = {"good": "推荐", "tight": "可用（勉强）", "too_big": "别下（跑不动）"}


# ---------------------------------------------------------------- 硬件画像


def gpu_vram_gb() -> float | None:
    """NVIDIA 单卡显存容量（GB）。没有 nvidia-smi 时返回 None。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=8,
        )
        first = decode_bytes(r.stdout).strip().splitlines()
        if not first:
            return None
        return round(float(first[0].strip()) / 1024.0, 1)
    except Exception:  # noqa: BLE001
        return None


def available_ram_gb() -> float | None:
    """当前**可用**物理内存（GB）。

    用 total 会严重高估：模型加载时真正能拿到的只有 avail。Windows 走
    GlobalMemoryStatusEx，Linux 读 /proc/meminfo，其余退回 total。
    """
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MSX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MSX()
            st.dwLength = ctypes.sizeof(_MSX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return round(st.ullAvailPhys / (1024**3), 1)
        else:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / (1024**2), 1)
    except Exception:  # noqa: BLE001
        pass
    t = total_ram_gb()
    return round(t * 0.6, 1) if t else None


@dataclass
class Hardware:
    os: str = ""
    cpu: int = 0
    threads: int = 0
    ram_total_gb: float | None = None
    ram_avail_gb: float | None = None
    gpu: bool = False
    vram_gb: float | None = None

    def budget_gb(self) -> float:
        """能安全喂给单个模型的规模预算（GB）。

        扣掉系统与常驻占用，再加上可卸载到显存的部分 ——
        有 GPU 时显存是“额外额度”，这是本机 8GB 卡能跑 7B 的关键。
        """
        base = self.ram_avail_gb or (self.ram_total_gb or 8.0) * 0.6 or 8.0
        reserve = max(2.0, base * 0.2)
        usable = max(1.0, base - reserve)
        if self.gpu and self.vram_gb:
            usable += self.vram_gb * 0.8
        return round(usable, 1)

    def summary(self) -> str:
        bits = [f"{self.cpu} 核 / {self.threads} 线程"]
        if self.ram_total_gb:
            avail = f"（可用 {self.ram_avail_gb}GB）" if self.ram_avail_gb else ""
            bits.append(f"内存 {self.ram_total_gb:.1f}GB{avail}")
        if self.gpu:
            bits.append(f"GPU 显存 {self.vram_gb or '?'}GB")
        else:
            bits.append("无 NVIDIA GPU（纯 CPU）")
        bits.append(f"预算约 {self.budget_gb()}GB")
        return " · ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["budget_gb"] = self.budget_gb()
        d["summary"] = self.summary()
        return d


def detect_hardware() -> Hardware:
    return Hardware(
        os=f"{platform.system()} {platform.release()}",
        cpu=cpu_count(),
        threads=default_threads(),
        ram_total_gb=total_ram_gb(),
        ram_avail_gb=available_ram_gb(),
        gpu=has_nvidia_gpu(),
        vram_gb=gpu_vram_gb(),
    )


# ---------------------------------------------------------------- 精选目录


@dataclass
class Candidate:
    key: str
    title: str
    role: str
    refs: list[str]
    quant: str = ""
    file: str = ""
    size_gb: float = 0.0
    """磁盘占用（GB），含 mmproj。"""
    ram_gb: float = 0.0
    """跑起来大致需要的可用内存+显存（GB）。"""
    mmproj: bool = False
    layers: int = 0
    """Transformer 层数（用于分层加载估算，0 = 不适用/未知）。"""
    dest: str = "models"
    """models = 语言模型库；image-models = 扩散模型目录（不登记进模型库）。
    image-models/<子目录> 表示落到扩散模型目录下的子目录（组件成套的模型用）。"""
    why: str = ""
    tags: list[str] = field(default_factory=list)
    items: list[dict[str, str]] = field(default_factory=list)
    """多文件成套下载（跨仓库）：[{"ref":..., "file":...}, ...]。"""

    def to_spec(self) -> dict[str, Any]:
        """下发给下载引擎的参数。"""
        return {
            "refs": self.refs,
            "quant": self.quant or None,
            "file": self.file or None,
            "mmproj": self.mmproj,
            "dest": self.dest,
            "register": self.dest == "models",
            "items": self.items or None,
        }


def _c(key: str, title: str, role: str, refs: Iterable[str], **kw: Any) -> Candidate:
    return Candidate(key=key, title=title, role=role, refs=list(refs), **kw)


CATALOG: list[Candidate] = [
    # ---------------- 工具调用
    _c(
        "qwen25-05b-tools",
        "Qwen2.5-0.5B-Instruct",
        "tools",
        ["Qwen/Qwen2.5-0.5B-Instruct-GGUF", "bartowski/Qwen2.5-0.5B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=0.40,
        ram_gb=1.0,
        why="最小可跑通的多轮工具调用模型。链路自检首选。",
        tags=["0.5B", "极省资源"],
    ),
    _c(
        "qwen25-15b-tools",
        "Qwen2.5-1.5B-Instruct",
        "tools",
        ["Qwen/Qwen2.5-1.5B-Instruct-GGUF", "bartowski/Qwen2.5-1.5B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=1.05,
        ram_gb=2.0,
        why="日常工具调用的性价比甜点：JSON 参数基本不出错，CPU 也能秒回。",
        tags=["1.5B", "推荐起步"],
    ),
    _c(
        "qwen25-3b-tools",
        "Qwen2.5-3B-Instruct",
        "tools",
        ["Qwen/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=2.00,
        ram_gb=3.2,
        why="3B 档指令遵循明显更稳，多步任务不容易跑偏。",
        tags=["3B", "多步任务"],
    ),
    _c(
        "llama3-groq-8b-tools",
        "Llama-3-Groq-8B-Tool-Use",
        "tools",
        ["bartowski/Llama-3-Groq-8B-Tool-Use-GGUF"],
        quant="q4_k_m",
        size_gb=4.90,
        ram_gb=6.6,
        why="专为函数调用微调（Berkeley Function-Calling 榜前列），工具名/参数几乎不臆造；代价是 8B 体量。",
        tags=["8B", "专精工具调用"],
    ),
    # ---------------- 通用对话
    _c(
        "qwen25-7b-chat",
        "Qwen2.5-7B-Instruct",
        "chat",
        ["Qwen/Qwen2.5-7B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=4.70,
        ram_gb=6.5,
        why="中文问答与长文整理的主力档位。",
        tags=["7B", "中文强"],
    ),
    _c(
        "qwen35-9b-chat",
        "Qwen3.5-9B-Instruct",
        "chat",
        ["unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"],
        quant="q4_k_m",
        size_gb=5.60,
        ram_gb=7.6,
        why="新一代 9B 档：推理跳步更少、长上下文更稳，适合当日常主力。",
        tags=["9B", "新一代"],
    ),
    _c(
        "qwen25-14b-chat",
        "Qwen2.5-14B-Instruct",
        "chat",
        ["Qwen/Qwen2.5-14B-Instruct-GGUF", "bartowski/Qwen2.5-14B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=9.00,
        ram_gb=12.5,
        why="质量接近云端小模型，但需要大内存/大显存；本机为独显笔记本时通常偏紧。",
        tags=["14B", "吃资源"],
    ),
    # ---------------- 代码
    _c(
        "qwen25-coder-7b",
        "Qwen2.5-Coder-7B-Instruct",
        "coding",
        ["Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=4.70,
        ram_gb=6.5,
        why="写代码/改文件/写 patch 比同尺寸通用模型稳。",
        tags=["7B", "代码"],
    ),
    # ---------------- 看图（多模态输入）
    _c(
        "qwen25-vl-3b",
        "Qwen2.5-VL-3B-Instruct",
        "vision",
        ["ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", "Qwen/Qwen2.5-VL-3B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=2.90,
        ram_gb=4.5,
        mmproj=True,
        why="视觉链路已验证可用（截图/图表识别全对）。下载会自动带上 mmproj 投影层。",
        tags=["3B", "含 mmproj"],
    ),
    _c(
        "qwen25-vl-7b",
        "Qwen2.5-VL-7B-Instruct",
        "vision",
        ["ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen/Qwen2.5-VL-7B-Instruct-GGUF"],
        quant="q4_k_m",
        size_gb=6.00,
        ram_gb=8.5,
        mmproj=True,
        why="细节识别（表格、K 线图注、模糊截图）比 3B 明显好，代价是显存吃满。",
        tags=["7B", "含 mmproj"],
    ),
    # ---------------- 文生图（走 sd.cpp，落在 image-models）
    _c(
        "sd15-q8",
        "Stable Diffusion 1.5（Q8_0）",
        "image",
        ["second-state/stable-diffusion-v1-5-GGUF", "green-sky/stable-diffusion-v1-5-GGUF"],
        quant="q8_0",
        file="stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
        size_gb=1.68,
        ram_gb=2.2,
        dest="image-models",
        why="512×512 原生分辨率，无需额外 VAE/CLIP 组件。",
        tags=["SD1.5", "免组件"],
    ),
    _c(
        "sdxl-base-q8",
        "Stable Diffusion XL 1.0（Q8_0）",
        "image",
        ["gpustack/stable-diffusion-xl-base-1.0-GGUF", "second-state/stable-diffusion-xl-base-1.0-GGUF"],
        quant="q8_0",
        file="stable-diffusion-xl-base-1.0-Q8_0.gguf",
        size_gb=4.70,
        ram_gb=6.2,
        dest="image-models",
        why="相对 SD1.5 的画质跃升档：原生 1024×1024，仍是单文件（自带 VAE 与文本编码器）。",
        tags=["SDXL", "1024px", "免组件"],
    ),
    _c(
        "flux1-schnell-q3",
        "FLUX.1-schnell（Q3_K_S 全套）",
        "image",
        ["city96/FLUX.1-schnell-gguf"],
        quant="q3_k_s",
        file="flux1-schnell-Q3_K_S.gguf",
        size_gb=8.5,
        ram_gb=9.8,
        dest="image-models/flux1-schnell",
        why="8GB 显存能装下的 FLUX 档位：主模型进显存、文本编码器走内存，4 步出图。",
        tags=["FLUX", "画质最佳", "4 步", "8GB 可跑"],
        items=[
            {"ref": "city96/FLUX.1-schnell-gguf", "file": "flux1-schnell-Q3_K_S.gguf"},
            {"ref": "city96/t5-v1_1-xxl-encoder-gguf", "file": "t5-v1_1-xxl-encoder-Q5_K_S.gguf"},
            {"ref": "comfyanonymous/flux_text_encoders", "file": "clip_l.safetensors"},
            {"ref": "calcuis/flux1-gguf", "file": "ae.safetensors"},
        ],
    ),
    _c(
        "flux1-schnell-q4",
        "FLUX.1-schnell（Q4_K_S 全套）",
        "image",
        ["city96/FLUX.1-schnell-gguf"],
        size_gb=9.95,
        ram_gb=11.5,
        dest="image-models/flux1-schnell",
        why="画质比 Q3 略好；但主模型在 8GB 显存装不下，只能权重留内存流式出图（实测慢一个数量级），显存 ≥12GB 再选它。",
        tags=["FLUX", "高精度", "需 12GB+ 显存"],
        items=[
            {"ref": "city96/FLUX.1-schnell-gguf", "file": "flux1-schnell-Q4_K_S.gguf"},
            {"ref": "city96/t5-v1_1-xxl-encoder-gguf", "file": "t5-v1_1-xxl-encoder-Q5_K_S.gguf"},
            {"ref": "comfyanonymous/flux_text_encoders", "file": "clip_l.safetensors"},
            {"ref": "calcuis/flux1-gguf", "file": "ae.safetensors"},
        ],
    ),
]

ROLE_ORDER = list(ROLES)

# 常见架构的 transformer 层数。GGUF 尚未下载时无法读取真实元数据，
# 这里按公开模型配置给估算值，界面会明确标注「估算」。
_LAYER_MAP: dict[str, int] = {
    "qwen25-05b-tools": 24,
    "qwen25-15b-tools": 28,
    "qwen25-3b-tools": 36,
    "llama3-groq-8b-tools": 32,
    "qwen25-7b-chat": 28,
    "qwen35-9b-chat": 32,
    "qwen25-14b-chat": 48,
    "qwen25-coder-7b": 28,
    "qwen25-vl-3b": 36,
    "qwen25-vl-7b": 28,
}
for _cand in CATALOG:
    _cand.layers = _LAYER_MAP.get(_cand.key, 0)


# ---------------------------------------------------------------- 分层加载（GPU offload）


LAYER_OVERHEAD_GB = 0.6   # KV cache / 计算缓冲 / 运行时固定开销
GPU_RESERVE = 0.15        # 显存给上下文与临时缓冲留的比例
RAM_RESERVE = 0.20        # 内存给系统留的比例
WEIGHT_GPU_FACTOR = 1.05  # 同一份权重进显存时略高于磁盘/内存占用

LAYER_VERDICT_LABEL = {"ok": "可分层加载", "tight": "临界", "no": "装不下"}


@dataclass
class OffloadConfig:
    """分层加载策略。对应 llama.cpp 的 --n-gpu-layers / -ngl。"""

    mode: str = "auto"  # auto | cpu | all | custom
    layers: int = 0     # 仅 custom 模式使用

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "OffloadConfig":
        d = d or {}
        mode = str(d.get("mode") or "auto").strip()
        if mode not in ("auto", "cpu", "all", "custom"):
            mode = "auto"
        try:
            layers = max(0, min(128, int(d.get("layers") or 0)))
        except (TypeError, ValueError):
            layers = 0
        return cls(mode=mode, layers=layers)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "layers": self.layers}


def layered_plan(c: Candidate, hw: Hardware, off: OffloadConfig) -> dict[str, Any]:
    """估算按指定层数卸载到 GPU 后的显存/内存分配是否可行。

    线性分摊只是工程近似：真实占用还受上下文长度、量化类型、KV cache 精度影响，
    层数也是估算值。结论用于「选档」，不能替代实际启动后的 nvidia-smi 观测。
    """
    total = c.layers
    if total <= 0:
        return {
            "applicable": False,
            "note": "扩散模型 / 非 GGUF 权重，不适用分层加载",
        }

    need = c.ram_gb or c.size_gb * 1.4
    weight = max(0.1, need - LAYER_OVERHEAD_GB)
    per_layer = weight / total * WEIGHT_GPU_FACTOR
    vram = (hw.vram_gb or 0.0) if hw.gpu else 0.0
    vram_budget = vram * (1 - GPU_RESERVE)
    ram_total = hw.ram_avail_gb or hw.ram_total_gb or 8.0
    ram_budget = ram_total * (1 - RAM_RESERVE)

    max_layers = int(vram_budget / per_layer) if per_layer > 0 else 0
    max_layers = max(0, min(total, max_layers))

    if off.mode == "cpu":
        chosen = 0
    elif off.mode == "all":
        chosen = total
    elif off.mode == "custom":
        chosen = min(off.layers, total)
    else:  # auto
        chosen = max_layers

    gpu_gb = chosen * per_layer
    cpu_gb = LAYER_OVERHEAD_GB + (total - chosen) * (weight / total) * WEIGHT_GPU_FACTOR

    gd = gpu_gb / vram_budget if vram_budget > 0 else (0.0 if gpu_gb == 0 else 9.9)
    rd = cpu_gb / ram_budget if ram_budget > 0 else 9.9
    worst = max(gd, rd)
    if worst <= 0.8:
        verdict, note = "ok", "显存与内存均有余量"
    elif worst <= 1.0:
        verdict, note = "tight", "接近显存/内存上限，长上下文可能 OOM"
    else:
        verdict, note = "no", "超出显存或内存预算，建议减少层数或换更小量化"

    return {
        "applicable": True,
        "mode": off.mode,
        "layers": chosen,
        "ngl": chosen,
        "total_layers": total,
        "max_layers": max_layers,
        "per_layer_gb": round(per_layer, 3),
        "gpu_gb": round(gpu_gb, 2),
        "cpu_gb": round(cpu_gb, 2),
        "vram_budget_gb": round(vram_budget, 2),
        "ram_budget_gb": round(ram_budget, 2),
        "verdict": verdict,
        "verdict_label": LAYER_VERDICT_LABEL.get(verdict, verdict),
        "note": note,
    }


# ---------------------------------------------------------------- 打分


def _fit_of(c: Candidate, hw: Hardware) -> tuple[str, str]:
    """返回 (档位, 理由)。"""
    need = c.ram_gb or c.size_gb * 1.4
    budget = hw.budget_gb()
    if need <= budget * 0.55:
        return "good", f"需约 {need:.1f}GB，预算 {budget}GB，余量充足"
    if need <= budget:
        return "tight", f"需约 {need:.1f}GB，接近预算 {budget}GB，可能要靠磁盘交换变慢"
    over = need / budget if budget else 9.9
    if hw.gpu and not hw.vram_gb:
        return "too_big", f"需约 {need:.1f}GB，超预算 {budget}GB；建议改用更小的量化或云端 API"
    return "too_big", f"需约 {need:.1f}GB，超预算 {budget}GB（约 {over:.1f} 倍）"


def _installed_keys(entries: Iterable[Any]) -> tuple[set[str], set[str]]:
    """从模型库条目里推断「已装」的候选 key 与仓库名。"""
    repos: set[str] = set()
    names: set[str] = set()
    for e in entries or []:
        repo = (getattr(e, "repo", "") or "").lower()
        if repo:
            repos.add(repo)
        names.add((getattr(e, "name", "") or "").lower())
        names.add((getattr(e, "alias", "") or "").lower())
    keys: set[str] = set()
    for c in CATALOG:
        for r in c.refs:
            if r.lower() in repos:
                keys.add(c.key)
                break
    return keys, names


def _looks_installed(c: Candidate, keys: set[str], names: set[str], image_names: set[str] = frozenset()) -> bool:
    if c.key in keys:
        return True
    # 文件名层面兜底：用户手动放进目录的也算
    probe = (c.file or "").lower()
    if probe and probe in names:
        return True
    if c.dest.startswith("image-models"):
        # 扩散模型不进模型库，只能按 image-models 目录里的文件名判断
        if c.items:
            # 成套下载：主模型在就算「已装」（组件可以后续补齐，别整条都藏起来）
            main = Path(c.items[0]["file"]).name.lower()
            return any(main == n or main in n for n in image_names)
        if probe and probe in image_names:
            return True
        q = (c.quant or "").lower()
        if q and any(q in n and "diffusion" in n for n in image_names):
            return True
    stem = (c.title or "").lower().replace(" ", "-")
    return any(stem and stem in n for n in names)


def recommend(
    hw: Hardware | None = None,
    role: str | None = None,
    entries: Iterable[Any] | None = None,
    image_files: Iterable[str] | None = None,
    limit: int | None = None,
    include_unfit: bool = True,
    offload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """给出推荐清单。

    entries     传模型库条目（标记语言模型「已装」）
    image_files 传 image-models 目录下的文件名（扩散模型不进模型库，单独传）
    """
    hw = hw or detect_hardware()
    off = OffloadConfig.from_dict(offload)
    keys, names = _installed_keys(entries or [])
    image_names = {(n or "").lower() for n in (image_files or [])}
    rows: list[dict[str, Any]] = []
    for c in CATALOG:
        if role and c.role != role:
            continue
        fit, note = _fit_of(c, hw)
        if not include_unfit and fit == "too_big":
            continue
        rows.append(
            {
                "key": c.key,
                "title": c.title,
                "role": c.role,
                "role_label": ROLES.get(c.role, c.role),
                "quant": c.quant,
                "file": c.file,
                "ref": c.refs[0] if c.refs else "",
                "refs": c.refs,
                "size_gb": c.size_gb,
                "size_h": f"{c.size_gb:.2f} GB",
                "ram_gb": c.ram_gb,
                "mmproj": c.mmproj,
                "dest": c.dest,
                "why": c.why,
                "tags": c.tags,
                "fit": fit,
                "fit_label": FIT_LABEL.get(fit, fit),
                "fit_note": note,
                "installed": _looks_installed(c, keys, names, image_names),
                "layered": layered_plan(c, hw, off),
                "spec": c.to_spec(),
            }
        )
    order = {r: i for i, r in enumerate(ROLE_ORDER)}
    rank = {"good": 0, "tight": 1, "too_big": 2}
    rows.sort(key=lambda r: (order.get(r["role"], 9), rank.get(r["fit"], 9), r["size_gb"]))
    if limit:
        rows = rows[:limit]
    return {
        "hardware": hw.to_dict(),
        "roles": [{"key": k, "label": v} for k, v in ROLES.items()],
        "recommendations": rows,
        "installed": sorted(keys),
        "offload": {
            **off.to_dict(),
            "modes": [
                {"key": "auto", "label": "自动（按显存取最大可卸载层数）"},
                {"key": "cpu", "label": "纯 CPU（0 层）"},
                {"key": "all", "label": "全部进 GPU"},
                {"key": "custom", "label": "自定义层数"},
            ],
            "estimated_layers": True,
            "note": "层数为公开架构的估算值，实际以 GGUF 元数据为准；显存/内存占用为线性近似，"
                    "未计入长上下文 KV cache 的额外开销。",
        },
    }


def best_for(role: str, hw: Hardware | None = None, entries: Iterable[Any] | None = None) -> dict[str, Any] | None:
    """取某用途下适配最好的一条（供 CLI / 引导提示直接用）。"""
    res = recommend(hw=hw, role=role, entries=entries)
    rows = [r for r in res["recommendations"] if r["fit"] != "too_big"]
    return rows[0] if rows else None
