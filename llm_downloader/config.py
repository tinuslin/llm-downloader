"""LLM Downloader 配置。

只保留「模型下载 / 推荐 / 本地模型库」三块功能需要的配置，去掉 harness 里的
provider / runtime / imagegen / video / safety 等无关模块。

配置分两层：
- **默认值**：写在 DEFAULT_SETTINGS 里（源顺序、镜像端点、目录、分层加载策略）。
- **用户设置**：持久化在 `$LLM_DOWNLOADER_HOME/settings.json`，由 Web「设置」页或
  `save_settings()` 写入；load_config() 每次读取，改动即时生效，无需重启。

home 目录约定：$LLM_DOWNLOADER_HOME > ~/.llm-downloader
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import ensure_dir, expand, read_json, write_json


APP_NAME = "llm-downloader"
VERSION = "0.1.0"

KNOWN_SOURCES = ("hf-mirror", "modelscope", "ollama")

DEFAULT_SETTINGS: dict[str, Any] = {
    # 空串 = 使用 home 下的默认目录
    "models_dir": "",
    "image_models_dir": "",
    "sources": {
        "order": ["hf-mirror", "ollama", "modelscope"],
        "hf_endpoint": "https://hf-mirror.com",
        "modelscope_endpoint": "https://www.modelscope.cn",
        "ollama_registry": "https://registry.ollama.ai",
    },
    # 分层加载（GPU offload）默认策略：auto = 按显存自动算最大可卸载层数
    "offload": {"mode": "auto", "layers": 0},
}

OFFLOAD_MODES = ("auto", "cpu", "all", "custom")


@dataclass
class SourceConfig:
    """多源下载配置：按顺序自动降级 + 端点地址。"""

    # 解析仓库时按此顺序尝试，第一个能列出文件清单的源即采用
    order: list[str] = field(default_factory=lambda: list(DEFAULT_SETTINGS["sources"]["order"]))
    hf_endpoint: str = "https://hf-mirror.com"
    modelscope_endpoint: str = "https://www.modelscope.cn"
    ollama_registry: str = "https://registry.ollama.ai"
    # 下载超时（秒）
    connect_timeout: float = 15.0
    read_timeout: float = 120.0
    # 源可用性探测结果缓存秒数
    probe_ttl: int = 900
    # 多候选时先并行测速再择优下载
    probe_speed: bool = True
    # 测速取样字节数
    probe_bytes: int = 384 * 1024

    def endpoint_of(self, source: str) -> str:
        return {
            "hf-mirror": self.hf_endpoint,
            "modelscope": self.modelscope_endpoint,
            "ollama": self.ollama_registry,
        }.get(source, "")


def _clean_dir(v: Any) -> str:
    s = str(v or "").strip().strip('"').strip("'")
    if s:
        s = str(expand(s))
    return s


def _clean_endpoint(v: Any, fallback: str) -> str:
    s = str(v or "").strip().rstrip("/")
    if not s:
        return fallback
    if not (s.startswith("http://") or s.startswith("https://")):
        raise ValueError(f"镜像地址必须以 http:// 或 https:// 开头：{s}")
    if len(s) > 300:
        raise ValueError("镜像地址过长")
    return s


def _clean_order(v: Any, fallback: list[str]) -> list[str]:
    if not isinstance(v, list):
        return list(fallback)
    out: list[str] = []
    for x in v:
        name = str(x).strip()
        if name in KNOWN_SOURCES and name not in out:
            out.append(name)
    if not out:
        raise ValueError("至少要启用一个下载源")
    return out


def normalize_settings(patch: dict[str, Any] | None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """把「部分更新」合并进现有设置，并做校验；返回完整设置字典。

    patch 只需带要改的字段（例如 {"models_dir": "D:\\\\models"}），
    其它字段沿用 base / 默认值。
    """
    out = copy.deepcopy(base or DEFAULT_SETTINGS)
    patch = patch or {}
    if not isinstance(patch, dict):
        raise ValueError("设置必须是对象")

    for k in ("models_dir", "image_models_dir"):
        if k in patch:
            out[k] = _clean_dir(patch[k])

    if "offload" in patch and isinstance(patch["offload"], dict):
        o = dict(out.get("offload") or {})
        if "mode" in patch["offload"]:
            mode = str(patch["offload"]["mode"]).strip()
            if mode not in OFFLOAD_MODES:
                raise ValueError(f"未知的分层加载模式：{mode}")
            o["mode"] = mode
        if "layers" in patch["offload"]:
            try:
                n = int(patch["offload"]["layers"])
            except (TypeError, ValueError) as e:
                raise ValueError("层数必须是 0-128 的整数") from e
            o["layers"] = max(0, min(128, n))
        out["offload"] = o

    if "sources" in patch and isinstance(patch["sources"], dict):
        sp = patch["sources"]
        src = dict(out.get("sources") or {})
        for k in ("hf_endpoint", "modelscope_endpoint", "ollama_registry"):
            if k in sp:
                src[k] = _clean_endpoint(sp[k], DEFAULT_SETTINGS["sources"][k])
        if "order" in sp:
            src["order"] = _clean_order(sp["order"], src.get("order", []))
        out["sources"] = src

    # 兜底：从磁盘读到的旧配置可能缺字段
    out.setdefault("sources", {})
    for k, v in DEFAULT_SETTINGS["sources"].items():
        out["sources"].setdefault(k, copy.deepcopy(v))
    out.setdefault("offload", copy.deepcopy(DEFAULT_SETTINGS["offload"]))
    return out


@dataclass
class Config:
    home: Path = field(
        default_factory=lambda: expand(
            os.environ.get("LLM_DOWNLOADER_HOME", f"~/.{APP_NAME}")
        )
    )
    sources: SourceConfig = field(default_factory=SourceConfig)
    settings: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SETTINGS))

    # ---- 用户设置文件 ----
    @property
    def settings_file(self) -> Path:
        return self.home / "settings.json"

    def apply_settings(self, data: dict[str, Any]) -> None:
        s = normalize_settings(data)
        self.settings = s
        src = s["sources"]
        self.sources.order = list(src["order"])
        self.sources.hf_endpoint = src["hf_endpoint"]
        self.sources.modelscope_endpoint = src["modelscope_endpoint"]
        self.sources.ollama_registry = src["ollama_registry"]

    def to_settings(self) -> dict[str, Any]:
        return copy.deepcopy(self.settings)

    @property
    def offload(self) -> dict[str, Any]:
        return dict(self.settings.get("offload") or DEFAULT_SETTINGS["offload"])

    # ---- 目录（全部惰性创建）----
    @property
    def models_dir(self) -> Path:
        raw = str(self.settings.get("models_dir") or "").strip()
        return ensure_dir(expand(raw) if raw else self.home / "models")

    @property
    def image_models_dir(self) -> Path:
        """扩散模型（SD / FLUX）目录，不进本地模型库索引。"""
        raw = str(self.settings.get("image_models_dir") or "").strip()
        return ensure_dir(expand(raw) if raw else self.home / "image-models")

    @property
    def bin_dir(self) -> Path:
        return ensure_dir(self.home / "bin")

    @property
    def cache_dir(self) -> Path:
        return ensure_dir(self.home / "cache")

    @property
    def downloads_dir(self) -> Path:
        return ensure_dir(self.cache_dir / "downloads")

    @property
    def logs_dir(self) -> Path:
        return ensure_dir(self.home / "logs")

    @property
    def index_file(self) -> Path:
        return self.models_dir / "index.json"

    @property
    def probe_cache_file(self) -> Path:
        return self.cache_dir / "source_probe.json"


def load_config() -> Config:
    """读取用户设置并返回生效配置（每次调用都重新读，改动即时生效）。

    settings.json 损坏时静默回退默认值，不让整个应用起不来。
    """
    cfg = Config()
    data = read_json(cfg.settings_file, {})
    if isinstance(data, dict) and data:
        try:
            cfg.apply_settings(data)
        except (ValueError, KeyError, TypeError):
            pass
    return cfg


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """合并写入用户设置，返回写入后的完整设置。"""
    cfg = load_config()
    merged = normalize_settings(patch, cfg.to_settings())
    write_json(cfg.settings_file, merged)
    return merged


def reset_settings() -> dict[str, Any]:
    cfg = Config()
    default = copy.deepcopy(DEFAULT_SETTINGS)
    write_json(cfg.settings_file, default)
    return default
