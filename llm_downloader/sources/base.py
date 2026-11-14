"""模型源抽象：统一“仓库 -> 文件 -> 可下载候选 URL”的解析。

支持的引用语法（下载时传给 ref）：
    Qwen/Qwen2.5-1.5B-Instruct-GGUF        # 默认走 hf-mirror
    hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF     # 显式 hf-mirror
    ms:Qwen/Qwen2.5-7B-Instruct-GGUF       # ModelScope
    ollama:qwen2.5:1.5b                    # Ollama registry

本文件只定义抽象基类、注册表与探测缓存，不依赖任何具体镜像实现。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from llm_downloader.http import Client, Timeout
from llm_downloader.config import Config
from llm_downloader.util import read_json, write_json


@dataclass
class RemoteFile:
    filename: str
    size: int | None = None
    sha256: str | None = None
    source: str = ""
    # 该文件在源里的角色：weights | mmproj | config | template | other
    role: str = "other"
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        from llm_downloader.util import human_size

        return f"{self.filename} ({human_size(self.size)})"


@dataclass
class Candidate:
    url: str
    source: str
    filename: str = ""
    size: int | None = None
    sha256: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class ModelRef:
    source: str  # "" 表示 auto
    repo: str
    revision: str = "main"
    filename: str = ""

    def __str__(self) -> str:
        base = f"{self.source}:{self.repo}" if self.source else self.repo
        return f"{base}@{self.revision}" + (f"/{self.filename}" if self.filename else "")


def parse_ref(text: str, default_source: str = "") -> ModelRef:
    """解析模型引用字符串。"""
    text = (text or "").strip()
    src = default_source
    if ":" in text:
        head, tail = text.split(":", 1)
        if head in ("hf", "hf-mirror", "ms", "modelscope", "ollama", "auto", "github"):
            src = "" if head == "auto" else ("hf-mirror" if head == "hf" else head)
            text = tail
    revision = "main"
    if "@" in text:
        text, revision = text.rsplit("@", 1)
        src = src or ""
    filename = ""
    # 区分 ollama 的 name:tag 与 HF 的 owner/name/file.gguf
    if src == "ollama":
        repo, _, tag = text.partition(":")
        if tag:
            revision = tag
        repo = repo if "/" in repo else f"library/{repo}"
        return ModelRef(source="ollama", repo=repo, revision=revision)

    if text.count("/") >= 2:
        text, filename = text.rsplit("/", 1)
    if ":" in text and "/" not in text:
        text, _, tag = text.partition(":")
        revision = tag
    return ModelRef(source=src, repo=text, revision=revision, filename=filename)


class Source(ABC):
    name: str = "base"
    kind: str = "model"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._http: Client | None = None

    # ---- http
    @property
    def http(self) -> Client:
        if self._http is None:
            s = self.cfg.sources
            self._http = Client(
                timeout=Timeout(
                    connect=s.connect_timeout, read=s.read_timeout, write=60.0, pool=30.0
                ),
                follow_redirects=True,
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 接口
    @abstractmethod
    def probe(self) -> tuple[bool, str]:
        """探测该源是否可用。返回 (可用, 说明)。"""

    @abstractmethod
    def list_files(self, ref: ModelRef) -> list[RemoteFile]:
        ...

    @abstractmethod
    def download_candidates(self, ref: ModelRef, filename: str) -> list[Candidate]:
        ...

    # ---- 便捷方法
    def pick_gguf(self, files: Iterable[RemoteFile]) -> list[RemoteFile]:
        return [f for f in files if f.filename.lower().endswith(".gguf")]


# ---------------------------------------------------------------- 注册表


_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str, cfg: Config) -> Source:
    _load_all()
    key = {"hf": "hf-mirror", "ms": "modelscope", "hf_hub": "hf-mirror"}.get(name, name)
    if key not in _REGISTRY:
        raise KeyError(f"未知的模型源: {name}（可用: {', '.join(sorted(_REGISTRY))}）")
    return _REGISTRY[key](cfg)


def all_sources(cfg: Config) -> list[Source]:
    _load_all()
    out = []
    for n in cfg.sources.order:
        try:
            out.append(get_source(n, cfg))
        except KeyError:
            continue
    return out


_loaded = False


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from llm_downloader.sources import hfmirror, modelscope, ollama  # noqa: F401


# ---------------------------------------------------------------- 可用性探测缓存


def load_probe_cache(cfg: Config) -> dict[str, Any]:
    return read_json(cfg.probe_cache_file, {}) or {}


def save_probe_cache(cfg: Config, cache: dict[str, Any]) -> None:
    write_json(cfg.probe_cache_file, cache)


def probe_source(src: Source, use_cache: bool = True) -> tuple[bool, str]:
    cfg = src.cfg
    cache = load_probe_cache(cfg)
    rec = cache.get(src.name)
    ttl = cfg.sources.probe_ttl
    if use_cache and rec and time.time() - rec.get("ts", 0) < ttl:
        return bool(rec.get("ok")), str(rec.get("msg", ""))
    try:
        ok, msg = src.probe()
    except Exception as e:  # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    cache[src.name] = {"ok": ok, "msg": msg, "ts": time.time()}
    save_probe_cache(cfg, cache)
    return ok, msg


def probe_all(cfg: Config, use_cache: bool = True) -> dict[str, tuple[bool, str]]:
    out: dict[str, tuple[bool, str]] = {}
    for s in all_sources(cfg):
        out[s.name] = probe_source(s, use_cache=use_cache)
    return out
