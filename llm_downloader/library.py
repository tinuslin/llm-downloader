"""本地模型库：扫描 / 索引 / 增删 / 别名。

索引落在 $HOME/models/index.json，同时支持直接扫描 models 目录，
所以你可以手动把 GGUF 丢进去再点「扫描」认领。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_downloader.config import Config
from llm_downloader.downloader import guess_quant
from llm_downloader.util import human_size, read_json, write_json


@dataclass
class ModelEntry:
    id: str
    name: str
    path: str
    size: int = 0
    quant: str = ""
    mmproj: str | None = None
    source: str = ""
    repo: str = ""
    revision: str = "main"
    alias: str = ""
    # 该模型是否支持视觉输入
    vision: bool = False
    ctx_default: int = 0
    chat_template: str = ""
    notes: str = ""
    added_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    @property
    def size_h(self) -> str:
        return human_size(self.size)

    @property
    def display(self) -> str:
        return self.alias or self.name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["exists"] = self.exists
        d["size_h"] = self.size_h
        d["display"] = self.display
        return d


class Library:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---- 读写
    def load(self) -> list[ModelEntry]:
        raw = read_json(self.cfg.index_file, []) or []
        out: list[ModelEntry] = []
        for d in raw:
            try:
                d = {k: v for k, v in d.items() if k in ModelEntry.__dataclass_fields__}
                out.append(ModelEntry(**d))
            except Exception:
                continue
        return out

    def save(self, entries: list[ModelEntry]) -> None:
        write_json(self.cfg.index_file, [asdict(e) for e in entries])

    # ---- 查询
    def list(self, only_existing: bool = False) -> list[ModelEntry]:
        items = self.load()
        if only_existing:
            items = [e for e in items if e.exists]
        return sorted(items, key=lambda e: (-e.last_used, e.name))

    def get(self, key: str) -> ModelEntry | None:
        key_l = (key or "").strip().lower()
        if not key_l:
            return None
        items = self.load()
        for e in items:
            if e.id == key or e.alias.lower() == key_l or e.name.lower() == key_l:
                return e
        for e in items:
            if key_l in e.name.lower() or key_l in Path(e.path).name.lower():
                return e
        # 退化：直接把 key 当文件路径
        p = Path(key)
        if p.exists() and p.suffix.lower() in (".gguf", ".bin"):
            return self.upsert_from_path(p)
        return None

    def find_by_basename(self, name: str) -> ModelEntry | None:
        for e in self.load():
            if Path(e.path).name == name:
                return e
        return None

    # ---- 写入
    def add(self, entry: ModelEntry) -> ModelEntry:
        items = self.load()
        for i, e in enumerate(items):
            if e.id == entry.id or e.path == entry.path:
                entry.added_at = e.added_at
                items[i] = entry
                self.save(items)
                return entry
        items.append(entry)
        self.save(items)
        return entry

    def remove(self, key: str, delete_file: bool = False) -> bool:
        items = self.load()
        target = self.get(key)
        if not target:
            return False
        keep = [e for e in items if e.id != target.id]
        self.save(keep)
        if delete_file:
            p = Path(target.path)
            if p.exists():
                p.unlink()
            if target.mmproj:
                mp = Path(target.mmproj)
                if mp.exists():
                    mp.unlink()
        return True

    def purge(self, key: str) -> dict[str, Any]:
        """移除条目并删除本地权重文件（主模型 + mmproj）。

        两道安全闸：
          1. 只删 cfg.models_dir 以内的文件——索引里指向别处的路径一律不碰，
             避免坏条目 / 手滑把整个硬盘拖下去；
          2. 文件删不掉时**保留索引条目**，绝不出现“记录没了、文件还在”的半截状态。
        """
        entry = self.get(key)
        if not entry:
            return {"ok": False, "error": f"未找到 {key}", "freed": 0,
                    "deleted": [], "failed": [], "removed_dirs": [], "outside": []}
        try:
            root = Path(self.cfg.models_dir).resolve()
        except OSError:
            root = Path(self.cfg.models_dir)

        files: list[Path] = []
        outside: list[str] = []
        for raw in (entry.path, entry.mmproj):
            if not raw:
                continue
            p = Path(raw)
            try:
                p.resolve().relative_to(root)
            except ValueError:
                outside.append(str(p))
                continue
            files.append(p)

        freed, deleted, failed = 0, [], []
        for p in files:
            if not p.exists():
                deleted.append(str(p))
                continue
            try:
                freed += p.stat().st_size
                p.unlink()
                deleted.append(str(p))
            except OSError as e:
                failed.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})

        # 只回收因本次删除而空掉的目录（逐层向上，永不碰 models_dir 本身）
        removed_dirs: list[str] = []
        if not failed:
            for p in files:
                d = p.parent
                while d != root and root in d.parents:
                    try:
                        next(d.iterdir())
                        break
                    except StopIteration:
                        try:
                            d.rmdir()
                        except OSError:
                            break
                        removed_dirs.append(str(d))
                        d = d.parent
                    except OSError:
                        break

        if failed:
            return {"ok": False, "error": "文件删除失败，索引已保留", "freed": freed,
                    "deleted": deleted, "failed": failed, "removed_dirs": removed_dirs,
                    "outside": outside, "entry": entry}
        self.remove(key)
        return {"ok": True, "error": "", "freed": freed, "deleted": deleted,
                "failed": [], "removed_dirs": removed_dirs, "outside": outside, "entry": entry}

    def touch(self, key: str) -> None:
        items = self.load()
        changed = False
        for e in items:
            if e.id == key or e.alias == key or e.name == key:
                e.last_used = time.time()
                changed = True
        if changed:
            self.save(items)

    # ---- 从文件构造
    def upsert_from_path(self, path: Path, **kw: Any) -> ModelEntry:
        path = Path(path)
        size = path.stat().st_size if path.exists() else 0
        name = _base_name(path)
        e = ModelEntry(
            id=name.lower().replace(" ", "-"),
            name=name,
            path=str(path),
            size=size,
            quant=guess_quant(path.name) or kw.get("quant", ""),
            source=kw.get("source", "manual"),
            repo=kw.get("repo", ""),
            revision=kw.get("revision", "main"),
            alias=kw.get("alias", ""),
            notes=kw.get("notes", ""),
        )
        # 同名 mmproj 自动关联
        for cand in path.parent.glob("*mmproj*.gguf"):
            if cand != path:
                e.mmproj = str(cand)
                e.vision = True
                break
        return self.add(e)

    # ---- 扫描
    def scan(self, prune: bool = False) -> tuple[list[ModelEntry], int]:
        """扫描 models 目录，认领未登记的 GGUF。返回 (当前列表, 新增数)。"""
        root = self.cfg.models_dir

        def is_mmproj(p: Path) -> bool:
            n = p.name.lower()
            return "mmproj" in n or "projector" in n

        known = {e.path for e in self.load()}
        added = 0
        for p in sorted(root.rglob("*.gguf")):
            # mmproj 不单独登记：它没有 weights 跑不起来，只会污染模型列表、
            # 还可能被误选。它通过下面的关联逻辑挂到主模型上。
            if is_mmproj(p):
                continue
            if str(p) not in known:
                self.upsert_from_path(p)
                added += 1
        items = self.load()

        # 清掉历史上被误登记成“模型”的 mmproj 条目
        drop = {e.id for e in items if is_mmproj(Path(e.path))}
        if drop:
            items = [e for e in items if e.id not in drop]

        if prune:
            keep = [e for e in items if e.exists]
            removed = len(items) - len(keep)
            if removed:
                self.save(keep)
            items = keep
        # 为已有条目补 mmproj 关联
        changed = False
        for e in items:
            if not e.mmproj:
                for cand in Path(e.path).parent.glob("*mmproj*.gguf"):
                    e.mmproj = str(cand)
                    e.vision = True
                    changed = True
                    break
        if changed or drop:
            self.save(items)
        return items, added

    # ---- 注册下载产物
    def register_download(
        self,
        weights: Path,
        mmproj: Path | None,
        source: str,
        repo: str,
        revision: str,
        alias: str = "",
        notes: str = "",
    ) -> ModelEntry:
        name = _base_name(weights)
        e = ModelEntry(
            id=name.lower().replace(" ", "-"),
            name=name,
            path=str(weights),
            size=weights.stat().st_size if weights.exists() else 0,
            quant=guess_quant(weights.name),
            mmproj=str(mmproj) if mmproj else None,
            source=source,
            repo=repo,
            revision=revision,
            alias=alias,
            vision=bool(mmproj),
            notes=notes,
        )
        return self.add(e)


def _base_name(path: Path) -> str:
    return path.stem
