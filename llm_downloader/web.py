"""FastAPI 后端：把「推荐 / 模型管理 / 下载」三块能力暴露成 HTTP/SSE。

端点
    GET  /                       单文件工作台 UI
    GET  /api/config             当前配置快照
    GET  /api/sources            源可用性探测
    POST /api/search             搜索模型（hf-mirror）
    GET  /api/files?ref=         仓库文件列表
    POST /api/download           下载（SSE 进度流；支持 refs 多仓库回落 / items 成套 / dest / register）
    POST /api/download/cancel    停止 / 取消下载任务（?purge=1 删除 .part）
    GET  /api/download/tasks     活动下载任务
    GET/POST /api/settings       自定义下载目录、源优先级、镜像地址、分层加载策略
    GET  /api/advisor            硬件自适应模型推荐（含分层加载评估）
    GET  /api/models             模型库（自动扫描）
    POST /api/models/scan        扫描目录认领新 GGUF
    DELETE /api/models/{key}     移除条目（?delete_file=1 连文件一起删）
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from llm_downloader.advisor import detect_hardware, recommend
from llm_downloader.config import (
    DEFAULT_SETTINGS,
    KNOWN_SOURCES,
    Config,
    load_config,
    reset_settings,
    save_settings,
)
from llm_downloader.downloader import CancelToken, DownloadCancelled, execute_plan, plan_download
from llm_downloader.http import Client, Timeout
from llm_downloader.library import Library
from llm_downloader.sources import all_sources, get_source, parse_ref, probe_all
from llm_downloader.util import human_size


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="llm-downloader", docs_url="/docs", redoc_url=None)


# ---------------------------------------------------------------- 基础


def _cfg() -> Config:
    return load_config()


def _sse(data: dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _image_model_files(cfg: Config) -> list[str]:
    """扫描 image-models 目录，给推荐模块判断「扩散模型是否已装」。"""
    try:
        return [p.name for p in sorted(Path(cfg.image_models_dir).rglob("*")) if p.is_file()]
    except Exception:  # noqa: BLE001
        return []


@app.get("/", response_class=HTMLResponse)
async def index():
    f = STATIC_DIR / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>llm-downloader</h1><p>static/index.html 缺失</p>")
    return HTMLResponse(f.read_text(encoding="utf-8"))


def _effective(cfg: Config) -> dict[str, Any]:
    return {
        "home": str(cfg.home),
        "models_dir": str(cfg.models_dir),
        "image_models_dir": str(cfg.image_models_dir),
        "sources_order": cfg.sources.order,
        "endpoints": {
            "hf-mirror": cfg.sources.hf_endpoint,
            "modelscope": cfg.sources.modelscope_endpoint,
            "ollama": cfg.sources.ollama_registry,
        },
        "settings_file": str(cfg.settings_file),
        "offload": cfg.offload,
    }


@app.get("/api/config")
async def api_config():
    return _effective(_cfg())


# ---------------------------------------------------------------- 设置


@app.get("/api/settings")
async def api_settings_get():
    cfg = _cfg()
    return {
        "settings": cfg.to_settings(),
        "defaults": DEFAULT_SETTINGS,
        "effective": _effective(cfg),
        "known_sources": [
            {"name": n, "endpoint": cfg.sources.endpoint_of(n), "enabled": n in cfg.sources.order}
            for n in KNOWN_SOURCES
        ],
    }


@app.post("/api/settings")
async def api_settings_save(payload: dict = Body(...)):
    """部分更新：只带要改的字段即可，其余沿用现有设置。"""
    try:
        merged = save_settings(payload)
    except (ValueError, OSError) as e:
        raise HTTPException(400, f"设置无效：{e}")
    cfg = load_config()
    return {"ok": True, "settings": merged, "effective": _effective(cfg)}


@app.post("/api/settings/reset")
async def api_settings_reset():
    default = reset_settings()
    cfg = load_config()
    return {"ok": True, "settings": default, "effective": _effective(cfg)}


# ---------------------------------------------------------------- 源与搜索


@app.get("/api/sources")
async def api_sources(refresh: bool = False):
    cfg = _cfg()
    res = probe_all(cfg, use_cache=not refresh)
    return {
        "order": cfg.sources.order,
        "sources": [
            {
                "name": s.name,
                "kind": s.kind,
                "endpoint": cfg.sources.endpoint_of(s.name),
                "ok": res.get(s.name, (False, ""))[0],
                "message": res.get(s.name, (False, ""))[1],
            }
            for s in all_sources(cfg)
        ],
    }


@app.post("/api/search")
async def api_search(payload: dict = Body(...)):
    cfg = _cfg()
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "query 不能为空")
    limit = int(payload.get("limit") or 25)
    ep = cfg.sources.hf_endpoint

    def _search() -> Any:
        # 标准库同步请求丢进线程池，避免阻塞 ASGI 事件循环（SSE 下载仍需实时推送）
        with Client(timeout=Timeout(read=25.0)) as c:
            r = c.get(
                f"{ep}/api/models",
                params={
                    "search": q,
                    "limit": limit,
                    "filter": "gguf",
                    "sort": "downloads",
                    "direction": "-1",
                },
            )
            r.raise_for_status()
            return r.json()

    try:
        data = await asyncio.to_thread(_search)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"搜索失败：{type(e).__name__}: {e}")
    return {
        "results": [
            {
                "id": m.get("id") or m.get("modelId", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "tags": (m.get("tags") or [])[:6],
                "modified": m.get("lastModified") or "",
            }
            for m in data
        ]
    }


@app.get("/api/files")
async def api_files(ref: str, source: Optional[str] = None):
    cfg = _cfg()
    parsed = parse_ref(ref, default_source=source or "")
    if source:
        parsed.source = source
    candidates = [get_source(parsed.source, cfg)] if parsed.source else all_sources(cfg)
    errors = []
    for src in candidates:
        try:
            items = src.list_files(parsed)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{src.name}: {type(e).__name__}: {e}")
            continue
        if not items:
            continue
        return {
            "ref": str(parsed),
            "source": src.name,
            "files": [
                {
                    "filename": f.filename,
                    "size": f.size,
                    "size_h": human_size(f.size),
                    "role": f.role,
                }
                for f in items
            ],
        }
    raise HTTPException(404, "拿不到文件清单：" + "; ".join(errors))


# ---------------------------------------------------------------- 下载（SSE）


@dataclass
class DownloadTask:
    id: str
    label: str = ""
    status: str = "running"  # running | cancelling | done | cancelled | error
    created: float = field(default_factory=time.time)
    token: CancelToken = field(default_factory=CancelToken)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "age_s": round(time.time() - self.created, 1),
            "cancel_requested": self.token.cancelled,
        }


_TASKS: dict[str, DownloadTask] = {}
_TASKS_LOCK = threading.Lock()


def _register_task(label: str) -> DownloadTask:
    t = DownloadTask(id=uuid.uuid4().hex[:12], label=label)
    with _TASKS_LOCK:
        _TASKS[t.id] = t
        ended = [k for k, v in _TASKS.items() if v.status not in ("running", "cancelling")]
        for k in ended[:-40]:
            _TASKS.pop(k, None)
    return t


def _resolve_dest_dir(cfg: Config, dest: str, parsed: Any) -> Path:
    """把 dest 描述符解析成真实落盘目录。

    dest 形式：
        models                  → 语言模型库（进索引）
        image-models            → 扩散模型根目录（不进索引）
        image-models/<子目录>    → 扩散模型子目录（成套组件凑齐一个目录）
        绝对路径                 → 直接用
    """
    if dest.startswith("image-models"):
        root = cfg.image_models_dir
        sub = dest.split("/", 1)[1].strip() if "/" in dest else ""
        return root / sub if sub else root
    if Path(dest).is_absolute():
        return Path(dest)
    return cfg.models_dir / parsed.repo.replace("/", "__")


@app.post("/api/download")
async def api_download(payload: dict = Body(...)):
    cfg = _cfg()
    raw_refs = payload.get("refs") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    single = (payload.get("ref") or "").strip()
    refs = [r for r in ([single] if single else []) + [str(r).strip() for r in raw_refs] if r]
    # items 模式（成套下载）只需要 items，不需要 ref
    if not refs and not (payload.get("items") or []):
        raise HTTPException(400, "ref / refs / items 不能为空")
    quant = payload.get("quant") or None
    filename = payload.get("file") or None
    source = payload.get("source") or None
    alias = payload.get("alias") or ""
    with_mmproj = bool(payload.get("mmproj", True))
    dest = (payload.get("dest") or "models").strip() or "models"
    register = payload.get("register")
    register = (dest == "models") if register is None else bool(register)
    raw_items = payload.get("items") or []
    items: list[dict[str, Any]] = [dict(x) for x in raw_items if isinstance(x, dict) and x.get("file")]
    dtask = _register_task("、".join(refs[:2]) or f"{len(items)} 个组件")

    async def gen() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done_marker = object()

        def on_progress(p) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "progress",
                    "phase": p.phase,
                    "filename": p.filename,
                    "downloaded": p.downloaded,
                    "total": p.total,
                    "percent": round(p.percent, 2),
                    "speed_bps": round(p.speed_bps),
                    "eta_s": p.eta_s,
                    "source": p.source,
                    "message": p.message,
                },
            )

        def work() -> None:
            try:
                # ---------- 模式 A：成套下载（items = [{ref, file}, …]）
                if items:
                    dest_dir = cfg.image_models_dir
                    if dest.startswith("image-models") and "/" in dest:
                        dest_dir = dest_dir / dest.split("/", 1)[1].strip()
                    elif Path(dest).is_absolute():
                        dest_dir = Path(dest)
                    done_paths: list[Path] = []
                    total_all = 0.0
                    for idx, it in enumerate(items, 1):
                        it_ref = str(it.get("ref") or "").strip()
                        it_file = str(it.get("file") or "").strip()
                        p = parse_ref(it_ref, default_source=source or "")
                        if source:
                            p.source = source
                        queue.put_nowait(
                            {
                                "type": "trying",
                                "ref": f"[{idx}/{len(items)}] {p} :: {it_file}",
                                "index": idx,
                            }
                        )
                        plan = plan_download(
                            cfg, p, quant=None, filename=it_file,
                            with_mmproj=False, on_progress=on_progress, cancel=dtask.token,
                        )
                        if not plan or not plan.items:
                            raise RuntimeError(
                                f"组件 {it_file} 解析失败（{p}）："
                                + "；".join((plan.errors if plan else []) or ["无匹配文件"])
                            )
                        total_all += plan.total_size
                        queue.put_nowait(
                            {
                                "type": "plan",
                                "ref": str(p),
                                "resolved_by": plan.resolved_by,
                                "total_size": plan.total_size,
                                "total_size_h": human_size(plan.total_size),
                                "items": [
                                    {"filename": i.file.filename, "size": i.file.size, "role": i.file.role}
                                    for i in plan.items
                                ],
                                "errors": plan.errors,
                                "step": f"{idx}/{len(items)}",
                            }
                        )
                        done_paths += execute_plan(
                            plan, dest_dir, on_progress=on_progress, cfg=cfg, cancel=dtask.token
                        )
                    queue.put_nowait(
                        {
                            "type": "done",
                            "model": None,
                            "paths": [str(p) for p in done_paths],
                            "dest": str(dest_dir),
                            "registered": False,
                            "total_size_h": human_size(int(total_all)),
                            "files": len(done_paths),
                        }
                    )
                    return

                # ---------- 模式 B：单模型（可多仓库回落）
                plan = None
                parsed = None
                tried: list[str] = []
                for ref in refs:
                    parsed = parse_ref(ref, default_source=source or "")
                    if source:
                        parsed.source = source
                    tried.append(str(parsed))
                    queue.put_nowait({"type": "trying", "ref": str(parsed), "index": len(tried)})
                    cand = plan_download(
                        cfg, parsed, quant=quant, filename=filename,
                        with_mmproj=with_mmproj, on_progress=on_progress, cancel=dtask.token,
                    )
                    if cand and cand.items:
                        plan = cand
                        break
                    if cand and cand.errors:
                        queue.put_nowait(
                            {"type": "note", "message": f"{parsed} 不可用：" + "；".join(cand.errors)}
                        )
                if plan is None:
                    raise RuntimeError("所有候选仓库都无法解析出文件：" + "、".join(tried))

                queue.put_nowait(
                    {
                        "type": "plan",
                        "ref": str(parsed),
                        "resolved_by": plan.resolved_by,
                        "total_size": plan.total_size,
                        "total_size_h": human_size(plan.total_size),
                        "items": [
                            {"filename": i.file.filename, "size": i.file.size, "role": i.file.role}
                            for i in plan.items
                        ],
                        "errors": plan.errors,
                    }
                )
                dest_dir = _resolve_dest_dir(cfg, dest, parsed)
                paths = execute_plan(
                    plan, dest_dir, on_progress=on_progress, cfg=cfg, cancel=dtask.token
                )

                entry = None
                if register:
                    weights = next((p for p in paths if "mmproj" not in p.name.lower()), paths[0])
                    mp = next((p for p in paths if "mmproj" in p.name.lower()), None)
                    entry = Library(cfg).register_download(
                        weights, mp, source=plan.resolved_by, repo=parsed.repo,
                        revision=parsed.revision, alias=alias,
                        notes=f"quant={quant or 'auto'}",
                    )
                queue.put_nowait(
                    {
                        "type": "done",
                        "model": entry.to_dict() if entry else None,
                        "paths": [str(p) for p in paths],
                        "dest": str(dest_dir),
                        "registered": bool(entry),
                    }
                )
            except DownloadCancelled:
                dtask.status = "cancelled"
                queue.put_nowait(
                    {
                        "type": "cancelled",
                        "task_id": dtask.id,
                        "purge": dtask.token.purge,
                        "message": "已取消，临时文件已删除" if dtask.token.purge
                        else "已停止，保留 .part 供下次断点续传",
                    }
                )
            except Exception as e:  # noqa: BLE001
                dtask.status = "error"
                queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                if dtask.status in ("running", "cancelling"):
                    dtask.status = "done"
                queue.put_nowait(done_marker)

        worker = asyncio.create_task(asyncio.to_thread(work))
        yield _sse({"type": "task", "id": dtask.id, "label": dtask.label}, event="download")
        yield _sse({"type": "start", "refs": refs, "dest": dest, "task_id": dtask.id}, event="download")
        try:
            while True:
                item = await queue.get()
                if item is done_marker:
                    break
                yield _sse(item, event="download")
        finally:
            if not worker.done():
                # SSE 断开视为「停止」：保留 .part，避免后台留下无人观看的下载
                if dtask.status == "running":
                    dtask.token.cancel(purge=False)
                    dtask.status = "cancelling"
                await asyncio.gather(worker, return_exceptions=True)
        yield _sse({"type": "end"}, event="download")

    return _sse_response(gen())


@app.post("/api/download/cancel")
async def api_download_cancel(id: Optional[str] = None, payload: dict = Body(default={})):
    """停止 / 取消一个下载任务。

    purge=false（停止）：保留 .part，下次自动续传；
    purge=true （取消）：删除 .part，不留半成品。
    """
    payload = payload or {}
    tid = str(id or payload.get("id") or "").strip()
    purge = bool(payload.get("purge"))
    if not tid:
        raise HTTPException(400, "缺少任务 id")
    with _TASKS_LOCK:
        t = _TASKS.get(tid)
    if not t:
        raise HTTPException(404, f"任务 {tid} 不存在或已清理")
    if t.status not in ("running", "cancelling"):
        return {"ok": True, "id": tid, "status": t.status, "note": "任务已结束"}
    t.token.cancel(purge=purge)
    t.status = "cancelling"
    return {
        "ok": True,
        "id": tid,
        "status": "cancelling",
        "purge": purge,
        "note": "将删除临时 .part 文件" if purge else "保留 .part，下次自动断点续传",
    }


@app.get("/api/download/tasks")
async def api_download_tasks():
    with _TASKS_LOCK:
        tasks = [t.to_dict() for t in _TASKS.values()]
    return {
        "tasks": tasks,
        "active": [t for t in tasks if t["status"] in ("running", "cancelling")],
    }


# ---------------------------------------------------------------- 模型库


@app.get("/api/models")
async def api_models(scan: bool = True):
    cfg = _cfg()
    lib = Library(cfg)
    if scan:
        lib.scan()
    return {"models": [e.to_dict() for e in lib.list()]}


@app.post("/api/models/scan")
async def api_models_scan(payload: dict = Body(default={})):
    cfg = _cfg()
    items, added = Library(cfg).scan(prune=bool(payload.get("prune")))
    return {"count": len(items), "added": added, "models": [e.to_dict() for e in items]}


@app.delete("/api/models/{key}")
async def api_models_delete(key: str, delete_file: bool = False):
    """移除模型条目。?delete_file=1 时连权重文件一起删（不可恢复）。

    注意：本项目不管理运行时服务，所以不做「模型被占用」的拦截——
    若文件被其他进程（如 llama-server）打开导致删除失败，purge 会保留索引并回传错误。
    """
    cfg = _cfg()
    lib = Library(cfg)
    entry = lib.get(key)
    if not entry:
        raise HTTPException(404, f"未找到 {key}")
    if not delete_file:
        lib.remove(key)
        return {"ok": True, "deleted_file": False, "freed_h": ""}
    res = lib.purge(key)
    if not res["ok"]:
        detail = res["failed"][0]["error"] if res["failed"] else res.get("error", "未知错误")
        raise HTTPException(409, f"文件删除失败：{detail}（索引已保留，可重试）")
    return {
        "ok": True,
        "deleted_file": True,
        "freed": res["freed"],
        "freed_h": human_size(res["freed"]),
        "deleted": res["deleted"],
        "removed_dirs": res["removed_dirs"],
        "outside": res["outside"],
    }


# ---------------------------------------------------------------- 模型推荐


@app.get("/api/advisor")
async def api_advisor(
    role: Optional[str] = None,
    installed_only: bool = False,
    offload_mode: Optional[str] = None,
    offload_layers: Optional[int] = None,
):
    """硬件自适应推荐 + 分层加载（GPU offload）评估。

    未传 offload_* 时沿用「设置」里保存的策略。
    """
    cfg = _cfg()
    saved = cfg.offload
    offload = {
        "mode": offload_mode or saved.get("mode") or "auto",
        "layers": saved.get("layers", 0) if offload_layers is None else offload_layers,
    }
    data = recommend(
        hw=detect_hardware(),
        role=role or None,
        entries=Library(cfg).list(),
        image_files=_image_model_files(cfg),
        offload=offload,
    )
    if installed_only:
        data["recommendations"] = [r for r in data["recommendations"] if r["installed"]]
    return data
