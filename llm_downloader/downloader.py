"""下载引擎：多源降级 + Range 断点续传 + 进度回调节流 + 校验。

设计要点
- 统一写 <dest>.part，校验通过后再原子改名，中断不会留下半个“看起来正常”的文件。
- 一个源失败自动切下一个候选 URL（同一文件跨源字节一致，可续传；若期望大小不符则重置）。
- 进度回调带节流（默认 0.25s 一次），供 CLI / SSE 复用。
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from llm_downloader.config import Config
from llm_downloader.http import Client, Response, Timeout
from llm_downloader.sources import Candidate, ModelRef, RemoteFile, all_sources, get_source
from llm_downloader.util import ensure_dir, human_size, sha256_file

ProgressFn = Callable[["Progress"], None]


@dataclass
class Progress:
    filename: str = ""
    phase: str = "download"  # resolve | download | verify | done | error | skipped
    downloaded: int = 0
    total: int | None = None
    speed_bps: float = 0.0
    eta_s: float | None = None
    source: str = ""
    message: str = ""
    attempt: int = 1

    @property
    def percent(self) -> float:
        if not self.total:
            return 0.0
        return min(100.0, self.downloaded * 100.0 / self.total)

    def line(self) -> str:
        bar_len = 28
        filled = int(bar_len * self.percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        tot = human_size(self.total) if self.total else "?"
        return (
            f"{bar} {self.percent:5.1f}%  {human_size(self.downloaded)}/{tot}"
            f"  {human_size(int(self.speed_bps))}/s"
            + (f"  ETA {int(self.eta_s)}s" if self.eta_s and self.eta_s < 86400 else "")
            + f"  [{self.source}]"
        )


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    """用户主动停止 / 取消（不算失败，.part 可按策略保留以便续传）。"""


class CancelToken:
    """跨线程取消信号。

    cancel(purge=False)  = 停止：保留 .part，下次自动断点续传
    cancel(purge=True)   = 取消：删除 .part，不留半成品
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._purge = False

    def cancel(self, purge: bool = False) -> None:
        with self._lock:
            if purge:
                self._purge = True
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def purge(self) -> bool:
        with self._lock:
            return self._purge

    def check(self) -> None:
        if self._event.is_set():
            raise DownloadCancelled("下载已取消")


# ---------------------------------------------------------------- 候选测速优选


@dataclass
class ProbeResult:
    candidate: Candidate
    bytes_per_sec: float = 0.0
    ok: bool = False
    note: str = ""


def probe_candidates(
    candidates: Sequence[Candidate],
    sample_bytes: int = 384 * 1024,
    timeout: float = 10.0,
    max_parallel: int = 6,
    cancel: CancelToken | None = None,
) -> list[ProbeResult]:
    """并行测速：对每个候选拉一小段，量出真实吞吐，用于挑最快的源。

    这一步很值：同一份文件在不同镜像上可能是 50 KB/s 与 1.7 MB/s 的差别，
    而“哪个镜像快”跟时间/网络/运营商都有关，写死顺序必然吃亏。
    """
    import concurrent.futures as cf

    def one(c: Candidate) -> ProbeResult:
        if cancel:
            cancel.check()
        headers = dict(c.headers or {})
        headers["Range"] = f"bytes=0-{sample_bytes - 1}"
        t0 = time.perf_counter()
        got = 0
        try:
            with Client(
                follow_redirects=True,
                timeout=Timeout(connect=8.0, read=timeout, write=timeout, pool=10.0),
                headers={"User-Agent": "llm-downloader/0.1 (+local)"},
            ) as client:
                with client.stream("GET", c.url, headers=headers) as r:
                    if r.status_code >= 400:
                        return ProbeResult(c, 0.0, False, f"HTTP {r.status_code}")
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        if cancel:
                            cancel.check()
                        got += len(chunk)
                        if got >= sample_bytes:
                            break
        except DownloadCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            return ProbeResult(c, 0.0, False, f"{type(e).__name__}")
        dt = time.perf_counter() - t0
        if got < 4096:  # 拿到太少，不可信
            return ProbeResult(c, 0.0, False, f"仅 {got} B")
        return ProbeResult(c, got / dt if dt > 0 else 0.0, True)

    if not candidates:
        return []
    with cf.ThreadPoolExecutor(max_workers=min(max_parallel, len(candidates))) as ex:
        results = list(ex.map(one, candidates))

    # 成功的按速度降序；失败的留在最后（仍值得一试，测速失败不代表下载失败）
    ok = sorted([r for r in results if r.ok], key=lambda r: r.bytes_per_sec, reverse=True)
    bad = [r for r in results if not r.ok]
    return ok + bad


def _distinct_hosts(candidates: Sequence[Candidate]) -> int:
    from urllib.parse import urlparse

    hosts = set()
    for c in candidates:
        try:
            hosts.add(urlparse(c.url).netloc)
        except Exception:
            continue
    return len(hosts)


# ---------------------------------------------------------------- 单文件


def download_file(
    candidates: Sequence[Candidate],
    dest: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    on_progress: ProgressFn | None = None,
    cfg: Config | None = None,
    timeout: float = 120.0,
    verify: bool = True,
    speed_probe: bool | None = None,
    probe_bytes: int = 384 * 1024,
    cancel: CancelToken | None = None,
) -> Path:
    """按候选顺序尝试下载到 dest，成功返回 dest。

    speed_probe: 有多个不同域名的候选时，先并行测速再按实测速度排序。
                 默认跟随 config.sources.probe_speed（默认 True）。
    """
    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")
    cands = list(candidates)

    def _cancelled() -> None:
        # 「取消」删半成品；「停止」保留 .part 供下次续传
        if cancel and cancel.purge:
            part.unlink(missing_ok=True)
        raise DownloadCancelled("下载已取消")

    if cancel and cancel.cancelled:
        _cancelled()

    # 已经完整下载过 -> 直接跳过（除非要求校验）
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        if on_progress:
            on_progress(
                Progress(
                    filename=dest.name,
                    phase="skipped",
                    downloaded=dest.stat().st_size,
                    total=expected_size,
                    message="已存在且大小匹配",
                )
            )
        return dest

    do_probe = speed_probe
    if do_probe is None:
        do_probe = bool(getattr(getattr(cfg, "sources", None), "probe_speed", True))
    # 只有一个域名就没有可选的，别白花流量
    if do_probe and _distinct_hosts(cands) < 2:
        do_probe = False

    if do_probe and len(cands) > 1:
        if on_progress:
            on_progress(
                Progress(
                    filename=dest.name,
                    phase="resolve",
                    message=f"对 {len(cands)} 个候选测速，择优下载…",
                )
            )
        pb = getattr(cfg.sources, "probe_bytes", probe_bytes) if cfg else probe_bytes
        try:
            results = probe_candidates(cands, sample_bytes=pb, cancel=cancel)
        except DownloadCancelled:
            _cancelled()
        if on_progress:
            for r in results[:6]:
                on_progress(
                    Progress(
                        filename=dest.name,
                        phase="probe",
                        source=r.candidate.source,
                        speed_bps=r.bytes_per_sec,
                        message=(
                            f"测速 {human_size(int(r.bytes_per_sec))}/s"
                            if r.ok
                            else f"不可用（{r.note}）"
                        ),
                    )
                )
        ordered = [r.candidate for r in results if r.ok] + [r.candidate for r in results if not r.ok]
        # 去重保序
        seen: set[str] = set()
        cands = [c for c in ordered if not (c.url in seen or seen.add(c.url))]
        if on_progress and cands:
            best = results[0]
            if best.ok:
                on_progress(
                    Progress(
                        filename=dest.name,
                        phase="resolve",
                        source=best.candidate.source,
                        speed_bps=best.bytes_per_sec,
                        message=(
                            f"选用最快源 {best.candidate.source}"
                            f"（{human_size(int(best.bytes_per_sec))}/s，{best.candidate.note or best.candidate.url[:60]}）"
                        ),
                    )
                )

    errors: list[str] = []
    for idx, cand in enumerate(cands, start=1):
        if cancel and cancel.cancelled:
            _cancelled()
        try:
            _download_one(cand, part, expected_size, on_progress, timeout, attempt=idx, cancel=cancel)
            break
        except DownloadCancelled:
            _cancelled()
        except Exception as e:  # noqa: BLE001
            errors.append(f"[{cand.source}] {type(e).__name__}: {e}")
            if on_progress:
                on_progress(
                    Progress(
                        filename=dest.name,
                        phase="error",
                        source=cand.source,
                        attempt=idx,
                        message=f"{type(e).__name__}: {e} —— 尝试下一个源",
                    )
                )
            continue
    else:
        raise DownloadError(
            f"所有源均失败（{dest.name}）：\n  " + "\n  ".join(errors) if errors else "无可用候选 URL"
        )

    # 大小校验
    got = part.stat().st_size
    if expected_size and got != expected_size:
        raise DownloadError(f"大小不符：期望 {expected_size}，实际 {got}（{dest.name}）")

    if cancel and cancel.cancelled:
        _cancelled()

    # 哈希校验
    if verify and expected_sha256:
        if on_progress:
            on_progress(
                Progress(filename=dest.name, phase="verify", downloaded=got, total=got, message="校验 sha256")
            )
        actual = sha256_file(part)
        if actual.lower() != expected_sha256.lower():
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 校验失败：期望 {expected_sha256[:16]}…，实际 {actual[:16]}…（{dest.name}）"
            )

    os.replace(part, dest)
    if on_progress:
        on_progress(
            Progress(filename=dest.name, phase="done", downloaded=got, total=got, message=str(dest))
        )
    return dest


def _download_one(
    cand: Candidate,
    part: Path,
    expected_size: int | None,
    on_progress: ProgressFn | None,
    timeout: float,
    attempt: int = 1,
    cancel: CancelToken | None = None,
) -> None:
    resume_from = 0
    if part.exists():
        resume_from = part.stat().st_size
        # 源之间大小不一致时重置，避免拼出坏文件
        if expected_size and resume_from > expected_size:
            part.unlink(missing_ok=True)
            resume_from = 0

    headers = dict(cand.headers or {})
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    with Client(
        follow_redirects=True,
        timeout=Timeout(connect=20.0, read=timeout, write=timeout, pool=30.0),
        headers={"User-Agent": "llm-downloader/0.1 (+local)"},
    ) as client:
        with client.stream("GET", cand.url, headers=headers) as r:
            if r.status_code == 416:
                # 已经下完
                return
            if r.status_code == 206:
                total = None
                cr = r.headers.get("content-range", "")
                if "/" in cr:
                    try:
                        total = int(cr.split("/")[-1])
                    except ValueError:
                        total = None
                mode = "ab"
            elif r.status_code == 200:
                if resume_from:
                    # 服务端不支持 Range，从头下
                    resume_from = 0
                total = _total_of(r, cand, expected_size)
                mode = "wb"
            else:
                r.raise_for_status()
                return

            if total is None:
                total = _total_of(r, cand, expected_size)

            done = resume_from
            t0 = time.perf_counter()
            last_emit = 0.0
            win: list[tuple[float, int]] = [(t0, done)]

            with open(part, mode) as f:
                for chunk in r.iter_bytes(chunk_size=1 << 18):
                    if cancel:
                        cancel.check()
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.perf_counter()
                    win.append((now, done))
                    while len(win) > 2 and now - win[0][0] > 3.0:
                        win.pop(0)
                    if on_progress and (now - last_emit > 0.25 or (total and done >= total)):
                        last_emit = now
                        speed = _speed(win)
                        eta = (total - done) / speed if (speed > 0 and total) else None
                        on_progress(
                            Progress(
                                filename=cand.filename or part.name,
                                phase="download",
                                downloaded=done,
                                total=total,
                                speed_bps=speed,
                                eta_s=eta,
                                source=cand.source,
                                attempt=attempt,
                            )
                        )


def _total_of(r: Response, cand: Candidate, expected_size: int | None) -> int | None:
    cl = r.headers.get("content-length")
    if cl and cl.isdigit():
        base = cand.size or expected_size
        n = int(cl)
        if base and abs(n - base) < 1:
            return n
        return n
    return cand.size or expected_size


def _speed(win: list[tuple[float, int]]) -> float:
    if len(win) < 2:
        return 0.0
    dt = win[-1][0] - win[0][0]
    if dt <= 0:
        return 0.0
    return (win[-1][1] - win[0][1]) / dt


# ---------------------------------------------------------------- 计划与多源编排


QUANT_ORDER = [
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "iq4_xs",
    "q4_0",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "fp16",
    "bf16",
    "f32",
]


def guess_quant(filename: str) -> str:
    low = filename.lower()
    for q in sorted(QUANT_ORDER, key=len, reverse=True):
        if q in low:
            return q.upper().replace("_K", "_K").replace("Q", "Q")
    if low.endswith(".gguf"):
        return "unknown"
    return ""


def select_gguf(
    files: Iterable[RemoteFile],
    want_quant: str | None = None,
    want_filename: str | None = None,
    prefer_smallest: bool = False,
) -> RemoteFile | None:
    """从文件清单里挑一个 GGUF 权重文件。"""
    ggufs = [f for f in files if f.filename.lower().endswith(".gguf")]
    # 排除 mmproj / projector
    weights = [f for f in ggufs if f.role != "mmproj" and "mmproj" not in f.filename.lower()]
    if not weights:
        return None
    if want_filename:
        for f in weights:
            if f.filename == want_filename or Path(f.filename).name == want_filename:
                return f
        return None
    if want_quant:
        wq = want_quant.lower()
        exact = [f for f in weights if wq in f.filename.lower()]
        if exact:
            return sorted(exact, key=lambda f: f.size or 0)[0]
    # 默认：优先 q4_k_m，其次按量化偏好顺序，最后挑最小的
    for pref in ("q4_k_m", "q5_k_m", "q4_k_s", "q8_0"):
        m = [f for f in weights if pref in f.filename.lower()]
        if m:
            return sorted(m, key=lambda f: f.size or 0)[0]
    if prefer_smallest or True:
        return sorted(weights, key=lambda f: f.size or 0)[0]
    return weights[0]


@dataclass
class PlanItem:
    file: RemoteFile
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def chosen(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass
class DownloadPlan:
    ref: ModelRef
    items: list[PlanItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    resolved_by: str = ""

    @property
    def total_size(self) -> int:
        return sum((i.file.size or 0) for i in self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


def plan_download(
    cfg: Config,
    ref: ModelRef,
    quant: str | None = None,
    filename: str | None = None,
    with_mmproj: bool = True,
    source_names: Sequence[str] | None = None,
    on_progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
) -> DownloadPlan:
    """解析仓库文件清单并选出要下载的文件，生成可执行的下载计划。

    按 config.sources.order（或显式 source_names）逐个源尝试，某个源解析成功即采用。
    """
    plan = DownloadPlan(ref=ref)

    # 显式文件名 -> 只需要那一个文件
    explicit = filename or ref.filename

    sources = [get_source(n, cfg) for n in source_names] if source_names else all_sources(cfg)
    if ref.source:
        sources = [get_source(ref.source, cfg)]

    files: list[RemoteFile] = []
    used: str = ""
    errs: list[str] = []
    for src in sources:
        if cancel:
            cancel.check()
        if on_progress:
            on_progress(
                Progress(phase="resolve", filename=ref.repo, source=src.name, message="解析文件清单…")
            )
        try:
            files = src.list_files(ref)
        except Exception as e:  # noqa: BLE001
            errs.append(f"[{src.name}] {type(e).__name__}: {e}")
            continue
        if files:
            used = src.name
            break

    if not files:
        plan.errors = errs or ["没有任何源返回文件清单"]
        return plan

    plan.resolved_by = used
    src_obj = get_source(used, cfg)

    picked: list[RemoteFile] = []
    if explicit:
        for f in files:
            if f.filename == explicit or Path(f.filename).name == explicit:
                picked.append(f)
        if not picked:
            plan.errors.append(f"在 {used} 中找不到文件 {explicit}")
            return plan
    else:
        w = select_gguf(files, want_quant=quant)
        if not w:
            gg = [f.filename for f in files if f.filename.lower().endswith(".gguf")]
            plan.errors.append(
                f"{used}/{ref.repo} 里没有 GGUF 权重文件。可用文件：{gg[:8] or [f.filename for f in files][:8]}"
            )
            return plan
        picked.append(w)
        if with_mmproj:
            mm = [f for f in files if f.role == "mmproj"]
            if mm:
                picked.append(sorted(mm, key=lambda f: f.size or 0)[0])

    for f in picked:
        cands = []
        try:
            cands = src_obj.download_candidates(ref, f.filename)
        except Exception as e:  # noqa: BLE001
            plan.errors.append(f"生成下载候选失败 [{f.filename}]：{e}")
        if not cands:
            plan.errors.append(f"无下载候选：[{f.filename}]")
        plan.items.append(PlanItem(file=f, candidates=cands))

    return plan


def execute_plan(
    plan: DownloadPlan,
    dest_dir: Path,
    on_progress: ProgressFn | None = None,
    cfg: Config | None = None,
    cancel: CancelToken | None = None,
) -> list[Path]:
    """执行下载计划，返回落盘路径列表。"""
    out: list[Path] = []
    dest_dir = Path(dest_dir)
    for item in plan.items:
        if cancel:
            cancel.check()
        dest = dest_dir / Path(item.file.filename).name
        p = download_file(
            item.candidates,
            dest,
            expected_size=item.file.size,
            expected_sha256=item.file.sha256,
            on_progress=on_progress,
            cfg=cfg,
            cancel=cancel,
        )
        out.append(p)
    return out
