"""LLM Downloader 命令行入口。

用法
    python -m llm_downloader                 # 启动 Web UI（默认 0.0.0.0:8765）
    python -m llm_downloader serve            # 同上，可加 --host / --port
    python -m llm_downloader recommend        # 打印硬件自适应推荐清单
    python -m llm_downloader download <ref>   # 下载一个仓库（支持 hf:/ms:/ollama: 前缀）
    python -m llm_downloader scan             # 扫描模型库目录认领新 GGUF
    python -m llm_downloader list             # 列出已登记模型

模型根目录可用环境变量 LLM_DOWNLOADER_HOME 覆盖（默认 ~/.llm-downloader）。
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from llm_downloader.advisor import ROLES, detect_hardware, recommend
from llm_downloader.config import load_config
from llm_downloader.downloader import execute_plan, plan_download
from llm_downloader.library import Library
from llm_downloader.sources import parse_ref
from llm_downloader.util import human_size


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


def _pick_port(preferred: int, host: str = DEFAULT_HOST, tries: int = 30) -> int:
    """端口被占用时顺延到下一个空闲端口（本机常同时跑着别的服务）。"""
    import socket

    bind_host = "127.0.0.1" if host in ("0.0.0.0", "", "localhost") else host
    for p in range(preferred, preferred + tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((bind_host, p))
            return p
        except OSError:
            continue
    return preferred


def _cmd_serve(args: argparse.Namespace) -> int:
    missing = []
    for mod in ("fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"Web UI 缺少依赖：{', '.join(missing)}", file=sys.stderr)
        print("  → 执行：pip install -r requirements.txt", file=sys.stderr)
        print("  → Windows 可直接双击 start.bat（自动建虚拟环境并装依赖）", file=sys.stderr)
        print("  提示：recommend / download / scan / list 命令零依赖，可直接使用。", file=sys.stderr)
        return 2
    import uvicorn

    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    picked = _pick_port(port, host)
    if picked != port:
        print(f"端口 {port} 已被占用，自动改用 {picked}")
        port = picked
    # 让 web 模块在模块路径下正确加载 .static
    try:
        from llm_downloader import web
    except Exception as e:  # noqa: BLE001
        print(f"加载 Web 模块失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{port}")

    print(f"LLM Downloader 已启动： http://{host}:{port}  (本地访问 http://127.0.0.1:{port})")
    uvicorn.run(web.app, host=host, port=port, log_level="info")
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    cfg = load_config()
    hw = detect_hardware()
    print(f"\n本机：{hw.summary()}\n")
    off = None
    if args.offload:
        v = str(args.offload).strip()
        off = {"mode": "custom", "layers": int(v)} if v.isdigit() else {"mode": v}
    data = recommend(hw=hw, role=args.role, entries=Library(cfg).list(), offload=off)
    rows = data["recommendations"]
    if args.installed_only:
        rows = [r for r in rows if r["installed"]]
    if not rows:
        print("（无匹配条目）")
        return 0
    role_label = {r["key"]: r["label"] for r in data["roles"]}
    last_role = None
    for r in rows:
        if r["role"] != last_role:
            last_role = r["role"]
            print(f"\n=== {role_label.get(r['role'], r['role'])} ===")
        mark = "✓已装" if r["installed"] else "    "
        print(
            f"[{mark}] {r['title']:<34} {r['quant'] or '-':<8} "
            f"{r['size_h']:>9}  {r['fit_label']:<10} {r['fit_note']}"
        )
        if r["why"]:
            print(f"        ↳ {r['why']}")
        lay = r.get("layered") or {}
        if lay.get("applicable"):
            print(
                f"        ↳ 分层 {lay['layers']}/{lay['total_layers']} 层"
                f"（显存 {lay['gpu_gb']}GB / 内存 {lay['cpu_gb']}GB）· {lay['verdict_label']}"
                f" · 建议 -ngl {lay['max_layers']}"
            )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    cfg = load_config()
    ref = parse_ref(args.ref, default_source=args.source or "")
    if args.source:
        ref.source = args.source
    with_mmproj = args.no_mmproj is False
    plan = plan_download(
        cfg, ref, quant=args.quant, filename=args.file,
        with_mmproj=with_mmproj,
    )
    if not plan or not plan.items:
        print("无法解析出下载文件：", "; ".join((plan.errors if plan else []) or ["无匹配文件"]), file=sys.stderr)
        return 1
    print(f"已解析（来源 {plan.resolved_by}），共 {len(plan.items)} 个文件，"
          f"总大小 {human_size(plan.total_size)}")
    dest = cfg.models_dir / ref.repo.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    paths = execute_plan(plan, dest, cfg=cfg)
    print("已下载：")
    for p in paths:
        print(f"  {p}")
    if args.register:
        mp = next((p for p in paths if "mmproj" in p.name.lower()), None)
        weights = next((p for p in paths if "mmproj" not in p.name.lower()), paths[0])
        Library(cfg).register_download(
            weights, mp, source=plan.resolved_by, repo=ref.repo, revision=ref.revision,
            notes=f"quant={args.quant or 'auto'}",
        )
        print("已登记进模型库。")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    cfg = load_config()
    items, added = Library(cfg).scan(prune=args.prune)
    print(f"模型库共 {len(items)} 个，本次新增 {added} 个：")
    for e in items:
        print(f"  {e.display:<34} {e.size_h:>9}  {e.quant or '-'}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config()
    items = Library(cfg).list()
    if not items:
        print("（模型库为空，先执行 scan 或下载模型）")
        return 0
    for e in items:
        print(f"  {e.display:<34} {e.size_h:>9}  {e.quant or '-'}  exists={e.exists}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-downloader",
        description="多源模型下载器 + 硬件自适应推荐 + 本地模型库",
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("serve", help="启动 Web UI（默认命令）")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    sp.set_defaults(func=_cmd_serve)

    rp = sub.add_parser("recommend", help="打印硬件自适应推荐清单")
    rp.add_argument("--role", default=None, help="只看某个用途：tools/chat/coding/vision/image")
    rp.add_argument("--installed-only", action="store_true")
    rp.add_argument("--offload", default=None, help="分层加载策略：auto / cpu / all / 整数层数（对应 -ngl）")
    rp.set_defaults(func=_cmd_recommend)

    dp = sub.add_parser("download", help="下载一个仓库（ref 支持 hf:/ms:/ollama: 前缀）")
    dp.add_argument("ref", help="仓库引用，如 hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    dp.add_argument("--quant", default=None, help="指定量化档，如 q4_k_m")
    dp.add_argument("--file", default=None, help="只下某个文件名")
    dp.add_argument("--source", default=None, help="限定源：hf-mirror / modelscope / ollama")
    dp.add_argument("--no-mmproj", action="store_true", help="不下载 mmproj 投影层")
    dp.add_argument("--register", action="store_true", help="下载后登记进模型库")
    dp.set_defaults(func=_cmd_download)

    sc = sub.add_parser("scan", help="扫描模型库目录认领新 GGUF")
    sc.add_argument("--prune", action="store_true", help="顺手移除已不存在的条目")
    sc.set_defaults(func=_cmd_scan)

    sub.add_parser("list", help="列出已登记模型").set_defaults(func=_cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # 默认进入 serve
        args = parser.parse_args(["serve"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
