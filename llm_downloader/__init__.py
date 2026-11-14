"""LLM Downloader —— 多源模型下载器 + 硬件自适应推荐 + 本地模型库。

把「推荐 / 模型管理 / 下载」三块能力从 agent-harness 抽离成独立开源项目，
不依赖 harness 的 provider / runtime / 文生图 / 视频等模块。

入口：
    python -m llm_downloader            # 启动 Web UI（默认 0.0.0.0:8765）
    python -m llm_downloader --help      # CLI 帮助
"""

from __future__ import annotations

from .config import VERSION, APP_NAME, Config, SourceConfig, load_config

__all__ = ["VERSION", "APP_NAME", "Config", "SourceConfig", "load_config"]
__version__ = VERSION
