# LLM Downloader

[![CI](https://github.com/tinuslin/llm-downloader/actions/workflows/ci.yml/badge.svg)](https://github.com/tinuslin/llm-downloader/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tinuslin/llm-downloader)](https://github.com/tinuslin/llm-downloader/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

多源模型下载器 + 硬件自适应推荐 + 本地模型库。把「**推荐 / 模型管理 / 下载**」三块能力从
内部 agent-harness 项目抽离成一个独立、可分享的开源小工具，不依赖 harness 的
provider / 运行时 / 文生图 / 视频等模块。

仓库地址：<https://github.com/tinuslin/llm-downloader>　·　License: MIT　·　Python 3.10+

> 面向本地跑 GGUF 模型的玩家：想下个 Qwen，但镜像站上几十个量化变体 + 多源地址让人头大；
> 这个工具负责「该下哪个」和「从哪个源稳定下下来」两件事。

## 功能

- **多源镜像下载**：hf-mirror / ModelScope / Ollama Registry 抽象成统一接口，按配置顺序自动降级；
  Range 断点续传、多源并行测速择优、sha256 校验，断网/换源都不白下。
- **硬件自适应推荐**：实时探测本机 CPU / 内存 / 显存，给出「推荐 / 可用（勉强）/ 别下」三档，并按用途
  （工具调用 / 对话 / 代码 / 看图 / 文生图）分类，已下载的自动标「已装」。
- **本地模型库**：扫描目录认领 GGUF、`index.json` 索引、mmproj 自动关联、增删与清理。
- **分层加载评估（GPU Offload）**：按 `n-gpu-layers` 估算显存/内存分配，支持「自动 / 纯 CPU / 全部进 GPU / 自定义层数」四档，给出建议 `-ngl`。
- **可视化设置**：自定义模型下载目录、启用的下载源与优先级、各镜像地址，写入 `settings.json` 后即时生效。
- **下载可停止 / 取消**：「停止」保留 `.part` 供断点续传，「取消」删除临时文件；SSE 连接断开时自动按停止处理。

## 安装 / 启动

**命令行零依赖**：Python 3.10+ 直接跑，不需要 `pip install` 任何东西。
HTTP 客户端用标准库 `urllib` 实现（自动继承系统代理、跟随 302/CDN 跳转、支持 Range 续传和 gzip），
所以 `clone` 下来就能执行 `recommend` / `download` / `scan` / `list`。

**Web UI** 才需要两个包（`fastapi` + `uvicorn`）：

```bash
pip install -r requirements.txt       # 源码目录直接跑
pip install -e ".[web]"               # 或装成可编辑包，Web 依赖一并安装
```

`[web]` 是可选依赖：`pip install .` 只装零依赖 CLI。
Web 页面 `static/index.html` 已声明为包数据，wheel / sdist 均会随包分发。

不想敲命令的话，用仓库里的启动脚本：

| 系统 | 操作 | 说明 |
| --- | --- | --- |
| Windows | **双击 `start.bat`** | 自动在项目内建 `.venv` → 装 Web 依赖 → 起服务 → 打开浏览器 |
| Linux / macOS | `bash start.sh` | 同上 |

端口默认 `8765`，**被占用时会自动顺延到下一个空闲端口**（终端会打印实际地址）；
也可直接指定：`start.bat 9000` / `bash start.sh 9000`。
脚本内置 **清华 → 阿里 → 官方 PyPI** 三级镜像自动降级（国内直连 PyPI 常超时），
设置了 `PIP_INDEX_URL` 则只用你指定的源；依赖已装好时不会重复安装，`.venv` 已被 `.gitignore` 忽略。
不想自动弹浏览器时设 `LLD_NO_OPEN=1`。

## 使用

### Web UI（推荐）

```bash
python -m llm_downloader            # 默认 0.0.0.0:8765
python -m llm_downloader serve --open   # 启动后自动打开浏览器
python -m llm_downloader serve --port 9000
```

浏览器打开 `http://127.0.0.1:8765`，三个标签页：

- **模型推荐**：先看本机硬件档位，再按用途挑模型，一键下载。
- **本地模型库**：扫描目录、移除记录、彻底删除权重文件。
- **多源下载**：搜索关键词 → 选仓库/量化 → 看文件 → 开始下载（带 SSE 实时进度）。

### 命令行

```bash
# 硬件自适应推荐清单（可加 --role tools/chat/coding/vision/image）
python -m llm_downloader recommend
python -m llm_downloader recommend --offload 12   # 指定 GPU 卸载 12 层；也可填 auto/cpu/all

# 下载一个仓库（ref 支持 hf:/ms:/ollama: 前缀）
python -m llm_downloader download hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF --quant q4_k_m --register

# 扫描模型库目录认领新 GGUF
python -m llm_downloader scan

# 列出已登记模型
python -m llm_downloader list
```

## 设置

Web UI 第 4 个标签「设置」，保存后立即生效，无需重启：

| 项 | 说明 |
| --- | --- |
| 下载目录 | 语言模型 / 扩散模型分别可填绝对路径或 `~`；留空回退默认目录。切换目录不会搬动或删除已有模型 |
| 下载源与优先级 | 勾选启用、↑↓ 调整顺序；解析仓库时按序自动降级，至少保留一个源 |
| 镜像地址 | hf-mirror / ModelScope / Ollama Registry 可换成自建镜像；留空保存即恢复默认 |
| 分层加载 | 在「模型推荐」页选择模式并自动保存，随设置一起持久化 |

设置文件为 `$LLM_DOWNLOADER_HOME/settings.json`，也可脚本化修改：

```bash
curl -X POST http://127.0.0.1:8765/api/settings \
  -H "Content-Type: application/json" \
  -d '{"models_dir":"D:/models","sources":{"order":["hf-mirror","ollama"]}}'
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET / POST | `/api/settings` | 读取 / 部分更新设置（POST 只带要改的字段） |
| POST | `/api/settings/reset` | 恢复默认 |
| GET | `/api/advisor?offload_mode=...&offload_layers=...` | 推荐 + 分层加载评估 |
| POST | `/api/download` | 下载，SSE 先返回 `task` 事件带任务 ID |
| POST | `/api/download/cancel` | `{"id":"...","purge":false}` 停止（保留 `.part`）／`true` 取消（删除） |
| GET | `/api/download/tasks` | 活动任务与状态 |

> **口径说明**：分层加载的层数取自公开模型配置的**估算值**（未下载前读不到 GGUF 元数据），
> 显存/内存按层线性分摊，未计入长上下文 KV cache 的额外开销；结论用于选档，
> 不能替代实际加载后 `nvidia-smi` 的观测。

## 数据目录

模型默认落在 `$LLM_DOWNLOADER_HOME`（缺省 `~/.llm-downloader`）：

```
~/.llm-downloader/
├── settings.json    # 用户设置（下载目录 / 源顺序 / 镜像 / 分层策略）
├── models/          # 语言模型（GGUF），含 index.json 索引
├── image-models/    # 扩散模型（SD / FLUX），不进索引
├── bin/             # 预留二进制目录
├── cache/           # 源探测缓存 / 下载临时文件
└── logs/
```

可用环境变量覆盖根目录：

```bash
LLM_DOWNLOADER_HOME=/data/llm python -m llm_downloader
```

## 引用语法（ref）

下载接口接受一个仓库引用，支持前缀区分源：

| 写法 | 含义 |
| --- | --- |
| `hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 指定 hf-mirror 源 |
| `ms:modelscope/xxx` | 指定 ModelScope 源 |
| `ollama:qwen2.5:1.5b` | 指定 Ollama Registry |
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 不指定，按配置顺序自动选源 |

## 项目结构

```
llm_downloader/
├── __init__.py      # 包入口
├── config.py        # 配置 + settings.json 持久化（目录 / 源 / 镜像 / 分层策略）
├── http.py          # 标准库 HTTP 客户端（代理 / 重定向 / 流式 / Range），CLI 零依赖的关键
├── util.py          # 通用工具（human_size / sha256 / 硬件探测等）
├── sources/         # 多源抽象：hf-mirror / modelscope / ollama
│   ├── base.py
│   ├── hfmirror.py
│   ├── modelscope.py
│   └── ollama.py
├── downloader.py    # 下载引擎（多源降级 / 断点续传 / 测速 / 校验 / CancelToken）
├── library.py       # 本地模型库（扫描 / 索引 / 增删）
├── advisor.py       # 硬件自适应推荐（CATALOG + 打分 + 分层加载评估）
├── web.py           # FastAPI 路由（SSE / 设置 / 任务注册表与取消 / 推荐 / 模型库）
├── main.py          # CLI 入口（serve / recommend / download / scan / list）
└── static/index.html  # Web UI（推荐 / 模型库 / 下载 三视图）

start.bat / start.sh   # 一键启动 Web UI（自动建虚拟环境 + 装依赖）
```

依赖一览：**CLI 无第三方依赖**；`fastapi` + `uvicorn` 仅供 Web UI 使用。

## 许可

[MIT](LICENSE)
