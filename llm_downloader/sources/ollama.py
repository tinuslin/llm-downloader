"""Ollama registry 源 —— 直接从 registry.ollama.ai 抽 GGUF。

实测可用。关键点：manifest 里 mediaType 为
    application/vnd.ollama.image.model      -> 权重（就是裸 GGUF）
    application/vnd.ollama.image.projector  -> 视觉投影层（VLM 的 mmproj）
的 layer 都能通过
    GET {registry}/v2/{repo}/blobs/{digest}
直接下载。也就是说 **不用装 Ollama 也能拿到 GGUF**。

注意：registry 会返回 307 跳转到 CDN，需要 follow_redirects。
"""

from __future__ import annotations

from llm_downloader.sources.base import Candidate, ModelRef, RemoteFile, Source, register
from llm_downloader.util import human_size

MEDIA_MODEL = "application/vnd.ollama.image.model"
MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_TEMPLATE = "application/vnd.ollama.image.template"
MEDIA_SYSTEM = "application/vnd.ollama.image.system"
MEDIA_PARAMS = "application/vnd.ollama.image.params"
MEDIA_LICENSE = "application/vnd.ollama.image.license"


@register
class OllamaSource(Source):
    name = "ollama"
    kind = "model"

    def probe(self) -> tuple[bool, str]:
        ep = self.cfg.sources.ollama_registry
        try:
            r = self.http.get(f"{ep}/v2/library/qwen2.5/manifests/1.5b", timeout=10.0)
            if r.status_code == 200:
                return True, f"OK ({ep})"
            if r.status_code == 401:
                # 有些仓库需要匿名 token；能拿到 401 说明域名通
                return True, f"可达但有鉴权 (HTTP 401)"
            return False, f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def manifest(self, ref: ModelRef) -> dict:
        ep = self.cfg.sources.ollama_registry
        r = self.http.get(
            f"{ep}/v2/{ref.repo}/manifests/{ref.revision}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        )
        r.raise_for_status()
        return r.json()

    def list_files(self, ref: ModelRef) -> list[RemoteFile]:
        m = self.manifest(ref)
        out: list[RemoteFile] = []
        for layer in m.get("layers") or []:
            mt = layer.get("mediaType") or ""
            digest = layer.get("digest") or ""
            size = layer.get("size")
            long = digest.replace("sha256:", "")[:12]
            if mt == MEDIA_MODEL:
                out.append(
                    RemoteFile(
                        filename="model.gguf",
                        size=size,
                        sha256=digest.replace("sha256:", ""),
                        source=self.name,
                        role="weights",
                        extra={"digest": digest, "mediaType": mt},
                    )
                )
            elif mt == MEDIA_PROJECTOR:
                out.append(
                    RemoteFile(
                        filename="mmproj.gguf",
                        size=size,
                        sha256=digest.replace("sha256:", ""),
                        source=self.name,
                        role="mmproj",
                        extra={"digest": digest, "mediaType": mt},
                    )
                )
            else:
                role = {
                    MEDIA_TEMPLATE: "template",
                    MEDIA_SYSTEM: "config",
                    MEDIA_PARAMS: "config",
                    MEDIA_LICENSE: "config",
                }.get(mt, "other")
                out.append(
                    RemoteFile(
                        filename=f"{role}-{long}.txt",
                        size=size,
                        sha256=digest.replace("sha256:", ""),
                        source=self.name,
                        role=role,
                        extra={"digest": digest, "mediaType": mt},
                    )
                )
        return out

    def download_candidates(self, ref: ModelRef, filename: str) -> list[Candidate]:
        m = self.manifest(ref)
        ep = self.cfg.sources.ollama_registry
        # 反查 digest：优先按角色匹配
        want_role = "weights"
        if filename == "mmproj.gguf":
            want_role = "mmproj"
        target = None
        for layer in m.get("layers") or []:
            mt = layer.get("mediaType") or ""
            if mt == MEDIA_MODEL and want_role == "weights":
                target = layer
                break
            if mt == MEDIA_PROJECTOR and want_role == "mmproj":
                target = layer
                break
            long = (layer.get("digest") or "").replace("sha256:", "")[:12]
            if want_role not in ("weights", "mmproj") and long in filename:
                target = layer
                break
        if target is None:
            # 退化为按 manifest 顺序
            layers = m.get("layers") or []
            target = layers[0] if layers else None
        if target is None:
            return []
        digest = target.get("digest")
        return [
            Candidate(
                url=f"{ep}/v2/{ref.repo}/blobs/{digest}",
                source=self.name,
                filename=filename,
                size=target.get("size"),
                sha256=digest.replace("sha256:", ""),
            )
        ]

    def describe(self, ref: ModelRef) -> dict:
        m = self.manifest(ref)
        info = {"repo": ref.repo, "tag": ref.revision, "layers": []}
        for layer in m.get("layers") or []:
            info["layers"].append(
                {
                    "mediaType": layer.get("mediaType"),
                    "size": layer.get("size"),
                    "size_h": human_size(layer.get("size")),
                    "digest": (layer.get("digest") or "")[:20],
                }
            )
        return info
