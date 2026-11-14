"""hf-mirror.com 源（HuggingFace 的国内镜像）。

API 形态（实测可用）：
    GET {endpoint}/api/models/{repo}?blobs=true
        -> {"siblings": [{"rfilename": "...gguf", "size": 1117320736, ...}]}
    GET {endpoint}/{repo}/resolve/{revision}/{filename}
        -> 302 到 CDN，客户端自动跟随重定向
"""

from __future__ import annotations

from llm_downloader.sources.base import Candidate, ModelRef, RemoteFile, Source, register


def _role_of(filename: str) -> str:
    low = filename.lower()
    if low.endswith(".gguf"):
        if "mmproj" in low or "projector" in low or ("vision" in low and "encoder" in low):
            return "mmproj"
        if "mmproj" in low:
            return "mmproj"
        return "weights"
    # 非 GGUF 权重格式：safetensors / pytorch bin / onnx。
    if low.endswith((".safetensors", ".pt", ".pth", ".onnx")):
        return "weights-nongguf"
    if low.endswith((".json", ".txt", ".md")):
        return "config"
    if low.endswith((".jinja", ".template")):
        return "template"
    return "other"


@register
class HfMirrorSource(Source):
    name = "hf-mirror"
    kind = "model"

    def probe(self) -> tuple[bool, str]:
        ep = self.cfg.sources.hf_endpoint
        try:
            r = self.http.get(f"{ep}/api/models/gpt2", timeout=10.0)
            if r.status_code == 200:
                return True, f"OK ({ep})"
            return False, f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def _api(self, repo: str) -> dict:
        ep = self.cfg.sources.hf_endpoint
        r = self.http.get(f"{ep}/api/models/{repo}", params={"blobs": "true"})
        r.raise_for_status()
        return r.json()

    def list_files(self, ref: ModelRef) -> list[RemoteFile]:
        data = self._api(ref.repo)
        files: list[RemoteFile] = []
        for sib in data.get("siblings") or []:
            fn = sib.get("rfilename") or ""
            if not fn:
                continue
            lfs = sib.get("lfs") or {}
            files.append(
                RemoteFile(
                    filename=fn,
                    size=sib.get("size") or lfs.get("size"),
                    sha256=(lfs.get("sha256") or sib.get("sha256") or None),
                    source=self.name,
                    role=_role_of(fn),
                    extra={"repo": ref.repo, "revision": ref.revision},
                )
            )
        return files

    def download_candidates(self, ref: ModelRef, filename: str) -> list[Candidate]:
        ep = self.cfg.sources.hf_endpoint
        urls = [
            f"{ep}/{ref.repo}/resolve/{ref.revision}/{filename}?download=true",
            f"{ep}/{ref.repo}/resolve/{ref.revision}/{filename}",
        ]
        return [Candidate(url=u, source=self.name, filename=filename) for u in urls]

    def repo_info(self, repo: str) -> dict:
        d = self._api(repo)
        return {
            "id": d.get("id") or repo,
            "downloads": d.get("downloads"),
            "likes": d.get("likes"),
            "pipeline_tag": d.get("pipeline_tag"),
            "tags": d.get("tags") or [],
            "lastModified": d.get("lastModified"),
        }
