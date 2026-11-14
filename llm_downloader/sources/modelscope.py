"""ModelScope（魔搭）源。

API 形态（接口按官方 v1 版本实现；不可达时自动跳过）：
    GET {endpoint}/api/v1/models/{repo}/repo/files?Revision={rev}&Recursive=true
        -> {"Data": {"Files": [{"Path": "xxx.gguf", "Size": 123, "Sha256": "..."}]}}
    GET {endpoint}/api/v1/models/{repo}/repo?Revision={rev}&FilePath={path}
        -> 302 到对象存储直链
"""

from __future__ import annotations

from llm_downloader.sources.base import Candidate, ModelRef, RemoteFile, Source, register
from llm_downloader.sources.hfmirror import _role_of


@register
class ModelScopeSource(Source):
    name = "modelscope"
    kind = "model"

    def probe(self) -> tuple[bool, str]:
        ep = self.cfg.sources.modelscope_endpoint
        try:
            r = self.http.get(f"{ep}/api/v1/models/Qwen/Qwen2.5-1.5B-Instruct", timeout=10.0)
            if r.status_code == 200:
                return True, f"OK ({ep})"
            return False, f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def list_files(self, ref: ModelRef) -> list[RemoteFile]:
        ep = self.cfg.sources.modelscope_endpoint
        r = self.http.get(
            f"{ep}/api/v1/models/{ref.repo}/repo/files",
            params={"Revision": ref.revision, "Recursive": "true"},
        )
        r.raise_for_status()
        data = (r.json() or {}).get("Data") or {}
        out: list[RemoteFile] = []
        for f in data.get("Files") or []:
            path = f.get("Path") or f.get("Name") or ""
            if not path or f.get("Type") == "tree":
                continue
            out.append(
                RemoteFile(
                    filename=path,
                    size=f.get("Size"),
                    sha256=f.get("Sha256") or f.get("sha256"),
                    source=self.name,
                    role=_role_of(path),
                    extra={"repo": ref.repo, "revision": ref.revision},
                )
            )
        return out

    def download_candidates(self, ref: ModelRef, filename: str) -> list[Candidate]:
        ep = self.cfg.sources.modelscope_endpoint
        return [
            Candidate(
                url=self.build_url(ep, ref.repo, ref.revision, filename),
                source=self.name,
                filename=filename,
                note="v1 文件接口",
            )
        ]

    @staticmethod
    def build_url(ep: str, repo: str, revision: str, filename: str) -> str:
        from urllib.parse import quote

        return (
            f"{ep}/api/v1/models/{repo}/repo"
            f"?Revision={quote(revision)}&FilePath={quote(filename)}"
        )
