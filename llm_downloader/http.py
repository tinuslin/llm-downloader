"""极简 HTTP 客户端：只用 Python 标准库，保持 CLI「零第三方依赖」。

为什么自己写：下载器的核心痛点在国内网络环境（代理、302/CDN 跳转、Range 续传），
为此引入 httpx 会让「clone 下来直接 python -m llm_downloader」这一步就崩。
这里实现 httpx 风格的**最小子集**：

    Client(timeout=Timeout(...), follow_redirects=True, headers={...})
      .get(url, params=..., headers=..., timeout=...) -> Response
      .stream("GET", url, headers=...)               -> Response (可作 with 上下文)
    Response: .status_code / .headers / .iter_bytes(n) / .json() / .raise_for_status()

要点
- 自动继承系统代理（HTTP_PROXY / HTTPS_PROXY），但访问 127.0.0.1 / localhost 时绕过代理，
  避免「本机服务被代理劫持」这类经典问题。
- 4xx/5xx 不抛异常，照样返回 Response（416 用于判断续传已完成），由调用方决定。
- 显式 Accept-Encoding: identity，仍对 gzip/deflate 做兜底解压。
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

UA = "llm-downloader/0.1 (+local)"


class HTTPError(Exception):
    """网络层错误（与 urllib 的 HTTPError 区分，避免上层误判）。"""


@dataclass
class Timeout:
    connect: float = 10.0
    read: float = 60.0
    write: float = 60.0
    pool: float = 10.0

    def socket_timeout(self) -> float:
        # urllib 只有一个 socket 级超时，用 read 为准，并保证不小于 5s
        return max(5.0, float(self.read or 60.0))


class _Headers:
    """大小写不敏感的只读头视图。"""

    def __init__(self, src: Any):
        self._m: dict[str, str] = {}
        try:
            items = src.items()
        except Exception:  # noqa: BLE001
            items = []
        for k, v in items:
            self._m[str(k).lower()] = str(v)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._m.get(key.lower(), default)

    def __getitem__(self, key: str) -> str:
        return self._m[key.lower()]

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._m


class Response:
    def __init__(self, raw: Any, url: str, status: int | None = None):
        self._raw = raw
        self.url = getattr(raw, "url", url) or url
        self.status_code = int(status if status is not None else getattr(raw, "status", 200))
        self.headers = _Headers(getattr(raw, "headers", None))
        enc = (self.headers.get("content-encoding") or "").lower()
        self._dec: Any = None
        if "gzip" in enc:
            self._dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif "deflate" in enc:
            self._dec = zlib.decompressobj()

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        dec = self._dec
        while True:
            try:
                chunk = self._raw.read(chunk_size)
            except Exception as e:  # noqa: BLE001
                raise HTTPError(f"读取响应失败：{type(e).__name__}: {e}") from e
            if not chunk:
                break
            if dec is not None:
                chunk = dec.decompress(chunk)
            if chunk:
                yield chunk
        if dec is not None:
            tail = dec.flush()
            if tail:
                yield tail

    def read(self) -> bytes:
        return b"".join(self.iter_bytes())

    @property
    def text(self) -> str:
        ctype = self.headers.get("content-type") or ""
        charset = "utf-8"
        for part in ctype.split(";"):
            p = part.strip().lower()
            if p.startswith("charset="):
                charset = p.split("=", 1)[1].strip('"\'') or "utf-8"
        return self.read().decode(charset, errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}: {self.url}")

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _is_loopback(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1")


class Client:
    def __init__(
        self,
        timeout: Timeout | float | None = None,
        follow_redirects: bool = True,
        headers: Mapping[str, str] | None = None,
        trust_env: bool = True,
    ):
        if isinstance(timeout, Timeout):
            self.timeout = timeout
        elif timeout:
            self.timeout = Timeout(read=float(timeout), connect=min(10.0, float(timeout)))
        else:
            self.timeout = Timeout()
        self.follow_redirects = follow_redirects
        self.trust_env = trust_env
        self.headers: dict[str, str] = {
            "User-Agent": UA,
            "Accept-Encoding": "identity",
        }
        if headers:
            self.headers.update({str(k): str(v) for k, v in headers.items()})
        self._openers: dict[bool, Any] = {}

    def _opener(self, url: str) -> Any:
        direct = (not self.trust_env) or _is_loopback(url)
        if direct in self._openers:
            return self._openers[direct]
        handlers: list[Any] = []
        if direct:
            handlers.append(urllib.request.ProxyHandler({}))
        else:
            handlers.append(urllib.request.ProxyHandler(urllib.request.getproxies()))
        if not self.follow_redirects:
            handlers.append(_NoRedirect())
        handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        op = urllib.request.build_opener(*handlers)
        self._openers[direct] = op
        return op

    @staticmethod
    def _with_params(url: str, params: Mapping[str, Any] | None) -> str:
        if not params:
            return url
        sep = "&" if "?" in url else "?"
        return url + sep + urllib.parse.urlencode(params, doseq=True)

    def request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Response:
        full = self._with_params(url, params)
        hdrs = dict(self.headers)
        if headers:
            hdrs.update({str(k): str(v) for k, v in headers.items()})
        req = urllib.request.Request(full, headers=hdrs, method=method.upper())
        t = float(timeout) if timeout else self.timeout.socket_timeout()
        try:
            raw = self._opener(full).open(req, timeout=t)
        except urllib.error.HTTPError as e:
            # 4xx/5xx 也是响应（416 = 续传已完成）
            return Response(e, full, status=e.code)
        except urllib.error.URLError as e:
            raise HTTPError(f"网络不可达：{e.reason}") from e
        except Exception as e:  # noqa: BLE001
            raise HTTPError(f"请求失败：{type(e).__name__}: {e}") from e
        return Response(raw, full)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Response:
        return self.request("GET", url, params=params, headers=headers, timeout=timeout)

    def stream(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Response:
        return self.request(method, url, params=params, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._openers.clear()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
