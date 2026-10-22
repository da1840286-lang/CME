"""Shared HTTP helpers for adapters. The transport is injectable so tests never touch the network."""
from __future__ import annotations
import base64
import json
import mimetypes
from pathlib import Path
from ..providers import InfraError, ProviderCallError


def urllib_transport(method, url, headers, body, timeout=180):
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise InfraError(f"network: {e}") from e


def data_uri(path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def check_status(name: str, status: int, raw: bytes) -> None:
    if status == 429 or status >= 500:
        raise InfraError(f"{name} HTTP {status}")
    if status >= 400:
        raise ProviderCallError(f"{name} HTTP {status}: {raw[:200]!r}")
