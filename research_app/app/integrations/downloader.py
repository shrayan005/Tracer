"""
Safe downloader (integration layer, step 4.3): fetch a full-text file from a URL we did not choose.

The URLs come from outside (APIs today, users tomorrow), so every download is treated as untrusted:

  SSRF guard   A URL like http://169.254.169.254/ (cloud server secrets) or http://localhost:5433
               would make OUR server fetch something private. So before EVERY request, including
               each redirect hop, the host is resolved and ALL its addresses must be public internet
               addresses. Only http/https on ports 80/443. Redirects are followed by hand (max 5),
               because an automatic redirect could jump to a private address unchecked.
  Size limit   streamed in pieces; stops as soon as it passes the limit (a 5 GB "PDF" is refused
               without ever being held in memory).
  Time limit   one deadline for the whole download, redirects included.
  Type check   the bytes must really be what we expected: a PDF starts with "%PDF-", JATS XML with
               "<?xml"/"<article". Publishers often answer with an HTML login page instead.

Known limit (documented, not hidden): the address is checked, then httpx resolves the name again
to connect. A hostile DNS server could answer differently the second time ("DNS rebinding").
Fully closing that needs connecting to the checked IP itself; for public paper sites the risk is low.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlsplit

import httpx

from app.domain.errors import DownloadError
from app.integrations.sources.base import USER_AGENT

Kind = Literal["pdf", "xml"]
Resolver = Callable[[str, int], Awaitable[list[str]]]     # (host, port) -> IP addresses

ALLOWED_PORTS = {"http": 80, "https": 443}
MAX_REDIRECTS = 5
CONTENT_TYPES = {"pdf": "application/pdf", "xml": "application/xml"}


@dataclass
class Downloaded:
    data: bytes
    kind: Kind
    content_type: str
    final_url: str                    # after redirects (what we actually fetched)


async def system_resolver(host: str, port: int) -> list[str]:
    """Ask the operating system's DNS for all addresses of a host (async, does not block the loop)."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list({info[4][0] for info in infos})


class SafeDownloader:
    def __init__(self, client: httpx.AsyncClient, *, max_bytes: int = 50 * 1024 * 1024,
                 timeout_s: float = 60.0, resolver: Resolver = system_resolver):
        self.client = client              # the SHARED async HTTP client (redirects handled here, by hand)
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.resolver = resolver          # injectable: tests use a fake DNS

    async def download(self, url: str, expect: Kind) -> Downloaded:
        try:
            return await asyncio.wait_for(self._download(url, expect), timeout=self.timeout_s)
        except TimeoutError:
            raise DownloadError(url, f"timed out after {self.timeout_s:.0f}s") from None
        except httpx.HTTPError as e:
            raise DownloadError(url, f"network error: {type(e).__name__}") from e

    async def _download(self, url: str, expect: Kind) -> Downloaded:
        for _ in range(MAX_REDIRECTS + 1):
            await self._check_url(url)
            async with self.client.stream("GET", url, follow_redirects=False,
                                          headers={"User-Agent": USER_AGENT}) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DownloadError(url, "redirect without a location")
                    url = urljoin(url, location)              # relative redirects are allowed
                    continue                                  # the new URL is checked at the loop start
                if response.status_code != 200:
                    raise DownloadError(url, f"HTTP {response.status_code}")
                data = await self._read_limited(response, url)
            kind = sniff(data)
            if kind != expect:
                found = kind or ("an HTML page" if looks_like_html(data) else "an unknown file type")
                raise DownloadError(url, f"expected {expect}, got {found}")
            return Downloaded(data=data, kind=kind, content_type=CONTENT_TYPES[kind], final_url=url)
        raise DownloadError(url, f"more than {MAX_REDIRECTS} redirects")

    async def _check_url(self, url: str) -> None:
        """The SSRF guard: scheme, port, and every resolved address must be public."""
        parts = urlsplit(url)
        if parts.scheme not in ALLOWED_PORTS:
            raise DownloadError(url, f"scheme '{parts.scheme}' not allowed")
        if not parts.hostname:
            raise DownloadError(url, "no host in URL")
        if parts.username or parts.password:
            raise DownloadError(url, "credentials in URL not allowed")
        port = parts.port or ALLOWED_PORTS[parts.scheme]
        if port not in ALLOWED_PORTS.values():
            raise DownloadError(url, f"port {port} not allowed")
        try:
            addresses = await self.resolver(parts.hostname, port)
        except OSError as e:
            raise DownloadError(url, f"cannot resolve host: {e}") from e
        if not addresses:
            raise DownloadError(url, "host has no addresses")
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%")[0])          # strip an IPv6 zone like %eth0
            if not ip.is_global or ip.is_multicast:
                raise DownloadError(url, f"blocked non-public address {ip}")

    async def _read_limited(self, response: httpx.Response, url: str) -> bytes:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            raise DownloadError(url, f"too large ({int(declared) // 1_000_000} MB declared)")
        chunks, total = [], 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.max_bytes:                    # the declared size can lie: count ourselves
                raise DownloadError(url, f"too large (over {self.max_bytes // 1_000_000} MB)")
            chunks.append(chunk)
        return b"".join(chunks)


def sniff(data: bytes) -> Kind | None:
    """What the bytes really are, from their first bytes (never trust the server's Content-Type)."""
    if b"%PDF-" in data[:1024]:                           # the PDF spec allows a little junk before it
        return "pdf"
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()     # skip a BOM and whitespace
    if looks_like_html(data):
        return None
    if head.startswith(b"<?xml") or head.startswith(b"<!doctype article") or head.startswith(b"<article"):
        return "xml"
    return None


def looks_like_html(data: bytes) -> bool:
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:300]
