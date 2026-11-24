"""
The safe downloader with a fake web (httpx.MockTransport) and a fake DNS: no network.
Most tests are about what must be REFUSED: private addresses, bad schemes/ports, sneaky redirects,
huge files, and HTML pages pretending to be PDFs.
"""

import httpx
import pytest

from app.domain.errors import DownloadError
from app.integrations.downloader import SafeDownloader, sniff

PDF = b"%PDF-1.7\n" + b"x" * 1000 + b"\n%%EOF"
XML = b'<?xml version="1.0"?>\n<!DOCTYPE article PUBLIC "-//NLM//DTD JATS">\n<article></article>'
HTML = b"<!DOCTYPE html><html><body>Please log in</body></html>"

# fake DNS: host -> addresses
DNS = {"papers.example.org": ["93.184.216.34"], "mirror.example.org": ["93.184.216.35"],
       "internal.example.org": ["10.0.0.5"], "metadata.example.org": ["169.254.169.254"],
       "localhost": ["127.0.0.1", "::1"], "mixed.example.org": ["93.184.216.36", "192.168.1.10"]}


async def fake_resolver(host: str, port: int) -> list[str]:
    if host in DNS:
        return DNS[host]
    try:                                                   # a literal IP address resolves to itself
        import ipaddress
        return [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        raise OSError(f"unknown host {host}") from None


def downloader(routes: dict, *, max_bytes: int = 1_000_000, calls: list | None = None) -> SafeDownloader:
    """routes: url -> httpx.Response (or a function returning one)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        route = routes.get(str(request.url))
        if route is None:
            return httpx.Response(404)
        return route() if callable(route) else route

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SafeDownloader(client, max_bytes=max_bytes, timeout_s=5, resolver=fake_resolver)


# ---------- what should work ----------

async def test_downloads_a_pdf():
    result = await downloader({"https://papers.example.org/a.pdf": httpx.Response(200, content=PDF)}) \
        .download("https://papers.example.org/a.pdf", "pdf")
    assert result.data == PDF and result.kind == "pdf" and result.content_type == "application/pdf"


async def test_downloads_jats_xml():
    result = await downloader({"https://papers.example.org/a.xml": httpx.Response(200, content=XML)}) \
        .download("https://papers.example.org/a.xml", "xml")
    assert result.kind == "xml" and result.content_type == "application/xml"


async def test_follows_a_relative_redirect_to_a_public_host():
    routes = {"https://papers.example.org/doi/1": httpx.Response(302, headers={"location": "/files/1.pdf"}),
              "https://papers.example.org/files/1.pdf": httpx.Response(200, content=PDF)}
    result = await downloader(routes).download("https://papers.example.org/doi/1", "pdf")
    assert result.final_url == "https://papers.example.org/files/1.pdf"


# ---------- SSRF guard ----------

@pytest.mark.parametrize("url, reason", [
    ("https://internal.example.org/a.pdf", "non-public address 10.0.0.5"),        # private network
    ("http://metadata.example.org/latest/", "non-public address 169.254.169.254"),  # cloud secrets
    ("http://localhost/a.pdf", "non-public address"),                              # this machine
    ("http://127.0.0.1/a.pdf", "non-public address 127.0.0.1"),
    ("http://[::1]/a.pdf", "non-public address ::1"),
    ("https://mixed.example.org/a.pdf", "non-public address 192.168.1.10"),        # ONE bad address is enough
    ("file:///etc/passwd", "scheme 'file' not allowed"),
    ("ftp://papers.example.org/a.pdf", "scheme 'ftp' not allowed"),
    ("https://papers.example.org:5433/a.pdf", "port 5433 not allowed"),            # e.g. our own Postgres
    ("https://user:pw@papers.example.org/a.pdf", "credentials in URL"),
    ("https://nowhere.invalid/a.pdf", "cannot resolve host"),
])
async def test_blocked_urls_never_get_a_request(url, reason):
    calls: list = []
    with pytest.raises(DownloadError, match=reason):
        await downloader({}, calls=calls).download(url, "pdf")
    assert calls == []                                      # refused BEFORE any request was sent


async def test_redirect_to_a_private_address_is_blocked():
    calls: list = []
    routes = {"https://papers.example.org/a.pdf":
              httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})}
    with pytest.raises(DownloadError, match="non-public address 169.254.169.254"):
        await downloader(routes, calls=calls).download("https://papers.example.org/a.pdf", "pdf")
    assert calls == ["https://papers.example.org/a.pdf"]   # the private address was never contacted


async def test_redirect_loop_is_stopped():
    routes = {"https://papers.example.org/loop": httpx.Response(302, headers={"location": "/loop"})}
    with pytest.raises(DownloadError, match="more than 5 redirects"):
        await downloader(routes).download("https://papers.example.org/loop", "pdf")


# ---------- size, status and type ----------

async def test_declared_size_over_the_limit_is_refused():
    routes = {"https://papers.example.org/big.pdf":
              httpx.Response(200, headers={"content-length": "999999999"}, content=PDF)}
    with pytest.raises(DownloadError, match="too large"):
        await downloader(routes).download("https://papers.example.org/big.pdf", "pdf")


async def test_actual_size_over_the_limit_is_refused_while_streaming():
    routes = {"https://papers.example.org/big.pdf": httpx.Response(200, content=PDF * 20)}   # no honest length
    with pytest.raises(DownloadError, match="too large"):
        await downloader(routes, max_bytes=5_000).download("https://papers.example.org/big.pdf", "pdf")


async def test_http_error_status():
    routes = {"https://papers.example.org/a.pdf": httpx.Response(403)}
    with pytest.raises(DownloadError, match="HTTP 403"):
        await downloader(routes).download("https://papers.example.org/a.pdf", "pdf")


async def test_html_login_page_instead_of_pdf_is_refused():
    routes = {"https://papers.example.org/a.pdf":
              httpx.Response(200, headers={"content-type": "application/pdf"}, content=HTML)}   # lying header
    with pytest.raises(DownloadError, match="expected pdf, got an HTML page"):
        await downloader(routes).download("https://papers.example.org/a.pdf", "pdf")


def test_sniff():
    assert sniff(PDF) == "pdf"
    assert sniff(b"\r\n junk %PDF-1.4") == "pdf"           # a little junk before the header is allowed
    assert sniff(XML) == "xml"
    assert sniff(b"\xef\xbb\xbf<article>x</article>") == "xml"
    assert sniff(HTML) is None
    assert sniff(b"PK\x03\x04 zip file") is None
