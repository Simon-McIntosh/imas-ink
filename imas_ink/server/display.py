"""HTTP display server for efit.ink charts.

Runs a lightweight HTTP server in a background daemon thread that serves
the most recently pushed chart HTML at http://localhost:PORT/. A
Server-Sent Events endpoint at ``/events`` notifies connected browsers
to reload when a new chart is pushed — no polling required.

Typical usage (remote SSH session)::

    from efit.ink.display import push_chart

    chart = render_alt(TimeSeries(...))
    url = push_chart(chart)          # starts server on first call
    print(url)                       # http://localhost:8766/

The server only occupies port 8766 when ``push_chart`` (or
``get_server``) is first called — not at import time.

SSH tunnel
----------
Forward port 8766 from the remote host to your laptop before calling
``push_chart``::

    imas-codex tunnel start iter --ink     # if --ink is available
    ssh -L 8766:localhost:8766 iter        # or manually

Then open http://localhost:8766/ in your browser.  Every subsequent
``push_chart`` call reloads the page automatically.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT: int = 8766

# ---------------------------------------------------------------------------
# Empty-page HTML (defined before _DisplayState uses it as default)
# ---------------------------------------------------------------------------

_EMPTY_PAGE: str = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>efit.ink display</title>
  <style>
    body {
      margin: 0;
      background: #1a1a1a;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      color: #888;
      font-family: sans-serif;
      font-size: 1.1rem;
    }
  </style>
</head>
<body>
  <p>Waiting for chart&hellip;</p>
  <script>
    const es = new EventSource('/events');
    es.onmessage = e => { if (e.data === 'update') location.reload(); };
    es.onerror   = () => setTimeout(() => { es.close(); location.reload(); }, 3000);
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Thread-safe state store
# ---------------------------------------------------------------------------


class _DisplayState:
    """Thread-safe container for the current chart HTML.

    Uses a ``threading.Condition`` so SSE handler threads block until a
    new chart is pushed, then wake immediately.
    """

    _cond: threading.Condition = threading.Condition()
    _html: str = _EMPTY_PAGE
    _version: int = 0

    @classmethod
    def set(cls, html: str) -> None:
        with cls._cond:
            cls._html = html
            cls._version += 1
            cls._cond.notify_all()

    @classmethod
    def get(cls) -> tuple[str, int]:
        with cls._cond:
            return cls._html, cls._version

    @classmethod
    def wait_for_version(cls, known_version: int, timeout: float = 15.0) -> int:
        """Block until ``_version > known_version`` or *timeout* expires.

        Returns the current version number (which may equal *known_version*
        on timeout — callers use this to distinguish update vs. heartbeat).
        """
        deadline = time.monotonic() + timeout
        with cls._cond:
            while cls._version <= known_version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return cls._version
                cls._cond.wait(timeout=remaining)
            return cls._version


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Handle chart-page and SSE requests."""

    def log_message(self, format: str, *args: object) -> None:  # silence access log
        pass

    def do_GET(self) -> None:
        if self.path == "/events":
            self._serve_events()
        elif self.path in ("/", "/index.html"):
            self._serve_chart()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/push":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            _DisplayState.set(body)
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_chart(self) -> None:
        html, _ = _DisplayState.get()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        """Stream Server-Sent Events until the client disconnects."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        _, known_version = _DisplayState.get()
        try:
            while True:
                new_version = _DisplayState.wait_for_version(known_version)
                if new_version > known_version:
                    self.wfile.write(b"data: update\n\n")
                    self.wfile.flush()
                    known_version = new_version
                else:
                    # heartbeat — keeps the connection alive through proxies
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


# ---------------------------------------------------------------------------
# Display server
# ---------------------------------------------------------------------------


class DisplayServer:
    """Background HTTP server that holds and serves the latest ink chart.

    Parameters
    ----------
    port : int
        TCP port to listen on (default :data:`DEFAULT_PORT`).
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}/"

    def start(self) -> None:
        """Start the server in a daemon background thread.

        Raises
        ------
        OSError
            If *port* is already in use.  The error message explains the
            likely cause and remediation.
        """
        if self._thread and self._thread.is_alive():
            return
        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), _Handler)
        except OSError as exc:
            raise OSError(
                f"Port {self._port} is already in use — check for another "
                "efit-ink process or a stale socket.\n"
                f"  Original error: {exc}"
            ) from exc

        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="ink-display"
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the server (blocks until the serving thread exits)."""
        if self._httpd is not None:
            self._httpd.shutdown()

    def push(self, html: str) -> str:
        """Push new chart HTML and return the server URL."""
        _DisplayState.set(html)
        return self.url

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_server: DisplayServer | None = None
_server_lock: threading.Lock = threading.Lock()


def get_server(port: int = DEFAULT_PORT) -> DisplayServer:
    """Return the module-level :class:`DisplayServer`, starting it if needed.

    Thread-safe: a ``threading.Lock`` guards the lazy-initialisation so
    two concurrent calls cannot bind the port twice.

    Parameters
    ----------
    port : int
        Server port.  Ignored if the server is already running.

    Returns
    -------
    DisplayServer
    """
    global _server
    with _server_lock:
        if _server is None or not _server.is_running:
            _server = DisplayServer(port=port)
            _server.start()
    return _server


def push_chart(html_or_chart: object, port: int = DEFAULT_PORT) -> str:
    """Push a chart to the display server and return its URL.

    Starts the server on first call (lazy initialisation).

    Parameters
    ----------
    html_or_chart
        Either a raw HTML string *or* any object with a ``.to_html()``
        method (e.g. an Altair ``Chart``).
    port : int
        Server port.  Forwarded to :func:`get_server`.

    Returns
    -------
    str
        URL where the chart is now served, e.g. ``"http://localhost:8766/"``.

    Examples
    --------
    >>> from efit.ink import render_alt, TimeSeries, push_chart
    >>> chart = render_alt(TimeSeries(time, ip, ylabel="Ip", units="MA"))
    >>> url = push_chart(chart)
    >>> print(url)
    http://localhost:8766/
    """
    if hasattr(html_or_chart, "to_html"):
        html: str = html_or_chart.to_html()  # type: ignore[union-attr]
    else:
        html = str(html_or_chart)

    url = f"http://localhost:{port}/"

    # Try to POST to an already-running server first (cross-process safe).
    try:
        import urllib.request

        data = html.encode("utf-8")
        req = urllib.request.Request(
            f"http://localhost:{port}/push",
            data=data,
            headers={"Content-Type": "text/html; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 204:
                return url
    except Exception:
        pass  # No server running — start one in this process

    server = get_server(port=port)
    return server.push(html)
