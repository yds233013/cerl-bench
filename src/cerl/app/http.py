"""A thin HTTP adapter. Standard library only.

Thin is the requirement and the design: the project carries three runtime
dependencies and a web framework would be the largest thing in it. This is a
router over ``http.server`` -- enough to serve JSON to a local frontend and
nothing more.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

Handler = Callable[[dict[str, str], dict[str, Any]], tuple[int, Any]]


class Router:
    """Method + path-pattern to handler. Patterns use ``{name}`` segments."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, re.Pattern[str], Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        regex = re.compile(
            "^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$",
        )
        self._routes.append((method, regex, handler))

    def resolve(
        self, method: str, path: str,
    ) -> tuple[Handler, dict[str, str]] | None:
        for route_method, regex, handler in self._routes:
            if route_method != method:
                continue
            match = regex.match(path)
            if match:
                return handler, match.groupdict()
        return None


#: Prefixes reserved for APIs. A request under one of these is answered by the
#: router or refused -- never by the single-page-app fallback. Otherwise this
#: server would return 200 and an HTML shell for ``/review/episodes``, which
#: reads as "the privileged API is here" when it is deliberately not.
API_PREFIXES: tuple[str, ...] = ("/api/", "/review/")

#: Extensions the server will hand back, and how to label them. Anything else
#: is refused rather than guessed at.
CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".woff2": "font/woff2",
    ".map": "application/json",
}


class StaticSite:
    """Serves a built single-page app from one directory.

    Present so the whole workspace is one command and one process. It resolves
    every request under the root and refuses anything that escapes it, because
    ``..`` in a URL should not be able to read the repository even on loopback.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> tuple[bytes, str] | None:
        relative = path.lstrip("/") or "index.html"
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            # Unknown path inside a single-page app means a client-side route,
            # so fall back to the shell rather than 404.
            candidate = (self.root / "index.html").resolve()
            if not candidate.is_file():
                return None
        suffix = candidate.suffix.lower()
        if suffix not in CONTENT_TYPES:
            return None
        return candidate.read_bytes(), CONTENT_TYPES[suffix]


class ApiError(Exception):
    """A handler's deliberate error response, with a status the client can act on."""

    def __init__(self, status: int, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


def make_handler(
    router: Router, name: str, site: StaticSite | None = None,
) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        server_version = f"cerl-{name}"

        def log_message(self, *_args: Any, **_kwargs: Any) -> None:
            """Silenced: the CLI prints one line per action, which is enough."""

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # The frontend dev server runs on another port.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send(204, {})

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            resolved = router.resolve(method, path)
            if resolved is None:
                self._send(404, {"error": "not found", "path": path})
                return
            handler, params = resolved
            body: dict[str, Any] = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    body = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self._send(400, {"error": "request body is not valid JSON"})
                    return
            try:
                status, payload = handler(params, body)
            except ApiError as error:
                self._send(
                    error.status,
                    {"error": error.message, "detail": error.detail},
                )
            except Exception as error:  # noqa: BLE001 - surfaced, never swallowed
                self._send(
                    500, {"error": f"{type(error).__name__}: {error}"},
                )
            else:
                self._send(status, payload)

        def do_GET(self) -> None:
            # API routes win. Anything else is a request for the built frontend,
            # which this server hosts so the whole workspace is one process.
            path = self.path.split("?", 1)[0]
            reserved = any(path.startswith(prefix) for prefix in API_PREFIXES)
            if site is not None and not reserved and router.resolve("GET", path) is None:
                asset = site.resolve(path)
                if asset is not None:
                    body, content_type = asset
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return RequestHandler


def serve(
    router: Router, port: int, name: str, site: StaticSite | None = None,
) -> ThreadingHTTPServer:
    """Bind to loopback only. A local workspace has no business on a public port."""
    return ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(router, name, site),
    )
