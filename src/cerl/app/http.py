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


class ApiError(Exception):
    """A handler's deliberate error response, with a status the client can act on."""

    def __init__(self, status: int, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


def make_handler(router: Router, name: str) -> type[BaseHTTPRequestHandler]:
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
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return RequestHandler


def serve(router: Router, port: int, name: str) -> ThreadingHTTPServer:
    """Bind to loopback only. A local workspace has no business on a public port."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(router, name))
