"""Polling HTTP control plane and resumable immutable downloads."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from .artifacts import MAX_ARTIFACT, MAX_RESULT, decode, encode
from .owner import ProtocolError


def make_server(owner, address, token):
    if not isinstance(token, str) or len(token) < 32 or "\n" in token or "\r" in token:
        raise ValueError("MUNET_SWARM_TOKEN must contain at least 32 characters and no newlines")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        def reply(self, status, value):
            data = encode(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized"})
                return False
            return True

        def do_POST(self):
            if not self.authorized():
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ProtocolError("chunked requests are unsupported", 411)
                length = int(self.headers.get("Content-Length", "-1"))
                limit = MAX_RESULT if self.path == "/v1/results" else 64 * 1024
                if not 0 < length <= limit:
                    raise ProtocolError("request size out of bounds", 413)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ProtocolError("incomplete body")
                body = decode(raw)
                if not isinstance(body, dict):
                    raise ProtocolError("expected a JSON object")
                routes = {"/v1/register": owner.register, "/v1/lease": owner.lease,
                          "/v1/heartbeat": owner.heartbeat, "/v1/results": owner.result}
                if self.path not in routes:
                    raise ProtocolError("unknown endpoint", 404)
                self.reply(200, routes[self.path](body))
            except ProtocolError as e:
                self.reply(e.status, {"error": str(e)})
            except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError):
                self.reply(400, {"error": "invalid request"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception:
                self.reply(500, {"error": "owner could not commit request; retry the identical payload"})

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/v1/status":
                self.reply(200, owner.status())
                return
            match = re.fullmatch(r"/v1/artifacts/([0-9a-f]{64})", self.path)
            if not match:
                self.reply(404, {"error": "unknown artifact"})
                return
            path = owner.artifacts.path(match[1])
            if not path.is_file() or path.stat().st_size > MAX_ARTIFACT:
                self.reply(404, {"error": "unknown artifact"})
                return
            size, offset = path.stat().st_size, 0
            byte_range = self.headers.get("Range")
            if byte_range:
                m = re.fullmatch(r"bytes=(\d+)-", byte_range)
                if not m or int(m[1]) >= size:
                    self.reply(416, {"error": "range not satisfiable"})
                    return
                offset = int(m[1])
            self.send_response(206 if byte_range else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"' + match[1] + '"')
            self.send_header("Content-Length", str(size - offset))
            if byte_range:
                self.send_header("Content-Range", f"bytes {offset}-{size - 1}/{size}")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(offset)
                    while block := f.read(64 * 1024):
                        self.wfile.write(block)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            self.close_connection = True

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    return server
