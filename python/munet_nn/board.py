"""MuNetBoard: realtime web metrics and model inspection for MuNet.

Example:
    board = munet.MuNetBoard.start("127.0.0.1", 8080)
    board.attach_model(model)
    board.log_scalar("train/loss", step, loss.item())
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


@dataclass
class ScalarPoint:
    step: int
    value: float
    wall_time: float


class MuNetBoard:
    """Modern, lightweight live board for MuNet experiments."""

    def __init__(self, interface: str = "127.0.0.1", port: int = 8080) -> None:
        self.interface = interface
        self.port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._metrics: Dict[str, List[ScalarPoint]] = defaultdict(list)
        self._model_summary: Dict[str, Any] = {}
        self._started_at = time.time()

    @classmethod
    def start(cls, interface: str = "127.0.0.1", port: int = 8080) -> "MuNetBoard":
        board = cls(interface=interface, port=port)
        board._start_server()
        return board

    def _start_server(self) -> None:
        board = self

        class Handler(BaseHTTPRequestHandler):
            def _write_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                self._write_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self._write_bytes(board._render_html().encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.OK)
                    return
                if self.path == "/api/metrics":
                    self._write_json(board._metrics_payload())
                    return
                if self.path == "/api/model":
                    self._write_json(board._model_payload())
                    return
                if self.path == "/api/summary":
                    self._write_json(board._summary_payload())
                    return
                self._write_json({"error": "not found"}, status=404)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((self.interface, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def log_scalar(self, tag: str, step: int, value: float) -> None:
        with self._lock:
            self._metrics[tag].append(ScalarPoint(int(step), float(value), time.time()))

    def attach_model(self, model: Any) -> None:
        named_modules: Dict[str, str] = {}
        named_parameters: Dict[str, Dict[str, Any]] = {}

        if hasattr(model, "named_modules"):
            named_modules = {str(n): m.__class__.__name__ for n, m in dict(model.named_modules()).items()}
        if hasattr(model, "named_parameters"):
            for n, t in dict(model.named_parameters()).items():
                named_parameters[str(n)] = {
                    "shape": list(getattr(t, "shape", [])),
                    "dtype": str(getattr(t, "dtype", "unknown")),
                    "device": str(getattr(t, "device", "unknown")),
                    "requires_grad": bool(getattr(t, "requires_grad", False)),
                }

        with self._lock:
            self._model_summary = {
                "root": model.__class__.__name__,
                "module_count": len(named_modules),
                "parameter_count": len(named_parameters),
                "modules": named_modules,
                "parameters": named_parameters,
            }

    def _metrics_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {"series": {k: [asdict(p) for p in v] for k, v in self._metrics.items()}}

    def _model_payload(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._model_summary)

    def _summary_payload(self) -> Dict[str, Any]:
        with self._lock:
            points = sum(len(v) for v in self._metrics.values())
            return {
                "uptime_sec": round(time.time() - self._started_at, 2),
                "series_count": len(self._metrics),
                "points": points,
                "model_attached": bool(self._model_summary),
            }

    def _render_html(self) -> str:
        return """<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/><title>MuNetBoard</title>
<style>
:root{--bg:#0b1020;--panel:#121a33;--muted:#98a2c7;--txt:#e8ecff;--acc:#53d2ff;--ok:#8ce99a}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top,#101a3f,var(--bg));color:var(--txt);font-family:Inter,system-ui,Arial;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.card{background:rgba(18,26,51,.92);border:1px solid #22305c;border-radius:14px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.26)}
h1{margin:0 0 12px;font-size:28px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em} .val{font-size:28px;font-weight:700}
.metric{margin:12px 0;padding:10px;border:1px solid #243869;border-radius:10px;background:#0f1730} .row{display:flex;justify-content:space-between;gap:8px}
canvas{width:100%;height:80px;border-radius:8px;background:#0a1128} table{width:100%;border-collapse:collapse;font-size:12px} td,th{padding:6px;border-bottom:1px solid #20315f;text-align:left}
small{color:var(--muted)}
</style></head><body>
<h1>MuNetBoard <small id='sub'>live training telemetry</small></h1>
<div class='grid'>
  <div class='card'><div class='label'>Uptime</div><div class='val' id='uptime'>-</div></div>
  <div class='card'><div class='label'>Series</div><div class='val' id='series'>-</div></div>
  <div class='card'><div class='label'>Points</div><div class='val' id='points'>-</div></div>
  <div class='card'><div class='label'>Model Attached</div><div class='val' id='modelok'>-</div></div>
</div>
<div class='card' style='margin-top:14px'><h3>Metrics</h3><div id='metrics'></div></div>
<div class='card' style='margin-top:14px'><h3>Model / Layers</h3><div id='model'></div></div>
<script>
function draw(canvas, values){const ctx=canvas.getContext('2d');const w=canvas.width=canvas.clientWidth*devicePixelRatio;const h=canvas.height=canvas.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);if(values.length<2)return;const min=Math.min(...values),max=Math.max(...values),r=max-min||1;ctx.strokeStyle='#53d2ff';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=(i/(values.length-1))*canvas.clientWidth;const y=canvas.clientHeight-((v-min)/r)*canvas.clientHeight; i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
async function refresh(){const [s,m,mod]=await Promise.all([fetch('/api/summary').then(r=>r.json()),fetch('/api/metrics').then(r=>r.json()),fetch('/api/model').then(r=>r.json())]);
uptime.textContent=s.uptime_sec+'s';series.textContent=s.series_count;points.textContent=s.points;modelok.textContent=s.model_attached?'Yes':'No';
const root=document.getElementById('metrics'); root.innerHTML=''; Object.entries(m.series).forEach(([k,pts])=>{const vals=pts.map(p=>p.value);const last=vals.at(-1);const min=Math.min(...vals),max=Math.max(...vals);const d=document.createElement('div');d.className='metric';d.innerHTML=`<div class='row'><b>${k}</b><small>last=${last?.toFixed(6)} min=${min.toFixed(6)} max=${max.toFixed(6)} n=${vals.length}</small></div><canvas></canvas>`;root.appendChild(d);draw(d.querySelector('canvas'),vals.slice(-120));});
const modelBox=document.getElementById('model'); if(!mod.root){modelBox.innerHTML='<small>No model attached yet.</small>';return;} const params=Object.entries(mod.parameters||{}); let rows=params.slice(0,200).map(([n,p])=>`<tr><td>${n}</td><td>${(p.shape||[]).join('x')}</td><td>${p.dtype}</td><td>${p.device}</td><td>${p.requires_grad}</td></tr>`).join(''); modelBox.innerHTML=`<div class='row'><div><b>${mod.root}</b></div><small>modules=${mod.module_count} params=${mod.parameter_count}</small></div><table><thead><tr><th>Name</th><th>Shape</th><th>Dtype</th><th>Device</th><th>Grad</th></tr></thead><tbody>${rows}</tbody></table>`;}
setInterval(refresh,500); refresh();
</script></body></html>"""
