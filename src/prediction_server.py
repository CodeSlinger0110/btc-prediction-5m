"""
WebSocket server for live intra-window BTC predictions (Up/Down %).

Listens on ``ws://127.0.0.1:8766`` by default (poly-arbitrage uses 8765 for the UI bridge).

Run:
  python src/prediction_server.py

Env:
  BTC_MODEL_WS_HOST=127.0.0.1
  BTC_MODEL_WS_PORT=8766
  BTC_MODEL_ARTIFACTS_DIR=models/artifacts
  BTC_MODEL_LIVE_LOG=optional path to append CSV ticks for offline retraining
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from inference import load_lightgbm_bundle, predict_final_window_pct
from live_inference import LiveMarketBuffer, pred_to_up_down_pct

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None  # type: ignore
    ConnectionClosed = Exception  # type: ignore[misc, assignment]


def _artifacts_dir() -> Path:
    raw = os.environ.get("BTC_MODEL_ARTIFACTS_DIR", "models/artifacts")
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO / p
    return p


def _maybe_log_row(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow(row)


class PredictionRuntime:
    """Loads model once; keeps one LiveMarketBuffer per active market slug."""

    def __init__(self) -> None:
        self.cfg = Config()
        art = _artifacts_dir()
        self.booster, self.scaler, self.feature_names = load_lightgbm_bundle(art)
        self.buffers: Dict[str, LiveMarketBuffer] = {}
        self.log_path: Optional[Path] = None
        lp = os.environ.get("BTC_MODEL_LIVE_LOG", "").strip()
        if lp:
            self.log_path = Path(lp) if Path(lp).is_absolute() else _REPO / lp

    def sync_buffer(
        self,
        market_id: str,
        bucket_start_sec: int,
        window_open: float,
    ) -> LiveMarketBuffer:
        buf = self.buffers.get(market_id)
        if buf is None or buf.bucket_start_sec != int(bucket_start_sec):
            self.buffers[market_id] = LiveMarketBuffer(market_id, int(bucket_start_sec), float(window_open))
            buf = self.buffers[market_id]
        else:
            if np.isfinite(window_open) and window_open > 0:
                buf.window_open = float(window_open)
        while len(self.buffers) > 8:
            oldest = next(iter(self.buffers))
            del self.buffers[oldest]
        return buf

    def handle_tick(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        market_id = str(msg.get("market") or msg.get("market_id") or "")
        bucket_start_sec = int(msg["bucket_start_sec"])
        remain_sec = float(msg.get("remain_sec") or msg.get("time_remaining") or 150)
        rsec = int(round(remain_sec))
        rsec = max(1, min(self.cfg.data.window_size, rsec))
        price = float(msg["price"])
        window_open = float(msg.get("window_open") or msg.get("window_open_price") or price)
        now_sec = int(msg.get("now_sec") or int(time.time()))

        if not market_id or bucket_start_sec <= 0:
            return {"type": "error", "message": "market and bucket_start_sec required"}

        buf = self.sync_buffer(market_id, bucket_start_sec, window_open)
        buf.add_second(now_sec, price)

        row = buf.feature_row(rsec, self.cfg)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(row.reshape(1, -1))
        pred = float(predict_final_window_pct(self.booster, x)[0])
        up_pct, down_pct = pred_to_up_down_pct(pred)

        out = {
            "type": "btc_model_prediction",
            "market": market_id,
            "pred_final_pct": pred,
            "up_pct": round(up_pct, 2),
            "down_pct": round(down_pct, 2),
            "remain_sec": rsec,
        }
        _maybe_log_row(
            self.log_path,
            {
                "ts_ms": int(time.time() * 1000),
                "market": market_id,
                "now_sec": now_sec,
                "bucket_start_sec": bucket_start_sec,
                "remain_sec": rsec,
                "price": price,
                "window_open": window_open,
                "pred_final_pct": pred,
                "up_pct": up_pct,
                "down_pct": down_pct,
            },
        )
        return out


async def _handler(runtime: PredictionRuntime, websocket: Any) -> None:
    """
    One client connection. Peers often drop TCP without a WebSocket close frame (bot restart,
    sleep, Windows WinError 64); that raises ConnectionClosed — handled so the server keeps running.
    """
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "invalid json"}))
                continue
            mtype = msg.get("type")
            if mtype == "ping":
                await websocket.send(json.dumps({"type": "pong", "t": time.time()}))
                continue
            if mtype != "tick":
                await websocket.send(json.dumps({"type": "error", "message": "expected type tick"}))
                continue
            try:
                resp = runtime.handle_tick(msg)
            except Exception as e:  # noqa: BLE001
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                continue
            await websocket.send(json.dumps(resp))
    except ConnectionClosed:
        return


def _bind_error_help(host: str, port: int, err: OSError) -> None:
    msg = str(err).lower()
    if getattr(err, "winerror", None) == 10048 or getattr(err, "errno", None) in (98, 48) or "address already in use" in msg:
        print(
            f"Cannot bind {host}:{port} — port already in use (another prediction_server may be running).\n"
            f"  Fix: stop the other process, or use a free port, e.g.\n"
            f"    set BTC_MODEL_WS_PORT=8767\n"
            f"  Windows: netstat -ano | findstr :{port}\n"
            f"           taskkill /PID <pid> /F\n",
            flush=True,
        )


async def _main_async() -> None:
    if websockets is None:
        raise RuntimeError("Install websockets: pip install websockets")
    runtime = PredictionRuntime()
    host = os.environ.get("BTC_MODEL_WS_HOST", "127.0.0.1")
    port = int(os.environ.get("BTC_MODEL_WS_PORT", "8766"))

    try:
        async with websockets.serve(lambda ws: _handler(runtime, ws), host, port):
            print(f"BTC model prediction server ws://{host}:{port}", flush=True)
            await asyncio.Future()
    except OSError as e:
        _bind_error_help(host, port, e)
        raise SystemExit(1) from e


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
