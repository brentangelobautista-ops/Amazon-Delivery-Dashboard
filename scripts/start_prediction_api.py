"""Start the local LogiTrack inference API, creating its local model artifact if needed."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "machine-learning" / "models" / "delivery_time_pipeline.joblib"
METADATA_PATH = ROOT / "machine-learning" / "models" / "model_metadata.json"


def ensure_prediction_artifacts() -> None:
    missing = [path for path in (MODEL_PATH, METADATA_PATH) if not path.exists()]
    if not missing:
        return

    print("[LogiTrack API] Prediction artifact missing; training the model from amazon_delivery.csv...")
    result = subprocess.run([sys.executable, "machine-learning/train_model.py"], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)

    remaining = [str(path.relative_to(ROOT)) for path in (MODEL_PATH, METADATA_PATH) if not path.exists()]
    if remaining:
        raise RuntimeError(f"Training finished without creating: {', '.join(remaining)}")


def local_service_is_ready(host: str, port: str) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with urlopen(f"http://{probe_host}:{port}/health", timeout=1.5) as response:
            health = json.load(response)
            return response.status == 200 and health.get("ready") is True and isinstance(health.get("model"), str)
    except (OSError, URLError, ValueError):
        return False


def port_is_in_use(host: str, port: str) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, int(port)), timeout=1.5):
            return True
    except OSError:
        return False


def main() -> int:
    host = os.getenv("LOGITRACK_API_HOST", "127.0.0.1")
    port = os.getenv("LOGITRACK_API_PORT", "8000")
    if local_service_is_ready(host, port):
        print(f"[LogiTrack API] A healthy API is already available at http://{host}:{port}; leaving it running.")
        return 0
    if port_is_in_use(host, port):
        print(
            f"[LogiTrack API] Port {port} is already in use, but it is not a healthy LogiTrack API. "
            "Stop the conflicting process or set LOGITRACK_API_PORT and LOGITRACK_API_PROXY_TARGET.",
            file=sys.stderr,
        )
        return 1

    try:
        ensure_prediction_artifacts()
    except RuntimeError as error:
        print(f"[LogiTrack API] {error}", file=sys.stderr)
        return 1

    print(f"[LogiTrack API] Starting FastAPI at http://{host}:{port}")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            host,
            "--port",
            port,
            "--reload",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
