"""Run an MLflow tracking server for a local store so artifacts stay portable.

The registry and the dashboard both use an HTTP tracking URI. MLflow then records
``mlflow-artifacts:/...`` locations instead of ``file:///...`` paths, so the store
can be copied to another machine and still load its models and artifacts.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

STATIC_PREFIX = "/mlflow"
STARTUP_TIMEOUT = 90.0


def free_port():
    """A currently free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _health_url(port):
    return "http://127.0.0.1:{}{}/health".format(port, STATIC_PREFIX)


def _log_tail(handle, limit=4000):
    handle.flush()
    handle.seek(0)
    return handle.read()[-limit:]


@contextmanager
def mlflow_server(store, timeout=STARTUP_TIMEOUT):
    """Serve ``store`` over HTTP and yield its tracking URI.

    The server always stops when the block exits, including on error. If it never
    becomes healthy, the raised error carries the tail of the server's stderr.
    """
    store = Path(store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    environment = dict(os.environ)
    environment.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    environment.setdefault("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    command = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///{}".format(store / "mlflow.db"),
        "--serve-artifacts",
        "--artifacts-destination", str(store / "artifacts"),
        "--static-prefix", STATIC_PREFIX,
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                               env=environment)
    tracking_uri = "http://127.0.0.1:{}{}".format(port, STATIC_PREFIX)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    "The MLflow server for {} exited with code {} before it was ready:\n{}".format(
                        store, process.returncode, _log_tail(log)))
            try:
                with urllib.request.urlopen(_health_url(port), timeout=2) as response:
                    if 200 <= response.status < 300:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "The MLflow server for {} did not become healthy within {:.0f}s at {}:\n{}".format(
                        store, timeout, _health_url(port), _log_tail(log)))
            time.sleep(0.2)
        yield tracking_uri
    finally:
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        finally:
            log.close()
