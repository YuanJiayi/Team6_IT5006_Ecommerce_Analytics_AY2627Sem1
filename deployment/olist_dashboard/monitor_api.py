"""HTTP responses for the Models-tab monitoring section.

The heavy lifting lives in phase3.monitor; this module only validates the
request, maps failures to clear HTTP errors and returns the payload or the
Evidently HTML report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))


class MonitorAPIUnavailable(Exception):
    """The monitor module or its dependencies could not be imported."""


def _load_monitor():
    try:
        from phase3 import monitor  # noqa: PLC0415 - lazy so the dashboard starts without Evidently
    except Exception as error:  # noqa: BLE001 - report a clear 503
        raise MonitorAPIUnavailable(
            "The monitoring dependencies are not available: {}".format(error)
        ) from error
    return monitor


def _error(status, code, message):
    return status, {"ok": False, "error": {"code": code, "message": message}}


def monitor_response(date, root=None):
    """Return (status, payload) for GET /api/models/monitor?date=YYYY-MM-DD."""
    try:
        module = _load_monitor()
    except MonitorAPIUnavailable as error:
        return _error(503, "monitor_unavailable", str(error))
    try:
        date_str = module.validate_date(date)
    except ValueError as error:
        return _error(400, "bad_request", str(error))
    try:
        payload = module.build_monitor(date_str, root=root or module.REPO_ROOT)
        return 200, payload
    except module.MonitorUnavailable as error:
        return _error(503, "monitor_unavailable", str(error))
    except Exception as error:  # noqa: BLE001 - a broken monitor must not look like success
        return _error(503, "monitor_unavailable",
                      "The monitoring data could not be prepared: {}".format(error))


def report_response(date, root=None):
    """Return (status, body bytes, content type) for the Evidently HTML report."""
    try:
        module = _load_monitor()
    except MonitorAPIUnavailable as error:
        return 503, _json_bytes(_error(503, "monitor_unavailable", str(error))[1]), "application/json; charset=utf-8"
    try:
        date_str = module.validate_date(date)
    except ValueError as error:
        return 400, _json_bytes(_error(400, "bad_request", str(error))[1]), "application/json; charset=utf-8"
    try:
        html = module.build_report_html(date_str, root=root or module.REPO_ROOT)
        return 200, html.encode("utf-8"), "text/html; charset=utf-8"
    except module.MonitorUnavailable as error:
        return 503, _json_bytes(_error(503, "monitor_unavailable", str(error))[1]), "application/json; charset=utf-8"
    except Exception as error:  # noqa: BLE001
        return 503, _json_bytes(_error(
            503, "monitor_unavailable",
            "The input-drift report could not be generated: {}".format(error),
        )[1]), "application/json; charset=utf-8"


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
