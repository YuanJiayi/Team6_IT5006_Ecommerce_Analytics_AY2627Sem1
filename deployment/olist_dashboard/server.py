#!/usr/bin/env python3
"""Local Vamos server: static app plus POST /api/ask.

Run with:
    python3 deployment/olist_dashboard/server.py
    python3 deployment/olist_dashboard/server.py --host 127.0.0.1 --port 8000

Provider settings come from VAMOS_LLM_BASE_URL, VAMOS_LLM_API_KEY and
VAMOS_LLM_MODEL in the environment, or from an uncommitted .env in the project
folder. The key is used only for the server-to-provider request and is never
sent back to the browser or written to logs.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sqlite3
import sys
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import actions_api
import models_api
import monitor_api
from analytics_sql import AnalyticsStore, QueryError
from ask_service import (
    AskService,
    LLMClient,
    ProviderError,
    WINDOW_CHOICE_LABELS,
    WINDOW_CHOICES,
    _default_window_start,
    _extract_repeated_totals,
    _incomplete_note,
    _incomplete_periods,
    _natural_date,
    _pin_period_label,
    _plain_evidence,
    _strip_fixed_period_title,
    _window_choice_start,
    classify_window,
    load_config,
    parse_iso,
    to_rolling_sql,
)


APP_ROOT = Path(__file__).resolve().parent
# VAMOS_ANALYTICS_DB lets a host point at a copy on fast local disk.
ANALYTICS_DB = Path(os.environ.get("VAMOS_ANALYTICS_DB") or APP_ROOT / "data" / "analytics.sqlite")
MAX_BODY_BYTES = 20_000
STATIC_SUFFIXES = {".html", ".js", ".json", ".css", ".svg", ".png", ".ico"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
TRUSTED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def trusted_hostname(host_header):
    """Return a bare hostname for a Host header, or None when it is malformed."""
    if not isinstance(host_header, str):
        return None
    value = host_header.strip().lower()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        rest = value[end + 1 :]
        if rest and not rest.startswith(":"):
            return None
        if rest and not rest[1:].isdigit():
            return None
        return value[1:end]
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)
        if port and not port.isdigit():
            return None
        return host
    if ":" in value:
        return None
    return value


def parse_allowed_hosts(value):
    """Return the hosts configured through VAMOS_ALLOWED_HOSTS, lowercased."""
    hosts = set()
    if not isinstance(value, str):
        return hosts
    for entry in value.split(","):
        entry = entry.strip().lower()
        if entry:
            hosts.add(entry)
    return hosts


def is_trusted_host(host_header, trusted_hosts=None):
    hosts = TRUSTED_HOSTS if trusted_hosts is None else trusted_hosts
    return trusted_hostname(host_header) in hosts


def is_trusted_origin(origin_header, trusted_hosts=None):
    hosts = TRUSTED_HOSTS if trusted_hosts is None else trusted_hosts
    if not isinstance(origin_header, str) or not origin_header.strip():
        return False
    value = origin_header.strip()
    if value.lower() == "null":
        return False
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    try:
        hostname = parsed.hostname
    except ValueError:
        return False
    return hostname in hosts


def is_json_content_type(content_type):
    if not isinstance(content_type, str):
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


MAX_PINS = 6
MAX_ANSWER_ROWS = 500
PIN_ID_RE_LENGTH = 32
SEED_CHATS_PATH = APP_ROOT / "data" / "ask-seed-chats.json"


class PinLimitError(Exception):
    """The browser already holds the maximum number of pinned charts."""


class SeedChatsError(Exception):
    """The ready-made Ask chats file is missing or malformed."""


def load_seed_answers(path=None):
    """Return the answers from the ready-made chats, keyed by answer id.

    Raises SeedChatsError when the file is missing or malformed so the server
    fails at startup instead of silently serving chats whose pins cannot work.
    """
    source = Path(path) if path is not None else SEED_CHATS_PATH
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as error:
        raise SeedChatsError(
            "Could not read seed chats file {}: {}".format(source, error)
        ) from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise SeedChatsError(
            "Seed chats file {} is not valid JSON: {}".format(source, error)
        ) from error
    if not isinstance(data, dict) or not isinstance(data.get("chats"), list):
        raise SeedChatsError(
            "Seed chats file {} must contain a chats list.".format(source)
        )
    answers = {}
    for index, chat in enumerate(data["chats"]):
        if not isinstance(chat, dict):
            raise SeedChatsError(
                "Seed chats file {} has a chat that is not an object.".format(source)
            )
        answer = chat.get("answer")
        if not isinstance(answer, dict):
            raise SeedChatsError(
                "Seed chat {} in {} has no answer object.".format(index, source)
            )
        answer_id = answer.get("answer_id")
        if not PinStore._valid_id(answer_id):
            raise SeedChatsError(
                "Seed chat {} in {} has an invalid answer_id.".format(index, source)
            )
        sql = answer.get("sql")
        chart = answer.get("chart")
        if not isinstance(sql, str) or not isinstance(chart, dict):
            raise SeedChatsError(
                "Seed chat {} in {} must have SQL and chart objects.".format(index, source)
            )
        answers[answer_id] = {
            "title": str(answer.get("title") or "Saved chart"),
            "question": str(chat.get("question") or ""),
            "sql": sql,
            "chart": chart,
        }
    return answers


class PinStore:
    """Server-side store for Ask answers and pinned charts.

    The browser only ever sees opaque ids. Stored SQL is re-validated and run
    through AnalyticsStore exactly like Ask when a pin is rendered.
    """

    def __init__(self, path, seed_answers=None):
        self.path = Path(path)
        self.seed_answers = dict(seed_answers or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS answers ("
                " id TEXT PRIMARY KEY, created REAL NOT NULL, title TEXT NOT NULL,"
                " question TEXT NOT NULL, sql TEXT NOT NULL, chart TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pins ("
                " id TEXT PRIMARY KEY, answer_id TEXT NOT NULL, created REAL NOT NULL,"
                " title TEXT NOT NULL, question TEXT NOT NULL, sql TEXT NOT NULL,"
                " chart TEXT NOT NULL)"
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self):
        return sqlite3.connect(str(self.path), timeout=5.0)

    def remember_answer(self, answer_id, answer):
        if not self._valid_id(answer_id):
            return
        sql = answer.get("sql")
        chart = answer.get("chart")
        if not isinstance(sql, str) or not isinstance(chart, dict):
            return
        connection = self._connect()
        try:
            connection.execute(
                "INSERT OR REPLACE INTO answers (id, created, title, question, sql, chart)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    answer_id,
                    time.time(),
                    str(answer.get("title") or "Saved chart"),
                    str(answer.get("question") or ""),
                    sql,
                    json.dumps(chart, separators=(",", ":")),
                ),
            )
            connection.execute(
                "DELETE FROM answers WHERE id NOT IN "
                "(SELECT id FROM answers ORDER BY created DESC LIMIT ?)",
                (MAX_ANSWER_ROWS,),
            )
            connection.commit()
        finally:
            connection.close()

    def create_pin(self, answer_id):
        if not self._valid_id(answer_id):
            raise KeyError(answer_id)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT title, question, sql, chart FROM answers WHERE id = ?",
                (answer_id,),
            ).fetchone()
            if row is None:
                seed = self.seed_answers.get(answer_id)
                if seed is None:
                    raise KeyError(answer_id)
                row = (
                    seed["title"],
                    seed["question"],
                    seed["sql"],
                    json.dumps(seed["chart"], separators=(",", ":")),
                )
            title, question, sql, chart = row
            window = classify_window(sql)
            pin_sql = to_rolling_sql(sql)
            if window["kind"] == "fixed":
                title = _strip_fixed_period_title(title, window.get("label"))
            count = connection.execute("SELECT COUNT(*) FROM pins").fetchone()[0]
            if count >= MAX_PINS:
                raise PinLimitError()
            pin_id = secrets.token_hex(16)
            connection.execute(
                "INSERT INTO pins (id, answer_id, created, title, question, sql, chart)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pin_id, answer_id, time.time(), title, question, pin_sql, chart),
            )
            connection.commit()
            return {"id": pin_id, "title": title}
        finally:
            connection.close()

    def get_pin(self, pin_id):
        if not self._valid_id(pin_id):
            return None
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, title, question, sql, chart, answer_id FROM pins WHERE id = ?",
                (pin_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            chart = json.loads(row[4])
        except (TypeError, ValueError):
            chart = None
        return {
            "id": row[0],
            "title": row[1],
            "question": row[2],
            "sql": row[3],
            "chart": chart,
            "answer_id": row[5],
        }

    def delete_pin(self, pin_id):
        if not self._valid_id(pin_id):
            return False
        connection = self._connect()
        try:
            cursor = connection.execute("DELETE FROM pins WHERE id = ?", (pin_id,))
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    @staticmethod
    def _valid_id(value):
        return (
            isinstance(value, str)
            and len(value) == PIN_ID_RE_LENGTH
            and all(character in "0123456789abcdef" for character in value)
        )


class VamosHandler(BaseHTTPRequestHandler):
    server_version = "Vamos/1.0"
    protocol_version = "HTTP/1.1"

    def _send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _safe_static_path(self, request_path):
        if request_path in ("", "/"):
            relative = "index.html"
        else:
            relative = unquote(request_path.lstrip("/"))
        parts = Path(relative).parts
        if not parts or any(part.startswith(".") for part in parts):
            return None
        candidate = (APP_ROOT / relative).resolve()
        try:
            candidate.relative_to(APP_ROOT)
        except ValueError:
            return None
        if not candidate.is_file() or candidate.suffix.lower() not in STATIC_SUFFIXES:
            return None
        return candidate

    def _pin_store(self):
        return getattr(self.server, "pin_store", None)

    def _check_pin_host(self):
        if is_trusted_host(self.headers.get("Host"), self.server.trusted_hosts):
            return True
        self.close_connection = True
        self._send_json(
            403,
            {
                "ok": False,
                "error": {
                    "code": "forbidden_request",
                    "message": "Pins accept requests only from a local host.",
                },
            },
        )
        return False

    def _check_pin_origin(self):
        origin = self.headers.get("Origin")
        if origin is None or is_trusted_origin(origin, self.server.trusted_hosts):
            return True
        self.close_connection = True
        self._send_json(
            403,
            {
                "ok": False,
                "error": {
                    "code": "forbidden_origin",
                    "message": "Pins accept requests only from a local origin.",
                },
            },
        )
        return False

    def _read_json_request(self):
        if not is_json_content_type(self.headers.get("Content-Type")):
            self.close_connection = True
            self._send_json(
                415,
                {
                    "ok": False,
                    "error": {
                        "code": "unsupported_media_type",
                        "message": "This endpoint requires Content-Type: application/json.",
                    },
                },
            )
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json(
                413,
                {"ok": False, "error": {"code": "bad_request", "message": "Request body is too large."}},
            )
            return None
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Request body must be valid JSON."}},
            )
            return None
        if not isinstance(payload, dict):
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Request body must be a JSON object."}},
            )
            return None
        return payload

    def _handle_get_pin(self, parsed):
        store = self._pin_store()
        if store is None:
            self._send_json(
                503,
                {"ok": False, "error": {"code": "pins_unavailable", "message": "Pinned charts are not available."}},
            )
            return
        if not self._check_pin_host():
            return
        pin_id = parsed.path[len("/api/pins/"):].strip("/")
        try:
            pin = store.get_pin(pin_id)
        except sqlite3.Error:
            self._send_json(
                503,
                {"ok": False, "error": {"code": "pins_unavailable", "message": "Pinned charts could not be read."}},
            )
            return
        if pin is None:
            self._send_json(
                404,
                {"ok": False, "error": {"code": "not_found", "message": "Pinned chart not found."}},
            )
            return
        query = parse_qs(parsed.query)
        try:
            as_of = parse_iso(query.get("as_of", [None])[0])
        except (TypeError, ValueError):
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Data as of must be a valid YYYY-MM-DD date."}},
            )
            return
        requested = query.get("window", [None])[0]
        if requested is not None and requested not in WINDOW_CHOICES:
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Unknown window choice."}},
            )
            return
        window = classify_window(pin["sql"])
        params = None
        window_label = None
        selected_window = None
        if window["kind"] == "rolling":
            if requested is not None:
                start = _window_choice_start(as_of, requested)
                selected_window = requested
                window_label = WINDOW_CHOICE_LABELS[requested]
            else:
                start = _default_window_start(as_of, window.get("default"))
                selected_window = window.get("default_button")
                window_label = window["default_label"]
            if start is None:
                self._send_json(
                    422,
                    {
                        "ok": False,
                        "error": {
                            "code": "pin_invalid",
                            "message": "This saved chart has no usable rolling-window start.",
                        },
                    },
                )
                return
            params = {"window_start": start.isoformat()}
        available = True
        partial = False
        note = ""
        if window["kind"] == "fixed":
            period_start = window.get("start")
            period_end = window.get("end")
            window_label = "Fixed: " + window["label"]
            if period_start is not None and as_of <= period_start:
                available = False
                note = (
                    "This fixed period starts after the selected date. "
                    "Choose a later Data as of date."
                )
            elif period_end is not None and as_of < period_end:
                partial = True
                note = "This fixed period is not complete at the selected date."
        elif window["kind"] == "all":
            window_label = "All history to " + _natural_date(as_of - timedelta(days=1))
        period_label = _pin_period_label(
            window["kind"], window_label, as_of, selected_window
        )
        if not available:
            self._send_json(
                200,
                {
                    "ok": True,
                    "available": False,
                    "as_of": as_of.isoformat(),
                    "pin": {
                        "id": pin["id"],
                        "title": pin["title"],
                        "answer_id": pin.get("answer_id"),
                        "kind": window["kind"],
                        "label": window_label,
                        "period": period_label,
                        "window": selected_window,
                        "window_label": window_label,
                        "windows": [],
                        "chart": pin["chart"],
                    },
                    "note": note,
                },
            )
            return
        analytics = getattr(self.server.ask_service, "store", None)
        if analytics is None or not analytics.ready():
            self._send_json(
                503,
                {
                    "ok": False,
                    "error": {
                        "code": "analytics_not_ready",
                        "message": "The local analytics database is not available.",
                    },
                },
            )
            return
        try:
            result = analytics.execute(pin["sql"], as_of, params)
            result, overall_totals = _extract_repeated_totals(result)
            if overall_totals:
                result["overall_totals"] = overall_totals
        except QueryError as error:
            status = 503 if error.code == "analytics_not_ready" else 422
            self._send_json(
                status,
                {
                    "ok": False,
                    "as_of": as_of.isoformat(),
                    "error": {
                        "code": error.code,
                        "message": (
                            "This saved chart could not be re-run against the current "
                            "data. Remove it and pin the question again. ({})"
                        ).format(error.message),
                    },
                },
            )
            return
        chart = pin.get("chart")
        incomplete = _incomplete_periods(result)
        table = {"columns": result["columns"], "rows": result["rows"]}
        if incomplete:
            table["incomplete_rows"] = [item["index"] for item in incomplete]
            table["incomplete_note"] = _incomplete_note(incomplete)
            if chart is not None:
                x_column = chart.get("x")
                if x_column in result["columns"]:
                    chart = dict(chart)
                    chart["incomplete"] = [
                        result["rows"][item["index"]].get(x_column, item["label"])
                        for item in incomplete
                    ]
        if overall_totals:
            table["overall_totals"] = overall_totals
        self._send_json(
            200,
            {
                "ok": True,
                "available": True,
                "as_of": as_of.isoformat(),
                "pin": {
                    "id": pin["id"],
                    "title": pin["title"],
                    "answer_id": pin.get("answer_id"),
                    "kind": window["kind"],
                    "label": window_label,
                    "period": period_label,
                    "window": selected_window,
                    "window_label": window_label,
                    "windows": list(WINDOW_CHOICES) if window["kind"] == "rolling" else [],
                    "window_labels": dict(WINDOW_CHOICE_LABELS),
                    "chart": chart,
                    "evidence": _plain_evidence(as_of, pin["sql"], result),
                },
                "table": table,
                "partial": partial,
                "note": note,
            },
        )

    def _handle_create_pin(self):
        store = self._pin_store()
        if store is None:
            self._send_json(
                503,
                {"ok": False, "error": {"code": "pins_unavailable", "message": "Pinned charts are not available."}},
            )
            return
        if not self._check_pin_host() or not self._check_pin_origin():
            return
        payload = self._read_json_request()
        if payload is None:
            return
        if set(payload) != {"answer_id"} or not isinstance(payload.get("answer_id"), str):
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Send exactly one answer_id."}},
            )
            return
        try:
            created = store.create_pin(payload["answer_id"])
        except sqlite3.Error:
            self._send_json(
                503,
                {"ok": False, "error": {"code": "pins_unavailable", "message": "Pinned charts could not be saved."}},
            )
            return
        except QueryError as error:
            self._send_json(
                422,
                {"ok": False, "error": {"code": error.code, "message": error.message}},
            )
            return
        except KeyError:
            self._send_json(
                404,
                {
                    "ok": False,
                    "error": {
                        "code": "answer_not_found",
                        "message": "That answer is no longer available. Ask the question again, then pin it.",
                    },
                },
            )
            return
        except PinLimitError:
            self._send_json(
                409,
                {
                    "ok": False,
                    "error": {
                        "code": "pin_limit",
                        "message": "Overview holds at most {} pinned charts. Remove one first.".format(MAX_PINS),
                    },
                },
            )
            return
        self._send_json(200, {"ok": True, "pin": created})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/pins/") and parsed.path != "/api/pins/":
            self._handle_get_pin(parsed)
            return
        if parsed.path == "/api/actions/queue":
            service = getattr(self.server, "actions_service", None)
            if service is None:
                self._send_json(503, {"ok": False, "error": {
                    "code": "actions_unavailable",
                    "message": "The champion queue is not loaded.",
                }})
                return
            date = parse_qs(parsed.query).get("date", [None])[0]
            status, payload = service.queue_response(date)
            self._send_json(status, payload)
            return
        if parsed.path == "/api/actions/promises":
            service = getattr(self.server, "actions_service", None)
            if service is None:
                self._send_json(503, {"ok": False, "error": {
                    "code": "actions_unavailable",
                    "message": "The champion queue is not loaded.",
                }})
                return
            date = parse_qs(parsed.query).get("date", [None])[0]
            status, payload = service.promises_response(date)
            self._send_json(status, payload)
            return
        if parsed.path == "/api/actions/order":
            service = getattr(self.server, "actions_service", None)
            if service is None:
                self._send_json(503, {"ok": False, "error": {
                    "code": "actions_unavailable",
                    "message": "The champion queue is not loaded.",
                }})
                return
            query = parse_qs(parsed.query)
            date = query.get("date", [None])[0]
            order_id = query.get("order_id", [None])[0]
            status, payload = service.order_response(date, order_id)
            self._send_json(status, payload)
            return
        if parsed.path == "/api/actions/predict":
            service = getattr(self.server, "actions_service", None)
            if service is None:
                self._send_json(503, {"ok": False, "error": {
                    "code": "actions_unavailable",
                    "message": "The champion queue is not loaded.",
                }})
                return
            query = parse_qs(parsed.query)
            date = query.get("date", [None])[0]
            order_id = query.get("order_id", [None])[0]
            status, payload = service.predict_response(date, order_id)
            self._send_json(status, payload)
            return
        if parsed.path == "/api/models":
            date = parse_qs(parsed.query).get("date", [None])[0]
            status, payload = models_api.models_response(date)
            self._send_json(status, payload)
            return
        if parsed.path in ("/api/models/monitor", "/api/models/monitor/report"):
            date = parse_qs(parsed.query).get("date", [None])[0]
            if parsed.path.endswith("/report"):
                status, body, content_type = monitor_api.report_response(date)
                self._send_bytes(status, body, content_type)
            else:
                status, payload = monitor_api.monitor_response(date)
                self._send_json(status, payload)
            return
        if parsed.path.startswith("/api/models/") and parsed.path.endswith("/importance"):
            name = unquote(parsed.path[len("/api/models/"):-len("/importance")]).strip("/")
            status, payload = models_api.importance_response(name)
            self._send_json(status, payload)
            return
        if parsed.path == "/mlflow" or parsed.path.startswith("/mlflow/"):
            models_api.proxy(self)
            return
        if parsed.path == "/api/ask":
            self._send_json(
                405,
                {"ok": False, "error": {"code": "method_not_allowed", "message": "Use POST for Ask."}},
            )
            return
        candidate = self._safe_static_path(parsed.path)
        if candidate is None:
            self._send_json(
                404,
                {"ok": False, "error": {"code": "not_found", "message": "Not found."}},
            )
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self._send_json(
                500,
                {"ok": False, "error": {"code": "read_error", "message": "Could not read the file."}},
            )
            return
        self._send_bytes(
            200,
            body,
            CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream"),
        )

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/mlflow" or parsed.path.startswith("/mlflow/"):
            models_api.proxy(self)
            return
        if parsed.path == "/api/pins":
            self._handle_create_pin()
            return
        if parsed.path != "/api/ask":
            self._send_json(
                404,
                {"ok": False, "error": {"code": "not_found", "message": "Not found."}},
            )
            return
        scope = "an allowed" if self.server.public_mode else "a local"
        if not is_trusted_host(
            self.headers.get("Host"), self.server.trusted_hosts
        ):
            self.close_connection = True
            self._send_json(
                403,
                {
                    "ok": False,
                    "error": {
                        "code": "forbidden_request",
                        "message": "Ask accepts requests only from {} host.".format(scope),
                    },
                },
            )
            return
        origin = self.headers.get("Origin")
        if origin is not None and not is_trusted_origin(
            origin, self.server.trusted_hosts
        ):
            self.close_connection = True
            self._send_json(
                403,
                {
                    "ok": False,
                    "error": {
                        "code": "forbidden_origin",
                        "message": "Ask accepts requests only from {} origin.".format(scope),
                    },
                },
            )
            return
        if not is_json_content_type(self.headers.get("Content-Type")):
            self.close_connection = True
            self._send_json(
                415,
                {
                    "ok": False,
                    "error": {
                        "code": "unsupported_media_type",
                        "message": "Ask requires Content-Type: application/json.",
                    },
                },
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json(
                413,
                {"ok": False, "error": {"code": "bad_request", "message": "Request body is too large."}},
            )
            return
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                400,
                {"ok": False, "error": {"code": "bad_request", "message": "Request body must be valid JSON."}},
            )
            return
        try:
            status, response = self.server.ask_service.handle(payload)
        except ProviderError as error:
            status, response = 502, {
                "ok": False,
                "error": {"code": "provider_error", "message": str(error)},
            }
        except Exception:
            status, response = 500, {
                "ok": False,
                "error": {
                    "code": "server_error",
                    "message": "The server could not complete the request.",
                },
            }
        pin_store = self._pin_store()
        if (
            pin_store is not None
            and isinstance(response, dict)
            and response.get("ok")
            and isinstance(response.get("answer"), dict)
        ):
            answer = response["answer"]
            pin = answer.get("pin")
            if isinstance(pin, dict) and isinstance(answer.get("answer_id"), str):
                try:
                    pin_store.remember_answer(answer["answer_id"], answer)
                except Exception:
                    pass
        self._send_json(status, response)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == "/mlflow" or parsed.path.startswith("/mlflow/"):
            models_api.proxy(self)
            return
        self._send_json(
            405,
            {"ok": False, "error": {"code": "method_not_allowed", "message": "Use GET for this path."}},
        )

    def _other_method(self):
        parsed = urlparse(self.path)
        if parsed.path == "/mlflow" or parsed.path.startswith("/mlflow/"):
            models_api.proxy(self)
            return
        self._send_json(
            405,
            {"ok": False, "error": {"code": "method_not_allowed", "message": "Method not allowed."}},
        )

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/pins/") and parsed.path != "/api/pins/":
            store = self._pin_store()
            if store is None:
                self._send_json(
                    503,
                    {"ok": False, "error": {"code": "pins_unavailable", "message": "Pinned charts are not available."}},
                )
                return
            if not self._check_pin_host() or not self._check_pin_origin():
                return
            pin_id = parsed.path[len("/api/pins/"):].strip("/")
            try:
                deleted = store.delete_pin(pin_id)
            except sqlite3.Error:
                self._send_json(
                    503,
                    {
                        "ok": False,
                        "error": {
                            "code": "pins_unavailable",
                            "message": "Pinned charts could not be changed.",
                        },
                    },
                )
                return
            if deleted:
                self._send_json(200, {"ok": True})
            else:
                self._send_json(
                    404,
                    {"ok": False, "error": {"code": "not_found", "message": "Pinned chart not found."}},
                )
            return
        self._other_method()

    def do_PUT(self):
        self._other_method()

    def do_PATCH(self):
        self._other_method()

    def do_OPTIONS(self):
        self._other_method()


class VamosServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address,
        ask_service,
        trusted_hosts=None,
        public_mode=False,
        actions_service=None,
        pin_store=None,
    ):
        self.ask_service = ask_service
        self.actions_service = actions_service
        self.pin_store = pin_store
        self.trusted_hosts = frozenset(
            TRUSTED_HOSTS if trusted_hosts is None else trusted_hosts
        )
        self.public_mode = bool(public_mode)
        super().__init__(address, VamosHandler)


def create_server(
    host="127.0.0.1",
    port=8000,
    ask_service=None,
    store=None,
    allowed_hosts=None,
    actions_service=None,
    pin_store=None,
    seed_answers=None,
):
    if allowed_hosts is None:
        allowed_hosts = os.environ.get("VAMOS_ALLOWED_HOSTS", "")
    configured_hosts = parse_allowed_hosts(allowed_hosts)
    trusted_hosts = TRUSTED_HOSTS | configured_hosts
    if ask_service is None:
        store = store or AnalyticsStore(ANALYTICS_DB)
        config = load_config()
        model = LLMClient(config) if config else None
        ask_service = AskService(store, model)
    if actions_service is None:
        actions_service = actions_api.ActionsService()
    if pin_store is None:
        if seed_answers is None:
            seed_answers = load_seed_answers()
        pin_store = PinStore(
            ANALYTICS_DB.with_name(ANALYTICS_DB.name + "-pins.db"),
            seed_answers=seed_answers,
        )
    return VamosServer(
        (host, port),
        ask_service,
        trusted_hosts=trusted_hosts,
        public_mode=bool(configured_hosts),
        actions_service=actions_service,
        pin_store=pin_store,
    )


def public_mode_problems(ask_service):
    """Return the reasons a public deployment may not start."""
    problems = []
    store = getattr(ask_service, "store", None)
    if store is None or not store.ready():
        problems.append(
            "the analytics database is not ready. Build it with "
            "python3 deployment/olist_dashboard/analytics_db.py."
        )
    if getattr(ask_service, "model", None) is None:
        problems.append(
            "the Ask provider is not configured. Set VAMOS_LLM_BASE_URL, "
            "VAMOS_LLM_API_KEY and VAMOS_LLM_MODEL."
        )
    return problems


DEFAULT_PREWARM_DATE = "2018-08-22"


def _prewarm_default_date(server, date=DEFAULT_PREWARM_DATE):
    """Warm the default demo date sequentially in the background after startup."""
    if os.environ.get("VAMOS_PREWARM", "1") == "0":
        return None

    def _run():
        try:
            models_api.models_response(date)
            actions_api._release_free_memory()
            service = getattr(server, "actions_service", None)
            if service is not None:
                service.queue_response(date)
                actions_api._release_free_memory()
                service.promises_response(date)
                actions_api._release_free_memory()
            monitor_api.monitor_response(date)
            actions_api._release_free_memory()
        except Exception:
            pass

    thread = threading.Thread(target=_run, name="vamos-prewarm", daemon=True)
    thread.start()
    return thread


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the Vamos app and local Ask API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args(argv)

    try:
        server = create_server(args.host, args.port)
    except SeedChatsError as error:
        print("Vamos cannot start: {}".format(error), file=sys.stderr)
        return 1
    except OSError as error:
        print("Could not start the server: {}".format(error), file=sys.stderr)
        return 1

    if server.public_mode:
        problems = public_mode_problems(server.ask_service)
        if problems:
            print("Vamos cannot start in public mode:", file=sys.stderr)
            for problem in problems:
                print("  - {}".format(problem), file=sys.stderr)
            server.server_close()
            return 1

    print("Vamos is running at http://{}:{}/".format(args.host, args.port))
    store = getattr(server.ask_service, "store", None)
    if store is not None and not store.ready():
        print(
            "Ask analytics database is missing. Build it with "
            "python3 deployment/olist_dashboard/analytics_db.py",
            file=sys.stderr,
        )
    if getattr(server.ask_service, "model", None) is None:
        print(
            "Ask provider is not configured. Set VAMOS_LLM_BASE_URL, "
            "VAMOS_LLM_API_KEY and VAMOS_LLM_MODEL in the environment or an "
            "uncommitted .env file.",
            file=sys.stderr,
        )
    _prewarm_default_date(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
