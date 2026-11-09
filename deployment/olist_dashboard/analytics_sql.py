"""Narrow, read-only SQL access to the local Olist analytics database.

Ask never accepts SQL from the browser. The configured model proposes one
statement, and this module is the only place it is executed. Guards:

* SQLite is opened in read-only URI mode and query_only mode.
* A SQLite authorizer denies every action except SELECT/READ/FUNCTION.
* Only one SELECT or WITH statement is accepted, and it must include :as_of.
* The selected cutoff is enforced by per-request temporary views over private
  _base_* tables, not by trusting the model's WHERE clause.
* Direct reads of private base tables, internal tables and unknown tables fail.
  A lexical guard also rejects any _base_* identifier, including quoted names and
  CTEs that shadow a public view.
* Personal identifiers, addresses and free-text review comments are unreadable.
* Identifier-like output columns and identifier-shaped values are refused.
* Constant-only queries that never read an analytics table are refused, and
  numeric constant-output projections are rejected even when a FROM clause is
  present.
* A 2-second progress deadline, row/column/cell/output-size bounds.

The model and the user both see the executed SQL; no writes are possible.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path

from analytics_db import (
    BASE_PREFIX,
    BLOCKED_COLUMNS,
    FORMAT_VERSION,
    OUTPUT_BLOCKED_COLUMNS,
    RELATIONSHIPS,
    TABLES,
)

MAX_SQL_CHARS = 12_000
MAX_ROWS = 200
MAX_COLUMNS = 30
MAX_CELL_CHARS = 2_000
MAX_OUTPUT_BYTES = 200_000
DEFAULT_TIMEOUT_SECONDS = 15.0
PROGRESS_INTERVAL = 50_000

ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", -1),
}
BLOCKED_FUNCTIONS = {
    "load_extension",
    "readfile",
    "writefile",
    "edit",
    "fts3_tokenizer",
    "zipfile",
    "substr",
    "substring",
    "printf",
    "format",
    "quote",
    "hex",
    "unhex",
    "char",
    "unicode",
    "replace",
}
INTERNAL_TABLE_PREFIXES = ("sqlite_", "_vamos_", "pragma_")
IDENTIFIER_VALUE_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{16,}(?![0-9a-f])")
PUBLIC_TABLES = frozenset(TABLES)
# The products view is server-authored, so a small set of string functions is
# allowed only when called from that view definition. They remain blocked in the
# model's own SQL, where the authorizer sees no trusted view name.
SAFE_VIEW_FUNCTIONS = {
    "products": frozenset({"substr", "replace"}),
}
ALLOWED_BIND_PARAMETERS = frozenset({":as_of", ":window_start"})
SQL_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
SELECT_TERMINATORS = frozenset(
    {
        "FROM",
        "WHERE",
        "GROUP",
        "HAVING",
        "ORDER",
        "LIMIT",
        "UNION",
        "EXCEPT",
        "INTERSECT",
        "WINDOW",
    }
)
AGGREGATE_FUNCTIONS = frozenset(
    {"count", "sum", "avg", "total", "min", "max", "group_concat"}
)
CUTOFF_METADATA_FUNCTIONS = frozenset({"date", "datetime", "time", "strftime", "coalesce"})
SQL_KEYWORDS = frozenset(
    {
        "ALL",
        "AND",
        "AS",
        "ASC",
        "BETWEEN",
        "BY",
        "CASE",
        "CAST",
        "COLLATE",
        "CURRENT",
        "DESC",
        "DISTINCT",
        "ELSE",
        "END",
        "ESCAPE",
        "EXISTS",
        "FALSE",
        "FILTER",
        "FOLLOWING",
        "GLOB",
        "IN",
        "IS",
        "LIKE",
        "MATCH",
        "NOCASE",
        "NOT",
        "NULL",
        "OFFSET",
        "OR",
        "OVER",
        "PARTITION",
        "PRECEDING",
        "RANGE",
        "REGEXP",
        "ROW",
        "ROWS",
        "THEN",
        "TRUE",
        "UNBOUNDED",
        "WHEN",
        "WITHIN",
        "BLOB",
        "INTEGER",
        "NUMERIC",
        "REAL",
        "TEXT",
    }
)


class QueryError(Exception):
    """The proposed SQL is unsafe, invalid, or outside the result bounds."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _code_only(sql):
    """Return SQL with comments and quoted literals blanked out.

    This is only a cheap guard for the bind parameter and statement count. The
    cutoff itself is enforced by the temporary views in the data access layer.
    """
    output = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if char == "-" and sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            if end == -1:
                break
            output.append(" ")
            index = end + 1
            continue
        if char == "/" and sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end == -1:
                return None
            output.append(" ")
            index = end + 2
            continue
        if char in ("'", '"', "`"):
            quote = char
            index += 1
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            output.append(" ")
            continue
        if char == "[":
            end = sql.find("]", index + 1)
            if end == -1:
                return None
            output.append(" ")
            index = end + 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _sql_tokens(sql):
    """Lex SQL source into words, identifiers, literals and punctuation.

    Comments and string literals are skipped. Quoted identifiers keep their
    contents, which lets the private-table guard see ``"main"."_base_orders"``
    and ``[_base_orders]`` as table references rather than blank text.
    """
    tokens = []
    index = 0
    length = len(sql)
    depth = 0
    while index < length:
        char = sql[index]
        if char.isspace():
            index += 1
            continue
        if char == "-" and sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            index = length if end == -1 else end + 1
            continue
        if char == "/" and sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end == -1:
                return None
            index = end + 2
            continue
        if char == "'":
            start = index
            index += 1
            while index < length:
                if sql[index] == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            tokens.append(("string", sql[start:index], depth, start, index))
            continue
        if char in ('"', "`"):
            quote = char
            start = index
            index += 1
            value = []
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        value.append(quote)
                        index += 2
                        continue
                    index += 1
                    break
                value.append(sql[index])
                index += 1
            else:
                return None
            tokens.append(("ident", "".join(value), depth, start, index))
            continue
        if char == "[":
            start = index
            end = sql.find("]", index + 1)
            if end == -1:
                return None
            tokens.append(("ident", sql[index + 1 : end], depth, start, end + 1))
            index = end + 1
            continue
        if char in (":", "@", "$") and index + 1 < length:
            if sql[index + 1].isalnum() or sql[index + 1] == "_":
                start = index
                index += 2
                while index < length and (sql[index].isalnum() or sql[index] == "_"):
                    index += 1
                tokens.append(("param", sql[start:index], depth, start, index))
                continue
        if char == "?":
            tokens.append(("param", "?", depth, index, index + 1))
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < length and sql[index + 1].isdigit()):
            match = SQL_NUMBER_RE.match(sql, index)
            end = match.end() if match else index + 1
            tokens.append(("number", sql[index:end], depth, index, end))
            index = end
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (sql[index].isalnum() or sql[index] in "_$"):
                index += 1
            tokens.append(("word", sql[start:index], depth, start, index))
            continue
        if char == "(":
            tokens.append(("punct", "(", depth, index, index + 1))
            depth += 1
            index += 1
            continue
        if char == ")":
            if depth > 0:
                depth -= 1
            tokens.append(("punct", ")", depth, index, index + 1))
            index += 1
            continue
        tokens.append(("punct", char, depth, index, index + 1))
        index += 1
    return tokens


def _is_function_call(tokens, position):
    return (
        position + 1 < len(tokens)
        and tokens[position + 1][0] == "punct"
        and tokens[position + 1][1] == "("
    )


def _strip_projection_alias(tokens, depth):
    """Remove a trailing ``AS alias`` or implicit output alias."""
    for position, token in enumerate(tokens):
        if (
            token[0] == "word"
            and token[1].upper() == "AS"
            and token[2] == depth
        ):
            return tokens[:position]
    if len(tokens) >= 2:
        last = tokens[-1]
        previous = tokens[-2]
        if last[0] == "ident" or (
            last[0] == "word" and last[1].upper() not in SQL_KEYWORDS
        ):
            if not (previous[0] == "punct" and previous[1] == "."):
                if not (
                    previous[0] == "punct"
                    and previous[1] in "+-*/%<>=!|&"
                ):
                    return tokens[:-1]
    return tokens


def _is_cutoff_metadata(tokens):
    if len(tokens) < 4:
        return False
    if not (tokens[0][0] == "word" and _is_function_call(tokens, 0)):
        return False
    if tokens[0][1].lower() not in CUTOFF_METADATA_FUNCTIONS:
        return False
    has_as_of = False
    for position, token in enumerate(tokens):
        if token[0] == "number":
            return False
        if token[0] == "param":
            if token[1].lower() not in ALLOWED_BIND_PARAMETERS:
                return False
            if token[1].lower() == ":as_of":
                has_as_of = True
        elif token[0] == "word":
            if _is_function_call(tokens, position):
                if token[1].lower() not in CUTOFF_METADATA_FUNCTIONS:
                    return False
            elif token[1].upper() not in SQL_KEYWORDS:
                return False
    return has_as_of


def _is_constant_projection(tokens, depth):
    """True for an output expression with no column/aggregate data lineage.

    This is deliberately narrow: aggregates and any non-keyword identifier are
    treated as data-bearing. A projection that only combines literals (or only
    uses :as_of) cannot become a trusted numeric answer. The guard does not
    prove lineage for every possible expression; the provenance text says so.
    """
    tokens = _strip_projection_alias(tokens, depth)
    for position, token in enumerate(tokens):
        if (
            token[0] == "word"
            and token[1].lower() in AGGREGATE_FUNCTIONS
            and _is_function_call(tokens, position)
        ):
            return False
    has_data = False
    for position, token in enumerate(tokens):
        if token[0] == "ident":
            has_data = True
            break
        if token[0] == "word":
            if token[1].upper() in SQL_KEYWORDS:
                continue
            if _is_function_call(tokens, position):
                continue
            has_data = True
            break
    if has_data:
        return False
    if _is_cutoff_metadata(tokens):
        return False
    for token in tokens:
        if token[0] == "number":
            return True
        if token[0] == "string" and re.search(r"\d", token[1]):
            return True
    for position, token in enumerate(tokens):
        if (
            token[0] == "word"
            and _is_function_call(tokens, position)
            and token[1].lower() not in CUTOFF_METADATA_FUNCTIONS
        ):
            return True
    return False


def _split_projection(tokens, depth):
    segments = []
    current = []
    for token in tokens:
        if token[0] == "punct" and token[1] == "," and token[2] == depth:
            segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _validate_lexical_guards(sql, tokens):
    for kind, value, _depth, _start, _end in tokens:
        if kind in ("word", "ident") and BASE_PREFIX in value.lower():
            raise QueryError(
                "sql_rejected",
                "Direct references to private _base_* tables are not allowed.",
            )
    for index, token in enumerate(tokens):
        if not (token[0] == "word" and token[1].upper() == "SELECT"):
            continue
        depth = token[2]
        end = index + 1
        while end < len(tokens):
            candidate = tokens[end]
            if candidate[2] < depth:
                break
            if (
                candidate[2] == depth
                and candidate[0] == "word"
                and candidate[1].upper() in SELECT_TERMINATORS
            ):
                break
            end += 1
        for segment in _split_projection(tokens[index + 1 : end], depth):
            if segment and _is_constant_projection(segment, depth):
                raise QueryError(
                    "sql_rejected",
                    "The query projects a constant value with no data lineage; "
                    "return a column or aggregate from the analytics tables.",
                )


def validate_sql_text(sql):
    if not isinstance(sql, str) or not sql.strip():
        raise QueryError("sql_rejected", "The model did not propose any SQL.")
    sql = sql.strip()
    if len(sql) > MAX_SQL_CHARS:
        raise QueryError("sql_rejected", "The proposed SQL is too long.")
    code = _code_only(sql)
    if code is None:
        raise QueryError("sql_rejected", "The proposed SQL has an unterminated comment or quote.")
    tokens = _sql_tokens(sql)
    if tokens is None:
        raise QueryError("sql_rejected", "The proposed SQL has an unterminated comment or quote.")
    body = code.strip()
    if not re.match(r"(?is)^(select|with)\b", body):
        raise QueryError(
            "sql_rejected", "Only a single read-only SELECT or WITH statement is allowed."
        )
    if ":as_of" not in body:
        raise QueryError(
            "sql_rejected",
            "The SQL must include the :as_of bind parameter outside comments or strings.",
        )
    parameters = {
        value.lower()
        for kind, value, _depth, _start, _end in tokens
        if kind == "param"
    }
    if "?" in parameters:
        raise QueryError(
            "sql_rejected",
            "Use named :as_of (and :window_start) bind parameters, not positional ? marks.",
        )
    unknown = parameters - ALLOWED_BIND_PARAMETERS
    if unknown:
        raise QueryError(
            "sql_rejected",
            "Unsupported bind parameter(s): {}. Only :as_of and :window_start are "
            "available.".format(", ".join(sorted(unknown))),
        )
    trimmed = body.rstrip().rstrip(";").rstrip()
    if ";" in trimmed:
        raise QueryError(
            "sql_rejected", "Only one SQL statement is allowed per question."
        )
    _validate_lexical_guards(sql, tokens)
    return sql


def _as_of_view_statements(as_of):
    """Per-request public views over the private base tables.

    The views apply the exclusive cutoff to the event tables. Timestamps for
    events on or after the cutoff are masked so a purchase-cohort query cannot
    see a future delivery or handover. Reviews are visible only once their answer
    timestamp is known and strictly before the cutoff, because review_score is
    not knowable until the review is answered. order_status is masked to
    'delivered' only when the delivery event is visible, and NULL otherwise.
    """
    cutoff = as_of.isoformat()
    literal = "'" + cutoff.replace("'", "''") + "'"
    statements = [
        """
        CREATE TEMP VIEW orders AS
        SELECT
          order_id,
          customer_id,
          CASE WHEN order_delivered_customer_date IS NOT NULL
                    AND order_delivered_customer_date < {cutoff}
               THEN 'delivered' ELSE NULL END AS order_status,
          order_purchase_timestamp,
          CASE WHEN order_approved_at IS NOT NULL AND order_approved_at < {cutoff}
               THEN order_approved_at ELSE NULL END AS order_approved_at,
          CASE WHEN order_delivered_carrier_date IS NOT NULL
                    AND order_delivered_carrier_date < {cutoff}
               THEN order_delivered_carrier_date ELSE NULL END AS order_delivered_carrier_date,
          CASE WHEN order_delivered_customer_date IS NOT NULL
                    AND order_delivered_customer_date < {cutoff}
               THEN order_delivered_customer_date ELSE NULL END AS order_delivered_customer_date,
          order_estimated_delivery_date
        FROM {base}orders
        WHERE order_purchase_timestamp IS NOT NULL
          AND order_purchase_timestamp < {cutoff}
        """.format(cutoff=literal, base=BASE_PREFIX),
        """
        CREATE TEMP VIEW order_items AS
        SELECT oi.order_id, oi.order_item_id, oi.product_id, oi.seller_id,
               oi.shipping_limit_date, oi.price, oi.freight_value
        FROM {base}order_items oi
        JOIN {base}orders o ON o.order_id = oi.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
          AND o.order_purchase_timestamp < {cutoff}
        """.format(cutoff=literal, base=BASE_PREFIX),
        """
        CREATE TEMP VIEW payments AS
        SELECT p.order_id, p.payment_sequential, p.payment_type,
               p.payment_installments, p.payment_value
        FROM {base}payments p
        JOIN {base}orders o ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
          AND o.order_purchase_timestamp < {cutoff}
        """.format(cutoff=literal, base=BASE_PREFIX),
        """
        CREATE TEMP VIEW reviews AS
        SELECT r.review_id, r.order_id, r.review_score,
               r.review_creation_date,
               CASE WHEN r.review_answer_timestamp IS NOT NULL
                         AND r.review_answer_timestamp < {cutoff}
                    THEN r.review_answer_timestamp ELSE NULL END AS review_answer_timestamp
        FROM {base}reviews r
        JOIN {base}orders o ON o.order_id = r.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
          AND o.order_purchase_timestamp < {cutoff}
          AND r.review_creation_date IS NOT NULL
          AND r.review_creation_date < {cutoff}
          AND r.review_answer_timestamp IS NOT NULL
          AND r.review_answer_timestamp < {cutoff}
        """.format(cutoff=literal, base=BASE_PREFIX),
    ]
    for table in ("customers", "sellers", "geolocation", "product_category_translation"):
        blocked = BLOCKED_COLUMNS.get(table, set())
        columns = ", ".join(
            name for name, _ in TABLES[table]["columns"] if name not in blocked
        )
        statements.append(
            "CREATE TEMP VIEW {table} AS SELECT {columns} FROM {base}{table}".format(
                table=table, columns=columns, base=BASE_PREFIX
            )
        )
    # Products carry a plain display name for category grouping. The English
    # translation is the source; the raw Portuguese slug is only a fallback.
    # SUBSTR/REPLACE are allowed inside this one trusted view (see the
    # authorizer) and stay blocked in model-proposed SQL.
    statements.append(
        """
        CREATE TEMP VIEW products AS
        SELECT
          p.product_id,
          p.product_category_name,
          UPPER(SUBSTR(REPLACE(COALESCE(t.product_category_name_english,
                                        p.product_category_name), '_', ' '), 1, 1))
          || LOWER(SUBSTR(REPLACE(COALESCE(t.product_category_name_english,
                                           p.product_category_name), '_', ' '), 2))
            AS product_category_display,
          p.product_name_lenght,
          p.product_description_lenght,
          p.product_photos_qty,
          p.product_weight_g,
          p.product_length_cm,
          p.product_height_cm,
          p.product_width_cm
        FROM {base}products p
        LEFT JOIN {base}product_category_translation t
          ON t.product_category_name = p.product_category_name
        """.format(base=BASE_PREFIX)
    )
    return statements


def _create_as_of_views(connection, as_of):
    for statement in _as_of_view_statements(as_of):
        connection.execute(statement)


def _make_authorizer(read_state):
    blocked = {
        table: {column.lower() for column in columns}
        for table, columns in BLOCKED_COLUMNS.items()
    }

    def authorizer(action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            column = (arg2 or "").lower()
            view = (trigger or "").lower()
            if table.startswith(INTERNAL_TABLE_PREFIXES):
                return sqlite3.SQLITE_DENY
            if table.startswith(BASE_PREFIX):
                public_table = table[len(BASE_PREFIX) :]
                if public_table not in PUBLIC_TABLES or view not in PUBLIC_TABLES:
                    return sqlite3.SQLITE_DENY
            else:
                public_table = table
                if public_table not in PUBLIC_TABLES:
                    return sqlite3.SQLITE_DENY
            if column and column in blocked.get(public_table, set()):
                return sqlite3.SQLITE_DENY
            read_state["data_read"] = True
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            name = (arg2 or "").lower()
            if name in BLOCKED_FUNCTIONS:
                allowed = SAFE_VIEW_FUNCTIONS.get((trigger or "").lower(), ())
                return sqlite3.SQLITE_OK if name in allowed else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action in ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorizer


def _open_read_only(path, query_only=True):
    connection = sqlite3.connect(
        "{}?mode=ro".format(Path(path).resolve().as_uri()),
        uri=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-16000")
    connection.execute("PRAGMA mmap_size=67108864")
    if query_only:
        connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    return connection


def load_metadata(path):
    connection = _open_read_only(path)
    try:
        row = connection.execute(
            "SELECT value FROM _vamos_meta WHERE key = 'metadata'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise QueryError(
            "analytics_not_ready",
            "The analytics database has no schema metadata. Rebuild it with "
            "python3 deployment/olist_dashboard/analytics_db.py.",
        )
    try:
        metadata = json.loads(row[0])
    except (TypeError, ValueError):
        raise QueryError(
            "analytics_not_ready",
            "The analytics database metadata is unreadable. Rebuild it.",
        ) from None
    if metadata.get("format_version") != FORMAT_VERSION:
        raise QueryError(
            "analytics_not_ready",
            "The analytics database uses an older format. Rebuild it with "
            "python3 deployment/olist_dashboard/analytics_db.py.",
        )
    return metadata


def _column_line(table):
    parts = []
    for name, kind in TABLES[table]["columns"]:
        note = ""
        if name in BLOCKED_COLUMNS.get(table, set()):
            note = " [blocked]"
        parts.append("{} {}{}".format(name, kind, note))
    return ", ".join(parts)


def schema_context(metadata, as_of):
    """Prompt-ready schema, relationships, metric definitions and SQL rules."""
    lines = ["Available tables (public Olist raw data, one local SQLite file):"]
    for table in TABLES:
        line = "- {}({}) -- {}".format(
            table, _column_line(table), TABLES[table]["description"]
        )
        if table == "products":
            line += (
                " Computed column product_category_display TEXT gives the English "
                "category name with underscores turned into spaces and only the "
                "first letter capitalised, for example 'Bed bath table'."
            )
        lines.append(line)
    lines.append("")
    lines.append("Cutoff enforcement (server-side, not model-controlled):")
    lines.append(
        "- Each public table above is a per-request temporary view over private "
        "_base_* tables. The selected exclusive cutoff is applied before your SQL "
        "runs; direct reads of _base_* tables are denied."
    )
    lines.append(
        "- orders contains only orders purchased before the cutoff. order_items and "
        "payments contain only rows tied to those visible orders. Approval, carrier "
        "and customer-delivery timestamps on or after the cutoff are NULL. Reviews "
        "appear only when review_answer_timestamp is non-null and strictly before "
        "the cutoff: review_score is not knowable until the review is answered, so "
        "reviews created but not yet answered are absent. review_answer_timestamp "
        "is therefore always a visible, non-null timestamp."
    )
    lines.append(
        "- customers, sellers, products, geolocation and "
        "product_category_translation are reference/catalog tables, not event "
        "tables, and are not date-filtered."
    )
    lines.append(
        "- order_status is masked to historically knowable state: it is 'delivered' "
        "only when the customer-delivery event is visible before the cutoff, and "
        "NULL otherwise. There is no status-event timestamp, so shipped, canceled "
        "and other earlier states cannot be reconstructed. Do not treat NULL as a "
        "cancellation or present any final status as the status at the cutoff."
    )
    lines.append("")
    lines.append("Relationships and key joins:")
    for left, right, cardinality in RELATIONSHIPS:
        lines.append("- {} = {} ({})".format(left, right, cardinality))
    lines.append("")
    bounds = metadata.get("event_bounds") or {}
    if bounds:
        lines.append("Observed data range:")
        for key, value in bounds.items():
            if value.get("min") or value.get("max"):
                lines.append(
                    "- {}: {} to {}".format(key, value.get("min"), value.get("max"))
                )
        lines.append("")
    lines.append("Metric definitions (be explicit and honest):")
    lines.append(
        "- Orders: COUNT(DISTINCT order_id) over rows whose order_purchase_timestamp "
        "is non-null and strictly before :as_of (purchase cohort)."
    )
    lines.append(
        "- Item price: SUM(order_items.price) over item rows joined to the purchase "
        "cohort. Freight is a separate column. Item price is not revenue, and "
        "payment_value is not item price."
    )
    lines.append(
        "- Deliveries: rows whose order_delivered_customer_date is non-null and "
        "strictly before :as_of (delivery cohort)."
    )
    lines.append(
        "- order_estimated_delivery_date is a calendar promise date, not a "
        "timestamp. Compare calendar dates, not raw timestamps."
    )
    lines.append(
        "- On-time delivery: date(order_delivered_customer_date) <= "
        "date(order_estimated_delivery_date), with both non-null. A delivery on "
        "the promised date is on time."
    )
    lines.append(
        "- Late delivery: date(order_delivered_customer_date) > "
        "date(order_estimated_delivery_date), with both non-null."
    )
    lines.append(
        "- Review score comes from reviews joined by order_id that were answered "
        "before the cutoff; review text is blocked. A review created before the "
        "cutoff but answered on or after it is not counted or scored."
    )
    lines.append(
        "- Product category English names come from product_category_translation. "
        "Use products.product_category_display for labels; do not return the raw "
        "underscore slug."
    )
    lines.append("")
    lines.append("Time basis and incomplete cohorts:")
    lines.append(
        "- Outcome measures over time (average delivery days, late rate, on-time "
        "rate, review score) are grouped by the outcome date: the customer delivery "
        "date for delivery outcomes, or the review answer date for review outcomes. "
        "This matches Overview's 'by delivery date'. Group by purchase date only "
        "when the person explicitly asks for a purchase cohort or purchase trend."
    )
    lines.append(
        "- If a purchase-date cohort is used for a delivery or review outcome, also "
        "return per period share_still_open = orders with no visible delivery before "
        "the cutoff divided by all orders in that period, as a 0..1 number. A period "
        "with more than 5% still open (below 95% delivered) is incomplete and must "
        "be compared cautiously. The server rejects a purchase-period outcome query "
        "that omits share_still_open."
    )
    lines.append(
        "- For weekly series return the week's Monday start date as a full ISO date "
        "using date(<date column>, 'weekday 0', '-6 days') AS week_start. Never "
        "return a year-week number such as 2018-14."
    )
    lines.append(
        "- A rolling window ('last 90 days', 'this year so far', 'last 6 months') "
        "compares the date column to :window_start and :as_of, and keeps the asked "
        "window as the default with COALESCE(:window_start, date(:as_of, '-90 days')). "
        "Use days, months or 'start of year' in that fallback. A fixed period "
        "('June 2018', 'in 2017') uses literal date bounds such as "
        ">= '2018-06-01' AND < '2018-07-01' and does not use :window_start."
    )
    lines.append("")
    lines.append("SQL rules:")
    lines.append(
        "- One SELECT or WITH statement only. No writes, ATTACH, PRAGMA or "
        "multiple statements."
    )
    lines.append(
        "- The statement MUST include the :as_of bind parameter outside comments and "
        "string literals; the server binds the selected YYYY-MM-DD cutoff. The views "
        "already enforce it, so do not hard-code a different date."
    )
    lines.append(
        "- Do not query private _base_* tables, sqlite_* tables or _vamos_meta. "
        "The query must read at least one public analytics table; constant-only "
        "SELECTs are rejected."
    )
    lines.append(
        "- Prefer aggregate results with clear aliases. Never select blocked columns "
        "or raw customer/order/seller/review identifiers, even with an alias or "
        "string concatenation. Identifier-shaped output values are rejected; do not "
        "use substr/printf/hex/quote/replace to transform identifiers."
    )
    lines.append(
        "- Every result column must come from the tables. Do not project a numeric "
        "literal (for example 999999 AS orders) or literal arithmetic and present it "
        "as data; such constant-output projections are rejected. Aggregates and "
        "expressions over columns are fine."
    )
    lines.append(
        "- Return only numbers and dates that appear in the query result or the "
        "selected cutoff. Do not derive window start dates in prose; if a date "
        "matters, return it as a column."
    )
    lines.append(
        "- Return at most 200 rows and 30 columns. Use LIMIT for rankings; aggregate "
        "high-cardinality dimensions."
    )
    lines.append(
        "- A time series should return a single date/period column plus numeric "
        "aggregate columns. A ranking should return one label column plus numeric "
        "aggregate columns."
    )
    lines.append(
        "- Observed associations are not causal. Do not add external data or "
        "invent numbers; if the tables cannot answer the question, say so."
    )
    lines.append("")
    lines.append(
        "Selected Data as of (exclusive cutoff): {}. Events strictly before this "
        "date are visible; the latest complete day is the day before.".format(
            as_of.isoformat()
        )
    )
    return "\n".join(lines)


def _clean_value(value, column):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QueryError(
                "invalid_result", "Query returned a non-finite number for {}.".format(column)
            )
        return value
    if isinstance(value, str):
        if "\x00" in value:
            raise QueryError("invalid_result", "Query returned a NUL character.")
        if len(value) > MAX_CELL_CHARS:
            raise QueryError(
                "invalid_result",
                "Query returned an oversized text value for {}. Aggregate instead.".format(
                    column
                ),
            )
        if IDENTIFIER_VALUE_RE.search(value):
            raise QueryError(
                "privacy_blocked",
                "Query returned an identifier-like value; aggregate or label it instead.",
            )
        return value
    raise QueryError(
        "invalid_result",
        "Query returned an unsupported value type for {}.".format(column),
    )


class AnalyticsStore:
    """Owns the database path and executes model-proposed SQL safely."""

    def __init__(
        self,
        path,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_rows=MAX_ROWS,
        max_columns=MAX_COLUMNS,
        max_output_bytes=MAX_OUTPUT_BYTES,
    ):
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_output_bytes = max_output_bytes
        self._metadata = None

    def ready(self):
        return self.path.is_file()

    def metadata(self):
        if self._metadata is None:
            if not self.ready():
                raise QueryError(
                    "analytics_not_ready",
                    "The local analytics database is missing. Build it with "
                    "python3 deployment/olist_dashboard/analytics_db.py.",
                )
            self._metadata = load_metadata(self.path)
        return self._metadata

    def schema_context(self, as_of):
        return schema_context(self.metadata(), as_of)

    def execute(self, sql, as_of, params=None):
        sql = validate_sql_text(sql)
        if not self.ready():
            raise QueryError(
                "analytics_not_ready",
                "The local analytics database is missing. Build it with "
                "python3 deployment/olist_dashboard/analytics_db.py.",
            )
        self.metadata()
        connection = _open_read_only(self.path, query_only=False)
        try:
            _create_as_of_views(connection, as_of)
            connection.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + self.timeout_seconds

            def progress():
                return 1 if time.monotonic() > deadline else 0

            read_state = {"data_read": False}
            connection.set_progress_handler(progress, PROGRESS_INTERVAL)
            connection.set_authorizer(_make_authorizer(read_state))
            bindings = {"as_of": as_of.isoformat()}
            if isinstance(params, dict):
                for key, value in params.items():
                    name = key if str(key).startswith(":") else ":" + str(key)
                    bindings[name.lstrip(":")] = value
            cursor = connection.execute(sql, bindings)
            if not read_state["data_read"]:
                raise QueryError(
                    "invalid_query",
                    "The query must read at least one analytics table; "
                    "constant-only queries are not accepted.",
                )
            description = cursor.description or []
            columns = [item[0] for item in description]
            if not columns:
                raise QueryError("invalid_result", "Query returned no columns.")
            if len(columns) > self.max_columns:
                raise QueryError(
                    "column_limit",
                    "Query returned more than {} columns.".format(self.max_columns),
                )
            if len({column.lower() for column in columns}) != len(columns):
                raise QueryError(
                    "invalid_result",
                    "Query returned duplicate column names; add distinct aliases.",
                )
            blocked = [
                column
                for column in columns
                if column.lower() in OUTPUT_BLOCKED_COLUMNS
            ]
            if blocked:
                raise QueryError(
                    "privacy_blocked",
                    "Query returned identifier columns ({}); aggregate instead.".format(
                        ", ".join(sorted(blocked))
                    ),
                )
            fetched = cursor.fetchmany(self.max_rows + 1)
            if len(fetched) > self.max_rows:
                raise QueryError(
                    "row_limit",
                    "Query returned more than {} rows; aggregate or limit it.".format(
                        self.max_rows
                    ),
                )
            rows = []
            for raw in fetched:
                rows.append(
                    {
                        column: _clean_value(value, column)
                        for column, value in zip(columns, raw)
                    }
                )
            result = {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
            size = len(
                json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
            )
            if size > self.max_output_bytes:
                raise QueryError(
                    "output_limit",
                    "Query result is larger than the output bound; aggregate more.",
                )
            return result
        except sqlite3.ProgrammingError as error:
            message = str(error)
            if "binding" in message.lower():
                raise QueryError(
                    "sql_rejected",
                    "The query needs a bind value the server did not supply: {}.".format(
                        message
                    ),
                ) from None
            raise QueryError(
                "sql_rejected",
                "Only one read-only SELECT or WITH statement is allowed.",
            ) from None
        except sqlite3.OperationalError as error:
            message = str(error)
            if "interrupted" in message.lower() or "timeout" in message.lower():
                raise QueryError(
                    "query_timeout",
                    "The query exceeded the {} second time limit; aggregate more.".format(
                        self.timeout_seconds
                    ),
                ) from None
            if (
                "no such" in message.lower()
                or "not authorized" in message.lower()
                or "prohibited" in message.lower()
            ):
                raise QueryError(
                    "sql_rejected",
                    "SQLite rejected the statement: {}.".format(message),
                ) from None
            raise QueryError("sql_error", "SQLite could not run the query: {}.".format(message)) from None
        except sqlite3.DatabaseError as error:
            message = str(error)
            if (
                "no such" in message.lower()
                or "not authorized" in message.lower()
                or "prohibited" in message.lower()
            ):
                raise QueryError(
                    "sql_rejected",
                    "SQLite rejected the statement: {}.".format(message),
                ) from None
            raise QueryError(
                "sql_error", "SQLite could not run the query: {}.".format(message)
            ) from None
        finally:
            connection.set_progress_handler(None, 0)
            connection.close()


__all__ = [
    "AnalyticsStore",
    "QueryError",
    "schema_context",
    "validate_sql_text",
]
