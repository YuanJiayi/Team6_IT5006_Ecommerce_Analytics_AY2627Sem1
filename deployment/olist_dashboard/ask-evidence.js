/* Ask evidence: concise period, source, measure and SQL. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AskEvidence = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const ISO_RE = /^\d{4}-\d{2}-\d{2}$/;

  function validIso(value) {
    if (!ISO_RE.test(String(value || ""))) return false;
    const parsed = new Date(String(value) + "T00:00:00Z");
    return !isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
  }

  function addDaysIso(value, count) {
    const parsed = new Date(String(value) + "T00:00:00Z");
    parsed.setUTCDate(parsed.getUTCDate() + count);
    return parsed.toISOString().slice(0, 10);
  }

  function formatDate(value) {
    if (!validIso(value)) return "";
    const parsed = new Date(String(value) + "T00:00:00Z");
    return parsed.getUTCDate() + " " + MONTHS[parsed.getUTCMonth()] + " " + parsed.getUTCFullYear();
  }

  function text(value) {
    return value === null || value === undefined ? "" : String(value).trim();
  }

  function firstIso(values) {
    for (let index = 0; index < values.length; index += 1) {
      if (validIso(values[index])) return values[index];
    }
    return "";
  }

  function plainSource(source) {
    const value = text(source);
    if (!value) return "Olist order data (local)";
    if (value.indexOf("analytics.sqlite") !== -1) {
      return "Olist order data (local database)";
    }
    return value.split(".")[0] + ".";
  }

  function plainMeasure(answer) {
    const table = answer && answer.table;
    const columns = table && Array.isArray(table.columns)
      ? table.columns.filter(function (column) { return text(column); })
      : [];
    if (!columns.length) return "";
    const rows = Array.isArray(table.rows) ? table.rows : [];
    const numeric = columns.filter(function (column) {
      return rows.some(function (row) {
        return row && typeof row[column] === "number" && isFinite(row[column]);
      });
    });
    if (!numeric.length) {
      return "Returned columns (no numeric values in the result): " +
        columns.join(", ") + ".";
    }
    return "Numeric measures returned by the query: " + numeric.join(", ") + ".";
  }

  function entry(label, value) {
    const clean = text(value);
    return clean ? [label, clean] : null;
  }

  function compact(entries) {
    return entries.filter(Boolean);
  }

  function build(payload, fallback) {
    const data = payload && typeof payload === "object" ? payload : {};
    const opts = fallback && typeof fallback === "object" ? fallback : {};
    const answer = data.answer && typeof data.answer === "object" ? data.answer : {};
    const provenance = answer.provenance && typeof answer.provenance === "object"
      ? answer.provenance : {};
    const asOf = firstIso([data.as_of, provenance.as_of, opts.asOf, opts.date]);
    const latest = asOf ? addDaysIso(asOf, -1) : "";
    const kind = data.ok
      ? text(answer.kind) || "query"
      : data.refused ? "refused" : data.error ? "error" : "unknown";

    const primary = [];
    let plain = "";
    const audit = [];
    let sql = null;
    let sqlLabel = "Executed SQL";

    if (kind === "query") {
      plain = text(provenance.plain);
      if (!plain) {
        plain = asOf
          ? "Data as of " + formatDate(asOf) + ". Counted the rows returned by this query."
          : "Counted the rows returned by this query.";
      }
      audit.push(entry("Rows", answer.row_count));
      audit.push(entry("Data available", latest ? "Data through " +
        formatDate(latest) + " (latest complete day)" : ""));
      audit.push(entry("Source", plainSource(provenance.source)));
      audit.push(entry("Measure", plainMeasure(answer)));
      sql = text(answer.sql) || null;
    } else if (kind === "message" || kind === "clarification") {
      primary.push(entry("Source", kind === "clarification"
        ? "No SQL was executed; the model asked for a clearer question."
        : "Conversational reply; no SQL was executed."));
    } else if (kind === "refused") {
      const refusal = data.refusal && typeof data.refusal === "object" ? data.refusal : {};
      primary.push(entry("Reason", refusal.reason || "unsupported"));
      primary.push(entry("Source", "No SQL was executed. Nothing was guessed."));
    } else if (kind === "error") {
      const error = data.error && typeof data.error === "object" ? data.error : {};
      const stage = text(error.stage).toLowerCase();
      let source = "The local Ask server did not return an answer.";
      if (stage === "planning") {
        source = "No query result was produced; no SQL was run for this answer.";
      } else if (stage === "synthesis") {
        source = "SQL ran, but the answer could not be completed.";
      }
      primary.push(entry("Source", source));
      sql = text(error.sql) || null;
      if (sql) sqlLabel = "Rejected SQL";
    }

    return {
      plain: plain,
      primary: compact(primary),
      audit: compact(audit),
      sql: sql,
      sqlLabel: sqlLabel
    };
  }

  function paragraph(doc, label, value) {
    const element = doc.createElement("p");
    const strong = doc.createElement("strong");
    strong.textContent = label + ": ";
    element.append(strong, doc.createTextNode(String(value)));
    return element;
  }

  function render(doc, model) {
    const details = doc.createElement("details");
    details.className = "ask-evidence";
    const summary = doc.createElement("summary");
    summary.textContent = "Evidence";
    const body = doc.createElement("div");
    body.className = "ask-evidence-body";
    if (model.plain) {
      const plain = doc.createElement("p");
      plain.className = "ask-evidence-plain";
      plain.textContent = model.plain;
      body.append(plain);
    }
    model.primary.forEach(function (item) {
      body.append(paragraph(doc, item[0], item[1]));
    });
    if (model.sql) {
      const label = doc.createElement("p");
      const strong = doc.createElement("strong");
      strong.textContent = model.sqlLabel + ": ";
      label.append(strong, doc.createTextNode(model.sqlLabel === "Rejected SQL"
        ? "statement the server did not run"
        : "read-only statement run for this answer"));
      const pre = doc.createElement("pre");
      const code = doc.createElement("code");
      code.textContent = model.sql;
      pre.append(code);
      body.append(label, pre);
    }
    model.audit.forEach(function (item) {
      body.append(paragraph(doc, item[0], item[1]));
    });
    details.append(summary, body);
    return details;
  }

  return {
    build: build,
    render: render,
    formatDate: formatDate,
    addDaysIso: addDaysIso,
    validIso: validIso
  };
});
