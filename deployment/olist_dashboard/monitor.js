(function () {
  "use strict";

  var state = {
    date: null,
    loading: false,
    stale: false,
    payload: null,
    requestId: 0
  };
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function $(id) { return document.getElementById(id); }

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function format(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
    return Number(value).toFixed(digits);
  }

  function percent(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
    return (Number(value) * 100).toFixed(digits) + "%";
  }

  function monthName(value) {
    var parts = String(value || "").split("-");
    var index = Number(parts[1]) - 1;
    return MONTHS[index] ? MONTHS[index] + " " + parts[0] : value;
  }

  function levelClass(level) {
    if (level === "act") return "is-act";
    if (level === "watch") return "is-watch";
    if (level === "early") return "is-early";
    return "is-ok";
  }

  function levelLabel(level) {
    if (level === "act") return "Act";
    if (level === "watch") return "Watch";
    if (level === "early") return "Early";
    return "OK";
  }

  function statusNode(level) {
    return node("span", "monitor-status " + levelClass(level), levelLabel(level));
  }

  function cell(text, className, title) {
    var td = node("td", className || "");
    td.textContent = text;
    if (title) td.title = title;
    return td;
  }

  function abbr(text, title, className) {
    var element = node("abbr", className || "", text);
    element.title = title;
    return element;
  }

  function labelled(text, key, suffix, prefix) {
    var span = node("span");
    if (prefix) span.appendChild(document.createTextNode(prefix));
    span.appendChild(abbr(text, DEFINITIONS[key]));
    if (suffix) span.appendChild(document.createTextNode(suffix));
    return span;
  }

  var DEFINITIONS = {
    ks: "Kolmogorov\u2013Smirnov distance: how far this month's spread of values is from training (0 = same, 1 = completely different).",
    tv: "Total variation: the share of orders whose category mix differs from training.",
    mae: "Mean absolute error: the average size of the prediction error.",
    rmse: "Root mean squared error: like MAE but penalises large errors more.",
    pr_auc: "Precision-recall area under the curve: how well late parcels rank above on-time ones; 1 = perfect.",
    roc_auc: "ROC area under the curve: how well the score separates late from on-time parcels; 0.5 = random."
  };

  function table(headers, rows) {
    var element = node("table", "monitor-table");
    var thead = node("thead");
    var headRow = node("tr");
    headers.forEach(function (header) {
      var text = typeof header === "string" ? header : header.text;
      var th = node("th", "", text);
      if (header && header.element) {
        th.textContent = "";
        th.appendChild(header.element);
      }
      if (header && header.title) th.title = header.title;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    element.appendChild(thead);
    var tbody = node("tbody");
    rows.forEach(function (row) {
      var tr = node("tr", row.className || "");
      if (row.hidden) tr.hidden = true;
      row.cells.forEach(function (item) {
        var td = cell(item.text, item.className || "", item.title || "");
        if (item.element) {
          td.textContent = "";
          td.appendChild(item.element);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    element.appendChild(tbody);
    return element;
  }

  function block(title) {
    var section = node("section", "monitor-block");
    section.appendChild(node("h3", "", title));
    return section;
  }

  function caption(text) {
    return node("p", "monitor-caption", text);
  }

  function methodDetails(paragraphs) {
    var details = node("details", "monitor-method");
    details.appendChild(node("summary", "", "How this is measured"));
    paragraphs.forEach(function (text) {
      if (text) details.appendChild(node("p", "", text));
    });
    return details;
  }

  function renderHeader(payload) {
    var head = node("div", "monitor-head");
    var left = node("div");
    left.appendChild(node("p", "eyebrow", "Model monitoring"));
    var title = node("h2", "", "Monitoring");
    title.id = "monitor-title";
    left.appendChild(title);
    left.appendChild(node("p", "monitor-note", payload.note));
    head.appendChild(left);

    var meta = node("div", "monitor-meta");
    meta.appendChild(node("span", "", "As of " + payload.as_of + " 09:00 \u00b7 " +
      (payload.reference ? payload.reference.orders.toLocaleString() : "0") +
      " training orders"));
    if (payload.report_url) {
      var link = node("a", "monitor-link", "Input-drift report (Evidently) \u2197");
      link.href = payload.report_url;
      link.target = "_blank";
      link.rel = "noopener";
      meta.appendChild(link);
    }
    head.appendChild(meta);
    return head;
  }

  function renderStatusCards(payload) {
    var section = block("Model status");
    var grid = node("div", "monitor-cards");
    (payload.status_cards || []).forEach(function (card) {
      var element = node("div", "monitor-card " + levelClass(card.level));
      var head = node("div", "monitor-card-head");
      head.appendChild(node("strong", "", card.title));
      head.appendChild(statusNode(card.level));
      element.appendChild(head);
      element.appendChild(node("p", "monitor-card-reason", card.reason));
      var action = card.action + (card.input_note ? " \u00b7 " + card.input_note : "");
      element.appendChild(node("p", "monitor-card-action", action));
      grid.appendChild(element);
    });
    section.appendChild(grid);
    return section;
  }

  function renderAlerts(payload) {
    var details = node("details", "monitor-method monitor-history");
    var count = payload.alerts ? payload.alerts.length : 0;
    details.appendChild(node("summary", "", "History (" + count + " alerts)"));
    var strip = node("div", "monitor-alerts");
    strip.setAttribute("aria-label", "Watch and Act alerts");
    if (!payload.alerts || !payload.alerts.length) {
      strip.appendChild(node("p", "monitor-empty", "No Watch or Act model alerts at this date."));
    } else {
      payload.alerts.forEach(function (alert) {
        var row = node("div", "monitor-alert " + levelClass(alert.level));
        row.appendChild(node("span", "monitor-badge", levelLabel(alert.level)));
        row.appendChild(node("span", "", alert.sentence));
        strip.appendChild(row);
      });
    }
    details.appendChild(strip);
    return details;
  }

  function renderInputs(payload) {
    var inputs = payload.inputs;
    var validation = inputs.validation || {};
    var section = block("Inputs (X): are orders changing?");
    if (!inputs.windows || !inputs.windows.length) {
      section.appendChild(node("p", "monitor-empty", "No later purchase month is known by this date."));
      return section;
    }
    var readings = validation.reading_count || 11;
    var readingNoun = validation.reading === "window" ? "windows" : "months";
    section.appendChild(caption(
      "Shaded = above its highest score in the " + readings + " validation " + readingNoun +
      ". Early warning only."));
    var windows = inputs.windows.slice(-6);  // latest six months only
    var headers = ["Input", "Used by"].concat(windows.map(monthName));
    var isFlagged = function (feature) {
      return windows.some(function (month) {
        var value = feature.months[month];
        return value && value.level === "unusual";
      });
    };
    var rows = inputs.features.map(function (feature) {
      var kind = feature.type === "categorical" ? "TV" : "KS";
      var title = "Unusual above " + format(feature.threshold, 3) +
        ", the highest score in the " + readings + " validation " + readingNoun +
        ". Validation scores: " +
        feature.validation.map(function (value) { return format(value, 3); }).join(", ") + ".";
      var label = node("span", "", feature.label + "  (");
      label.appendChild(abbr(kind, DEFINITIONS[kind.toLowerCase()]));
      label.appendChild(document.createTextNode(")"));
      var cells = [{
        element: label,
        className: ""
      }, { text: feature.used_by.join(", "), className: "" }];
      windows.forEach(function (month) {
        var value = feature.months[month];
        if (!value) {
          cells.push({ text: "n/a", className: "" });
        } else {
          cells.push({
            text: format(value.score, 3),
            className: value.level === "unusual" ? "is-unusual" : "",
            title: title
          });
        }
      });
      return { cells: cells, hidden: !isFlagged(feature) };
    });
    var shown = rows.filter(function (row) { return !row.hidden; }).length;
    var tableElement = table(headers, rows);
    section.appendChild(tableElement);
    if (shown < rows.length) {
      var toggle = node("button", "monitor-toggle", "Show all " + rows.length + " inputs");
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.addEventListener("click", function () {
        var open = toggle.getAttribute("aria-expanded") === "true";
        var tableRows = tableElement.querySelectorAll("tbody tr");
        tableRows.forEach(function (tr, index) {
          if (rows[index].hidden) tr.hidden = open;
        });
        toggle.setAttribute("aria-expanded", open ? "false" : "true");
        toggle.textContent = open ? "Show all " + rows.length + " inputs" : "Show flagged only";
      });
      section.appendChild(toggle);
    }
    var paragraphs = [
      inputs.threshold_note,
      inputs.importance_note,
      inputs.recalibration_note,
      inputs.source
    ];
    section.appendChild(methodDetails(paragraphs));
    return section;
  }

  function rangeText(bounds, formatter) {
    if (!bounds) return "n/a";
    return formatter(bounds.min) + " to " + formatter(bounds.max);
  }

  function renderOutcomes(payload) {
    var outcomes = payload.outcomes;
    var ranges = outcomes.reference_ranges || {};
    var days = function (v) { return format(v, 2); };
    var pct = function (v) { return percent(v, 2); };
    var minOrders = outcomes.reference_min_orders || 500;
    var rule = "A month's late rate is ready at 95% delivered; orders still open past their promise " +
      "count as late. Normal range = lowest to highest of the training months with at least " +
      minOrders + " orders.";
    var section = block("Outcomes (Y): are deliveries changing?");
    section.appendChild(caption("Shaded = outside the normal range."));
    var headers = ["Month", "Orders", "Delivered", "Still open", "Avg delivery days",
      "Avg promised days", "Late rate"];
    var shade = function (level) { return level === "none" ? "" : levelClass(level); };
    var rows = [{
      className: "is-range",
      cells: [
        { text: "Normal range" }, { text: "" }, { text: "" }, { text: "" },
        { text: rangeText(ranges.mean_delivery_days, days) },
        { text: rangeText(ranges.mean_promised_days, days) },
        { text: rangeText(ranges.late_rate, pct) }
      ]
    }].concat(outcomes.months.map(function (month) {
      var notReady = { text: "Not ready", title: "Under 95% of this month's orders are delivered." };
      var late = month.late_mature ?
        { text: pct(month.late_rate), className: shade(month.levels.late_rate),
          title: month.late_orders + " late of " + month.known_orders + " orders with a known outcome" } :
        { text: notReady.text, title: notReady.title };
      return {
        className: !month.late_mature ? "is-arriving" : "",
        cells: [
          { text: monthName(month.month) },
          { text: month.orders.toLocaleString() },
          { text: month.matured_orders.toLocaleString() + " (" + percent(month.share_matured, 0) + ")" },
          { text: month.open_orders.toLocaleString(),
            title: month.open_past_promise + " open past the promise date count as late; " +
              month.unknown_orders + " open not yet past it are unknown" },
          month.late_mature ?
            { text: days(month.mean_delivery_days), className: shade(month.levels.mean_delivery_days) } :
            { text: notReady.text, title: notReady.title },
          { text: days(month.mean_promised_days), className: shade(month.levels.mean_promised_days) },
          late
        ]
      };
    }));
    section.appendChild(table(headers, rows));
    section.appendChild(methodDetails([rule]));
    return section;
  }

  function ruleEdge(gap, digits) {
    if (!gap || gap.gap === null || gap.gap === undefined) return { text: "n/a", title: "" };
    var low = gap.low;
    var high = gap.high;
    return {
      text: format(gap.gap, digits) + " (" + format(low, digits) + " to " + format(high, digits) + ")",
      title: "Negative means the rule is better; positive means the model is better. The interval is 95%."
    };
  }

  function modelValidation(model) {
    var validation = model.validation || {};
    if (model.kind === "regression") {
      var text = model.title + ": validation MAE " + format(validation.mean, 2) + " \u00b1 " +
        format(validation.sd, 2) + " (" + validation.n_windows + " windows)";
      if (model.bias_bounds) {
        text += "; validation bias " + format(model.bias_bounds.min, 2) + " to " +
          format(model.bias_bounds.max, 2) + " days";
      }
      return text + ".";
    }
    return "Late-risk blend: validation ROC-AUC " + format(validation.roc_auc.mean, 3) + " \u00b1 " +
      format(validation.roc_auc.sd, 3) + " (" + validation.n_windows + " windows).";
  }

  function renderModelChecks(payload) {
    var checks = payload.model_checks;
    var start = String(checks.test_start || "");
    var testMonth = function (value) {
      // The first test month is partial: name the days it covers.
      if (!start || start.slice(0, 7) !== value || start.slice(8, 10) === "01") return monthName(value);
      var parts = value.split("-");
      var lastDay = new Date(Date.UTC(Number(parts[0]), Number(parts[1]), 0)).getUTCDate();
      return Number(start.slice(8, 10)) + "\u2013" + lastDay + " " + monthName(value);
    };
    var section = block("Model checks (X\u2192Y): are the models still right?");
    section.appendChild(caption((start ? "Models deployed from " + Number(start.slice(8, 10)) + " " + monthName(start.slice(0, 7)) + ". " : "") +
      "Each row is one month; the advantage adds up the months so far."));
    checks.models.forEach(function (model) {
      var card = node("div", "monitor-model");
      var head = node("div", "monitor-model-head");
      head.appendChild(node("strong", "", model.title));
      head.appendChild(node("span", "", "Rule: " + model.rule_label));
      card.appendChild(head);

      if (!model.months || !model.months.length) {
        card.appendChild(node("p", "monitor-empty", "No mature month yet; outcomes are still arriving."));
        section.appendChild(card);
        return;
      }
      var headers, rows;
      if (model.kind === "regression") {
        headers = [
          "Month",
          "Orders",
          { element: labelled("MAE", "mae", " (days)", "Model "), title: "The model's error; lower is better." },
          { element: labelled("MAE", "mae", " (days)", "Rule "), title: "The operations rule's error; lower is better." },
          { text: "Too long (+) / too short (\u2212), days", title: "Average of prediction minus actual. Shaded when outside the validation range, in either direction." },
          { text: "Model advantage (95% range)", title: "Positive = model better, negative = rule better." },
          "Status"
        ];
        rows = model.months.map(function (month) {
          if (month.too_few) {
            return {
              cells: [
                { text: testMonth(month.month) },
                { text: month.orders.toLocaleString() },
                { text: "n/a" }, { text: "n/a" }, { text: "n/a" }, { text: "n/a" },
                { text: "Too few to judge", title: month.reason }
              ]
            };
          }
          var edge = ruleEdge(month.rule_gap, 2);
          return {
            cells: [
              { text: testMonth(month.month) },
              { text: month.orders.toLocaleString() },
              { text: format(month.model.mae, 2),
                className: month.levels.performance === "watch" ? "is-watch" : "" },
              { text: format(month.rule.mae, 2) },
              { text: format(month.model.signed_error, 2),
                className: month.levels.bias === "none" ? "" : levelClass(month.levels.bias),
                title: "Positive means the model predicts too long." },
              { text: edge.text, title: edge.title,
                className: month.levels.rule === "act" ? "is-act" : "" },
              { text: levelLabel(month.level), className: month.level === "none" ? "" : levelClass(month.level),
                title: month.reason }
            ]
          };
        });
      } else {
        var rocHead = node("span", "", "Blend ");
        rocHead.appendChild(abbr("ROC-AUC", DEFINITIONS.roc_auc));
        var blendHead = node("span", "", "Blend score (");
        blendHead.appendChild(abbr("PR-AUC", DEFINITIONS.pr_auc));
        blendHead.appendChild(document.createTextNode(")"));
        var ruleHead = node("span", "", "Rule's score (");
        ruleHead.appendChild(abbr("PR-AUC", DEFINITIONS.pr_auc));
        ruleHead.appendChild(document.createTextNode(")"));
        headers = [
          "Month",
          "Orders",
          "Late orders",
          { element: rocHead, title: "Checked against validation; does not move with the late rate. Higher is better." },
          { element: blendHead, title: "Higher is better." },
          { element: ruleHead, title: "Higher is better." },
          { text: "Blend advantage (95% range)", title: "Positive = blend better, negative = rule better." },
          "Status"
        ];
        rows = model.months.map(function (month) {
          if (month.too_few) {
            return {
              cells: [
                { text: testMonth(month.month) },
                { text: month.orders.toLocaleString() },
                { text: month.late_orders.toLocaleString() },
                { text: "n/a" }, { text: "n/a" }, { text: "n/a" }, { text: "n/a" },
                { text: "Too few to judge", title: month.reason }
              ]
            };
          }
          var edge = ruleEdge(month.rule_gap, 3);
          return {
            cells: [
              { text: testMonth(month.month) },
              { text: month.orders.toLocaleString() },
              { text: month.late_orders.toLocaleString() },
              { text: format(month.model.roc_auc, 3),
                className: month.levels.performance === "watch" ? "is-watch" : "" },
              { text: format(month.model.pr_auc, 3) },
              { text: format(month.rule.pr_auc, 3) },
              { text: edge.text, title: edge.title,
                className: month.levels.rule === "act" ? "is-act" : "" },
              { text: levelLabel(month.level), className: month.level === "none" ? "" : levelClass(month.level),
                title: month.reason }
            ]
          };
        });
      }
      card.appendChild(table(headers, rows));
      section.appendChild(card);
    });
    var paragraphs = [
      checks.test_start_note,
      checks.rule_note,
      checks.min_note,
      checks.bias_note
    ].concat(checks.models.map(modelValidation));
    section.appendChild(methodDetails(paragraphs));
    return section;
  }

  function render(payload) {
    var root = $("monitor-root");
    if (!root) return;
    root.replaceChildren();
    root.appendChild(renderHeader(payload));
    root.appendChild(renderStatusCards(payload));
    root.appendChild(renderAlerts(payload));
    root.appendChild(renderInputs(payload));
    root.appendChild(renderOutcomes(payload));
    root.appendChild(renderModelChecks(payload));
  }

  function renderLoading(date) {
    var root = $("monitor-root");
    if (!root) return;
    root.replaceChildren();
    var panel = node("section", "panel models-loading");
    panel.setAttribute("role", "status");
    panel.textContent = "Computing monitoring for " + date + "\u2026";
    root.appendChild(panel);
  }

  function renderError(message) {
    var root = $("monitor-root");
    if (!root) return;
    root.replaceChildren();
    var panel = node("section", "panel models-error");
    panel.appendChild(node("p", "eyebrow", "Model monitoring"));
    panel.appendChild(node("h2", "", "Monitoring is unavailable"));
    panel.appendChild(node("p", "", message));
    root.appendChild(panel);
  }

  function currentDate() {
    var input = $("asof-date");
    return input && input.value ? input.value : null;
  }

  function viewVisible() {
    var view = $("view-drift");
    return !!view && !view.hidden;
  }

  function refresh() {
    var date = currentDate();
    if (!date) return;
    if (state.loading && state.date === date) return;
    state.date = date;
    state.stale = false;
    state.loading = true;
    state.requestId += 1;
    var requestId = state.requestId;
    renderLoading(date);
    fetch("/api/models/monitor?date=" + encodeURIComponent(date))
      .then(function (response) {
        return response.json().then(function (payload) {
          return { ok: response.ok, payload: payload };
        });
      })
      .then(function (result) {
        if (requestId !== state.requestId) return;
        state.loading = false;
        if (!result.ok) {
          throw new Error(result.payload && result.payload.error ?
            result.payload.error.message : "The monitoring data is unavailable.");
        }
        state.payload = result.payload;
        render(result.payload);
      })
      .catch(function (error) {
        if (requestId !== state.requestId) return;
        state.loading = false;
        renderError(error.message);
      });
  }

  function markStale() {
    if (viewVisible()) refresh();
    else state.stale = true;
  }

  function bindDateEvents() {
    var input = $("asof-date");
    if (input) input.addEventListener("change", markStale);
    ["prev-day", "next-day"].forEach(function (id) {
      var button = $(id);
      if (button) {
        button.addEventListener("click", function () { window.setTimeout(markStale, 0); });
      }
    });
  }

  function bindViewEvents() {
    var view = $("view-drift");
    if (!view) return;
    var observer = new MutationObserver(function () {
      if (!view.hidden && (state.stale || !state.payload || state.date !== currentDate())) refresh();
    });
    observer.observe(view, { attributes: true, attributeFilter: ["hidden"] });
    if (!view.hidden) refresh();
  }

  window.PrazoMonitor = { refresh: refresh };

  bindDateEvents();
  bindViewEvents();
})();
