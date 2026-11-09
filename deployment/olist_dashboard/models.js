(function () {
  "use strict";

  var state = {
    loaded: false,
    loading: false,
    models: null,
    selected: null,
    importance: {},
    hashBound: false
  };
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var VERDICT_NOTE = "On test, a difference counts only if its 95% confidence interval " +
    "(2,000 resamples of whole purchase weeks) excludes zero.";

  function $(id) { return document.getElementById(id); }

  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function all(selector) { return Array.prototype.slice.call(document.querySelectorAll(selector)); }

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  var ABBR_TEXT = { mae: "MAE", rmse: "RMSE", pr_auc: "PR-AUC", roc_auc: "ROC-AUC" };

  var DEFINITIONS = {
    mae: "Mean absolute error: the average size of the prediction error.",
    rmse: "Root mean squared error: like MAE but penalises large errors more.",
    pr_auc: "Precision-recall area under the curve: how well late parcels rank above on-time ones; 1 = perfect.",
    roc_auc: "ROC area under the curve: how well the score separates late from on-time parcels; 0.5 = random."
  };

  function labelled(text, key, suffix) {
    var span = node("span");
    var abbr = node("abbr", "", text);
    abbr.title = DEFINITIONS[key];
    span.appendChild(abbr);
    if (suffix) span.appendChild(document.createTextNode(suffix));
    return span;
  }

  function svgNode(tag, attrs) {
    var element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (key) {
      element.setAttribute(key, attrs[key]);
    });
    return element;
  }

  function format(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "\u2014";
    var text = Number(value).toFixed(digits);
    return /^-0\.?0*$/.test(text) ? Number(0).toFixed(digits) : text;
  }

  function metricDigits(key) {
    return (key === "mae" || key === "rmse") ? 2 : 3;
  }

  function compactNumber(value) {
    var number = Number(value);
    if (number >= 1000 && number % 1000 === 0) return (number / 1000) + "k";
    return String(number);
  }

  function monthName(value) {
    var parts = String(value).split("-");
    var index = Number(parts[1]) - 1;
    return MONTHS[index] ? MONTHS[index] + " " + parts[0] : value;
  }

  function monthRange(months) {
    if (!months || !months.length) return "";
    var first = months[0].month;
    var last = months[months.length - 1].month;
    if (first === last) return monthName(first);
    var firstYear = first.slice(0, 4);
    var lastYear = last.slice(0, 4);
    if (firstYear === lastYear) return monthName(first).split(" ")[0] + "\u2013" + monthName(last);
    return monthName(first) + "\u2013" + monthName(last);
  }

  function monthRangeValues(values) {
    return monthRange((values || []).map(function (value) { return { month: value }; }));
  }

  function shareNumber(model) {
    return model.review_share_percent;
  }

  function shareLabel(model) {
    var share = shareNumber(model);
    if (share === null || share === undefined) return "";
    var rounded = Math.round(Number(share) * 10) / 10;
    return " @" + format(rounded, Number.isInteger(rounded) ? 0 : 1) + "%";
  }

  function primaryUnit(model) {
    return model.kind === "regression" ? "days" : "";
  }

  function primaryLabel(model) {
    return model.kind === "regression" ? "MAE (days)" : "PR-AUC";
  }

  function primaryHover(model) {
    return model.kind === "regression" ? "MAE" :
      "How well the model ranks late parcels above on-time ones; 1 = perfect, the share of late parcels = random.";
  }

  function summaryText(model, value, withUnit) {
    var text = format(value, 2);
    var unit = primaryUnit(model);
    if (withUnit && unit) text += " " + unit;
    return text;
  }

  function modelDisplayName(model) {
    if (model.name === "late_risk") return "Logistic regression";
    if (model.kind === "classification") return "Phase 2 model";
    var phrase = chosenPhrase(model).replace(/^(a|the) /, "");
    return phrase.charAt(0).toUpperCase() + phrase.slice(1);
  }

  function championTestValue(model) {
    var key = model.primary_metric;
    var champion = championRung(model);
    var value = champion ? rungTestValue(model, champion, key) : null;
    if (value !== null) return value;
    var test = model.metrics && model.metrics.test;
    if (!test) return null;
    if (model.kind === "classification") return test.model ? metricNumber(test.model[key]) : null;
    return test.overall ? metricNumber(test.overall[key]) : null;
  }

  function rungTestValue(model, rung, key) {
    if (model.kind === "classification" && rung && rung.candidate === model.candidate &&
        model.metrics && model.metrics.test && model.metrics.test.model &&
        model.metrics.test.model[key] !== undefined) {
      return metricNumber(model.metrics.test.model[key]);
    }
    return metricNumber(ladderValue(rung, "test", key));
  }

  function rungVerdict(rung) {
    if (!rung || !rung.test_gap_verdict) return null;
    return {
      gap: Number(rung.test_gap_vs_headline),
      low: Number(rung.test_gap_low_vs_headline),
      high: Number(rung.test_gap_high_vs_headline),
      verdict: rung.test_gap_verdict
    };
  }

  function gapDescriptor(model, verdict) {
    if (!verdict) return null;
    var title = tileGapTitle(model, verdict);
    if (verdict.verdict === "level") return { tone: "same", text: "\u2248 About the same", title: title };
    var ahead = verdict.verdict === "better";
    return {
      tone: ahead ? "good" : "amber",
      text: (ahead ? "\u25B2 Model better" : "\u25BC Rule better"),
      title: title
    };
  }

  function gapNode(descriptor) {
    if (!descriptor) return null;
    var span = node("span", "model-delta" + (descriptor.tone === "same" ? " is-same" : " " + descriptor.tone));
    span.textContent = descriptor.text;
    if (descriptor.title) span.title = descriptor.title;
    return span;
  }

  function familyNames(model) {
    return Object.keys(model.ladders || {});
  }

  function familyTitle(family) {
    return family === "Tree" ? "Full-depth tree" : family;
  }

  function ladderValue(rung, bucket, key) {
    if (bucket === "validation") {
      return rung.validation && rung.validation.mean ? rung.validation.mean[key] : null;
    }
    var values = rung[bucket];
    return values && values[key] !== undefined ? values[key] : null;
  }

  function metricNumber(value) {
    return value === null || value === undefined ? null : Number(value);
  }

  function isBetter(a, b, lower) {
    if (a === null || b === null) return false;
    return lower ? a < b - 1e-12 : a > b + 1e-12;
  }

  function championRung(model) {
    var found = null;
    familyNames(model).forEach(function (family) {
      (model.ladders[family] || []).forEach(function (rung) {
        if (rung.candidate === model.candidate) found = rung;
      });
    });
    return found;
  }

  function paramText(params) {
    return Object.keys(params).sort().filter(function (key) {
      return key !== "family";
    }).map(function (key) {
      return key + " " + params[key];
    }).join(" \u00b7 ");
  }

  function renderHeader(models) {
    var sub = $("models-sub");
    if (!sub) return;
    var cutoff = models[0] && models[0].train_cutoff ? models[0].train_cutoff : "";
    sub.textContent = "The three models behind Actions, checked on later orders.";
    sub.title = "Trained on orders up to " + cutoff + " and checked on later orders.";
  }

  function renderError(message) {
    var root = $("models-root");
    if (!root) return;
    root.replaceChildren();
    var panel = node("section", "panel models-error");
    panel.setAttribute("aria-labelledby", "models-error-title");
    panel.appendChild(node("p", "eyebrow", "Model registry"));
    panel.appendChild(node("h2", "", "Models are unavailable"));
    var title = panel.querySelector("h2");
    title.id = "models-error-title";
    panel.appendChild(node("p", "", message));
    panel.appendChild(node("p", "", "Build the store with python -m phase3.registry, then start the dashboard with deployment/olist_dashboard/run_local.sh."));
    root.appendChild(panel);
  }

  function comparisonColumns(model) {
    var columns = [];
    familyNames(model).forEach(function (family) {
      (model.ladders[family] || []).forEach(function (rung) {
        columns.push({ family: family, rung: rung });
      });
    });
    return columns;
  }

  function slimComparisonTable(model) {
    var key = model.primary_metric;
    var wrap = node("div", "model-compare-wrap");
    var table = node("table", "model-compare-table model-compare-slim");
    var head = node("thead");
    var headRow = node("tr");
    function primaryHeader(prefix) {
      var span = node("span", "", prefix + " ");
      span.appendChild(labelled(
        model.kind === "regression" ? "MAE" : "PR-AUC",
        model.kind === "regression" ? "mae" : "pr_auc",
        model.kind === "regression" ? " (days)" : ""));
      return span;
    }
    [node("span", "", "Model"), primaryHeader("Validation"), primaryHeader("Test")].forEach(function (label, index) {
      var cell = node("th", "", typeof label === "string" ? label : "");
      if (typeof label !== "string") cell.appendChild(label);
      cell.setAttribute("scope", "col");
      if (index === 1 || index === 2) cell.title = primaryHover(model);
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);

    var body = node("tbody");
    familyNames(model).forEach(function (family) {
      var groupRow = node("tr", "model-family-row");
      var groupCell = node("th", "", familyTitle(family) + " family");
      groupCell.setAttribute("scope", "colgroup");
      groupCell.colSpan = 3;
      groupRow.appendChild(groupCell);
      body.appendChild(groupRow);
      (model.ladders[family] || []).forEach(function (rung) {
        var row = node("tr");
        if (rung.candidate === model.candidate) row.classList.add("is-champion");
        var headCell = node("th", "", rung.short_label || rung.label);
        headCell.setAttribute("scope", "row");
        headCell.title = rung.label + (rung.candidate === model.candidate ? " (chosen)" : "");
        row.appendChild(headCell);
        var validation = metricNumber(ladderValue(rung, "validation", key));
        var chosen = rung.candidate === model.candidate;
        var test = chosen ? rungTestValue(model, rung, key) : null;
        row.appendChild(node("td", "", validation === null ? "\u2014" : format(validation, 2)));
        row.appendChild(node("td", "", test === null ? "\u2014" : format(test, 2)));
        body.appendChild(row);
      });
    });

    var references = visibleReferences(model);
    if (references.length) {
      var groupRow = node("tr", "model-family-row");
      var groupCell = node("th", "", "Operations rule");
      groupCell.setAttribute("scope", "colgroup");
      groupCell.colSpan = 3;
      groupRow.appendChild(groupCell);
      body.appendChild(groupRow);
      references.forEach(function (reference) {
        var row = node("tr", "is-reference");
        var headCell = node("th", "", reference.label);
        headCell.setAttribute("scope", "row");
        headCell.title = reference.hover || reference.label;
        row.appendChild(headCell);
        var validation = referenceValidation(reference, key);
        var test = referenceTest(reference, key);
        row.appendChild(node("td", "", validation === null ? "\u2014" : format(validation, 2)));
        row.appendChild(node("td", "", test === null ? "\u2014" : format(test, 2)));
        body.appendChild(row);
      });
    }
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function windowSection(model) {
    if (!model.metrics.validation.windows.length) return null;
    var regression = model.kind === "regression";
    var keys = regression ? ["mae", "rmse"] : ["pr_auc", "roc_auc", "mcc", "precision", "recall"];
    var share = shareLabel(model);
    var labels = regression ? [labelled("MAE", "mae", " (days)"), labelled("RMSE", "rmse")] :
      [labelled("PR-AUC", "pr_auc"), labelled("ROC-AUC", "roc_auc"),
        "MCC" + share, "Precision" + share, "Recall" + share];
    var section = node("section", "model-detail-section");
    section.appendChild(node("h3", "model-detail-title", "Validation windows"));
    var wrap = node("div", "model-window-wrap");
    var table = node("table", "model-window-table");
    var head = node("thead");
    var headRow = node("tr");
    var windowHead = node("th", "", "Window");
    windowHead.setAttribute("scope", "col");
    headRow.appendChild(windowHead);
    labels.forEach(function (label) {
      var cell = node("th", "", typeof label === "string" ? label : "");
      if (typeof label !== "string") cell.appendChild(label);
      cell.setAttribute("scope", "col");
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    var body = node("tbody");
    model.metrics.validation.windows.forEach(function (window, index) {
      var row = node("tr");
      var cell = node("th", "", String(index + 1));
      cell.setAttribute("scope", "row");
      row.appendChild(cell);
      keys.forEach(function (key) {
        row.appendChild(node("td", "", format(window[key], metricDigits(key))));
      });
      body.appendChild(row);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    section.appendChild(wrap);
    return section;
  }

  function testMonthsSection(model) {
    var months = model.kind === "regression" ? model.metrics.test.months : [];
    if (!months.length) return null;
    var section = node("section", "model-detail-section");
    section.appendChild(node("h3", "model-detail-title", "Test months"));
    var wrap = node("div", "model-window-wrap");
    var table = node("table", "model-window-table");
    var head = node("thead");
    var headRow = node("tr");
    [node("span", "", "Month"), labelled("MAE", "mae", " (days)"), labelled("RMSE", "rmse")].forEach(function (label) {
      var cell = node("th", "", typeof label === "string" ? label : "");
      if (typeof label !== "string") cell.appendChild(label);
      cell.setAttribute("scope", "col");
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    var body = node("tbody");
    months.forEach(function (month) {
      var row = node("tr");
      var cell = node("th", "", monthName(month.month));
      cell.setAttribute("scope", "row");
      row.appendChild(cell);
      ["mae", "rmse"].forEach(function (key) {
        row.appendChild(node("td", "", format(month.metrics[key], metricDigits(key))));
      });
      body.appendChild(row);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    section.appendChild(wrap);
    return section;
  }

  function parametersSection(model) {
    var section = node("section", "model-detail-section");
    section.appendChild(node("h3", "model-detail-title", "Full hyperparameters"));
    var list = node("dl", "model-params");
    Object.keys(model.params).sort().forEach(function (key) {
      list.appendChild(node("dt", "", key));
      list.appendChild(node("dd", "", model.params[key]));
    });
    section.appendChild(list);
    return section;
  }

  function importanceSection(model) {
    var section = node("section", "model-detail-section");
    section.appendChild(node("h3", "model-detail-title", "What the model leans on"));
    section.appendChild(node("p", "model-sentence",
      "How much worse the validation score gets when this input, or a group of inputs that move " +
      "together, is shuffled. Not causal."));
    var host = node("div", "model-leans-host");
    host.id = "model-leans-" + model.name;
    host.appendChild(node("p", "models-loading", "Loading importance\u2026"));
    section.appendChild(host);
    return section;
  }

  function renderImportanceList(host, payload) {
    var inputs = (payload.inputs || []).slice(0, 10);
    if (!inputs.length) {
      host.appendChild(node("p", "model-sentence", "No permutation importance was recorded for this model."));
      return;
    }
    var maximum = Math.max.apply(null, inputs.map(function (row) { return row.importance_mean; })) || 1;
    var list = node("ul", "model-leans");
    inputs.forEach(function (row) {
      var item = node("li", "model-leans-item");
      var name = node("span", "model-leans-name", row.input);
      if (row.members && row.members.indexOf(",") !== -1) {
        name.title = "Inputs moved together: " + row.members.split(",").join(", ");
      }
      item.appendChild(name);
      item.appendChild(node("span", "model-leans-value", format(row.importance_mean, 3)));
      item.appendChild(node("span", "model-leans-direction",
        row.direction || "mixed/overlapping"));
      var bar = node("span", "model-leans-bar");
      var fill = node("span");
      fill.style.width = (Math.max(row.importance_mean, 0) / maximum * 100) + "%";
      bar.appendChild(fill);
      item.appendChild(bar);
      list.appendChild(item);
    });
    host.appendChild(list);
  }

  function renderImportanceFromCache(name) {
    var host = $("model-leans-" + name);
    if (!host) return;
    var entry = state.importance[name];
    if (!entry || entry.loading) return;
    host.replaceChildren();
    if (entry.error) {
      host.appendChild(node("p", "model-sentence", "The permutation importance could not be loaded: " + entry.error));
      return;
    }
    renderImportanceList(host, entry.payload);
  }

  function ensureImportance(model) {
    var cached = state.importance[model.name];
    if (cached) {
      if (!cached.loading) renderImportanceFromCache(model.name);
      return;
    }
    state.importance[model.name] = { loading: true };
    fetch("/api/models/" + encodeURIComponent(model.name) + "/importance")
      .then(function (response) {
        return response.json().then(function (payload) { return { ok: response.ok, payload: payload }; });
      })
      .then(function (result) {
        if (!result.ok) {
          state.importance[model.name] = {
            error: result.payload && result.payload.error ? result.payload.error.message : "The request failed."
          };
        } else {
          state.importance[model.name] = { payload: result.payload };
        }
        if (state.selected === model.name) renderImportanceFromCache(model.name);
      })
      .catch(function (error) {
        state.importance[model.name] = { error: error.message };
        if (state.selected === model.name) renderImportanceFromCache(model.name);
      });
  }

  function referencesOf(model) {
    return model.references || [];
  }

  function referenceByRole(model, role) {
    var found = null;
    referencesOf(model).forEach(function (reference) {
      if (!found && reference.role === role) found = reference;
    });
    return found;
  }

  function headlineReference(model) {
    return referenceByRole(model, "headline");
  }

  function visibleReferences(model) {
    var headline = headlineReference(model);
    return headline ? [headline] : [];
  }

  function referenceValidation(reference, key) {
    if (!reference || !reference.validation || !reference.validation.mean) return null;
    return reference.validation.mean[key] === undefined ? null : metricNumber(reference.validation.mean[key]);
  }

  function referenceTest(reference, key) {
    if (!reference || !reference.test || reference.test[key] === undefined) return null;
    return metricNumber(reference.test[key]);
  }

  function signedGap(value, digits) {
    var sign = value > 0 ? "+" : (value < 0 ? "\u2212" : "");
    return sign + format(Math.abs(value), digits);
  }

  function tileGapTitle(model, verdict) {
    var digits = model.kind === "regression" ? 2 : 3;
    var unit = model.kind === "regression" ? " days" : " PR-AUC";
    return "Test gap " + signedGap(verdict.gap, digits) + unit +
      ", 95% interval " + signedGap(verdict.low, digits) + " to " +
      signedGap(verdict.high, digits) + unit + ".";
  }

  function tileCompareLine(model, headline) {
    var line = node("p", "model-tile-rule");
    var rule = node("span", "model-tile-rule-text", headline.label);
    var full = headline.short || headline.label;
    if (headline.hover && full.indexOf(headline.hover) === -1) full += " " + headline.hover;
    rule.title = full;
    line.appendChild(rule);
    var verdict = rungVerdict(championRung(model));
    if (verdict) {
      var descriptor = gapDescriptor(model, verdict);
      if (descriptor) {
        line.appendChild(document.createTextNode(" "));
        line.appendChild(gapNode(descriptor));
      }
    }
    return line;
  }

  function regressionTileBlock(model) {
    var headline = headlineReference(model);
    if (!headline) return [];
    return [tileCompareLine(model, headline)];
  }

  function lateRiskTileBlock(model) {
    var headline = headlineReference(model);
    if (!headline) return [];
    return [tileCompareLine(model, headline)];
  }

  function renderTile(model) {
    var button = node("button", "model-tile");
    button.type = "button";
    button.setAttribute("data-model", model.name);
    button.setAttribute("aria-pressed", state.selected === model.name ? "true" : "false");
    button.addEventListener("click", function () {
      selectModel(model.name, true);
    });

    var head = node("div", "model-tile-head");
    head.appendChild(node("span", "model-tile-title", model.title));
    head.appendChild(node("p", "model-tile-chosen", modelDisplayName(model)));
    button.appendChild(head);

    var metric = node("div", "model-tile-metric");
    var caption = node("span", "model-tile-caption", "Test " + primaryLabel(model));
    caption.title = primaryHover(model);
    metric.appendChild(caption);
    metric.appendChild(node("span", "model-tile-value", summaryText(model, championTestValue(model), false)));
    button.appendChild(metric);

    var block = model.kind === "regression" ? regressionTileBlock(model) : lateRiskTileBlock(model);
    block.forEach(function (line) { button.appendChild(line); });

    var line = model.champion_line || {};
    button.appendChild(node("p", "model-tile-footer",
      "Pick of " + line.candidate_count + " models"));
    return button;
  }

  function renderTiles(models) {
    var row = node("div", "model-tiles");
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", "Models");
    models.forEach(function (model) { row.appendChild(renderTile(model)); });
    return row;
  }

  function updateTileState() {
    all(".model-tile").forEach(function (button) {
      button.setAttribute("aria-pressed",
        button.getAttribute("data-model") === state.selected ? "true" : "false");
    });
  }

  function showTip(tip, model, rung, cx, cy, width, height) {
    var key = model.primary_metric;
    var digits = metricDigits(key);
    tip.replaceChildren();
    tip.appendChild(node("strong", "model-tip-title", rung.label));
    tip.appendChild(node("span", "model-tip-candidate", rung.candidate));
    tip.appendChild(node("span", "model-tip-params", paramText(rung.params)));
    function row(name, value, suffix) {
      if (value === null || value === undefined) return;
      var line = node("span", "model-tip-row");
      line.appendChild(node("span", "model-tip-label", name));
      line.appendChild(node("span", "model-tip-value", format(value, digits) + (suffix || "")));
      tip.appendChild(line);
    }
    row("Train", ladderValue(rung, "train", key));
    row("Validation", ladderValue(rung, "validation", key));
    if (rung.candidate === model.candidate) row("Test", ladderValue(rung, "test", key));
    tip.hidden = false;
    tip.classList.toggle("is-left", cx > width * 0.62);
    tip.style.left = (cx / width * 100) + "%";
    tip.style.top = (cy / height * 100) + "%";
  }

  function showReferenceTip(tip, model, reference, cx, cy, width, height) {
    var key = model.primary_metric;
    var digits = metricDigits(key);
    tip.replaceChildren();
    tip.appendChild(node("strong", "model-tip-title", reference.label));
    function row(name, value) {
      if (value === null || value === undefined) return;
      var line = node("span", "model-tip-row");
      line.appendChild(node("span", "model-tip-label", name));
      line.appendChild(node("span", "model-tip-value", format(value, digits)));
      tip.appendChild(line);
    }
    row("Validation", referenceValidation(reference, key));
    row("Test", referenceTest(reference, key));
    tip.hidden = false;
    tip.classList.toggle("is-left", cx > width * 0.62);
    tip.style.left = (cx / width * 100) + "%";
    tip.style.top = (cy / height * 100) + "%";
  }

  function niceStep(raw) {
    if (!isFinite(raw) || raw <= 0) return 1;
    var exponent = Math.floor(Math.log(raw) / Math.LN10);
    var fraction = raw / Math.pow(10, exponent);
    var nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
    return nice * Math.pow(10, exponent);
  }

  function tickDecimals(step) {
    if (Math.abs(step - Math.round(step)) < 1e-9) return 0;
    if (Math.abs(step * 10 - Math.round(step * 10)) < 1e-9) return 1;
    if (Math.abs(step * 100 - Math.round(step * 100)) < 1e-9) return 2;
    return 3;
  }

  function ladderDomain(model) {
    var key = model.primary_metric;
    var values = [];
    familyNames(model).forEach(function (family) {
      (model.ladders[family] || []).forEach(function (rung) {
        ["validation", "test"].forEach(function (bucket) {
          var value = metricNumber(ladderValue(rung, bucket, key));
          if (value !== null) values.push(value);
        });
      });
    });
    visibleReferences(model).forEach(function (reference) {
      [referenceValidation(reference, key), referenceTest(reference, key)].forEach(function (value) {
        if (value !== null) values.push(value);
      });
    });
    values = values.filter(function (value) { return isFinite(value); });
    if (!values.length) return { lower: 0, upper: 1, step: 0.2, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
    var minimum = Math.min.apply(null, values);
    var maximum = Math.max.apply(null, values);
    if (maximum - minimum < 1e-9) {
      var pad = Math.abs(maximum) || 1;
      minimum -= pad * 0.1;
      maximum += pad * 0.1;
    }
    var step = niceStep((maximum - minimum) / 5);
    var lower = Math.floor(minimum / step) * step;
    var upper = Math.ceil(maximum / step) * step;
    if (upper <= lower) upper = lower + step;
    var ticks = [];
    for (var value = lower; value <= upper + step * 1e-9; value += step) {
      ticks.push(Math.round(value / step) * step);
    }
    return { lower: lower, upper: upper, step: step, ticks: ticks };
  }

  function markRowHot(svg, row) {
    var active = row === null || row === undefined ? null : String(row);
    svg.querySelectorAll(".ladder-point").forEach(function (point) {
      point.classList.toggle("is-hot", active !== null && point.getAttribute("data-row") === active);
    });
  }

  function ladderChart(model) {
    var key = model.primary_metric;
    var domain = ladderDomain(model);
    var families = familyNames(model);
    var width = 1100;
    var margin = { left: 210, right: 34, top: 56, bottom: 46 };
    var rowHeight = 38;
    var familyHeight = 28;
    var plotWidth = width - margin.left - margin.right;
    var rows = [];
    var cursor = margin.top;
    families.forEach(function (family) {
      rows.push({ kind: "family", title: familyTitle(family) + " family", y: cursor });
      cursor += familyHeight;
      (model.ladders[family] || []).forEach(function (rung) {
        rows.push({ kind: "rung", family: family, rung: rung, y: cursor });
        cursor += rowHeight;
      });
    });
    var references = visibleReferences(model);
    if (references.length) {
      rows.push({ kind: "family", title: "Operations rule", y: cursor });
      cursor += familyHeight;
      references.forEach(function (reference) {
        rows.push({ kind: "reference", reference: reference, y: cursor });
        cursor += rowHeight;
      });
    }
    var plotBottom = cursor;
    var height = plotBottom + margin.bottom;
    var selectedGrid = (model.grid || []).filter(function (row) { return row.selected; })[0];
    var atGridEdge = !!(selectedGrid && selectedGrid.grid_edge);

    function x(value) {
      return margin.left + ((value - domain.lower) / (domain.upper - domain.lower)) * plotWidth;
    }
    function center(row) { return row.y + rowHeight / 2; }

    var canvas = node("div", "model-ladder-canvas");
    var svg = svgNode("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": primaryLabel(model) +
        (model.lower_is_better ? ", lower is better." : ", higher is better.") +
        " Rows: " + rows.filter(function (row) { return row.kind !== "family"; })
          .map(function (row) {
            return row.kind === "rung" ? row.rung.label : row.reference.label;
          }).join(", ") + "."
    });
    var title = svgNode("title");
    title.textContent = "Model scores: filled dot = validation, hollow dot = test score (chosen model and rule).";
    svg.appendChild(title);

    rows.forEach(function (row) {
      if (row.kind === "rung" && row.rung.candidate === model.candidate) {
        svg.appendChild(svgNode("rect", {
          x: margin.left, y: row.y + 2, width: plotWidth, height: rowHeight - 4,
          rx: 5, class: "ladder-row-bg"
        }));
      }
    });

    var axisTitle = svgNode("text", {
      x: margin.left, y: 17, "text-anchor": "start", class: "model-axis-title"
    });
    axisTitle.textContent = primaryLabel(model) +
      (model.lower_is_better ? " \u00b7 lower is better" : " \u00b7 higher is better");
    svg.appendChild(axisTitle);

    domain.ticks.forEach(function (value) {
      var cx = x(value);
      svg.appendChild(svgNode("line", {
        x1: cx, x2: cx, y1: margin.top - 4, y2: plotBottom, class: "model-grid"
      }));
      var label = svgNode("text", {
        x: cx, y: plotBottom + 18, "text-anchor": "middle", class: "model-axis"
      });
      label.textContent = format(value, tickDecimals(domain.step));
      svg.appendChild(label);
    });
    svg.appendChild(svgNode("line", {
      x1: margin.left, x2: width - margin.right, y1: plotBottom, y2: plotBottom, class: "model-grid"
    }));

    var rowIndex = 0;
    var tip = node("div", "model-tip");
    tip.hidden = true;

    rows.forEach(function (row) {
      if (row.kind === "family") {
        var familyLabel = svgNode("text", {
          x: margin.left, y: row.y + familyHeight - 9, "text-anchor": "start",
          class: "ladder-family-title"
        });
        familyLabel.textContent = row.title;
        svg.appendChild(familyLabel);
        svg.appendChild(svgNode("line", {
          x1: margin.left, x2: width - margin.right,
          y1: row.y + familyHeight - 3, y2: row.y + familyHeight - 3, class: "ladder-separator"
        }));
        return;
      }

      var cy = center(row);
      var index = rowIndex;
      rowIndex += 1;

      if (row.kind === "reference") {
        var reference = row.reference;
        var referenceLabel = svgNode("text", {
          x: margin.left - 12, y: cy + 4, "text-anchor": "end",
          class: "ladder-row-label is-reference"
        });
        referenceLabel.textContent = reference.label;
        var referenceTitle = svgNode("title");
        referenceTitle.textContent = reference.hover || reference.label;
        referenceLabel.appendChild(referenceTitle);
        svg.appendChild(referenceLabel);

        var referenceTestValue = referenceTest(reference, key);
        var referenceValidationValue = referenceValidation(reference, key);

        if (referenceTestValue !== null) {
          svg.appendChild(svgNode("circle", {
            cx: x(referenceTestValue), cy: cy, r: 4.5, "data-row": index,
            class: "ladder-point test is-reference"
          }));
        }
        if (referenceValidationValue !== null) {
          svg.appendChild(svgNode("circle", {
            cx: x(referenceValidationValue), cy: cy, r: 5, "data-row": index,
            class: "ladder-point validation is-reference"
          }));
        }

        var referenceAnchor = referenceValidationValue;
        if (referenceAnchor === null) referenceAnchor = referenceTestValue;
        if (referenceAnchor === null) referenceAnchor = domain.lower;
        var referenceHit = svgNode("circle", {
          cx: x(referenceAnchor), cy: cy, r: rowHeight / 2,
          class: "ladder-hit", tabindex: "0", role: "button",
          "aria-label": reference.label + ": validation " + format(referenceValidationValue, 2) +
            ", test " + format(referenceTestValue, 2)
        });
        referenceHit.addEventListener("mouseenter", function () {
          markRowHot(svg, index);
          showReferenceTip(tip, model, reference, x(referenceAnchor), cy, width, height);
        });
        referenceHit.addEventListener("focus", function () {
          markRowHot(svg, index);
          showReferenceTip(tip, model, reference, x(referenceAnchor), cy, width, height);
        });
        referenceHit.addEventListener("mouseleave", function () {
          markRowHot(svg, null);
          tip.hidden = true;
        });
        referenceHit.addEventListener("blur", function () {
          markRowHot(svg, null);
          tip.hidden = true;
        });
        svg.appendChild(referenceHit);
        return;
      }

      var rung = row.rung;
      var chosen = rung.candidate === model.candidate;

      if (chosen) {
        var badge = svgNode("text", {
          x: 10, y: cy + 4, "text-anchor": "start", class: "ladder-chosen"
        });
        badge.textContent = "Chosen";
        svg.appendChild(badge);
      }

      var label = svgNode("text", {
        x: margin.left - 12, y: cy + 4, "text-anchor": "end",
        class: "ladder-row-label" + (chosen ? " is-chosen" : "")
      });
      label.textContent = rung.short_label || rung.label;
      svg.appendChild(label);

      var testValue = metricNumber(ladderValue(rung, "test", key));
      var validationValue = metricNumber(ladderValue(rung, "validation", key));
      var trainValue = metricNumber(ladderValue(rung, "train", key));

      if (chosen && testValue !== null) {
        svg.appendChild(svgNode("circle", {
          cx: x(testValue), cy: cy, r: 4.5, "data-row": index,
          class: "ladder-point test"
        }));
      }
      if (validationValue !== null) {
        svg.appendChild(svgNode("circle", {
          cx: x(validationValue), cy: cy, r: 5, "data-row": index,
          class: "ladder-point validation"
        }));
        if (chosen) {
          svg.appendChild(svgNode("circle", {
            cx: x(validationValue), cy: cy, r: 9.5, class: "ladder-ring"
          }));
        }
      }

      if (chosen && atGridEdge) {
        var edge = svgNode("text", {
          x: (validationValue !== null ? x(validationValue) : margin.left) + 14,
          y: cy + 4, "text-anchor": "start", class: "model-edge-label"
        });
        edge.textContent = "grid edge";
        svg.appendChild(edge);
      }

      var anchorValue = validationValue;
      if (anchorValue === null) anchorValue = testValue;
      if (anchorValue === null) anchorValue = trainValue;
      if (anchorValue === null) anchorValue = domain.lower;
      var hit = svgNode("circle", {
        cx: x(anchorValue), cy: cy, r: rowHeight / 2,
        class: "ladder-hit", tabindex: "0", role: "button",
        "aria-label": rung.label + ": train " + format(trainValue, 2) +
          ", validation " + format(validationValue, 2) +
          (chosen ? ", test " + format(testValue, 2) : "")
      });
      hit.addEventListener("mouseenter", function () {
        markRowHot(svg, index);
        showTip(tip, model, rung, x(anchorValue), cy, width, height);
      });
      hit.addEventListener("focus", function () {
        markRowHot(svg, index);
        showTip(tip, model, rung, x(anchorValue), cy, width, height);
      });
      hit.addEventListener("mouseleave", function () {
        markRowHot(svg, null);
        tip.hidden = true;
      });
      hit.addEventListener("blur", function () {
        markRowHot(svg, null);
        tip.hidden = true;
      });
      svg.appendChild(hit);
    });

    canvas.appendChild(svg);
    canvas.appendChild(tip);
    return canvas;
  }

  function legendStatic(className, text) {
    var span = node("span", "model-legend-static");
    span.appendChild(node("i", "legend-swatch " + className));
    span.appendChild(node("span", "", text));
    return span;
  }

  function ladderLegend() {
    var legend = node("div", "model-legend");
    legend.setAttribute("role", "group");
    legend.setAttribute("aria-label", "Chart marks");
    legend.appendChild(legendStatic("validation", "Validation"));
    legend.appendChild(legendStatic("test", "Test"));
    legend.appendChild(legendStatic("chosen", "Chosen"));
    return legend;
  }

  function ladderSection(model) {
    var section = node("div", "model-ladder");
    var chart = ladderChart(model);
    section.appendChild(chart);
    var note = node("p", "model-ladder-note",
      "Best setting of each model type, against the operations rule.");
    note.title = "Each row is the best setting tried for that model type. Filled dot = validation score " +
      "(used to pick the model); hollow dot = test score (chosen model and rule).";
    section.appendChild(note);
    section.appendChild(ladderLegend());
    return section;
  }

  function compareMetrics(model) {
    if (model.kind === "regression") {
      return [
        { key: "mae", label: "MAE (days)", lower: true },
        { key: "rmse", label: "RMSE (days)", lower: true }
      ];
    }
    return [
      { key: "pr_auc", label: "PR-AUC", lower: false },
      { key: "roc_auc", label: "ROC-AUC", lower: false },
      { key: "mcc", label: "MCC", lower: false },
      { key: "precision", label: "Precision", lower: false },
      { key: "recall", label: "Recall", lower: false }
    ];
  }

  function splitMetricLabel(model, metric) {
    if (model.kind === "classification" &&
        (metric.key === "mcc" || metric.key === "precision" || metric.key === "recall")) {
      return metric.label + shareLabel(model);
    }
    return metric.label;
  }

  function splitMetricValue(rung, split, key) {
    if (split === "validation") {
      if (!rung.validation || !rung.validation.mean || rung.validation.mean[key] === undefined) return null;
      return rung.validation.mean[key];
    }
    var values = rung[split];
    return values && values[key] !== undefined ? values[key] : null;
  }

  function splitReferenceValue(reference, split, key) {
    if (split === "validation") return referenceValidation(reference, key);
    if (split === "test") return referenceTest(reference, key);
    return null;
  }

  function fullComparisonTable(model) {
    var families = familyNames(model);
    var columns = comparisonColumns(model);
    var references = visibleReferences(model);
    var entries = columns.map(function (column) { return { kind: "rung", rung: column.rung }; });
    references.forEach(function (reference) { entries.push({ kind: "reference", reference: reference }); });
    var metrics = compareMetrics(model);
    var wrap = node("div", "model-compare-wrap");
    var table = node("table", "model-compare-table model-compare-full");
    var head = node("thead");
    var familyRow = node("tr");
    var corner = node("th", "", "Split / measure");
    corner.setAttribute("scope", "col");
    corner.rowSpan = 2;
    familyRow.appendChild(corner);
    families.forEach(function (family) {
      var cell = node("th", "model-family-head", familyTitle(family) + " family");
      cell.setAttribute("scope", "colgroup");
      cell.colSpan = (model.ladders[family] || []).length;
      familyRow.appendChild(cell);
    });
    if (references.length) {
      var referenceHead = node("th", "model-family-head", "Operations rule");
      referenceHead.setAttribute("scope", "colgroup");
      referenceHead.colSpan = references.length;
      familyRow.appendChild(referenceHead);
    }
    head.appendChild(familyRow);
    var rungRow = node("tr");
    entries.forEach(function (entry) {
      var cell;
      if (entry.kind === "rung") {
        cell = node("th", "model-rung-head" +
          (entry.rung.candidate === model.candidate ? " is-champion" : ""));
        cell.title = entry.rung.label + (entry.rung.candidate === model.candidate ? " (chosen)" : "");
        cell.textContent = entry.rung.short_label || entry.rung.label;
      } else {
        cell = node("th", "model-rung-head is-reference");
        cell.title = entry.reference.hover || entry.reference.label;
        cell.textContent = entry.reference.label;
      }
      cell.setAttribute("scope", "col");
      rungRow.appendChild(cell);
    });
    head.appendChild(rungRow);
    table.appendChild(head);

    var body = node("tbody");
    [
      { split: "train", label: "Train" },
      { split: "validation", label: "Validation" },
      { split: "test", label: "Test" }
    ].forEach(function (split) {
      var groupRow = node("tr", "model-split-row");
      var groupCell = node("th", "", split.label);
      groupCell.setAttribute("scope", "colgroup");
      groupCell.colSpan = entries.length + 1;
      groupRow.appendChild(groupCell);
      body.appendChild(groupRow);
      metrics.forEach(function (metric) {
        var row = node("tr");
        var headCell = node("th", "");
        if (ABBR_TEXT[metric.key]) {
          var suffix = metric.key === "mae" || metric.key === "rmse" ? " (days)" : "";
          headCell.appendChild(labelled(ABBR_TEXT[metric.key], metric.key, suffix));
        } else {
          headCell.textContent = splitMetricLabel(model, metric);
        }
        headCell.setAttribute("scope", "row");
        row.appendChild(headCell);
        var values = entries.map(function (entry) {
          if (entry.kind === "rung") {
            if (split.split === "test") return rungTestValue(model, entry.rung, metric.key);
            return metricNumber(splitMetricValue(entry.rung, split.split, metric.key));
          }
          return metricNumber(splitReferenceValue(entry.reference, split.split, metric.key));
        });
        var present = values.filter(function (value) { return value !== null; });
        var best = null;
        present.forEach(function (value) {
          if (best === null || isBetter(value, best, metric.lower)) best = value;
        });
        values.forEach(function (value, index) {
          var cell = node("td", "");
          var hidden = split.split === "test" && entries[index].kind === "rung" &&
            entries[index].rung.candidate !== model.candidate;
          if (entries[index].kind === "rung" &&
              entries[index].rung.candidate === model.candidate) {
            cell.classList.add("is-champion");
          }
          if (hidden || value === null) {
            cell.textContent = "\u2014";
          } else {
            var text = format(value, metricDigits(metric.key));
            if (best !== null && split.split !== "train" && Math.abs(value - best) < 1e-12) {
              cell.appendChild(node("strong", "", text));
            } else {
              cell.textContent = text;
            }
          }
          row.appendChild(cell);
        });
        body.appendChild(row);
      });
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function chosenPhrase(model) {
    var family = model.params && model.params.family;
    if (family === "ridge") return "ridge regression";
    if (family === "linear") return "linear regression";
    if (family === "tree") return "a decision tree";
    if (family === "forest") return "a random forest";
    if (family === "boost") return "gradient boosting";
    if (model.candidate === "mean_baseline") return "the training mean";
    return model.param_summary || model.candidate || "the chosen model";
  }

  function lateRiskChoice(model) {
    var days = model.warning_days || {};
    if (days.blend === null || days.blend === undefined ||
        days.slack === null || days.slack === undefined) return null;
    return "We use a 50/50 blend of the model and the remaining-slack rule. " +
      "It ranks as well as the rule and finds late parcels earlier (" +
      format(days.blend, 1) + " vs " + format(days.slack, 1) + " days left).";
  }

  function verdictText(model) {
    if (model.kind === "classification") return lateRiskChoice(model) || "";
    var champion = championRung(model);
    if (!champion) return "";
    var key = model.primary_metric;
    var value = metricNumber(ladderValue(champion, "validation", key));
    var sentences = [];
    sentences.push("We use " + chosenPhrase(model) + ".");
    if (value !== null) {
      sentences.push(model.kind === "regression"
        ? "In validation it was off by " + format(value, 2) + " days on average."
        : "In validation its PR-AUC was " + format(value, 3) + ".");
    }
    var headline = headlineReference(model);
    var ruleTest = referenceTest(headline, key);
    var modelTest = championTestValue(model);
    var verdict = rungVerdict(champion);
    if (ruleTest !== null && modelTest !== null && verdict) {
      var pair = format(modelTest, 2) + " vs " + format(ruleTest, 2) +
        (model.kind === "regression" ? " days" : "");
      if (verdict.verdict === "level") {
        sentences.push("On test it was about the same as the rule (" + pair + ").");
      } else if (verdict.verdict === "better") {
        sentences.push("On test it was better than the rule (" + pair + ").");
      } else {
        sentences.push("On test the rule was better (" + pair + ").");
      }
    }
    return sentences.join(" ");
  }

  function verdictSection(model) {
    var section = node("section", "model-verdict");
    section.appendChild(node("h3", "model-verdict-title", "Verdict"));
    section.appendChild(node("p", "model-verdict-text", verdictText(model)));
    return section;
  }

  function tuningSection(model) {
    var family = model.params.family;
    var key = family === "ridge" ? "alpha" : (family === "linear" ? "C" : null);
    if (!key) return null;
    var rows = model.grid.filter(function (row) {
      if (row.validation_mean === null || row.tuning_value === null || row.tuning_value === undefined) return false;
      if (row.params.family !== family) return false;
      if (key === "C" && String(row.params.class_weight || "None") !== String(model.params.class_weight || "None")) {
        return false;
      }
      return Number(row.tuning_value) > 0 && isFinite(Number(row.tuning_value));
    });
    if (rows.length < 2) return null;
    rows.sort(function (a, b) { return Number(a.tuning_value) - Number(b.tuning_value); });
    var lowerIsBetter = !!model.lower_is_better;
    var gapLabel = lowerIsBetter ? "Validation MAE minus chosen (days; lower is better)" :
      "Validation PR-AUC minus chosen (higher is better)";
    var section = node("section", "model-detail-section");
    section.appendChild(node("h3", "model-detail-title", "Tuning range"));
    section.appendChild(node("p", "model-sentence",
      "Each point is that setting's " + (lowerIsBetter ? "validation MAE" : "validation PR-AUC") +
      " minus the chosen setting's, on the same five windows; the bar is the standard error of the " +
      "paired difference."));

    var width = 560;
    var height = 190;
    var margin = { left: 74, right: 18, top: 28, bottom: 40 };
    var plotWidth = width - margin.left - margin.right;
    var plotHeight = height - margin.top - margin.bottom;
    var digits = metricDigits(model.primary_metric);
    var values = [0];
    rows.forEach(function (row) {
      var gap = row.gap_signed === null || row.gap_signed === undefined ? 0 : row.gap_signed;
      var se = row.se_of_gap || 0;
      values.push(gap - se, gap + se);
    });
    var minimum = Math.min.apply(null, values);
    var maximum = Math.max.apply(null, values);
    var span = maximum - minimum || Math.abs(maximum) || 1;
    var lower = minimum - span * 0.12;
    var upper = maximum + span * 0.12;
    var logs = rows.map(function (row) { return Math.log(Number(row.tuning_value)); });
    var logMin = Math.min.apply(null, logs);
    var logMax = Math.max.apply(null, logs);
    function x(value) {
      if (logMax === logMin) return margin.left + plotWidth / 2;
      return margin.left + ((Math.log(Number(value)) - logMin) / (logMax - logMin)) * plotWidth;
    }
    function y(value) {
      return margin.top + (1 - (value - lower) / (upper - lower)) * plotHeight;
    }
    var svg = svgNode("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": gapLabel + " over the " + (key === "alpha" ? "ridge alpha" : "logistic C") +
        " tuning grid, with the paired standard error."
    });
    [lower, 0, upper].forEach(function (value) {
      svg.appendChild(svgNode("line", {
        x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: "model-grid"
      }));
      var label = svgNode("text", {
        x: margin.left - 9, y: y(value) + 4.5, "text-anchor": "end", class: "model-axis"
      });
      label.textContent = format(value, digits);
      svg.appendChild(label);
    });
    svg.appendChild(svgNode("line", {
      x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), class: "model-grid"
    }));
    var axis = svgNode("text", {
      x: margin.left, y: 15, "text-anchor": "start", class: "model-axis-title"
    });
    axis.textContent = gapLabel;
    svg.appendChild(axis);
    svg.appendChild(svgNode("line", {
      x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotHeight, class: "model-grid"
    }));
    svg.appendChild(svgNode("line", {
      x1: width - margin.right, x2: width - margin.right,
      y1: margin.top, y2: margin.top + plotHeight, class: "model-grid"
    }));
    var points = rows.map(function (row) {
      var gap = row.gap_signed === null || row.gap_signed === undefined ? 0 : row.gap_signed;
      return x(row.tuning_value) + "," + y(gap);
    });
    svg.appendChild(svgNode("polyline", { points: points.join(" "), class: "model-series validation" }));
    rows.forEach(function (row, index) {
      var cx = x(row.tuning_value);
      var gap = row.gap_signed === null || row.gap_signed === undefined ? 0 : row.gap_signed;
      var cy = y(gap);
      var se = row.se_of_gap || 0;
      svg.appendChild(svgNode("line", {
        x1: cx, x2: cx, y1: y(gap - se), y2: y(gap + se),
        class: "model-error"
      }));
      svg.appendChild(svgNode("line", {
        x1: cx - 3, x2: cx + 3, y1: y(gap - se), y2: y(gap - se),
        class: "model-error"
      }));
      svg.appendChild(svgNode("line", {
        x1: cx - 3, x2: cx + 3, y1: y(gap + se), y2: y(gap + se),
        class: "model-error"
      }));
      svg.appendChild(svgNode("circle", {
        cx: cx, cy: cy, r: row.selected ? 4.5 : 3.2,
        class: "ladder-point validation" + (row.selected ? " is-chosen" : "")
      }));
      if (row.selected) {
        svg.appendChild(svgNode("circle", { cx: cx, cy: cy, r: 8, class: "ladder-ring" }));
        var chosen = svgNode("text", {
          x: cx, y: cy - 12, "text-anchor": "middle", class: "ladder-chosen"
        });
        chosen.textContent = "Chosen";
        svg.appendChild(chosen);
      }
      var tick = svgNode("text", {
        x: cx, y: height - margin.bottom + 13, "text-anchor": "middle", class: "ladder-x-label"
      });
      tick.textContent = compactNumber(row.tuning_value);
      svg.appendChild(tick);
      if (index === 0 || index === rows.length - 1) {
        var edge = svgNode("text", {
          x: cx, y: height - 8, "text-anchor": "middle", class: "model-edge-label"
        });
        edge.textContent = "grid edge";
        svg.appendChild(edge);
      }
    });
    var wrap = node("div", "model-window-wrap model-tuning");
    wrap.appendChild(svg);
    section.appendChild(wrap);
    return section;
  }

  function detailsSection(model) {
    var details = node("details", "model-details");
    details.appendChild(node("summary", "", "How this is measured"));
    var body = node("div", "model-details-body");
    body.appendChild(node("p", "model-sentence", VERDICT_NOTE));
    body.appendChild(node("h3", "model-detail-title", "All measures by split"));
    body.appendChild(fullComparisonTable(model));
    body.appendChild(node("p", "model-sentence",
      model.kind === "classification" ?
        "Precision, recall and MCC use the top " +
          format(shareNumber(model), 0) +
          "% of each purchase month on every split. The rule is not trained, so it has no training row." :
        "The rule is not trained, so it has no training row; its other cells use the same splits as the model."));
    var windows = windowSection(model);
    if (windows) body.appendChild(windows);
    var months = testMonthsSection(model);
    if (months) body.appendChild(months);
    body.appendChild(parametersSection(model));
    body.appendChild(importanceSection(model));
    var tuning = tuningSection(model);
    if (tuning) body.appendChild(tuning);
    body.appendChild(footnoteSection(model));
    details.appendChild(body);
    return details;
  }

  function footnoteSection(model) {
    var windows = model.metrics.validation.windows.length;
    var cutoff = model.train_cutoff || "the cutoff";
    var range = monthRangeValues(model.test_months || []);
    var text = "Chosen on " + windows + " validation windows (training up to " + cutoff + ").";
    if (range) text += " Test months (" + range + ") are for reference.";
    return node("p", "model-footnote model-panel-footnote", text);
  }

  function liveModelLine(model) {
    if (model.name !== "late_risk" || !model.live_model || !model.live_model.label) return null;
    return node("p", "model-sentence model-live", model.live_model.label);
  }

  function renderSelected(model) {
    var panel = node("section", "panel model-selected");
    panel.setAttribute("aria-labelledby", "model-selected-title");
    var head = node("div", "model-selected-head");
    var titleWrap = node("div", "model-selected-title-wrap");
    var title = node("h2", "model-selected-title", model.title);
    title.id = "model-selected-title";
    titleWrap.appendChild(title);
    head.appendChild(titleWrap);
    var links = node("div", "model-links");
    function link(text, href) {
      var anchor = node("a", "model-link", text + " \u2197");
      anchor.href = href;
      anchor.target = "_blank";
      anchor.rel = "noopener";
      return anchor;
    }
    links.appendChild(link("Open in MLflow", model.links.run_path));
    links.appendChild(link("Compare runs in MLflow", model.links.compare_path));
    head.appendChild(links);
    panel.appendChild(head);
    var live = liveModelLine(model);
    if (live) panel.appendChild(live);
    panel.appendChild(ladderSection(model));
    panel.appendChild(slimComparisonTable(model));
    panel.appendChild(verdictSection(model));
    panel.appendChild(detailsSection(model));
    return panel;
  }

  function selectedModel() {
    var models = state.models || [];
    for (var index = 0; index < models.length; index += 1) {
      if (models[index].name === state.selected) return models[index];
    }
    return models[0] || null;
  }

  function animateSelection(panel) {
    if (!panel || reducedMotion()) return;
    var svgs = Array.prototype.slice.call(panel.querySelectorAll(".model-ladder svg"));
    svgs.forEach(function (svg) {
      svg.classList.add("is-entering");
    });
    window.setTimeout(function () {
      svgs.forEach(function (svg) {
        svg.classList.remove("is-entering");
      });
    }, 360);
  }

  function renderSelectedPanel() {
    var host = $("model-selected");
    if (!host) return;
    var model = selectedModel();
    if (!model) return;
    var panel = renderSelected(model);
    host.replaceChildren(panel);
    animateSelection(panel);
    ensureImportance(model);
  }

  function modelFromHash() {
    var match = /^#models\/([A-Za-z0-9_]+)$/.exec(window.location.hash || "");
    return match ? match[1] : null;
  }

  function selectModel(name, updateHash) {
    if (!state.models || !state.models.some(function (model) { return model.name === name; })) return;
    state.selected = name;
    updateTileState();
    renderSelectedPanel();
    if (updateHash) {
      var target = "#models/" + name;
      if (window.location.hash !== target) {
        if (window.history && window.history.replaceState) {
          window.history.replaceState(null, "", target);
        } else {
          window.location.hash = target;
        }
      }
    }
  }

  function activateModelsView() {
    var host = $("view-drift");
    var button = document.querySelector('[data-view="drift"]');
    if (host && host.hidden && button) button.click();
  }

  function handleHash() {
    var name = modelFromHash();
    if (!name) return;
    activateModelsView();
    if (state.models) selectModel(name, false);
  }

  function renderModels(models) {
    var root = $("models-root");
    if (!root) return;
    renderHeader(models);
    var requested = modelFromHash();
    state.selected = models.some(function (model) { return model.name === requested; }) ?
      requested : models[0].name;
    root.replaceChildren();
    root.appendChild(renderTiles(models));
    var selected = node("div", "model-selected-host");
    selected.id = "model-selected";
    root.appendChild(selected);
    updateTileState();
    renderSelectedPanel();
  }

  function selectedDate() {
    var input = $("asof-date");
    return input && input.value ? input.value : "";
  }

  function open() {
    if (state.loading) return;
    var root = $("models-root");
    if (!root) return;
    state.loading = true;
    root.replaceChildren(node("p", "models-loading", "Loading models from the local registry\u2026"));
    var date = selectedDate();
    fetch("/api/models" + (date ? "?date=" + encodeURIComponent(date) : ""))
      .then(function (response) {
        return response.json().then(function (payload) { return { ok: response.ok, payload: payload }; });
      })
      .then(function (result) {
        state.loading = false;
        if (!result.ok) {
          throw new Error(result.payload && result.payload.error ?
            result.payload.error.message : "The model registry is unavailable.");
        }
        if (!Array.isArray(result.payload) || !result.payload.length) {
          throw new Error("The model registry returned no models.");
        }
        state.models = result.payload;
        state.loaded = true;
        renderModels(state.models);
      })
      .catch(function (error) {
        state.loading = false;
        renderError(error.message);
      });
  }

  window.PrazoModels = {
    open: open,
    gapDescriptor: gapDescriptor,
    visibleReferences: visibleReferences
  };

  function refreshIfOpen() {
    var view = $("view-drift");
    if (state.loaded && view && !view.hidden) open();
  }

  var dateInput = $("asof-date");
  if (dateInput) dateInput.addEventListener("change", refreshIfOpen);
  ["prev-day", "next-day"].forEach(function (id) {
    var button = $(id);
    if (button) {
      button.addEventListener("click", function () {
        window.setTimeout(refreshIfOpen, 0);
      });
    }
  });

  var host = $("view-drift");
  if (!state.hashBound) {
    window.addEventListener("hashchange", handleHash);
    state.hashBound = true;
  }
  if (modelFromHash()) activateModelsView();
  if (host && !host.hidden) open();
})();
