/* Actions tab: the daily parcel queue and the monthly promise policy.
   Both sections call the local Actions API for the selected as-of date. */
(function () {
  "use strict";

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var REGION_NAMES = {
    SP: "S\u00e3o Paulo",
    SE: "Southeast",
    S: "South",
    CO: "Center-West",
    NE: "Northeast",
    N: "North"
  };
  var cache = {};
  var state = {
    date: null,
    current: null,
    showAll: false,
    filter: "act",
    lookup: null,
    lookupValue: "",
    prediction: null,
    token: 0
  };

  function $(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtInt(value) {
    return Number(value || 0).toLocaleString("en-US");
  }

  function fmtOne(value) {
    if (value === null || value === undefined || !isFinite(Number(value))) return "\u2014";
    return Number(value).toFixed(1);
  }

  function fmtPct(value) {
    if (value === null || value === undefined || !isFinite(Number(value))) return "\u2014";
    return (Number(value) * 100).toFixed(1) + "%";
  }

  function fmtSignedOne(value) {
    var number = Number(value);
    if (!isFinite(number)) return "\u2014";
    var rounded = Math.round(number * 10) / 10;
    if (rounded === 0) return "0.0";
    return (rounded > 0 ? "+" : "") + rounded.toFixed(1);
  }

  function ordinal(value) {
    var number = Math.round(Number(value));
    if (!isFinite(number)) return "\u2014";
    var mod100 = number % 100;
    if (mod100 >= 11 && mod100 <= 13) return number + "th";
    var mod10 = number % 10;
    if (mod10 === 1) return number + "st";
    if (mod10 === 2) return number + "nd";
    if (mod10 === 3) return number + "rd";
    return number + "th";
  }

  function parts(iso) {
    var pieces = String(iso || "").slice(0, 10).split("-");
    return { year: Number(pieces[0]), month: Number(pieces[1]), day: Number(pieces[2]) };
  }

  function fmtTableDate(iso) {
    var p = parts(iso);
    if (!p.month) return "\u2014";
    return p.day + " " + MONTHS[p.month - 1] + " " + p.year;
  }

  function fmtRange(startIso, endIso) {
    var start = parts(startIso);
    var end = parts(endIso);
    if (!start.month || !end.month) return "";
    if (start.year === end.year && start.month === end.month) {
      return start.day + "\u2013" + end.day + " " + MONTHS[end.month - 1] + " " + end.year;
    }
    if (start.year === end.year) {
      return start.day + " " + MONTHS[start.month - 1] + "\u2013" +
        end.day + " " + MONTHS[end.month - 1] + " " + end.year;
    }
    return start.day + " " + MONTHS[start.month - 1] + " " + start.year + "\u2013" +
      end.day + " " + MONTHS[end.month - 1] + " " + end.year;
  }

  function regionName(code) {
    return REGION_NAMES[code] || code || "Unknown";
  }

  function promiseModelLine(payload) {
    return payload.model_label || "";
  }

  function queueModelLine(payload) {
    var versions = payload.model_versions || {};
    var pieces = [];
    if (versions.late_risk !== undefined) pieces.push("Late risk model v" + versions.late_risk);
    if (versions.handover_eta !== undefined) pieces.push("Delivery-time model v" + versions.handover_eta);
    return pieces.join(" \u00b7 ");
  }

  function errorFor(result) {
    if (!result) return "The server did not respond.";
    if (result.status >= 500) return "The Actions data could not be loaded. Please try again.";
    if (result.payload && result.payload.error && result.payload.error.message) {
      return result.payload.error.message;
    }
    return "The server returned HTTP " + result.status + ".";
  }

  var WHY_LABELS = {
    "Spare time": "Remaining slack",
    "Spare time below zero": "Remaining slack below zero"
  };

  function whyLabel(item) {
    var label = item && item.label;
    return WHY_LABELS[label] || label;
  }

  function riskInputs(row) {
    return (row.why || []).map(whyLabel);
  }

  function actionLine(row, payload) {
    var capacity = Number(payload.capacity || 0);
    if (row.past_promise) return "Update the customer: its promised date has passed.";
    if (row.tier === "act") {
      return "Act today: ranked " + ordinal(row.rank) + " of " + fmtInt(row.open_orders) +
        ", inside today's list of " + fmtInt(capacity) + ".";
    }
    if (row.tier === "watch") {
      return "Watch: ranked " + ordinal(row.rank) + " of " + fmtInt(row.open_orders) +
        ", in the next block of " + fmtInt(capacity) + " after today's list.";
    }
    return "Monitor: ranked " + ordinal(row.rank) + " of " + fmtInt(row.open_orders) +
      ", outside today's action and watch lists.";
  }

  function whyRankBlock(row) {
    if (row.past_promise) {
      return '<div class="actions-why-rank"><p class="actions-card-line actions-why-rank-title">' +
        "Why this rank</p><p class=\"actions-card-line\">Its promised date has passed, so it is not ranked; " +
        "it is in the customer-update list.</p></div>";
    }
    var late = row.late_risk_percentile;
    var spare = row.spare_time_percentile;
    var lines = [];
    lines.push('<p class="actions-card-line">Riskier than <strong>' +
      esc(fmtPct(late)) + "</strong> of training parcels (model).</p>");
    lines.push('<p class="actions-card-line">Remaining slack: <strong>' +
      esc(fmtOne(row.spare_time)) + " days</strong> (rule: promise minus recent route time; " +
      "higher priority than <strong>" + esc(fmtPct(spare)) + "</strong> of training parcels).</p>");
    lines.push('<p class="actions-card-line">Priority = average of the two \u2192 rank <strong>' +
      esc(fmtInt(row.rank)) + "</strong> of " + esc(fmtInt(row.open_orders)) + ".</p>");
    return '<div class="actions-why-rank"><p class="actions-card-line actions-why-rank-title">' +
      "Why this rank</p>" + lines.join("") + "</div>";
  }

  function whyDetail(row, payload) {
    var labels = riskInputs(row);
    var facts = '<div class="actions-card-facts">' +
      '<div class="actions-card-fact"><span class="actions-card-label">Predicted delivery</span>' +
      '<span class="actions-card-value">' + esc(fmtTableDate(row.predicted_delivery_date)) + "</span></div>" +
      '<div class="actions-card-fact"><span class="actions-card-label">Promised</span>' +
      '<span class="actions-card-value">' + esc(fmtTableDate(row.promised_date)) + "</span></div>" +
      '<div class="actions-card-fact"><span class="actions-card-label">Remaining slack</span>' +
      '<span class="actions-card-value">' + esc(fmtOne(row.spare_time)) + " days</span></div></div>";
    var inputs = '<p class="actions-card-line">' +
      (labels.length
        ? "Main inputs raising late risk: <strong>" + labels.map(esc).join("</strong>, <strong>") + "</strong>."
        : "No input raised late risk.") + "</p>";
    var action = '<p class="actions-card-line actions-card-action">' + esc(actionLine(row, payload)) + "</p>";
    return '<div class="actions-card">' + whyRankBlock(row) + facts + inputs + action + "</div>";
  }

  function tierBadge(row) {
    if (row.past_promise) return '<span class="actions-badge update">Update customer</span>';
    if (row.tier === "act") return '<span class="actions-badge act">Act today</span>';
    if (row.tier === "watch") return '<span class="actions-badge watch">Watch</span>';
    return '<span class="actions-badge none">Monitor</span>';
  }

  function setRowExpanded(row, open) {
    var detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("actions-why-row")) return;
    detail.hidden = !open;
    row.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function toggleRow(row) {
    var detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("actions-why-row")) return;
    setRowExpanded(row, detail.hidden);
  }

  function queueRow(row, index, payload) {
    var promise = fmtTableDate(row.promised_date);
    var predicted = fmtTableDate(row.predicted_delivery_date);
    var daysLeft = row.days_left === null || row.days_left === undefined ? "\u2014" : row.days_left + " d";
    var rank = row.past_promise ? "\u2014" : row.rank;
    return '<tr class="actions-row" data-order-id="' + esc(row.order_id) +
      '" data-actions-row="' + index + '" tabindex="0" aria-expanded="false"' +
      ' aria-controls="actions-why-' + index + '">' +
      '<td class="actions-rank">' + esc(rank) + "</td>" +
      '<td class="actions-order">' + esc(row.order_id) + "</td>" +
      '<td class="actions-num num">' + esc(promise) + "</td>" +
      '<td class="actions-num num">' + esc(daysLeft) + "</td>" +
      '<td class="actions-num num">' + esc(predicted) + "</td>" +
      "<td>" + tierBadge(row) + "</td>" +
      "</tr>" +
      '<tr class="actions-why-row" id="actions-why-' + index + '" data-actions-why-row="' + index + '" hidden>' +
      '<td colspan="6">' + whyDetail(row, payload) + "</td></tr>";
  }

  function predictionCard(data) {
    if (!data || (!data.on_list && !data.customer_update && data.rank === null)) return "";
    var facts = [
      ['Late risk', data.late_risk_percentile === null || data.late_risk_percentile === undefined
        ? "not scored" : "Riskier than " + fmtPct(data.late_risk_percentile) + " of training parcels"],
      ['Predicted delivery', fmtTableDate(data.predicted_delivery_date)],
      ['Remaining slack', data.spare_time === null || data.spare_time === undefined
        ? "\u2014" : fmtOne(data.spare_time) + " days"],
      ['Days left', data.days_left === null || data.days_left === undefined
        ? "\u2014" : data.days_left + " d"],
      [data.on_list ? "Where it ranks today" : "Where it would rank today",
        data.rank === null || data.rank === undefined
          ? "not ranked" : ordinal(data.rank) + " of " + fmtInt(data.open_orders)],
      ["On today's list", data.customer_update ? "Customer updates" : (data.on_list ? "Yes" : "No")]
    ];
    var labels = (data.why || []).map(whyLabel);
    var inputs = labels.length
      ? "Main inputs raising late risk: <strong>" +
        labels.map(esc).join("</strong>, <strong>") + "</strong>."
      : "No input raised late risk.";
    return '<div class="actions-prediction">' +
      '<p class="actions-card-line actions-why-rank-title">Live prediction</p>' +
      '<div class="actions-card-facts">' + facts.map(function (fact) {
        return '<div class="actions-card-fact"><span class="actions-card-label">' + esc(fact[0]) +
          '</span><span class="actions-card-value">' + esc(fact[1]) + "</span></div>";
      }).join("") + "</div>" +
      '<p class="actions-card-line">' + inputs + "</p>" +
      (data.message ? '<p class="actions-card-line">' + esc(data.message) + "</p>" : "") +
      "</div>";
  }

  function filterButton(value, label, current) {
    return '<button type="button" data-actions-filter="' + value + '" aria-pressed="' +
      (current === value ? "true" : "false") + '">' + esc(label) + "</button>";
  }

  function csvField(value) {
    var text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function downloadCustomerList() {
    var entry = state.current;
    var payload = entry && entry.queue && entry.queue.payload;
    if (!payload) return;
    var date = payload.date || entry.date;
    var past = (payload.rows || []).filter(function (row) { return row.past_promise; });
    var lines = ["order_id,promised_date,days_past_promise"];
    past.forEach(function (row) {
      var days = row.days_left === null || row.days_left === undefined
        ? "" : Math.abs(Number(row.days_left));
      lines.push([csvField(row.order_id), csvField(row.promised_date), csvField(days)].join(","));
    });
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "customer_updates_" + date + ".csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 0);
  }

  function queueSection(result) {
    if (!result) {
      return '<section class="panel actions-panel"><div class="actions-error">' +
        esc(errorFor(result)) + "</div></section>";
    }
    if (!result.ok) {
      return '<section class="panel actions-panel"><div class="actions-error">' +
        esc(errorFor(result)) + "</div></section>";
    }
    var payload = result.payload || {};
    var capacity = Number(payload.capacity || 0);
    var rows = Array.isArray(payload.rows) ? payload.rows : [];
    var past = rows.filter(function (row) { return row.past_promise; });
    var ranked = rows.filter(function (row) { return !row.past_promise; });
    var noRanked = ranked.length === 0;
    var filter = state.filter || "all";
    var filtered = filter === "all" ? ranked : ranked.filter(function (row) {
      return row.tier === filter;
    });
    var defaultCount = capacity > 0 ? capacity * 2 : ranked.length;
    var visible = filter === "all"
      ? (state.showAll ? ranked : ranked.slice(0, defaultCount))
      : filtered;
    var tiers = payload.tiers || {};
    var performance = payload.performance || {};
    var blend = performance.blend || {};
    var slack = performance.slack || {};
    var notice = payload.capacity_note && !noRanked
      ? '<p class="actions-note">' + esc(payload.capacity_note) + "</p>" : "";

    var lookupMessage = state.lookup && !state.prediction ? state.lookup.message || "" : "";
    var lookup = '<div class="actions-lookup">' +
      '<label for="actions-order-find">Predict an order</label>' +
      '<input id="actions-order-find" type="text" autocomplete="off" spellcheck="false" ' +
      'placeholder="Paste an order ID" value="' + esc(state.lookupValue || "") + '" />' +
      '<button type="button" class="actions-why-btn" data-actions-find="1">Predict</button>' +
      '<p class="actions-lookup-msg" id="actions-lookup-msg" role="status">' + esc(lookupMessage) + "</p>" +
      '<div class="actions-lookup-result" id="actions-lookup-result">' +
      predictionCard(state.prediction) + "</div></div>";

    var customer = "";
    if (past.length) {
      var shown = past.slice(0, 12);
      var more = past.length > shown.length
        ? '<span class="actions-customer-item">+' + fmtInt(past.length - shown.length) + " more</span>" : "";
      var summary = "Show " + fmtInt(shown.length) +
        (past.length > shown.length ? " of " + fmtInt(past.length) : "") + " order IDs";
      customer = '<div class="actions-customer"><div class="actions-customer-head">' +
        '<h3>Update customer <span class="actions-muted">' + fmtInt(past.length) + "</span></h3>" +
        '<button type="button" class="actions-why-btn" data-actions-download="1">Download list</button>' +
        "</div>" +
        '<p>Already past the promised date at 09:00.</p>' +
        '<details class="actions-customer-details"><summary>' + esc(summary) + "</summary>" +
        '<div class="actions-customer-list">' + shown.map(function (row) {
          var promise = row.promised_date ? fmtTableDate(row.promised_date) : "no promise";
          var late = row.days_left === null || row.days_left === undefined ? "" :
            " \u00b7 " + Math.abs(row.days_left) + " d past";
          return '<span class="actions-customer-item">' + esc(row.order_id) + " \u00b7 " +
            esc(promise) + esc(late) + "</span>";
        }).join("") + more + "</div></details></div>";
    }

    var filterBar = '<div class="actions-filters"><div class="seg" role="group" ' +
      'aria-label="Filter parcels">' +
      filterButton("all", "All", filter) +
      filterButton("act", "Act today", filter) +
      filterButton("watch", "Watch", filter) +
      "</div></div>";

    var table = '<div class="actions-table-wrap"><table class="actions-table">' +
      '<caption class="sr-only">Ranked parcels to act on at the selected date</caption>' +
      "<thead><tr><th scope=\"col\" title=\"Priority: blend of late risk and remaining slack\">Rank</th>" +
      "<th scope=\"col\">Order</th>" +
      "<th scope=\"col\" class=\"num\">Promised date</th>" +
      "<th scope=\"col\" class=\"num\">Days left</th>" +
      "<th scope=\"col\" class=\"num\" title=\"The purchase time plus the model's predicted total days from purchase to delivery, shown as a calendar date.\">Predicted delivery</th>" +
      "<th scope=\"col\">Action</th></tr></thead><tbody>" +
      visible.map(function (row) { return queueRow(row, rows.indexOf(row), payload); }).join("") +
      "</tbody></table></div>";

    var footer = "";
    if (filter === "all" && capacity > 0 && !state.showAll && ranked.length > visible.length) {
      footer = '<button type="button" class="actions-why-btn" data-actions-all="1">Show all ' +
        fmtInt(ranked.length) + " ranked parcels</button>";
    } else if (filter === "all" && capacity > 0 && state.showAll && ranked.length > capacity * 2) {
      footer = '<button type="button" class="actions-why-btn" data-actions-all="1">Show the top ' +
        fmtInt(capacity * 2) + " only</button>";
    } else if (filter !== "all" && ranked.length > visible.length) {
      footer = '<button type="button" class="actions-why-btn" data-actions-all="1">Show all ' +
        fmtInt(ranked.length) + " ranked parcels</button>";
    }

    var rankingIntro = "Ranked each morning. Act today = the top " + fmtInt(capacity) +
      ", a set 10% of yesterday's handovers; Watch = the next " + fmtInt(capacity) + ".";

    var performanceLine = "Last 4 weeks (" + esc(fmtRange(performance.start, performance.end)) + "): <strong>" +
      fmtInt(blend.caught_3plus_days) + "</strong> late parcels flagged 3+ days ahead (remaining-slack rule: <strong>" +
      fmtInt(slack.caught_3plus_days) + "</strong>).";
    var performanceDefinition = "Outcomes known by the selected date, excluding parcels already past their " +
      "promise. All late parcels found: blend " +
      fmtInt(blend.caught) + ", " + fmtOne(blend.mean_days_left) + " days ahead on average; rule " +
      fmtInt(slack.caught) + ", " + fmtOne(slack.mean_days_left) + " days.";
    var listBlock = noRanked
      ? '<p class="actions-note">No open parcels to rank.</p>'
      : filterBar + table;

    return '<section class="panel actions-panel" aria-labelledby="actions-queue-title">' +
      '<div class="actions-head"><div><h2 id="actions-queue-title">Parcels to act on today</h2>' +
      "<p>" + esc(rankingIntro) + "</p></div>" +
      '<div class="actions-head-right"><span class="actions-model">' +
      esc(queueModelLine(payload)) + "</span></div></div>" +
      notice +
      '<div class="actions-tier-grid">' +
      '<div class="actions-tier act"><div class="actions-tier-label">Act today</div><div class="actions-tier-value">' +
      fmtInt(tiers.act) + "</div></div>" +
      '<div class="actions-tier watch"><div class="actions-tier-label">Watch</div><div class="actions-tier-value">' +
      fmtInt(tiers.watch) + "</div></div>" +
      '<div class="actions-tier none"><div class="actions-tier-label">Monitor</div><div class="actions-tier-value">' +
      fmtInt(tiers.none) + "</div></div></div>" +
      lookup + customer + listBlock +
      '<div class="actions-performance" title="' + esc(performanceDefinition) + '">' +
      performanceLine + " " + footer + "</div></section>";
  }

  function promiseRow(region) {
    var known = region.our_mean_promise_days !== null && region.our_mean_promise_days !== undefined &&
      region.olist_mean_promise_days !== null && region.olist_mean_promise_days !== undefined;
    var diff = known
      ? Number(region.our_mean_promise_days) - Number(region.olist_mean_promise_days) : NaN;
    var shown = Math.round(diff * 10) / 10;
    var tone = shown < 0 ? "is-shorter" : (shown > 0 ? "is-longer" : "");
    return "<tr>" +
      '<td class="actions-region">' + esc(region.region) +
      '<span class="actions-muted"> ' + esc(regionName(region.region)) + "</span></td>" +
      '<td class="actions-num num">' + esc(fmtOne(region.buffer_days)) + " d</td>" +
      '<td class="actions-num num">' + esc(fmtOne(region.our_mean_promise_days)) + " d</td>" +
      '<td class="actions-num num">' + esc(fmtOne(region.olist_mean_promise_days)) + " d</td>" +
      '<td class="actions-num num actions-diff ' + tone + '">' + esc(fmtSignedOne(diff)) + " d</td>" +
      '<td class="actions-num num">' + fmtInt(region.n_orders) + "</td></tr>";
  }

  function promisesSection(result) {
    if (!result) {
      return '<section class="panel actions-panel"><div class="actions-error">' +
        esc(errorFor(result)) + "</div></section>";
    }
    if (!result.ok) {
      return '<section class="panel actions-panel"><div class="actions-error">' +
        esc(errorFor(result)) + "</div></section>";
    }
    var payload = result.payload || {};
    var regions = Array.isArray(payload.regions) ? payload.regions : [];
    var performance = payload.performance || null;
    var targetPct = payload.target !== undefined ? (Number(payload.target) * 100).toFixed(1) : "3.5";
    var promiseLine = "Our delivery-time estimate plus a regional buffer, reset each month, aiming for " +
      targetPct + "% late. The result line shows the outcome once enough orders are due.";

    var table = regions.length
      ? '<div class="actions-region-table-wrap"><table class="actions-region-table">' +
        '<caption class="sr-only">Proposed promise dates by region for the selected month</caption>' +
        "<thead><tr><th scope=\"col\">Region</th><th scope=\"col\" class=\"num\">Buffer</th>" +
        "<th scope=\"col\" class=\"num\">Our promise</th><th scope=\"col\" class=\"num\">Olist promise</th>" +
        "<th scope=\"col\" class=\"num\">Difference</th><th scope=\"col\" class=\"num\">Orders</th></tr></thead><tbody>" +
        regions.map(promiseRow).join("") + "</tbody></table></div>"
      : '<p class="actions-note">No orders this month.</p>';

    var resultLine = "Not enough orders yet.";
    if (performance && performance.enough) {
      resultLine = "Orders bought " + esc(fmtRange(performance.start, performance.end)) +
        ": ours <strong>" +
        esc(fmtOne(performance.our_mean_promise_days)) + " days</strong>, <strong>" +
        esc(fmtPct(performance.our_late_rate)) + "</strong> late; Olist <strong>" +
        esc(fmtOne(performance.olist_mean_promise_days)) + " days</strong>, <strong>" +
        esc(fmtPct(performance.olist_late_rate)) + "</strong> late.";
    }
    if (performance && performance.start && performance.end) {
      resultLine += '<span class="actions-muted"> \u00b7 ' + fmtInt(performance.orders) + " orders</span>";
    }
    var notYetDue = performance && performance.not_yet_due ? Number(performance.not_yet_due) : 0;
    var performanceTitle = "Orders bought in the latest 4-week window with at least 99% of promises passed. " +
      fmtInt(notYetDue) + " orders not yet due, left out. Late means arrived after the promise, or still open past it.";

    return '<section class="panel actions-panel" aria-labelledby="actions-promises-title">' +
      '<div class="actions-head"><div><h2 id="actions-promises-title">Promise dates this month</h2>' +
      '<p class="actions-promise-result" title="' + esc(performanceTitle) + '">' + resultLine + "</p>" +
      "<p>" + esc(promiseLine) + "</p></div>" +
      '<div class="actions-head-right"><span class="actions-model">' +
      esc(promiseModelLine(payload)) + "</span></div></div>" +
      table + "</section>";
  }

  function clearFound() {
    var host = $("actions-root");
    if (!host) return;
    var found = host.querySelectorAll(".actions-row.is-found");
    for (var i = 0; i < found.length; i += 1) found[i].classList.remove("is-found");
  }

  function revealLookup(host) {
    var lookup = state.lookup;
    if (!lookup || !lookup.id) return;
    if (lookup.open) {
      var row = host.querySelector('.actions-row[data-order-id="' + lookup.id + '"]');
      if (row) {
        row.classList.add("is-found");
        setRowExpanded(row, true);
        if (lookup.scroll) row.scrollIntoView({ block: "center" });
        lookup.scroll = false;
      }
    }
    var message = $("actions-lookup-msg");
    if (message) message.textContent = state.prediction ? "" : lookup.message || "";
  }

  function render(entry) {
    var host = $("actions-root");
    if (!host) return;
    host.innerHTML = queueSection(entry.queue) + promisesSection(entry.promises);
    revealLookup(host);
  }

  function closeLookupCard() {
    var host = $("actions-root");
    if (!host || !state.lookup || !state.lookup.id) return;
    var row = host.querySelector('.actions-row[data-order-id="' + state.lookup.id + '"]');
    if (!row) return;
    row.classList.remove("is-found");
    setRowExpanded(row, false);
  }

  function setLookup(id, open, message, scroll) {
    state.lookup = { id: id, open: open, message: message, scroll: !!scroll };
  }

  function showLookupMessage(message) {
    var host = $("actions-root");
    if (host) clearFound();
    var target = $("actions-lookup-msg");
    if (target) target.textContent = message;
  }

  function findOrder() {
    var input = $("actions-order-find");
    var orderId = input ? input.value.trim().toLowerCase() : "";
    state.lookupValue = orderId;
    closeLookupCard();
    var entry = state.current;
    var payload = entry && entry.queue && entry.queue.payload;
    if (!/^[0-9a-f]{32}$/.test(orderId)) {
      state.prediction = null;
      setLookup(orderId, false, "Order IDs are 32 characters long, using letters a\u2013f and digits 0\u20139.", false);
      if (entry) render(entry);
      else showLookupMessage(state.lookup.message);
      return;
    }
    if (!payload) return;
    var date = state.date || entry.date;
    fetchJson("/api/actions/predict?date=" + encodeURIComponent(date) + "&order_id=" + encodeURIComponent(orderId))
      .then(function (result) {
        if (state.date !== date) return;
        var data = result.payload || {};
        if (!result.ok) {
          state.prediction = null;
          setLookup(orderId, false, errorFor(result), false);
          render(entry);
          return;
        }
        var message = data.message || "";
        if (data.on_list) {
          state.prediction = data;
          state.filter = "all";
          var capacity = Number(payload.capacity || 0);
          if (capacity > 0 && Number(data.rank) > capacity * 2) state.showAll = true;
          setLookup(orderId, true, message, true);
        } else {
          state.prediction = data.found && (data.rank !== null || data.customer_update) ? data : null;
          setLookup(orderId, false, message, false);
        }
        render(entry);
      });
  }

  function fetchJson(path) {
    return fetch(path, { headers: { Accept: "application/json" } })
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (payload) {
          return { ok: response.ok, status: response.status, payload: payload };
        });
      })
      .catch(function (error) {
        return { ok: false, status: 0, payload: { error: { message: error.message } } };
      });
  }

  function load(date) {
    var host = $("actions-root");
    if (!host) return;
    var token = ++state.token;
    host.innerHTML = '<p class="actions-loading" role="status">Loading today\u2019s actions\u2026</p>';
    Promise.all([
      fetchJson("/api/actions/queue?date=" + encodeURIComponent(date)),
      fetchJson("/api/actions/promises?date=" + encodeURIComponent(date))
    ]).then(function (results) {
      if (token !== state.token) return;
      var entry = { date: date, queue: results[0], promises: results[1] };
      cache[date] = entry;
      state.current = entry;
      render(entry);
    });
  }

  function open(date) {
    state.date = date;
    var entry = cache[date];
    if (entry) {
      state.current = entry;
      render(entry);
      return;
    }
    load(date);
  }

  function setDate(date) {
    state.date = date;
    state.showAll = false;
    state.filter = "act";
    state.lookup = null;
    state.lookupValue = "";
    state.prediction = null;
    var view = $("view-actions");
    if (!view || view.hidden) return;
    open(date);
  }

  function bind() {
    var host = $("actions-root");
    if (!host) return;
    var downX = 0;
    var downY = 0;
    host.addEventListener("mousedown", function (event) {
      downX = event.clientX;
      downY = event.clientY;
    });
    host.addEventListener("click", function (event) {
      var find = event.target.closest ? event.target.closest("[data-actions-find]") : null;
      if (find) {
        findOrder();
        return;
      }
      var filter = event.target.closest ? event.target.closest("[data-actions-filter]") : null;
      if (filter && state.current) {
        state.filter = filter.getAttribute("data-actions-filter");
        render(state.current);
        return;
      }
      var download = event.target.closest ? event.target.closest("[data-actions-download]") : null;
      if (download) {
        downloadCustomerList();
        return;
      }
      var all = event.target.closest ? event.target.closest("[data-actions-all]") : null;
      if (all && state.current) {
        if (state.filter !== "all") {
          state.filter = "all";
          state.showAll = true;
        } else {
          state.showAll = !state.showAll;
        }
        render(state.current);
        return;
      }
      var row = event.target.closest ? event.target.closest("[data-actions-row]") : null;
      if (row) {
        var dragged = Math.hypot(event.clientX - downX, event.clientY - downY) > 6;
        var selection = window.getSelection && window.getSelection().toString();
        if (dragged && selection) return;
        if (event.detail > 1 && event.target.closest(".actions-order") && selection) return;
        toggleRow(row);
      }
    });
    host.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && event.target && event.target.id === "actions-order-find") {
        event.preventDefault();
        findOrder();
        return;
      }
      if ((event.key === "Enter" || event.key === " ") &&
          event.target && event.target.matches && event.target.matches("[data-actions-row]")) {
        event.preventDefault();
        toggleRow(event.target);
      }
    });
    host.addEventListener("input", function (event) {
      if (event.target && event.target.id === "actions-order-find") {
        state.lookupValue = event.target.value;
      }
    });
  }

  window.PrazoActions = {
    open: open,
    setDate: setDate
  };

  bind();
})();
