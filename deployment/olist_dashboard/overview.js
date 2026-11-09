/* Historical Overview, pinned charts and the Ask UI.
   Reads window.OVERVIEW_DATA; Ask calls POST /api/ask on the local server. */
(function () {
  "use strict";

  const DATA = window.OVERVIEW_DATA;
  const Core = window.MorningCore;
  const Evidence = window.AskEvidence;

  function $(id) {
    return document.getElementById(id);
  }

  function all(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  if (!DATA || !Core || !Array.isArray(DATA.orders)) {
    const host = $("metrics");
    if (host) {
      host.innerHTML = '<p class="empty">Overview data did not load. Run <code>python3 deployment/olist_dashboard/build_overview_data.py</code> and reload.</p>';
    }
    return;
  }

  const META = DATA.meta;
  const STATE_NAMES = META.state_names || {};
  const MODEL = Core.createModel(DATA);
  const MIN_SAMPLE = Core.minSample(MODEL);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const PIN_KEY = "olistOps.pins.v2";
  const MAX_PINS = 6;

  function hasSample(count) {
    return MIN_SAMPLE === null || (typeof count === "number" && count >= MIN_SAMPLE);
  }

  const state = {
    view: "overview",
    asOf: META.default_as_of,
    deliveryDays: 30,
    priceDays: 30,
    hotspotDays: 90,
    pins: [],
    pinContexts: Object.create(null),
    askChats: [],
    askActiveChatId: null,
    askSeedsAdded: [],
    askRenamingId: null
  };
  let pinRenderToken = 0;

  let snapshot = null;
  let resizeTimer = 0;
  let tipsBound = false;

  function fmtInt(value) {
    return Math.round(value).toLocaleString("en-US");
  }

  function fmtPct(value, decimals) {
    if (value === null || value === undefined || !isFinite(value)) return "\u2014";
    return (value * 100).toFixed(decimals == null ? 1 : decimals) + "%";
  }

  function fmtBRLExact(cents) {
    return "R$ " + (cents / 100).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  }

  function fmtCompactBRL(cents) {
    const value = cents / 100;
    if (value >= 1000000) return "R$ " + (value / 1000000).toFixed(2).replace(/\.?0+$/, "") + "m";
    if (value >= 1000) return "R$ " + (value / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return "R$ " + Math.round(value);
  }

  function fmtCompactNumber(value) {
    if (value >= 1000000) return (value / 1000000).toFixed(1).replace(/\.0$/, "") + "m";
    if (value >= 1000) return (value / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(Math.round(value));
  }

  function fmtPrice(cents) {
    return "R$ " + (cents / 100).toFixed(2);
  }

  function fmtTableDate(iso) {
    const date = Core.parseIso(iso);
    return date.getUTCDate() + " " + MONTHS[date.getUTCMonth()] + " " + date.getUTCFullYear();
  }

  function fmtDayMonth(iso) {
    const date = Core.parseIso(iso);
    return date.getUTCDate() + " " + MONTHS[date.getUTCMonth()];
  }

  function fmtRange(startIso, endIso) {
    const start = Core.parseIso(startIso);
    const end = Core.parseIso(endIso);
    if (start.getUTCFullYear() === end.getUTCFullYear() && start.getUTCMonth() === end.getUTCMonth()) {
      return start.getUTCDate() + "\u2013" + end.getUTCDate() + " " + MONTHS[end.getUTCMonth()] + " " + end.getUTCFullYear();
    }
    if (start.getUTCFullYear() === end.getUTCFullYear()) {
      return start.getUTCDate() + " " + MONTHS[start.getUTCMonth()] + "\u2013" + end.getUTCDate() + " " + MONTHS[end.getUTCMonth()] + " " + end.getUTCFullYear();
    }
    return start.getUTCDate() + " " + MONTHS[start.getUTCMonth()] + " " + start.getUTCFullYear() +
      "\u2013" + end.getUTCDate() + " " + MONTHS[end.getUTCMonth()] + " " + end.getUTCFullYear();
  }

  function stateName(code) {
    return STATE_NAMES[code] || code || "Unknown state";
  }

  function scopeLabel(code) {
    return code ? code + " (" + stateName(code) + ")" : "National";
  }

  function signedInt(value) {
    return value > 0 ? "+" + fmtInt(value) : fmtInt(value);
  }

  function signedFixed(value, decimals) {
    return (value > 0 ? "+" : "") + value.toFixed(decimals);
  }

  function fmtDays(value) {
    return value === null || value === undefined || !isFinite(value) ? "\u2014" : value.toFixed(1) + " days";
  }

  function fmtScore(value) {
    return value === null || value === undefined || !isFinite(value) ? "\u2014" : value.toFixed(1);
  }

  function signedScore(value) {
    if (value > 0) return "+" + value.toFixed(1);
    if (value < 0) return "\u2212" + Math.abs(value).toFixed(1);
    return value.toFixed(1);
  }

  function reviewComparison(currentOnTime, previousOnTime, currentLate, previousLate) {
    if ((previousOnTime === null || previousOnTime === undefined) &&
        (previousLate === null || previousLate === undefined)) {
      return "Previous period before available reviews";
    }
    const parts = [];
    if (currentOnTime !== null && currentOnTime !== undefined &&
        previousOnTime !== null && previousOnTime !== undefined) {
      parts.push("On time " + signedScore(currentOnTime - previousOnTime));
    }
    if (currentLate !== null && currentLate !== undefined &&
        previousLate !== null && previousLate !== undefined) {
      parts.push("late " + signedScore(currentLate - previousLate));
    }
    if (!parts.length) return "Previous period before available reviews";
    return parts.join(" \u00b7 ") + " vs prior 30 days";
  }

  function changeClass(current, previous, higherIsBad) {
    if (current === null || current === undefined ||
        previous === null || previous === undefined || current === previous) return "";
    const up = current > previous;
    if (higherIsBad) return up ? "is-bad" : "is-good";
    return up ? "is-good" : "is-bad";
  }

  function averageComparison(current, previous, unit) {
    if (previous === null || previous === undefined) return "Previous period before available data";
    if (current === null || current === undefined) return "No late deliveries in the window";
    return signedFixed(current - previous, 1) + " " + unit + " vs prior 30 days";
  }

  function pctChange(current, previous) {
    if (previous === null || previous === undefined || previous === 0) return null;
    return (current - previous) / previous;
  }

  function countComparison(current, previous) {
    if (previous === null || previous === undefined) return "Previous period before available series";
    if (previous === 0) return "No previous-period base";
    const change = pctChange(current, previous);
    return signedInt(current - previous) + " (" + signedFixed(change * 100, 1) + "%) vs prior 30 days";
  }

  function windowComparison(current, previous) {
    if (previous === null || previous === undefined) return "No data 30 days earlier";
    if (previous === 0) return "No base 30 days earlier";
    const change = pctChange(current, previous);
    return signedInt(current - previous) + " (" + signedFixed(change * 100, 1) + "%) vs 30 days earlier";
  }

  function shareComparison(current, previous) {
    if (previous === null || previous === undefined) return "Previous period before available series";
    if (current === null || current === undefined) return "No current-period deliveries";
    return signedFixed((current - previous) * 100, 1) + " pp vs prior 30 days";
  }

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function prefersReducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function scrollToElement(element, block) {
    if (!element) return;
    element.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: block || "nearest"
    });
  }

  function fmtMetricValue(metric, value, full) {
    if (value === null || value === undefined || !isFinite(value)) return "\u2014";
    if (metric === "sales") return full ? fmtBRLExact(value) : fmtCompactBRL(value);
    if (metric === "onTimeShare" || metric === "lateShare") return fmtPct(value);
    return fmtInt(value);
  }

  function metricLabel(metric) {
    return metric === "sales" ? "Item value" : Core.metricLabel(metric);
  }

  /* ---------- Pins ---------- */

  const PIN_ID_RE = /^[0-9a-f]{32}$/;

  function loadPins() {
    try {
      const raw = window.localStorage.getItem(PIN_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      const seen = Object.create(null);
      const pins = [];
      parsed.forEach(function (item) {
        if (typeof item !== "string" || !PIN_ID_RE.test(item)) return;
        if (seen[item]) return;
        seen[item] = true;
        pins.push(item);
      });
      return pins.slice(0, MAX_PINS);
    } catch (error) {
      return [];
    }
  }

  function savePins() {
    try {
      window.localStorage.setItem(PIN_KEY, JSON.stringify(state.pins));
      return true;
    } catch (error) {
      return false;
    }
  }

  function pinAnswerId(pinId) {
    const context = state.pinContexts[pinId];
    return context && context.answerId ? context.answerId : "";
  }

  function hasPinForAnswer(answerId) {
    return state.pins.some(function (pinId) {
      return pinAnswerId(pinId) === answerId;
    });
  }

  function addPinId(pinId, answerId) {
    if (!PIN_ID_RE.test(String(pinId || ""))) return false;
    state.pinContexts[pinId] = { answerId: answerId || "" };
    if (state.pins.indexOf(pinId) === -1) {
      state.pins.push(pinId);
      if (state.pins.length > MAX_PINS) {
        const dropped = state.pins.slice(0, state.pins.length - MAX_PINS);
        dropped.forEach(function (id) { delete state.pinContexts[id]; });
        state.pins = state.pins.slice(state.pins.length - MAX_PINS);
      }
    }
    const saved = savePins();
    renderPins();
    updatePinButtons();
    return saved;
  }

  function removePinId(pinId) {
    state.pins = state.pins.filter(function (id) { return id !== pinId; });
    delete state.pinContexts[pinId];
    savePins();
    renderPins();
    updatePinButtons();
  }

  function deletePin(pinId) {
    removePinId(pinId);
    fetch("/api/pins/" + encodeURIComponent(pinId), { method: "DELETE" }).catch(function () {});
  }

  function updatePinButtons() {
    all(".pin-btn").forEach(function (button) {
      const answerId = button.getAttribute("data-answer-id") || "";
      const label = button.getAttribute("data-pin-label") || "Pin to Overview";
      const pinned = answerId ? hasPinForAnswer(answerId) : false;
      button.classList.toggle("is-pinned", pinned);
      button.setAttribute("aria-pressed", pinned ? "true" : "false");
      button.textContent = pinned ? "Unpin from Overview" : label;
    });
  }

  /* ---------- Chrome and navigation ---------- */

  function renderChrome() {
    const dateInput = $("asof-date");
    if (dateInput) {
      dateInput.min = META.as_of_min;
      dateInput.max = META.as_of_max;
      dateInput.value = state.asOf;
    }
    const previous = $("prev-day");
    if (previous) previous.disabled = state.asOf <= META.as_of_min;
    const next = $("next-day");
    if (next) next.disabled = state.asOf >= META.as_of_max;
  }

  function setView(name) {
    state.view = name;
    all("[data-view]").forEach(function (button) {
      if (button.getAttribute("data-view") === name) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    ["overview", "actions", "drift", "ask"].forEach(function (view) {
      const host = $("view-" + view);
      if (host) host.hidden = view !== name;
    });
    const dateMessage = $("date-message");
    if (dateMessage) dateMessage.textContent = "";
    if (name === "overview") {
      renderMetrics();
      drawDeliveryChart();
      drawPriceChart();
      renderHotspots();
      renderPins();
    } else if (name === "actions") {
      if (window.PrazoActions) window.PrazoActions.open(state.asOf);
    } else if (name === "drift") {
      if (window.PrazoModels) window.PrazoModels.open();
    } else if (name === "ask") {
      renderAsk();
    }
  }

  function setDate(value, skipAskSync) {
    const message = $("date-message");
    if (!Core.isValidIso(value)) {
      if (message) message.textContent = "That date is not valid. Selectable dates run " +
        fmtTableDate(META.as_of_min) + " to " + fmtTableDate(META.as_of_max) + ".";
      return;
    }
    if (value < META.as_of_min || value > META.as_of_max) {
      if (message) message.textContent = "Selectable dates run " + fmtTableDate(META.as_of_min) +
        " to " + fmtTableDate(META.as_of_max) + ".";
      return;
    }
    if (message) message.textContent = "";
    if (value === state.asOf) return;
    state.asOf = value;
    if (!skipAskSync && AskChats) {
      const change = AskChats.applyDateChange(state.askChats, state.askActiveChatId, value);
      state.askChats = change.chats;
      state.askActiveChatId = change.activeChatId;
      state.askRenamingId = null;
      persistAsk();
      renderAsk();
    }
    renderAll();
  }

  /* ---------- Metrics ---------- */

  function renderMetricCards(host, cards, className) {
    if (!host) return;
    host.innerHTML = cards.map(function (card) {
      return '<article class="' + className + '" title="' + esc(card.title) + '">' +
        '<p class="metric-label">' + esc(card.label) + "</p>" +
        '<p class="metric-value">' + esc(card.value) + "</p>" +
        '<div class="metric-meta">' + card.meta.map(function (line, index) {
          const text = typeof line === "string" ? line : line.text;
          const extra = typeof line === "string" ? "" : (line.className ? " " + line.className : "");
          const cls = index > 0 ? "metric-note" : "metric-change" + extra;
          return '<span class="' + cls + '">' + esc(text) + "</span>";
        }).join("") + "</div>" +
        (card.subline ? '<p class="metric-subline">' + esc(card.subline) + "</p>" : "") +
        "</article>";
    }).join("");
  }

  function renderMetrics() {
    const host = $("metrics");
    const secondaryHost = $("metrics-secondary");
    if (!host) return;
    const window = Core.buildWindow(MODEL, state.asOf, 30, null);
    const totals = window.totals;
    const previous = window.previous;
    const windowText = "Last 30 complete days, " + fmtRange(window.start, window.end) + ".";
    const asOfText = "Only events dated before " + fmtTableDate(state.asOf) + " are counted.";
    const overdue = MODEL.overdueNotDelivered.get(state.asOf) || null;
    const carrier = overdue ? overdue.carrier : 0;
    const notShipped = overdue ? overdue.count - overdue.carrier : 0;

    const enoughDelivered = hasSample(totals.delivered);
    const enoughPriorDelivered = hasSample(previous ? previous.delivered : null);
    const enoughLate = hasSample(totals.late);
    const enoughPriorLate = hasSample(previous ? previous.late : null);
    const enoughOnTimeReviews = hasSample(totals.onTimeReviewCount);
    const enoughPriorOnTimeReviews = hasSample(previous ? previous.onTimeReviewCount : null);
    const enoughLateReviews = hasSample(totals.lateReviewCount);
    const enoughPriorLateReviews = hasSample(previous ? previous.lateReviewCount : null);
    const enoughOrders = hasSample(totals.orders);
    const enoughPriorOrders = hasSample(previous ? previous.orders : null);
    const enoughOverdue = hasSample(overdue ? overdue.count : null);
    const enoughPriorOverdue = hasSample(overdue ? overdue.previous : null);
    const onTimeReviewText = enoughOnTimeReviews ? fmtScore(totals.onTimeReviewAvg) : "\u2014";
    const lateReviewText = enoughLateReviews ? fmtScore(totals.lateReviewAvg) : "\u2014";

    const prior = previous || {};
    const reviewMove = (totals.onTimeReviewAvg !== null && totals.onTimeReviewAvg !== undefined &&
      prior.onTimeReviewAvg !== null && prior.onTimeReviewAvg !== undefined)
      ? totals.onTimeReviewAvg - prior.onTimeReviewAvg
      : ((totals.lateReviewAvg !== null && totals.lateReviewAvg !== undefined &&
        prior.lateReviewAvg !== null && prior.lateReviewAvg !== undefined)
        ? totals.lateReviewAvg - prior.lateReviewAvg : null);
    const reviewClass = reviewMove === null || reviewMove === 0 ? "" : (reviewMove > 0 ? "is-good" : "is-bad");
    const reviewText = (enoughOnTimeReviews && enoughPriorOnTimeReviews)
      ? reviewComparison(totals.onTimeReviewAvg, prior.onTimeReviewAvg,
        totals.lateReviewAvg, prior.lateReviewAvg)
      : ((enoughLateReviews && enoughPriorLateReviews)
        ? reviewComparison(null, null, totals.lateReviewAvg, prior.lateReviewAvg) : "\u2014");

    const cards = [
      {
        label: "Late deliveries",
        value: enoughDelivered ? fmtPct(totals.lateShare) : "\u2014",
        title: windowText + " Share of completed deliveries whose customer delivery date is after the promised date. Late plus on-time is 100%. " + asOfText,
        meta: [
          enoughDelivered && enoughPriorDelivered
            ? { text: shareComparison(totals.lateShare, prior.lateShare),
              className: changeClass(totals.lateShare, prior.lateShare, true) }
            : "\u2014",
          enoughDelivered
            ? fmtInt(totals.late) + " late of " + fmtInt(totals.delivered) + " deliveries"
            : "\u2014"
        ]
      },
      {
        label: "Average days late",
        value: enoughLate ? fmtDays(totals.lateDaysAvg) : "\u2014",
        title: windowText + " Mean calendar days from the promised date to the customer delivery date, over late deliveries only. " + asOfText,
        meta: [
          enoughLate && enoughPriorLate
            ? { text: averageComparison(totals.lateDaysAvg, prior.lateDaysAvg, "d"),
              className: changeClass(totals.lateDaysAvg, prior.lateDaysAvg, true) }
            : "\u2014",
          enoughLate ? "Across " + fmtInt(totals.late) + " late deliveries" : "\u2014"
        ]
      },
      {
        label: "Review score: on time vs late",
        value: onTimeReviewText === "\u2014" && lateReviewText === "\u2014"
          ? "\u2014"
          : onTimeReviewText + " vs " + lateReviewText,
        title: windowText + " Mean review score for on-time and late deliveries. Reviews count only when both their creation date and answer date are before the selected date. " + asOfText,
        meta: [
          { text: reviewText, className: reviewClass },
          fmtInt(totals.onTimeReviewCount) + " on-time \u00b7 " + fmtInt(totals.lateReviewCount) + " late visible reviews"
        ]
      },
      {
        label: "Deliveries",
        value: fmtInt(totals.delivered),
        title: windowText + " Orders with a customer delivery event in the window. " + asOfText,
        meta: [
          enoughDelivered && enoughPriorDelivered
            ? { text: countComparison(totals.delivered, prior.delivered), className: "" }
            : "\u2014"
        ]
      }
    ];

    const secondaryCards = [
      {
        label: "Orders",
        value: fmtInt(totals.orders),
        title: windowText + " Distinct orders with a purchase event in the window. " + asOfText,
        meta: [
          enoughOrders && enoughPriorOrders
            ? { text: countComparison(totals.orders, prior.orders), className: "" }
            : "\u2014"
        ]
      },
      {
        label: "Overdue, not yet delivered",
        value: overdue ? fmtInt(overdue.count) : "\u2014",
        title: windowText + " Orders due in the last 30 days that hadn't reached the customer by the selected date. Cancelled orders are included, because the data doesn't say when they were cancelled. 'With carrier' means handed to the carrier before the selected date; the rest are not yet shipped. The change compares with the same count 30 days earlier. " + asOfText,
        meta: [
          overdue && enoughOverdue && enoughPriorOverdue
            ? { text: windowComparison(overdue.count, overdue.previous),
              className: changeClass(overdue.count, overdue.previous, true) }
            : "\u2014"
        ],
        subline: overdue ? fmtInt(carrier) + " with carrier \u00b7 " + fmtInt(notShipped) + " not yet shipped" : ""
      }
    ];

    renderMetricCards(host, cards, "metric");
    renderMetricCards(secondaryHost, secondaryCards, "metric metric-compact");
  }

  /* ---------- Chart helpers ---------- */

  function labelIndices(length, count) {
    if (length <= count) {
      return Array.from({ length: length }, function (_, index) { return index; });
    }
    const indexes = [];
    for (let index = 0; index < count; index += 1) {
      indexes.push(Math.round((length - 1) * (index / (count - 1))));
    }
    return Array.from(new Set(indexes));
  }

  function chartGeometry(host, height, margins) {
    const width = Math.max(320, host.clientWidth || 720);
    return {
      width: width,
      height: height,
      m: margins,
      iw: width - margins.l - margins.r,
      ih: height - margins.t - margins.b
    };
  }

  function chartTipText(metric, point) {
    const value = Core.metricValue(point, metric);
    if (metric === "sales") {
      return fmtTableDate(point.day) + ": " + fmtBRLExact(value) + " total item value \u00b7 " + fmtInt(point.orders) + " orders";
    }
    if (metric === "onTimeShare") {
      return fmtTableDate(point.day) + ": " + fmtPct(value) + " on time \u00b7 " +
        fmtInt(point.onTime) + " of " + fmtInt(point.delivered) + " deliveries";
    }
    if (metric === "lateShare") {
      return fmtTableDate(point.day) + ": " + fmtPct(value) + " late \u00b7 " +
        fmtInt(point.late) + " of " + fmtInt(point.delivered) + " deliveries";
    }
    if (metric === "late") {
      return fmtTableDate(point.day) + ": " + fmtInt(value) + " late \u00b7 " + fmtInt(point.delivered) + " deliveries";
    }
    if (metric === "delivered") {
      return fmtTableDate(point.day) + ": " + fmtInt(value) + " deliveries \u00b7 " +
        fmtPct(point.delivered ? point.onTime / point.delivered : null) + " on time";
    }
    if (metric === "onTime") {
      return fmtTableDate(point.day) + ": " + fmtInt(value) + " on time of " + fmtInt(point.delivered);
    }
    return fmtTableDate(point.day) + ": " + fmtInt(value) + " orders placed \u00b7 " + fmtBRLExact(point.cents) + " total item value";
  }

  function addHitRects(svg, g, band, points, metric) {
    points.forEach(function (point, index) {
      const tip = chartTipText(metric, point);
      svg.push('<rect class="hit" x="' + (g.m.l + band * index).toFixed(1) + '" y="' + g.m.t +
        '" width="' + band.toFixed(1) + '" height="' + g.ih + '" tabindex="0" role="img" aria-label="' +
        esc(tip) + '" data-tip="' + esc(tip) + '"/>');
    });
    return svg.join("");
  }

  function pointLabel(point) {
    return point && point.label ? point.label : fmtDayMonth(point.day);
  }

  function pointTitle(point) {
    return point && point.label ? point.label : fmtTableDate(point.day);
  }

  function drawDeliveryChart() {
    const host = $("delivery-chart");
    if (!host) return;
    const window = Core.buildChartWindow(MODEL, state.asOf, state.deliveryDays, null);
    const points = window.points;
    const monthly = window.granularity === "month";
    const rates = points.map(function (point) {
      if (point.deliveriesNoData || !point.delivered || !hasSample(point.delivered)) return null;
      return (point.onTime / point.delivered) * 100;
    });
    const valid = rates.filter(function (value) { return value !== null; });
    const rawMin = valid.length ? Math.min.apply(null, valid) : 80;
    const rawMax = valid.length ? Math.max.apply(null, valid) : 100;
    let yMin = Math.max(0, Math.floor((rawMin - 2) / 5) * 5);
    let yMax = Math.min(100, Math.ceil((rawMax + 1) / 5) * 5);
    if (yMax - yMin < 10) {
      yMin = Math.max(0, yMin - 5);
      yMax = Math.min(100, yMax + 5);
    }
    const maxVolume = Math.max.apply(null, points.map(function (point) { return point.delivered; }).concat([1]));
    const rightMax = maxVolume * 1.12;
    const g = chartGeometry(host, host.clientWidth < 520 ? 230 : 290, { l: 46, r: 54, t: 24, b: 32 });
    const band = g.iw / points.length;
    const xAt = function (index) { return g.m.l + band * index + band / 2; };
    const yRate = function (value) { return g.m.t + g.ih - ((value - yMin) / (yMax - yMin)) * g.ih; };
    const yVolume = function (value) { return g.m.t + g.ih - (value / rightMax) * g.ih; };
    const parts = [];
    parts.push('<svg viewBox="0 0 ' + g.width + " " + g.height + '" role="group" aria-labelledby="delivery-svg-title delivery-svg-desc">');
    parts.push('<title id="delivery-svg-title">Deliveries and on-time share</title>');
    parts.push('<desc id="delivery-svg-desc">Deliveries by delivery date and on-time share for the ' +
      (monthly ? points.length + " months" : points.length + " days") +
      " ending " + esc(fmtTableDate(window.end)) + ".</desc>");

    for (let index = 0; index <= 4; index += 1) {
      const fraction = index / 4;
      const y = g.m.t + g.ih - fraction * g.ih;
      parts.push('<line class="grid-line" x1="' + g.m.l + '" y1="' + y.toFixed(1) + '" x2="' +
        (g.width - g.m.r) + '" y2="' + y.toFixed(1) + '"/>');
      parts.push('<text class="axis-text" x="' + (g.m.l - 8) + '" y="' + (y + 4).toFixed(1) +
        '" text-anchor="end">' + (yMin + (yMax - yMin) * fraction).toFixed(0) + "%</text>");
      parts.push('<text class="axis-text" x="' + (g.width - g.m.r + 8) + '" y="' + (y + 4).toFixed(1) +
        '" text-anchor="start">' + fmtCompactNumber(rightMax * fraction) + "</text>");
    }
    parts.push('<text class="axis-title" x="' + g.m.l + '" y="' + (g.m.t - 8) + '">On-time %</text>');
    parts.push('<text class="axis-title" x="' + (g.width - g.m.r) + '" y="' + (g.m.t - 8) + '" text-anchor="end">Delivered</text>');

    points.forEach(function (point, index) {
      if (point.deliveriesNoData) return;
      const barWidth = Math.max(2, band * 0.56);
      const x = g.m.l + band * index + (band - barWidth) / 2;
      const y = yVolume(point.delivered);
      const height = Math.max(1, g.m.t + g.ih - y);
      const last = index === points.length - 1;
      const partial = monthly && point.partial;
      parts.push('<rect class="volume-bar' + (last ? " is-current" : "") +
        (partial ? " is-partial" : "") + '" x="' + x.toFixed(1) +
        '" y="' + y.toFixed(1) + '" width="' + barWidth.toFixed(1) + '" height="' + height.toFixed(1) + '" rx="2"/>');
    });

    let line = "";
    let pen = false;
    rates.forEach(function (rate, index) {
      if (rate === null) {
        pen = false;
        return;
      }
      line += (pen ? "L" : "M") + xAt(index).toFixed(1) + "," + yRate(rate).toFixed(1);
      pen = true;
    });
    if (line) parts.push('<path class="rate-line" d="' + line + '"/>');

    if (points.length <= 31) {
      rates.forEach(function (rate, index) {
        if (rate === null) return;
        const last = index === points.length - 1;
        const partial = monthly && points[index].partial;
        parts.push('<circle class="rate-dot' + (last ? " is-current" : "") +
          (partial ? " is-partial" : "") + '" cx="' + xAt(index).toFixed(1) +
          '" cy="' + yRate(rate).toFixed(1) + '" r="' + (last ? 4 : 2.4) + '"/>');
      });
    } else if (rates[rates.length - 1] !== null) {
      parts.push('<circle class="rate-dot is-current" cx="' + xAt(points.length - 1).toFixed(1) +
        '" cy="' + yRate(rates[rates.length - 1]).toFixed(1) + '" r="4"/>');
    }

    labelIndices(points.length, g.width < 520 ? 4 : 6).forEach(function (index) {
      parts.push('<text class="axis-text" x="' + xAt(index).toFixed(1) + '" y="' + (g.height - 8) +
        '" text-anchor="middle">' + esc(pointLabel(points[index])) + "</text>");
    });

    const hits = points.map(function (point, index) {
      let tip;
      if (point.deliveriesNoData) {
        tip = pointTitle(point) + ": no data";
      } else {
        const rateText = point.delivered && hasSample(point.delivered)
          ? fmtPct(point.onTime / point.delivered) + " on time"
          : "\u2014 on-time rate";
        tip = pointTitle(point) + ": " + fmtInt(point.delivered) + " delivered \u00b7 " +
          rateText +
          " \u00b7 " + fmtInt(point.onTime) + " on time" + (monthly && point.partial ? " (partial period)" : "");
      }
      return '<rect class="hit" x="' + (g.m.l + band * index).toFixed(1) + '" y="' + g.m.t +
        '" width="' + band.toFixed(1) + '" height="' + g.ih + '" tabindex="0" role="img" aria-label="' +
        esc(tip) + '" data-tip="' + esc(tip) + '"/>';
    });
    parts.push(hits.join(""));
    parts.push("</svg>");
    host.innerHTML = parts.join("");

    const note = $("delivery-note");
    if (note) {
      note.textContent = monthly && points.length && points[points.length - 1].partial
        ? points[points.length - 1].label + " is partial because the selected date is " + state.asOf + "."
        : "";
    }
  }

  function drawPriceChart() {
    const host = $("price-chart");
    if (!host) return;
    const window = Core.buildChartWindow(MODEL, state.asOf, state.priceDays, null);
    const points = window.points;
    const monthly = window.granularity === "month";
    const maxCents = Math.max.apply(null, points.map(function (point) { return point.cents; }).concat([1]));
    const maxOrders = Math.max.apply(null, points.map(function (point) { return point.orders; }).concat([1]));
    const g = chartGeometry(host, host.clientWidth < 520 ? 230 : 290, { l: 52, r: 44, t: 24, b: 32 });
    const band = g.iw / points.length;
    const xAt = function (index) { return g.m.l + band * index + band / 2; };
    const yCents = function (value) { return g.m.t + g.ih - (value / (maxCents * 1.12)) * g.ih; };
    const yOrders = function (value) { return g.m.t + g.ih - (value / (maxOrders * 1.12)) * g.ih; };
    const parts = [];
    parts.push('<svg viewBox="0 0 ' + g.width + " " + g.height + '" role="group" aria-labelledby="price-svg-title price-svg-desc">');
    parts.push('<title id="price-svg-title">Item value and orders</title>');
    parts.push('<desc id="price-svg-desc">Total item value and orders placed by purchase date for the ' +
      (monthly ? points.length + " months" : points.length + " days") +
      " ending " + esc(fmtTableDate(window.end)) + ". Freight is excluded and this is not revenue.</desc>");

    for (let index = 0; index <= 2; index += 1) {
      const fraction = index / 2;
      const y = g.m.t + g.ih - fraction * g.ih;
      parts.push('<line class="grid-line" x1="' + g.m.l + '" y1="' + y.toFixed(1) + '" x2="' +
        (g.width - g.m.r) + '" y2="' + y.toFixed(1) + '"/>');
      parts.push('<text class="axis-text" x="' + (g.m.l - 8) + '" y="' + (y + 4).toFixed(1) +
        '" text-anchor="end">' + fmtCompactBRL(maxCents * 1.12 * fraction) + "</text>");
      parts.push('<text class="axis-text" x="' + (g.width - g.m.r + 8) + '" y="' + (y + 4).toFixed(1) +
        '" text-anchor="start">' + fmtCompactNumber(maxOrders * 1.12 * fraction) + "</text>");
    }
    parts.push('<text class="axis-title" x="' + g.m.l + '" y="' + (g.m.t - 8) + '">Item value</text>');
    parts.push('<text class="axis-title" x="' + (g.width - g.m.r) + '" y="' + (g.m.t - 8) + '" text-anchor="end">Orders</text>');

    points.forEach(function (point, index) {
      if (point.ordersNoData) return;
      const barWidth = Math.max(2, band * 0.56);
      const x = g.m.l + band * index + (band - barWidth) / 2;
      const y = yCents(point.cents);
      const height = Math.max(1, g.m.t + g.ih - y);
      const last = index === points.length - 1;
      const partial = monthly && point.partial;
      parts.push('<rect class="sales-bar' + (last ? " is-current" : "") +
        (partial ? " is-partial" : "") + '" x="' + x.toFixed(1) +
        '" y="' + y.toFixed(1) + '" width="' + barWidth.toFixed(1) + '" height="' + height.toFixed(1) + '" rx="2"/>');
    });

    let line = "";
    let pen = false;
    points.forEach(function (point, index) {
      if (point.ordersNoData || !point.orders) {
        pen = false;
        return;
      }
      line += (pen ? "L" : "M") + xAt(index).toFixed(1) + "," + yOrders(point.orders).toFixed(1);
      pen = true;
    });
    if (line) parts.push('<path class="demand-line" d="' + line + '"/>');
    if (points.length <= 31) {
      points.forEach(function (point, index) {
        if (point.ordersNoData || !point.orders) return;
        const last = index === points.length - 1;
        const partial = monthly && point.partial;
        parts.push('<circle class="rate-dot' + (last ? " is-current" : "") +
          (partial ? " is-partial" : "") + '" cx="' + xAt(index).toFixed(1) +
          '" cy="' + yOrders(point.orders).toFixed(1) + '" r="' + (last ? 3.6 : 2.1) + '" style="stroke:#8C5A12"/>');
      });
    }

    labelIndices(points.length, g.width < 520 ? 4 : 5).forEach(function (index) {
      parts.push('<text class="axis-text" x="' + xAt(index).toFixed(1) + '" y="' + (g.height - 8) +
        '" text-anchor="middle">' + esc(pointLabel(points[index])) + "</text>");
    });

    const hits = points.map(function (point, index) {
      const tip = point.ordersNoData
        ? pointTitle(point) + ": no data"
        : pointTitle(point) + ": " + fmtBRLExact(point.cents) + " total item value \u00b7 " +
          fmtInt(point.orders) + " orders placed" + (monthly && point.partial ? " (partial period)" : "");
      return '<rect class="hit" x="' + (g.m.l + band * index).toFixed(1) + '" y="' + g.m.t +
        '" width="' + band.toFixed(1) + '" height="' + g.ih + '" tabindex="0" role="img" aria-label="' +
        esc(tip) + '" data-tip="' + esc(tip) + '"/>';
    });
    parts.push(hits.join(""));
    parts.push("</svg>");
    host.innerHTML = parts.join("");

    const note = $("price-note");
    if (note) {
      note.textContent = monthly && points.length && points[points.length - 1].partial
        ? points[points.length - 1].label + " is partial because the selected date is " + state.asOf + "."
        : "";
    }
  }

  /* ---------- Hotspots ---------- */

  function renderHotspots() {
    const host = $("hotspot-list");
    if (!host) return;
    const window = Core.buildHotspotsWindow(MODEL, state.asOf, state.hotspotDays);
    const overall = $("hotspot-overall");
    if (overall) {
      const enough = hasSample(window.totals.delivered);
      overall.innerHTML = window.totals.delivered
        ? "<strong>" + esc(enough ? fmtPct(window.totals.rate) : "\u2014") + "</strong> late \u00b7 " +
          (enough
            ? esc(fmtInt(window.totals.late)) + " of " + esc(fmtInt(window.totals.delivered)) + " deliveries"
            : esc(fmtInt(window.totals.delivered)) + " deliveries")
        : "No completed deliveries";
    }
    const windowLabel = $("hotspot-window");
    if (windowLabel) windowLabel.textContent = fmtRange(window.start, window.end);

    if (window.rows.length === 0) {
      host.innerHTML = '<p class="empty">No completed deliveries in this window.</p>';
      return;
    }

    const top = window.rows.slice(0, 8);
    const maxLate = Math.max(1, top[0].late);
    let html = '<div class="hotspot-head" aria-hidden="true"><span>Buyer state</span><span>Late deliveries</span><span>Delivered</span><span>Late rate</span></div>' +
      '<ol class="hotspot-list" aria-label="Buyer states ranked by late deliveries">';
    top.forEach(function (row, index) {
      const rate = row.delivered ? row.late / row.delivered : 0;
      const width = Math.max(4, (row.late / maxLate) * 100);
      html += '<li class="hotspot-row">' +
        '<div class="hotspot-state"><span class="hotspot-rank">' + String(index + 1).padStart(2, "0") + "</span>" +
        '<span class="hotspot-code">' + esc(row.state) + '</span><span class="hotspot-name">' +
        esc(stateName(row.state)) + "</span></div>" +
        '<div class="hotspot-bar-wrap"><span class="hotspot-bar" aria-hidden="true"><i style="width:' +
        width.toFixed(1) + '%"></i></span><span class="hotspot-late">' + fmtInt(row.late) + "</span></div>" +
        '<div class="hotspot-denom">' + fmtInt(row.delivered) + " delivered</div>" +
        '<div class="hotspot-rate">' + (hasSample(row.delivered) ? fmtPct(rate) : "\u2014") + "</div></li>";
    });
    html += "</ol>";
    host.innerHTML = html;
  }

  /* ---------- Pinned charts ---------- */

  function pinCardFrame(pinId) {
    const article = document.createElement("article");
    article.className = "panel pinned-card";
    const head = document.createElement("div");
    head.className = "panel-head";
    const copy = document.createElement("div");
    const title = document.createElement("h2");
    title.textContent = "Pinned chart";
    title.setAttribute("data-pin-title", "");
    const kind = document.createElement("p");
    kind.className = "pin-kind";
    kind.setAttribute("data-pin-kind", "");
    const sub = document.createElement("p");
    sub.className = "panel-sub";
    sub.textContent = "Loading\u2026";
    sub.setAttribute("data-pin-evidence", "");
    copy.append(title, kind, sub);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "pin-remove";
    remove.textContent = "Remove";
    remove.setAttribute("data-pin-id", pinId);
    head.append(copy, remove);
    const tools = document.createElement("div");
    tools.className = "pin-windows";
    tools.setAttribute("data-pin-windows", "");
    const chart = document.createElement("div");
    chart.className = "chart pinned-chart";
    chart.style.marginTop = "0";
    const note = document.createElement("p");
    note.className = "pin-note";
    note.setAttribute("data-pin-note", "");
    article.append(head, tools, chart, note);
    return { article: article, title: title, kind: kind, sub: sub, tools: tools, chart: chart, note: note };
  }

  function showPinError(frame, message) {
    frame.title.textContent = "This saved chart could not load";
    frame.kind.textContent = "";
    frame.sub.textContent = "";
    frame.tools.textContent = "";
    chartEmpty(frame.chart, message);
    frame.note.textContent = "Remove this pin and pin the question again.";
  }

  function renderPinWindowButtons(frame, payload, context, pinId) {
    frame.tools.textContent = "";
    const windows = payload.pin.windows || [];
    if (!windows.length) return;
    const group = document.createElement("div");
    group.className = "seg";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Pinned chart window");
    windows.forEach(function (choice) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = (payload.pin.window_labels || {})[choice] || choice;
      button.setAttribute("data-pin-window", choice);
      button.setAttribute("data-pin-window-for", pinId);
      const selected = payload.pin.window === choice;
      button.setAttribute("aria-pressed", selected ? "true" : "false");
      group.append(button);
    });
    frame.tools.append(group);
  }

  function loadPinCard(pinId, context, frame, token) {
    let url = "/api/pins/" + encodeURIComponent(pinId) +
      "?as_of=" + encodeURIComponent(state.asOf);
    if (context.window) url += "&window=" + encodeURIComponent(context.window);
    fetch(url, { headers: { Accept: "application/json" } })
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (payload) {
          return { ok: response.ok, payload: payload };
        });
      })
      .then(function (outcome) {
        if (token !== pinRenderToken) return;
        const payload = outcome.payload;
        if (!payload || !payload.ok) {
          const message = payload && payload.error && payload.error.message
            ? payload.error.message
            : "This saved chart could not be re-run for this date.";
          showPinError(frame, message);
          return;
        }
        if (payload.pin && payload.pin.answer_id) {
          context.answerId = payload.pin.answer_id;
          updatePinButtons();
        }
        frame.title.textContent = payload.pin.title || "Pinned chart";
        frame.kind.textContent = payload.pin.period || payload.pin.label || "";
        frame.sub.textContent = payload.pin.evidence || "";
        renderPinWindowButtons(frame, payload, context, pinId);
        if (payload.available === false) {
          chartEmpty(frame.chart, payload.note || "This period is not available at the selected date.");
          frame.note.textContent = payload.note || "";
          return;
        }
        drawAskChart(frame.chart, payload.pin.chart, payload.table, payload.as_of);
        const notes = [];
        if (payload.partial && payload.note) notes.push(payload.note);
        if (payload.table && payload.table.incomplete_note) notes.push(payload.table.incomplete_note);
        frame.note.textContent = notes.join(" ");
      })
      .catch(function () {
        if (token !== pinRenderToken) return;
        showPinError(frame, "This saved chart could not be loaded.");
      });
  }

  function renderPins() {
    const section = $("pinned-section");
    const grid = $("pinned-grid");
    const count = $("pinned-count");
    if (!section || !grid) return;
    section.hidden = state.pins.length === 0;
    if (count) count.textContent = state.pins.length + (state.pins.length === 1 ? " chart" : " charts");
    grid.textContent = "";
    const token = (pinRenderToken += 1);
    state.pins.forEach(function (pinId) {
      const context = state.pinContexts[pinId] || (state.pinContexts[pinId] = {});
      const frame = pinCardFrame(pinId);
      grid.append(frame.article);
      loadPinCard(pinId, context, frame, token);
    });
  }

  /* ---------- Ask ---------- */

  const AskChats = window.AskChats;
  let askStorageLoadFailed = false;

  function setAskStatus(message, kind) {
    const status = $("ask-status");
    if (!status) return;
    status.textContent = message || "";
    status.className = "ask-status" + (kind ? " is-" + kind : "");
  }

  function safeStorage() {
    try {
      return window.localStorage;
    } catch (error) {
      return null;
    }
  }

  function setAskStorageNote(message) {
    const note = $("ask-storage-note");
    if (note) note.textContent = message || "";
  }

  function askChatById(id) {
    return state.askChats.filter(function (chat) { return chat.id === id; })[0] || null;
  }

  function activeAskChat() {
    return askChatById(state.askActiveChatId);
  }

  function latestAskSql(chat) {
    if (!chat || !Array.isArray(chat.turns)) return "";
    for (let index = chat.turns.length - 1; index >= 0; index -= 1) {
      const payload = chat.turns[index].payload;
      if (payload && payload.ok && payload.answer &&
          payload.answer.kind === "query" && payload.answer.sql) {
        return payload.answer.sql;
      }
    }
    return "";
  }

  function persistAsk() {
    if (!AskChats) return;
    const storage = safeStorage();
    if (!storage) {
      setAskStorageNote("Chats cannot be saved in this browser. This session is still usable.");
      return;
    }
    const result = AskChats.saveStore(storage, state.askChats, state.askActiveChatId, {
      seedsAdded: state.askSeedsAdded
    });
    if (!result.ok) {
      setAskStorageNote("Chats could not be saved. This session is still usable.");
      return;
    }
    if (result.chats) state.askChats = result.chats;
    if (Array.isArray(result.seedsAdded)) state.askSeedsAdded = result.seedsAdded;
    if (result.pruned) {
      setAskStorageNote("Older chats were removed to keep browser storage small.");
      return;
    }
    if (askStorageLoadFailed) {
      setAskStorageNote("Chats could not be loaded. New chats are saved from now on.");
      return;
    }
    setAskStorageNote("");
  }

  function restoreAsk() {
    if (!AskChats) {
      setAskStorageNote("Chat history is unavailable. Questions still work in this session.");
      return;
    }
    const loaded = AskChats.loadStore(safeStorage());
    if (!loaded.ok) askStorageLoadFailed = true;
    state.askChats = loaded.chats || [];
    state.askActiveChatId = loaded.activeChatId || null;
    state.askSeedsAdded = loaded.seedsAdded || [];
    let active = activeAskChat();
    if (!active) {
      const draft = AskChats.emptyChat(state.asOf, AskChats.makeId("chat"), Date.now());
      state.askChats.push(draft);
      state.askActiveChatId = draft.id;
    } else {
      state.asOf = active.date;
    }
    if (askStorageLoadFailed) {
      setAskStorageNote("Chats could not be loaded. New chats are saved from now on.");
    }
    renderChrome();
    renderAsk();
    mergeAskSeeds();
  }

  let askSeedsRequest = null;

  function mergeAskSeeds() {
    if (!AskChats || askSeedsRequest) return askSeedsRequest;
    askSeedsRequest = fetch("data/ask-seed-chats.json", { cache: "no-store" })
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (payload) {
        if (!payload) return;
        const previousSeeds = state.askSeedsAdded.length;
        const merged = AskChats.applySeeds(state.askChats, payload, state.askSeedsAdded);
        state.askChats = merged.chats;
        state.askSeedsAdded = merged.seedsAdded;
        if (!merged.changed && state.askSeedsAdded.length === previousSeeds) return;
        persistAsk();
        renderAsk();
      })
      .catch(function () {
        askSeedsRequest = null;
      });
    return askSeedsRequest;
  }

  function foldAvatar(mode) {
    const template = $("ask-fold-template");
    let avatar = null;
    if (template && template.content && template.content.firstElementChild) {
      avatar = template.content.firstElementChild.cloneNode(true);
    }
    if (!avatar) {
      avatar = document.createElement("span");
      avatar.className = "avatar";
      avatar.setAttribute("aria-hidden", "true");
    }
    if (mode === "thinking") avatar.classList.add("is-thinking");
    else if (mode === "answer") avatar.classList.add("is-answer");
    return avatar;
  }

  function emptyAskState() {
    const wrap = document.createElement("div");
    wrap.className = "ask-empty";
    const content = document.createElement("div");
    content.className = "assistant-content";
    const heading = document.createElement("h2");
    heading.textContent = "Ask about this date";
    const copy = document.createElement("p");
    copy.textContent = "Answers show their evidence.";
    content.append(heading, copy);
    wrap.append(foldAvatar("idle"), content);
    return wrap;
  }

  function userTurn(raw) {
    const turn = document.createElement("div");
    turn.className = "ask-turn user";
    const bubble = document.createElement("p");
    bubble.className = "user-bubble";
    bubble.textContent = raw;
    turn.append(bubble);
    return turn;
  }

  function updateAskStatus(chat) {
    if (!chat) {
      setAskStatus("");
      return;
    }
    const last = chat.turns[chat.turns.length - 1];
    if (!last) {
      setAskStatus("");
      return;
    }
    if (last.status === "loading") {
      setAskStatus("Working on your answer. Usually 10\u201320 seconds.", "loading");
      return;
    }
    const payload = last.payload;
    if (payload && payload.ok && payload.answer && payload.answer.kind === "clarification") {
      setAskStatus("Clarification needed.", "refusal");
      return;
    }
    if (payload && payload.refused) {
      setAskStatus("The data cannot answer that.", "refusal");
      return;
    }
    if (payload && payload.error) {
      setAskStatus(payload.error.message || "Ask could not answer this question.", "error");
      return;
    }
    if (last.status === "interrupted") {
      setAskStatus(last.error || "That answer stopped. Ask again.", "error");
      return;
    }
    setAskStatus("");
  }

  function renderAskHistory() {
    const list = $("ask-chat-list");
    if (!list) return;
    list.textContent = "";
    const saved = state.askChats.filter(function (chat) {
      return chat.turns.length > 0;
    }).sort(function (a, b) {
      return b.updatedAt - a.updatedAt;
    });
    if (!saved.length) {
      const item = document.createElement("li");
      item.className = "ask-chat-empty";
      item.textContent = "Saved chats appear here.";
      list.append(item);
      return;
    }
    saved.forEach(function (chat) {
      const item = document.createElement("li");
      item.className = "ask-chat-item" + (chat.id === state.askActiveChatId ? " is-active" : "");
      if (state.askRenamingId === chat.id) {
        const form = document.createElement("form");
        form.className = "ask-chat-rename";
        form.setAttribute("data-ask-rename-form", chat.id);
        const input = document.createElement("input");
        input.type = "text";
        input.maxLength = 80;
        input.value = chat.title;
        input.setAttribute("data-ask-rename-input", "");
        input.setAttribute("aria-label", "Chat name");
        const save = document.createElement("button");
        save.type = "submit";
        save.textContent = "Save";
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "Cancel";
        cancel.setAttribute("data-ask-rename-cancel", "");
        form.append(input, save, cancel);
        item.append(form);
        list.append(item);
        return;
      }
      const select = document.createElement("button");
      select.type = "button";
      select.className = "ask-chat-select";
      select.setAttribute("data-ask-select", chat.id);
      if (chat.id === state.askActiveChatId) select.setAttribute("aria-current", "true");
      const title = document.createElement("span");
      title.className = "ask-chat-title";
      title.textContent = chat.title;
      const date = document.createElement("span");
      date.className = "ask-chat-date";
      date.textContent = fmtTableDate(chat.date);
      select.append(title, date);
      const actions = document.createElement("div");
      actions.className = "ask-chat-actions";
      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "ask-chat-action";
      rename.textContent = "Rename";
      rename.setAttribute("data-ask-rename", chat.id);
      rename.setAttribute("aria-label", "Rename " + chat.title);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ask-chat-action is-delete";
      remove.textContent = "Delete";
      remove.setAttribute("data-ask-delete", chat.id);
      remove.setAttribute("aria-label", "Delete " + chat.title);
      actions.append(rename, remove);
      item.append(select, actions);
      list.append(item);
    });
  }

  function renderAskThread() {
    const thread = $("ask-thread");
    if (!thread) return;
    thread.textContent = "";
    const chat = activeAskChat();
    if (!chat) return;
    if (!chat.turns.length) {
      thread.append(emptyAskState());
    } else {
      chat.turns.forEach(function (turn) {
        thread.append(userTurn(turn.question));
        thread.append(renderAskTurn(chat, turn));
      });
    }
    updateAskStatus(chat);
  }

  function renderAskTurn(chat, turn) {
    const row = document.createElement("div");
    row.className = "ask-turn assistant";
    const content = document.createElement("div");
    content.className = "assistant-content";
    if (turn.status === "loading") {
      row.classList.add("is-thinking");
      const heading = document.createElement("h2");
      heading.textContent = "Working on your answer\u2026";
      const copy = document.createElement("p");
      copy.textContent = "Usually 10\u201320 seconds.";
      content.append(heading, copy);
      row.append(foldAvatar("thinking"), content);
      return row;
    }
    const answer = document.createElement("div");
    answer.className = "ask-answer";
    const payload = turn.payload;
    if (payload && payload.ok) {
      renderAskAnswer(answer, payload, chat.date);
    } else if (payload && payload.refused) {
      addAskRefusal(answer, payload, chat.date);
    } else if (payload && payload.error) {
      addAskError(answer, payload.error, chat.date);
    } else {
      addAskError(answer, { message: turn.error || "That answer was not saved. Ask again." }, chat.date);
    }
    content.append(answer);
    row.append(foldAvatar("answer"), content);
    return row;
  }

  function renderAsk() {
    renderAskHistory();
    renderAskThread();
    syncAskComposer();
  }

  function syncAskComposer() {
    const input = $("ask-input");
    const submit = $("ask-submit");
    const chat = activeAskChat();
    const pending = !!(chat && chat.turns.some(function (turn) { return turn.status === "loading"; }));
    if (input) {
      input.disabled = pending;
      input.placeholder = pending ? "Waiting for this answer\u2026" : "Ask about the selected date\u2026";
    }
    if (submit) submit.disabled = pending;
  }

  function selectAskChat(id) {
    const chat = askChatById(id);
    if (!chat) return;
    state.askActiveChatId = id;
    state.askRenamingId = null;
    if (chat.date !== state.asOf) setDate(chat.date, true);
    else renderChrome();
    renderAsk();
    persistAsk();
    const input = $("ask-input");
    if (input && !input.disabled) input.focus();
  }

  function startNewAskChat() {
    if (!AskChats) return;
    const active = activeAskChat();
    state.askRenamingId = null;
    if (!active || active.turns.length || active.date !== state.asOf) {
      const draft = AskChats.emptyChat(state.asOf, AskChats.makeId("chat"), Date.now());
      state.askChats.push(draft);
      state.askActiveChatId = draft.id;
    }
    persistAsk();
    renderAsk();
    const input = $("ask-input");
    if (input) input.focus();
  }

  function deleteAskChat(id) {
    const chat = askChatById(id);
    if (!chat) return;
    if (window.confirm && !window.confirm('Delete "' + chat.title + '"? This cannot be undone.')) return;
    state.askChats = AskChats.removeChat(state.askChats, id);
    state.askRenamingId = null;
    if (state.askActiveChatId === id) {
      const draft = AskChats.emptyChat(state.asOf, AskChats.makeId("chat"), Date.now());
      state.askChats.push(draft);
      state.askActiveChatId = draft.id;
    }
    persistAsk();
    renderAsk();
  }

  function answerHeader(host, title, tools) {
    const head = document.createElement("div");
    head.className = "ask-answer-head";
    const copy = document.createElement("div");
    const heading = document.createElement("h3");
    heading.textContent = title;
    copy.append(heading);
    head.append(copy);
    if (tools) head.append(tools);
    host.append(head);
    return copy;
  }

  function addText(host, text) {
    const paragraph = document.createElement("p");
    paragraph.className = "answer-text";
    paragraph.textContent = text;
    host.append(paragraph);
    return paragraph;
  }

  function addAskEvidence(host, payload, fallback) {
    if (!Evidence) return;
    host.append(Evidence.render(document, Evidence.build(payload, fallback)));
  }

  function addAskError(answer, error, dateIso) {
    const detail = error && typeof error === "object" ? error : {};
    const message = detail.message || "Ask could not answer this question.";
    const evidenceError = { message: message, sql: detail.sql || null };
    if (detail.stage) evidenceError.stage = detail.stage;
    if (detail.finish_reason) evidenceError.finish_reason = detail.finish_reason;
    answer.textContent = "";
    answer.classList.add("ask-unsupported");
    answerHeader(answer, "No answer");
    addText(answer, message);
    addAskEvidence(answer, {
      ok: false,
      as_of: dateIso || state.asOf,
      error: evidenceError
    }, { date: dateIso, asOf: state.asOf });
  }

  function addAskRefusal(answer, payload, dateIso) {
    const refusal = payload.refusal || {};
    answer.textContent = "";
    answer.classList.add("ask-unsupported");
    answerHeader(answer, refusal.title || "No grounded answer");
    addText(answer, refusal.message || "The available data cannot support this question.");
    addAskEvidence(answer, payload, { date: dateIso, asOf: state.asOf });
  }

  function fmtAskCell(value, column, unit, decimals) {
    if (value === null || value === undefined) return "\u2014";
    if (typeof value === "number") {
      if (Core.moneyUnit(column)) return Core.formatAskMoney(value, false);
      if (unit) {
        const rate = Core.formatRateValue(value, unit, decimals);
        if (rate !== null) return rate;
      }
      const formatted = Core.formatAskNumber(value);
      return formatted === null ? "\u2014" : formatted;
    }
    return String(value);
  }

  function numericValue(value) {
    if (typeof value === "number" && isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
      const cleaned = value.replace(/[^0-9.+-]/g, "");
      const parsed = Number(cleaned);
      return isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  function truncateLabel(value, max) {
    const text = value === null || value === undefined ? "\u2014" : String(value);
    return text.length > max ? text.slice(0, max - 1) + "\u2026" : text;
  }

  function renderAskResultTable(answer, table, rowCount, labels) {
    if (!table || !Array.isArray(table.rows) || !Array.isArray(table.columns)) return;
    const wrap = document.createElement("div");
    wrap.className = "ask-table-wrap";
    const element = document.createElement("table");
    element.className = "ask-table";
    const caption = document.createElement("caption");
    caption.className = "sr-only";
    caption.textContent = table.columns.map(function (column) {
      return Core.columnLabel(column, labels);
    }).join(", ");
    element.append(caption);
    const numeric = {};
    const units = {};
    const rateDecimals = {};
    table.columns.forEach(function (column) {
      numeric[column] = table.rows.some(function (row) {
        return typeof row[column] === "number";
      });
      const values = table.rows.map(function (row) { return row[column]; });
      if (numeric[column]) {
        units[column] = Core.rateUnit(column, values);
        if (units[column]) rateDecimals[column] = Core.rateLabelDecimals(values, units[column]);
      }
    });
    const incompleteRows = Array.isArray(table.incomplete_rows)
      ? table.incomplete_rows
      : [];
    const incomplete = Object.create(null);
    incompleteRows.forEach(function (index) { incomplete[index] = true; });
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    table.columns.forEach(function (heading) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = Core.columnLabel(heading, labels);
      if (numeric[heading]) th.className = "num";
      headRow.append(th);
    });
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    table.rows.forEach(function (row, rowIndex) {
      const tr = document.createElement("tr");
      if (incomplete[rowIndex]) tr.className = "is-incomplete";
      table.columns.forEach(function (column, columnIndex) {
        const cell = document.createElement("td");
        if (numeric[column]) cell.className = "num";
        let text = fmtAskCell(Core.displayValue(column, row[column]), column, units[column], rateDecimals[column]);
        if (incomplete[rowIndex] && columnIndex === 0) text += " (incomplete)";
        cell.textContent = text;
        tr.append(cell);
      });
      tbody.append(tr);
    });
    element.append(thead, tbody);
    wrap.append(element);
    const shown = table.rows.length;
    const storedTotal = typeof table.total_rows === "number" && isFinite(table.total_rows) ? table.total_rows : null;
    const answerTotal = typeof rowCount === "number" && isFinite(rowCount) ? rowCount : null;
    const total = Math.max(shown, storedTotal || 0, answerTotal || 0);
    if (total > shown) {
      const note = document.createElement("p");
      note.className = "ask-table-note";
      note.textContent = "Showing the first " + fmtInt(shown) + " of " + fmtInt(total) + " rows.";
      wrap.append(note);
    }
    if (Array.isArray(table.overall_totals) && table.overall_totals.length) {
      table.overall_totals.forEach(function (total) {
        const note = document.createElement("p");
        note.className = "ask-table-note";
        const values = [total.value];
        const unit = Core.rateUnit(total.column, values);
        const decimals = unit ? Core.rateLabelDecimals(values, unit) : undefined;
        note.textContent = "Overall " + Core.columnLabel(total.column, labels) + ": " +
          fmtAskCell(total.value, total.column, unit, decimals) + ".";
        wrap.append(note);
      });
    }
    if (table.incomplete_note) {
      const note = document.createElement("p");
      note.className = "ask-table-note";
      note.textContent = table.incomplete_note;
      wrap.append(note);
    }
    answer.append(wrap);
  }

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgElement(name, attributes) {
    const element = document.createElementNS(SVG_NS, name);
    if (attributes) {
      Object.keys(attributes).forEach(function (key) {
        element.setAttribute(key, attributes[key]);
      });
    }
    return element;
  }

  function svgText(value, attributes) {
    const text = svgElement("text", attributes);
    text.textContent = value;
    return text;
  }

  function chartEmpty(host, message) {
    host.textContent = "";
    const empty = document.createElement("p");
    empty.className = "chart-empty";
    empty.textContent = message;
    host.append(empty);
  }

  function drawAskChart(host, spec, table, asOf) {
    if (!host || !spec || !table || !Array.isArray(table.rows)) return;
    const columns = table.columns || [];
    if (columns.indexOf(spec.x) === -1 || columns.indexOf(spec.y) === -1) {
      chartEmpty(host, "Chart columns are not in the result.");
      return;
    }
    const points = table.rows.map(function (row) {
      return { x: row[spec.x], y: numericValue(row[spec.y]) };
    }).filter(function (point) { return point.y !== null; });
    if (!points.length) {
      chartEmpty(host, "No numeric values to chart.");
      return;
    }
    host.textContent = "";
    const width = Math.max(320, host.clientWidth || 720);
    const values = points.map(function (point) { return point.y; });
    const rawMin = Math.min.apply(null, values);
    const rawMax = Math.max.apply(null, values);
    const scale = Core.askChartScale(values, spec.y);
    const partialIndex = spec.type === "line" ? Core.partialBucketIndex(points, asOf) : -1;
    const partialNote = partialIndex >= 0 ? Core.partialBucketNote(points, partialIndex, asOf) : "";
    const incompleteLabels = Object.create(null);
    (Array.isArray(spec.incomplete) ? spec.incomplete : []).forEach(function (value) {
      incompleteLabels[String(value)] = true;
    });
    const isPartialPoint = function (point, index) {
      return index === partialIndex || incompleteLabels[String(point.x)] === true;
    };
    const rateDecimals = scale && scale.kind === "rate"
      ? Core.rateLabelDecimals(values, scale.unit) : undefined;
    const pointLabel = function (value) {
      if (scale && scale.kind === "rate") {
        return Core.formatRateValue(value, scale.unit, rateDecimals);
      }
      if (scale && scale.kind === "money") return Core.formatAskMoney(value, true);
      return fmtAskCell(value);
    };
    const svg = svgElement("svg", {
      viewBox: "0 0 " + width + " " + (spec.type === "line" ? 260 : Math.max(180, points.length * 28 + 42)),
      role: "group",
      "aria-label": Core.columnLabel(spec.y, spec.labels) + " by " + Core.columnLabel(spec.x, spec.labels)
    });

    if (spec.type === "line") {
      const height = 260;
      const m = { l: 58, r: 20, t: 24, b: 46 };
      const iw = width - m.l - m.r;
      const ih = height - m.t - m.b;
      const min = scale.min;
      const max = scale.max;
      const xScale = Core.chartXScale(points);
      const xAt = function (index) {
        return xScale.at(index, m.l, iw);
      };
      const yAt = function (value) {
        return m.t + ih - ((value - min) / (max - min)) * ih;
      };
      scale.ticks.forEach(function (tick) {
        const y = yAt(tick.value);
        svg.append(svgElement("line", {
          class: "grid-line", x1: m.l, y1: y.toFixed(1),
          x2: width - m.r, y2: y.toFixed(1)
        }));
        svg.append(svgText(tick.label, {
          class: "axis-text", x: m.l - 8, y: (y + 4).toFixed(1), "text-anchor": "end"
        }));
      });
      if (scale.axisTitle) {
        svg.append(svgText(scale.axisTitle, {
          class: "axis-title", x: m.l, y: m.t - 8
        }));
      }
      let line = "";
      points.forEach(function (point, index) {
        line += (index ? "L" : "M") + xAt(index).toFixed(1) + "," + yAt(point.y).toFixed(1);
      });
      svg.append(svgElement("path", { class: "rate-line", d: line }));
      points.forEach(function (point, index) {
        const partial = isPartialPoint(point, index);
        const label = fmtAskCell(Core.displayValue(spec.x, point.x)) + ": " + pointLabel(point.y) +
          (partial ? " (partial period)" : "");
        svg.append(svgElement("circle", {
          class: "rate-dot" + (partial ? " is-partial" : ""),
          cx: xAt(index).toFixed(1), cy: yAt(point.y).toFixed(1),
          r: points.length > 60 ? 1.6 : 3,
          "data-tip": label,
          "aria-label": label,
          role: "img",
          tabindex: "0"
        }));
      });
      labelIndices(points.length, width < 520 ? 4 : 7).forEach(function (index) {
        svg.append(svgText(truncateLabel(Core.displayValue(spec.x, points[index].x), 16), {
          class: "axis-text", x: xAt(index).toFixed(1), y: height - 10, "text-anchor": "middle"
        }));
      });
    } else {
      const rowHeight = 28;
      const height = Math.max(180, points.length * rowHeight + 42);
      const labelWidth = Math.min(190, Math.max(90, Math.round(width * 0.28)));
      const m = { l: labelWidth, r: 62, t: 18, b: 24 };
      const iw = width - m.l - m.r;
      let min = Math.min(0, rawMin);
      let max = Math.max(0, rawMax);
      if (min === max) max = min + 1;
      const xAt = function (value) {
        return m.l + ((value - min) / (max - min)) * iw;
      };
      const zeroX = xAt(0);
      svg.append(svgElement("line", {
        class: "grid-line", x1: zeroX.toFixed(1), y1: m.t,
        x2: zeroX.toFixed(1), y2: height - m.b
      }));
      points.forEach(function (point, index) {
        const y = m.t + index * rowHeight + 5;
        const barHeight = rowHeight - 10;
        const end = xAt(point.y);
        const x = Math.min(zeroX, end);
        const barWidth = Math.max(1, Math.abs(end - zeroX));
        const partial = isPartialPoint(point, index);
        svg.append(svgElement("rect", {
          class: "volume-bar" + (partial ? " is-partial" : ""), x: x.toFixed(1), y: y.toFixed(1),
          width: barWidth.toFixed(1), height: barHeight.toFixed(1), rx: "2",
          "data-tip": fmtAskCell(Core.displayValue(spec.x, point.x)) + ": " + pointLabel(point.y) +
            (partial ? " (partial period)" : ""), tabindex: "0"
        }));
        svg.append(svgText(truncateLabel(Core.displayValue(spec.x, point.x), 24), {
          class: "axis-text", x: m.l - 8, y: (y + barHeight / 2 + 4).toFixed(1), "text-anchor": "end"
        }));
        svg.append(svgText(pointLabel(point.y), {
          class: "axis-text", x: (Math.max(end, zeroX) + 7).toFixed(1), y: (y + barHeight / 2 + 4).toFixed(1)
        }));
      });
    }
    host.append(svg);
    const notes = [];
    if (scale.note) notes.push(scale.note);
    if (partialNote) notes.push(partialNote);
    if (notes.length) {
      const note = document.createElement("p");
      note.className = "ask-chart-note";
      note.textContent = notes.join(" ");
      host.append(note);
    }
  }

  function makeAskPinButton(answerId, pin) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn pin-btn";
    button.setAttribute("data-answer-id", answerId);
    let label = pin.button_label;
    if (!label) {
      const fixed = String(pin.label || "").replace(/^Fixed:\s*/, "") || "period";
      if (pin.kind === "rolling") {
        label = "Pin to Overview \u00b7 moves with the date (" + (pin.label || "window") + ")";
      } else if (pin.kind === "fixed") {
        label = "Pin to Overview \u00b7 fixed to " + fixed;
      } else {
        const all = String(pin.label || "All history");
        label = "Pin to Overview \u00b7 " + all.charAt(0).toLowerCase() + all.slice(1);
      }
    }
    button.setAttribute("data-pin-label", label);
    button.textContent = hasPinForAnswer(answerId) ? "Unpin from Overview" : label;
    button.addEventListener("click", function () {
      if (hasPinForAnswer(answerId)) {
        const existing = state.pins.filter(function (id) {
          return pinAnswerId(id) === answerId;
        })[0];
        if (existing) deletePin(existing);
        return;
      }
      button.disabled = true;
      fetch("/api/pins", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer_id: answerId })
      })
        .then(function (response) {
          return response.json().catch(function () { return null; }).then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (outcome) {
          button.disabled = false;
          if (!outcome.payload || !outcome.payload.ok) {
            const message = outcome.payload && outcome.payload.error
              ? outcome.payload.error.message
              : "This chart could not be pinned.";
            setAskStatus(message, "error");
            return;
          }
          addPinId(outcome.payload.pin.id, answerId);
          setAskStatus("Pinned to Overview.", "");
        })
        .catch(function () {
          button.disabled = false;
          setAskStatus("Could not reach the local server to pin this chart.", "error");
        });
    });
    return button;
  }

  function renderAskAnswer(answer, payload, dateIso) {
    const data = payload.answer || {};
    answer.textContent = "";
    answer.classList.remove("ask-unsupported");
    let tools = null;
    if (data.pin && data.chart && data.answer_id) {
      tools = document.createElement("div");
      tools.className = "ask-answer-tools";
      tools.append(makeAskPinButton(data.answer_id, data.pin));
    }
    answerHeader(answer, data.title || "Answer", tools);
    addText(answer, data.text || "");
    if (data.kind === "query") {
      const labels = data.chart && data.chart.labels ? data.chart.labels : null;
      if (data.chart) {
        const chartHost = document.createElement("div");
        chartHost.className = "chart ask-chart";
        answer.append(chartHost);
        drawAskChart(chartHost, data.chart, data.table, payload.as_of || dateIso || state.asOf);
      }
      if (data.table) renderAskResultTable(answer, data.table, data.row_count, labels);
      addAskEvidence(answer, payload, { date: dateIso, asOf: state.asOf });
    } else {
      addAskEvidence(answer, payload, { date: dateIso, asOf: state.asOf });
    }
  }

  async function askQuestion(raw) {
    const text = String(raw || "").trim();
    const chat = activeAskChat();
    if (!AskChats || !chat || !text) return false;
    if (chat.turns.some(function (turn) { return turn.status === "loading"; })) return false;
    const targetChatId = chat.id;
    const targetDate = chat.date;
    const history = AskChats.cleanHistory(chat.history);
    const priorSql = latestAskSql(chat);
    const turn = AskChats.addTurn(chat, text, AskChats.makeId("turn"), Date.now());
    if (!turn) return false;
    persistAsk();
    renderAsk();
    scrollToElement($("ask-thread") ? $("ask-thread").lastElementChild : null, "nearest");
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          as_of: targetDate,
          history: history,
          prior_sql: priorSql || null
        })
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }
      if (!payload) {
        throw new Error("The Ask server returned an unreadable response.");
      }
      const target = askChatById(targetChatId);
      if (target) {
        AskChats.commitTurn(target, turn.id, payload,
          AskChats.nextHistory(history, text, payload), Date.now());
      }
    } catch (error) {
      const message = "Ask could not reach the local server. Start it with python3 deployment/olist_dashboard/server.py and try again.";
      const target = askChatById(targetChatId);
      if (target) {
        const failure = { error: { message: message } };
        AskChats.commitTurn(target, turn.id, failure,
          AskChats.nextHistory(history, text, failure), Date.now());
      }
    } finally {
      persistAsk();
      if (state.askActiveChatId === targetChatId) {
        renderAsk();
        const input = $("ask-input");
        if (input && !input.disabled) input.focus();
        scrollToElement($("ask-thread") ? $("ask-thread").lastElementChild : null, "nearest");
      } else {
        renderAskHistory();
        updateAskStatus(activeAskChat());
      }
    }
    return true;
  }

  /* ---------- Tooltips ---------- */

  function setupTips() {
    if (tipsBound) return;
    const tip = $("chart-tip");
    if (!tip) return;
    tipsBound = true;

    function hide() {
      tip.style.display = "none";
    }

    function show(target) {
      const text = target.getAttribute("data-tip");
      if (!text) return;
      tip.textContent = text;
      tip.style.display = "block";
      const rect = target.getBoundingClientRect();
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      let left = rect.left + rect.width / 2 - width / 2;
      let top = rect.bottom + 10;
      if (top + height > window.innerHeight - 8) top = rect.top - height - 10;
      left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
      top = Math.max(8, Math.min(top, window.innerHeight - height - 8));
      tip.style.left = left + "px";
      tip.style.top = top + "px";
    }

    document.addEventListener("focusin", function (event) {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target) show(target);
    });
    document.addEventListener("focusout", hide);
    document.addEventListener("pointerover", function (event) {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target) show(target);
    });
    document.addEventListener("pointerout", function (event) {
      const target = event.target.closest ? event.target.closest("[data-tip]") : null;
      if (target) hide();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") hide();
    });
    window.addEventListener("scroll", hide, { passive: true });
  }

  /* ---------- Wiring ---------- */

  function redrawAllCharts() {
    drawDeliveryChart();
    drawPriceChart();
    renderPins();
  }

  function renderAll() {
    snapshot = Core.compute(MODEL, state.asOf, 30);
    renderChrome();
    renderMetrics();
    drawDeliveryChart();
    drawPriceChart();
    renderHotspots();
    renderPins();
    if (window.PrazoActions) window.PrazoActions.setDate(state.asOf);
  }

  function setupEvents() {
    const dateInput = $("asof-date");
    if (dateInput) {
      dateInput.addEventListener("change", function () {
        setDate(dateInput.value);
        renderChrome();
      });
      // Open the calendar on any click, not only on the small icon.
      dateInput.addEventListener("click", function () {
        if (typeof dateInput.showPicker === "function") {
          try { dateInput.showPicker(); } catch (error) { /* picker already open or not allowed */ }
        }
      });
    }
    const previous = $("prev-day");
    if (previous) {
      previous.addEventListener("click", function () {
        setDate(Core.addDaysIso(state.asOf, -1));
      });
    }
    const next = $("next-day");
    if (next) {
      next.addEventListener("click", function () {
        setDate(Core.addDaysIso(state.asOf, 1));
      });
    }

    all("[data-view]").forEach(function (button) {
      button.addEventListener("click", function () {
        setView(button.getAttribute("data-view"));
      });
    });
    const brand = document.querySelector("[data-nav='overview']");
    if (brand) {
      brand.addEventListener("click", function (event) {
        event.preventDefault();
        setView("overview");
      });
    }

    all("[data-chart-window]").forEach(function (button) {
      button.addEventListener("click", function () {
        const target = button.getAttribute("data-chart-window");
        const raw = button.getAttribute("data-days");
        const days = raw === "12mo" ? "12mo" : Number(raw);
        if (target === "delivery") {
          state.deliveryDays = days;
          all("[data-chart-window='delivery']").forEach(function (sibling) {
            sibling.setAttribute("aria-pressed", sibling === button ? "true" : "false");
          });
          drawDeliveryChart();
        } else if (target === "price") {
          state.priceDays = days;
          all("[data-chart-window='price']").forEach(function (sibling) {
            sibling.setAttribute("aria-pressed", sibling === button ? "true" : "false");
          });
          drawPriceChart();
        }
      });
    });

    all("[data-hotspot-window]").forEach(function (button) {
      button.addEventListener("click", function () {
        const raw = button.getAttribute("data-days");
        state.hotspotDays = raw === "12mo" ? "12mo" : Number(raw);
        all("[data-hotspot-window]").forEach(function (sibling) {
          sibling.setAttribute("aria-pressed", sibling === button ? "true" : "false");
        });
        renderHotspots();
      });
    });

    const askForm = $("ask-form");
    if (askForm) {
      askForm.addEventListener("submit", function (event) {
        event.preventDefault();
        const input = $("ask-input");
        if (!input) return;
        const value = input.value;
        askQuestion(value).then(function (started) {
          if (started && input.value === value) input.value = "";
        });
      });
    }
    const newChat = $("ask-new-chat");
    if (newChat) {
      newChat.addEventListener("click", function () {
        startNewAskChat();
      });
    }
    const chatList = $("ask-chat-list");
    if (chatList) {
      chatList.addEventListener("click", function (event) {
        if (!event.target.closest) return;
        const select = event.target.closest("[data-ask-select]");
        if (select) {
          selectAskChat(select.getAttribute("data-ask-select"));
          return;
        }
        const rename = event.target.closest("[data-ask-rename]");
        if (rename) {
          state.askRenamingId = rename.getAttribute("data-ask-rename");
          renderAskHistory();
          const input = chatList.querySelector("[data-ask-rename-input]");
          if (input) {
            input.focus();
            input.select();
          }
          return;
        }
        const remove = event.target.closest("[data-ask-delete]");
        if (remove) {
          deleteAskChat(remove.getAttribute("data-ask-delete"));
          return;
        }
        if (event.target.closest("[data-ask-rename-cancel]")) {
          state.askRenamingId = null;
          renderAskHistory();
        }
      });
      chatList.addEventListener("submit", function (event) {
        const form = event.target.closest ? event.target.closest("[data-ask-rename-form]") : null;
        if (!form) return;
        event.preventDefault();
        const input = form.querySelector("[data-ask-rename-input]");
        if (input) {
          AskChats.renameChat(state.askChats, form.getAttribute("data-ask-rename-form"), input.value);
        }
        state.askRenamingId = null;
        persistAsk();
        renderAsk();
      });
      chatList.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && state.askRenamingId) {
          state.askRenamingId = null;
          renderAskHistory();
        }
      });
    }

    const pinnedGrid = $("pinned-grid");
    if (pinnedGrid) {
      pinnedGrid.addEventListener("click", function (event) {
        if (!event.target.closest) return;
        const remove = event.target.closest("[data-pin-id]");
        if (remove) {
          deletePin(remove.getAttribute("data-pin-id"));
          return;
        }
        const windowButton = event.target.closest("[data-pin-window]");
        if (windowButton) {
          const pinId = windowButton.getAttribute("data-pin-window-for");
          const context = state.pinContexts[pinId] || (state.pinContexts[pinId] = {});
          context.window = windowButton.getAttribute("data-pin-window");
          renderPins();
        }
      });
    }

    window.addEventListener("resize", function () {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(redrawAllCharts, 120);
    });
  }

  setupTips();
  setupEvents();
  state.pins = loadPins();
  restoreAsk();
  renderAll();
  setView("overview");
})();
