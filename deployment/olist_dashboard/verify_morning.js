#!/usr/bin/env node
/* Focused checks for the as-of queue, windows, hotspots, charts and pins. */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const Core = require("./morning-core.js");

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "overview.json"), "utf8"));
const model = Core.createModel(data);

function expectedOutcomes(asOf, days) {
  const end = Core.addDaysIso(asOf, -1);
  const start = Core.addDaysIso(end, -(days - 1));
  const result = {
    delivered: 0,
    onTime: 0,
    late: 0,
    lateDaysSum: 0,
    onTimeReviewCount: 0,
    onTimeReviewScoreSum: 0,
    lateReviewCount: 0,
    lateReviewScoreSum: 0
  };
  data.daily_deliveries.forEach(function (row) {
    if (row[0] < start || row[0] > end) return;
    result.delivered += row[1];
    result.onTime += row[2];
    result.lateDaysSum += row[3];
  });
  result.late = result.delivered - result.onTime;
  data.daily_reviews.forEach(function (row) {
    if (row[0] < start || row[0] > end || row[1] >= asOf) return;
    if (row[2] === 1) {
      result.lateReviewCount += row[3];
      result.lateReviewScoreSum += row[4];
    } else {
      result.onTimeReviewCount += row[3];
      result.onTimeReviewScoreSum += row[4];
    }
  });
  return result;
}

function assertOutcomeTotals(totals, expected, label) {
  ["delivered", "onTime", "late", "lateDaysSum", "onTimeReviewCount",
    "onTimeReviewScoreSum", "lateReviewCount", "lateReviewScoreSum"].forEach(function (key) {
    assert.strictEqual(totals[key], expected[key], label + " " + key);
  });
  assert.strictEqual(totals.late + totals.onTime, totals.delivered,
    label + " late plus on-time must equal deliveries");
  if (totals.delivered) {
    assert.ok(Math.abs(totals.lateShare + totals.onTimeShare - 1) < 1e-12,
      label + " late share and on-time share must total 100%");
  }
  if (totals.late) {
    assert.ok(totals.lateDaysAvg > 0, label + " average days late must be positive");
  }
  if (totals.onTimeReviewCount) {
    assert.ok(Math.abs(totals.onTimeReviewAvg -
      totals.onTimeReviewScoreSum / totals.onTimeReviewCount) < 1e-12,
      label + " on-time review average must use visible reviews only");
  }
  if (totals.lateReviewCount) {
    assert.ok(Math.abs(totals.lateReviewAvg -
      totals.lateReviewScoreSum / totals.lateReviewCount) < 1e-12,
      label + " late review average must use visible reviews only");
  }
}

const synthetic = {
  meta: { state_names: { SP: "Sao Paulo", RJ: "Rio de Janeiro" } },
  daily_purchases: [],
  daily_deliveries: [],
  state_deliveries: {},
  orders: [
    ["o1-closed", "SP", "2018-08-01", "2018-08-10", "2018-08-21", "2018-08-15", 100],
    ["o2-delivered-on-day", "SP", "2018-08-01", "2018-08-10", "2018-08-22", "2018-08-15", 200],
    ["o3-delivered-later", "RJ", "2018-08-01", "2018-08-20", "2018-08-25", "2018-08-15", 300],
    ["o4-purchased-on-day", "SP", "2018-08-22", null, "2018-08-30", "2018-08-20", 400],
    ["o5-due-soon", "RJ", "2018-08-10", "2018-08-21", null, "2018-08-25", 500],
    ["o6-no-handover", "SP", "2018-08-05", null, null, "2018-08-19", 600],
    ["o7-older", "RJ", "2018-06-01", "2018-07-20", null, "2018-08-01", 700],
    ["o8-no-promise", "SP", "2018-08-10", null, null, null, 800]
  ]
};

const small = Core.compute(Core.createModel(synthetic), "2018-08-22", 30);
assert.strictEqual(small.counts.open, 6, "cutoff must exclude delivered-before and purchased-on-day orders");
assert.strictEqual(small.counts.overdue, 4, "overdue count");
assert.strictEqual(small.counts.dueSoon, 1, "due-soon count");
assert.strictEqual(small.counts.overdueHandover, 3, "handover split");
assert.strictEqual(small.counts.overdueNoHandover, 1, "no-handover split");
assert.strictEqual(small.counts.recentOverdue, 3, "recent overdue count");
assert.strictEqual(small.counts.recentOverdueHandover, 2, "recent handover split");
assert.strictEqual(small.counts.olderOverdue, 1, "older overdue count");
assert.strictEqual(small.counts.olderOverdueHandover, 1, "older handover split");
assert.strictEqual(small.counts.noPromise, 1, "open orders without a promise");
assert.deepStrictEqual(
  small.queue.map(function (item) { return item.orderId; }),
  ["o5-due-soon", "o6-no-handover", "o2-delivered-on-day", "o3-delivered-later", "o7-older"],
  "queue order must put due soon first, then nearest recent miss, then older records"
);
assert.deepStrictEqual(
  small.priorityQueue.map(function (item) { return item.orderId; }),
  ["o5-due-soon", "o6-no-handover", "o2-delivered-on-day", "o3-delivered-later"],
  "near-term queue must exclude older than seven days"
);
assert.strictEqual(small.openByState.SP, 3, "SP open count");
assert.strictEqual(small.overdueByState.RJ, 2, "RJ overdue count");
assert.strictEqual(small.noPromiseByState.SP, 1, "SP no-promise count");
assert.strictEqual(small.oldestOverdue.orderId, "o7-older", "oldest overdue includes older records");

const asOf = "2018-08-22";
const current = Core.compute(model, asOf, 30);
const previous = Core.compute(model, Core.addDaysIso(asOf, -1), 30);
const newDay = Core.addDaysIso(asOf, -1);
const purchases = (model.purchases.get(newDay) || { orders: 0 }).orders;
const deliveries = (model.deliveries.get(newDay) || { delivered: 0 }).delivered;
assert.strictEqual(
  current.counts.open,
  previous.counts.open + purchases - deliveries,
  "open count must reconcile across one replayed day"
);
assert.strictEqual(current.windowEnd, "2018-08-21", "context must end at the prior day");
assert.strictEqual(current.hotspotEnd, "2018-08-21", "hotspots must end at the prior day");
assert.strictEqual(current.rangeDays, 30, "default context window");

assert.strictEqual(current.counts.dueSoon, 255, "due-soon count on 22 Aug");
assert.strictEqual(current.counts.dueSoonHandover, 189, "due-soon handover count on 22 Aug");
assert.strictEqual(current.counts.recentOverdue, 117, "recent overdue count on 22 Aug");
assert.strictEqual(current.counts.recentOverdueHandover, 72, "recent overdue handover count on 22 Aug");
assert.strictEqual(current.counts.olderOverdue, 2823, "older overdue count on 22 Aug");
assert.strictEqual(current.counts.overdue, 2940, "total overdue count on 22 Aug");
assert.strictEqual(current.counts.open, 5074, "total open count on 22 Aug");
const overdueNotDelivered = model.overdueNotDelivered.get(asOf);
assert.ok(overdueNotDelivered, "overdue, not yet delivered series must cover 22 Aug");
assert.strictEqual(overdueNotDelivered.count, 301, "overdue, not yet delivered on 22 Aug");
assert.strictEqual(overdueNotDelivered.previous, 85, "overdue, not yet delivered 30 days earlier on 22 Aug");
assert.strictEqual(
  overdueNotDelivered.previous,
  model.overdueNotDelivered.get("2018-07-23").count,
  "prior value must equal the same measure 30 days earlier"
);
assert.strictEqual(overdueNotDelivered.due, 8766, "orders due in the 22 Aug window");
assert.strictEqual(overdueNotDelivered.carrier, 167, "with carrier on 22 Aug");
assert.strictEqual(
  overdueNotDelivered.count - overdueNotDelivered.carrier,
  134,
  "not yet shipped on 22 Aug"
);
assert.strictEqual(current.overdueByState.SP, 1230, "SP overdue count on 22 Aug");
assert.strictEqual(current.openByState.SP, 2051, "SP open count on 22 Aug");
assert.strictEqual(
  current.counts.overdue,
  current.counts.recentOverdue + current.counts.olderOverdue,
  "overdue total must split into recent and older"
);
assert.strictEqual(
  current.counts.overdueHandover,
  current.counts.recentOverdueHandover + current.counts.olderOverdueHandover,
  "overdue handover split must reconcile"
);
assert.strictEqual(
  current.counts.recentOverdue,
  current.counts.recentOverdueHandover + current.counts.recentOverdueNoHandover,
  "recent overdue handover split must reconcile"
);
assert.strictEqual(
  current.counts.olderOverdue,
  current.counts.olderOverdueHandover + current.counts.olderOverdueNoHandover,
  "older overdue handover split must reconcile"
);

const priority = current.priorityQueue;
const top12 = priority.slice(0, 12);
assert.strictEqual(priority.length, current.counts.dueSoon + current.counts.recentOverdue, "near-term queue size");
assert.strictEqual(top12.length, 12, "default queue must have a full top list");
assert.ok(top12.every(function (item) { return item.bucket === "soon" || item.bucket === "overdue"; }),
  "default top 12 must contain only due-soon and recent overdue orders");
assert.ok(top12.every(function (item) { return item.promiseDay >= "2018-08-15"; }),
  "default top 12 must not contain promises older than seven days");
assert.ok(priority.every(function (item) { return item.bucket !== "older"; }),
  "near-term queue must exclude older exceptions");
assert.ok(priority.slice(0, current.counts.dueSoon).every(function (item) { return item.bucket === "soon"; }),
  "due-soon orders must lead the near-term queue");
assert.ok(current.queue.some(function (item) { return item.bucket === "older"; }),
  "older exceptions must remain discoverable in the full attention queue");

const sumByState = function (map) {
  return Object.keys(map).reduce(function (total, key) { return total + map[key]; }, 0);
};
assert.strictEqual(sumByState(current.overdueByState), current.counts.overdue, "overdue state counts reconcile");
assert.strictEqual(sumByState(current.dueSoonByState), current.counts.dueSoon, "due-soon state counts reconcile");
assert.strictEqual(sumByState(current.recentOverdueByState), current.counts.recentOverdue, "recent state counts reconcile");
assert.strictEqual(sumByState(current.olderOverdueByState), current.counts.olderOverdue, "older state counts reconcile");
assert.strictEqual(sumByState(current.openByState), current.counts.open, "open state counts reconcile");
assert.strictEqual(sumByState(current.noPromiseByState), current.counts.noPromise, "no-promise state counts reconcile");

assert.strictEqual(Core.isValidIso("2018-08-22"), true, "valid calendar date");
assert.strictEqual(Core.isValidIso("2020-02-29"), true, "valid leap day");
assert.strictEqual(Core.isValidIso("2018-02-29"), false, "non-leap day must be rejected");
assert.strictEqual(Core.isValidIso("2018-02-30"), false, "day beyond month end must be rejected");
assert.strictEqual(Core.isValidIso("2018-13-01"), false, "month beyond December must be rejected");
assert.strictEqual(Core.isValidIso("2018-00-10"), false, "month zero must be rejected");
assert.strictEqual(Core.isValidIso("22-08-2018"), false, "wrong shape must be rejected");

const window30 = Core.buildWindow(model, asOf, 30, null);
assert.strictEqual(window30.start, "2018-07-23", "30-day window start");
assert.strictEqual(window30.end, "2018-08-21", "30-day window end");
assert.strictEqual(window30.totals.orders, current.context.orders, "window orders must match context");
assert.strictEqual(window30.totals.cents, current.context.salesCents, "window item price must match context");
assert.strictEqual(window30.totals.delivered, current.context.delivered, "window deliveries must match context");
assert.strictEqual(window30.totals.onTime, current.context.onTime, "window on-time count must match context");
assert.ok(Math.abs(window30.totals.onTimeShare - current.context.onTimeShare) < 1e-12, "window on-time share must match context");
assert.strictEqual(window30.previous.orders, 5626, "previous 30-day orders");
assert.strictEqual(window30.previous.delivered, 4895, "previous 30-day deliveries");
assert.strictEqual(window30.previous.late, 62, "previous 30-day late deliveries");
assert.strictEqual(
  window30.totals.late,
  window30.totals.delivered - window30.totals.onTime,
  "national late count must reconcile with delivered minus on time"
);
assertOutcomeTotals(window30.totals, expectedOutcomes(asOf, 30), "30-day headline outcomes");
assert.ok(window30.totals.onTimeReviewAvg !== null && window30.previous.onTimeReviewAvg !== null,
  "the review tile needs both on-time comparisons");
assert.ok(window30.totals.lateReviewAvg !== null && window30.previous.lateReviewAvg !== null,
  "the review tile needs both late comparisons");
assert.notStrictEqual(window30.previous.lateReviewAvg, window30.previous.onTimeReviewAvg,
  "the late review comparison must carry its own value");
const earlierAsOf = "2018-08-01";
const earlierWindow = Core.buildWindow(model, earlierAsOf, 30, null);
assertOutcomeTotals(earlierWindow.totals, expectedOutcomes(earlierAsOf, 30),
  "earlier 30-day headline outcomes");
const futureVisibleReviews = data.daily_reviews.filter(function (row) {
  return row[0] >= earlierWindow.start && row[0] <= earlierWindow.end && row[1] >= earlierAsOf;
});
assert.ok(futureVisibleReviews.length > 0,
  "the review fixture must contain reviews that are not visible at the earlier date");

const nationalDailySales = window30.points.map(function (point) {
  return Core.metricValue(point, "sales");
});
assert.ok(
  nationalDailySales.every(function (value) { return typeof value === "number" && isFinite(value); }),
  "national daily item price values must be finite numbers"
);
assert.strictEqual(
  nationalDailySales.reduce(function (total, value) { return total + value; }, 0),
  window30.totals.cents,
  "national daily item price values must reconcile to the window item price total"
);

const sp30 = Core.buildWindow(model, asOf, 30, "SP");
assert.strictEqual(sp30.totals.orders, 3824, "SP 30-day orders");
assert.strictEqual(sp30.totals.cents, 47201149, "SP 30-day item price");
assert.strictEqual(sp30.totals.delivered, 3677, "SP 30-day deliveries");
assert.strictEqual(sp30.totals.onTime, 3402, "SP 30-day on-time deliveries");
assert.strictEqual(sp30.totals.late, 275, "SP 30-day late deliveries");
assert.ok(Math.abs(sp30.totals.onTimeShare - 0.9252107696491705) < 1e-12, "SP on-time share");
assert.strictEqual(
  sp30.totals.onTimeShare,
  Core.metricValue({ delivered: 3677, onTime: 3402 }, "onTimeShare"),
  "metric share helper must match totals"
);
const spDailySales = sp30.points.map(function (point) {
  return Core.metricValue(point, "sales");
});
assert.strictEqual(
  spDailySales.reduce(function (total, value) { return total + value; }, 0),
  sp30.totals.cents,
  "SP daily item price values must reconcile to the window item price total"
);

const ordersByState = Core.buildStateRanking(model, asOf, 30, "orders");
assert.strictEqual(ordersByState.rows[0].state, "SP", "state ranking orders leader");
assert.strictEqual(ordersByState.rows[0].value, 3824, "SP ranked orders value");
assert.strictEqual(ordersByState.rows.length, 27, "all states with purchase data are ranked");
assert.strictEqual(
  ordersByState.rows.reduce(function (total, row) { return total + row.value; }, 0),
  ordersByState.national.orders,
  "state orders ranking must sum to the national total"
);
const salesByState = Core.buildStateRanking(model, asOf, 30, "sales");
assert.strictEqual(salesByState.rows[0].state, "SP", "state ranking sales leader");
assert.strictEqual(salesByState.rows[0].value, 47201149, "SP ranked item price value");
assert.strictEqual(
  salesByState.rows.reduce(function (total, row) { return total + row.value; }, 0),
  salesByState.national.cents,
  "state item price ranking must sum to the national total"
);
const deliveriesByState = Core.buildStateRanking(model, asOf, 30, "delivered");
assert.strictEqual(deliveriesByState.rows[0].state, "SP", "state ranking delivery leader");
assert.strictEqual(
  deliveriesByState.rows.reduce(function (total, row) { return total + row.value; }, 0),
  deliveriesByState.national.delivered,
  "state delivery ranking must sum to the national total"
);
const shareByState = Core.buildStateRanking(model, asOf, 30, "onTimeShare");
assert.strictEqual(shareByState.rows[0].value, 1, "state share ranking uses the highest share");
assert.ok(shareByState.rows.every(function (row) { return row.totals.delivered > 0; }),
  "state share ranking must only include states with deliveries");

const lateCountByState = Core.buildStateRanking(model, asOf, 90, "late");
const lateRateByState = Core.buildStateRanking(model, asOf, 90, "lateShare");
assert.strictEqual(lateCountByState.rows[0].state, "SP", "late count ranking leader");
assert.strictEqual(lateCountByState.rows[0].value, 427, "late count ranking value");
assert.strictEqual(lateRateByState.rows[0].state, "PI", "late rate ranking leader differs from count");
assert.strictEqual(lateRateByState.rows[0].value, 18 / 95, "late rate ranking uses late over delivered");
assert.ok(lateRateByState.rows.every(function (row) { return row.totals.delivered > 0; }),
  "late rate ranking must only include states with deliveries");
assert.strictEqual(Core.metricValue({ delivered: 0, late: 0 }, "lateShare"), null,
  "late share must be undefined without completed deliveries");
assert.strictEqual(Core.metricValue({ delivered: 8, late: 2 }, "lateShare"), 0.25,
  "late share must divide late deliveries by completed deliveries");
assert.strictEqual(Core.defaultViz("lateShare"), "line", "late share defaults to a line chart");
assert.strictEqual(Core.metricLabel("lateShare"), "Late-delivery share", "late share label");

const hotspots90 = Core.buildHotspots(model, asOf, 90);
assert.strictEqual(hotspots90.rows[0].state, "SP", "hotspots rank the highest late count first");
assert.strictEqual(hotspots90.rows[0].late, 427, "SP late count in hotspot window");
assert.strictEqual(hotspots90.rows[0].delivered, 8520, "SP delivered count in hotspot window");
assert.strictEqual(
  hotspots90.rows.reduce(function (total, row) { return total + row.late; }, 0),
  hotspots90.totals.late,
  "hotspot numerator must sum to the total"
);
assert.strictEqual(
  hotspots90.rows.reduce(function (total, row) { return total + row.delivered; }, 0),
  hotspots90.totals.delivered,
  "hotspot denominator must sum to the total"
);
assert.strictEqual(hotspots90.totals.late, 998, "hotspot late total");
assert.strictEqual(hotspots90.totals.delivered, 18928, "hotspot delivered total");

assert.strictEqual(Core.normalizePin, undefined, "legacy metric pin normalization must be retired");
assert.strictEqual(Core.pinKey, undefined, "legacy metric pin keys must be retired");
assert.strictEqual(Core.defaultViz("onTimeShare"), "line", "share defaults to a line chart");
assert.strictEqual(Core.defaultViz("orders"), "bar", "counts default to a bar chart");

const twelveMonths = Core.buildChartWindow(model, asOf, "12mo", null);
assert.strictEqual(twelveMonths.granularity, "month", "12 mo uses monthly points");
assert.strictEqual(twelveMonths.points.length, 12, "12 mo returns twelve monthly points");
assert.strictEqual(twelveMonths.start, "2017-09-01", "12 mo starts at the first of the month");
assert.strictEqual(twelveMonths.end, "2018-08-21", "12 mo ends at the latest complete day");
assert.strictEqual(twelveMonths.points[11].month, "2018-08", "the last monthly point is the selected month");
assert.strictEqual(twelveMonths.points[11].partial, true, "the selected month is marked partial");
assert.strictEqual(twelveMonths.points[10].partial, false, "an earlier full month is not partial");
assert.strictEqual(Core.minSample(model), 30, "the artifact carries the shared minimum sample");

function monthlyTotals(window) {
  return window.points.reduce(function (totals, point) {
    totals.orders += point.orders;
    totals.delivered += point.delivered;
    totals.late += point.late;
    return totals;
  }, { orders: 0, delivered: 0, late: 0 });
}

const twelveTotals = monthlyTotals(twelveMonths);
assert.strictEqual(twelveTotals.delivered, 72777, "12 mo to 22 Aug 2018 must include Sep-Oct 2017 deliveries");
assert.strictEqual(twelveTotals.late, 5702, "12 mo to 22 Aug 2018 late deliveries");
assert.strictEqual(twelveTotals.orders, 75422, "12 mo to 22 Aug 2018 orders placed");
assert.ok(Math.abs(twelveTotals.late / twelveTotals.delivered - 0.0783) < 0.0001,
  "12 mo to 22 Aug 2018 late share must be 7.8%");
assert.ok(twelveMonths.points.every(function (point) {
  return point.ordersNoData === false && point.deliveriesNoData === false;
}), "every month in an allowed 12 mo window must have data");

const firstMayMonths = Core.buildChartWindow(model, "2018-05-01", "12mo", null);
assert.strictEqual(firstMayMonths.start, "2017-05-01", "12 mo to 1 May 2018 starts in May 2017");
const firstMayTotals = monthlyTotals(firstMayMonths);
assert.strictEqual(firstMayTotals.delivered, 62187, "12 mo to 1 May 2018 deliveries");
assert.strictEqual(firstMayTotals.late, 4806, "12 mo to 1 May 2018 late deliveries");
assert.strictEqual(firstMayTotals.orders, 65582, "12 mo to 1 May 2018 orders placed");
assert.ok(firstMayMonths.points.every(function (point) {
  return point.ordersNoData === false && point.deliveriesNoData === false;
}), "12 mo to the earliest selectable date must still be backed by data");

const noDataModel = Core.createModel({
  meta: {
    state_names: {},
    min_sample: 30,
    series_start: { purchases: "2018-07-01", deliveries: "2018-07-01" }
  },
  daily_purchases: [],
  daily_deliveries: [],
  daily_reviews: [],
  state_deliveries: {},
  state_purchases: {},
  orders: []
});
const noDataWindow = Core.buildMonthlyWindow(noDataModel, "2018-08-22", 12, null);
assert.strictEqual(noDataWindow.points[0].ordersNoData, true,
  "a month before the first source month must be marked no data");
assert.strictEqual(noDataWindow.points[9].ordersNoData, true,
  "June 2018 is still before a July 2018 source start");
assert.strictEqual(noDataWindow.points[10].ordersNoData, false,
  "the first source month must not be marked no data");
const ninetyDays = Core.buildChartWindow(model, asOf, 90, null);
assert.strictEqual(ninetyDays.points.length, 90, "90 d keeps daily points");

const hotspotsTwelve = Core.buildHotspotsWindow(model, asOf, "12mo");
assert.strictEqual(hotspotsTwelve.start, "2017-09-01", "12 mo hotspot window starts at the first of the month");
assert.strictEqual(hotspotsTwelve.end, "2018-08-21", "12 mo hotspot window ends at the latest complete day");
assert.strictEqual(hotspotsTwelve.totals.delivered, 72777, "12 mo hotspot deliveries include Sep-Oct 2017");
assert.strictEqual(hotspotsTwelve.totals.late, 5702, "12 mo hotspot late deliveries");
const hotspotsThirty = Core.buildHotspotsWindow(model, asOf, 30);
assert.strictEqual(hotspotsThirty.start, "2018-07-23", "30 d hotspot window starts 30 days back");

const weeklyPoints = [{ x: "2018-08-06" }, { x: "2018-08-13" }, { x: "2018-08-20" }];
assert.strictEqual(
  Core.partialBucketIndex(weeklyPoints, "2018-08-22"),
  2,
  "an incomplete final week must be marked partial"
);
assert.ok(
  Core.partialBucketNote(weeklyPoints, 2, "2018-08-22").indexOf("partial week") !== -1,
  "the partial-week note must say week"
);

const nearSeriesStart = Core.buildWindow(model, "2016-10-01", 30, null);
assert.strictEqual(nearSeriesStart.previousAvailable, false,
  "window near the series start must omit the previous period");
assert.strictEqual(nearSeriesStart.previous, null,
  "previous totals must be suppressed when their full window predates the series");
assert.strictEqual(window30.previousAvailable, true, "normal window keeps its previous period");

const askChartWindow = Core.buildWindow(model, "2018-08-20", 30, "SP");
assert.strictEqual(askChartWindow.end, "2018-08-19", "an Ask chart keeps its answer as-of date");
assert.strictEqual(sp30.end, "2018-08-21", "a state window uses the global as-of date");
assert.strictEqual(
  Core.metricLabel("sales"),
  "Item price total",
  "sales metric label must not claim revenue"
);
assert.strictEqual(
  Core.metricMethod("sales").indexOf("not revenue") !== -1,
  true,
  "sales method must state that it is not revenue"
);

assert.strictEqual(Core.interpretAsk, undefined, "legacy Ask parser must be retired");
assert.strictEqual(Core.interpret, undefined, "legacy queue parser must be retired");

const irregular = [
  { x: "2018-01-01" },
  { x: "2018-01-02" },
  { x: "2018-03-01" }
];
const irregularScale = Core.chartXScale(irregular);
assert.strictEqual(irregularScale.temporal, true, "date x values must use a temporal scale");
const irregularPositions = irregular.map(function (point, index) {
  return irregularScale.at(index, 0, 100);
});
const firstGap = irregularPositions[1] - irregularPositions[0];
const secondGap = irregularPositions[2] - irregularPositions[1];
assert.ok(Math.abs(firstGap * 58 - secondGap) < 1e-9,
  "Jan 1, Jan 2 and Mar 1 must be spaced by actual days, not row index");
assert.ok(firstGap > 0 && secondGap > firstGap,
  "irregular date intervals must not look evenly spaced");

const categorical = [{ x: "SP" }, { x: "RJ" }, { x: "MG" }];
const categoricalScale = Core.chartXScale(categorical);
assert.strictEqual(categoricalScale.temporal, false, "labels must keep categorical spacing");
assert.deepStrictEqual(
  categorical.map(function (point, index) { return categoricalScale.at(index, 0, 100); }),
  [0, 50, 100],
  "categorical points must remain evenly spaced"
);
assert.strictEqual(Core.parseTemporalX("2018-02"), Date.UTC(2018, 1, 1), "YYYY-MM starts on the first");
assert.strictEqual(Core.parseTemporalX("2018-02-30"), null, "invalid calendar dates are categorical");

const monthlyPoints = [{ x: "2018-06" }, { x: "2018-07" }, { x: "2018-08" }];
assert.strictEqual(
  Core.partialBucketIndex(monthlyPoints, "2018-08-22"), 2,
  "an incomplete final month must be marked partial"
);
assert.strictEqual(
  Core.partialBucketNote(monthlyPoints, 2, "2018-08-22"),
  "2018-08 is a partial month because the cutoff is 2018-08-22.",
  "the partial-month note must name the bucket and cutoff"
);
assert.strictEqual(
  Core.partialBucketIndex(monthlyPoints, "2018-09-01"), -1,
  "a month ending before the cutoff must not be marked partial"
);
assert.strictEqual(
  Core.partialBucketIndex(monthlyPoints.concat([{ x: "2018-09" }]), "2018-08-22"), -1,
  "a future month must not be marked partial"
);
assert.strictEqual(
  Core.partialBucketIndex(monthlyPoints, "2018-08-01"), -1,
  "a month starting exactly at the cutoff must not be marked partial"
);
assert.strictEqual(
  Core.partialBucketNote(monthlyPoints, -1, "2018-09-01"), "",
  "complete months must not produce a partial note"
);
const yearlyPoints = [{ x: "2016" }, { x: "2017" }, { x: "2018" }];
assert.strictEqual(
  Core.partialBucketIndex(yearlyPoints, "2018-08-22"), 2,
  "an incomplete final year must be marked partial"
);
assert.strictEqual(
  Core.partialBucketNote(yearlyPoints, 2, "2018-08-22"),
  "2018 is a partial year because the cutoff is 2018-08-22.",
  "the partial-year note must name the bucket and cutoff"
);
assert.strictEqual(
  Core.partialBucketIndex(yearlyPoints, "2019-01-01"), -1,
  "a year ending before the cutoff must not be marked partial"
);
assert.strictEqual(
  Core.partialBucketIndex([{ x: "2018-08-21" }], "2018-08-22"), -1,
  "daily buckets are not treated as partial months or years"
);
assert.strictEqual(
  Core.partialBucketIndex(monthlyPoints, "not-a-date"), -1,
  "an invalid cutoff must not mark a partial bucket"
);
assert.strictEqual(Core.formatAskNumber(0.004), "0.004", "small values must not render as zero");
assert.strictEqual(Core.formatAskNumber(0.25), "0.25", "ordinary decimals keep their value");
assert.strictEqual(Core.formatAskNumber(1.23456), "1.23", "larger values keep two decimals");
assert.strictEqual(Core.formatAskNumber(1234), "1,234", "integers keep grouping");

const summerRate = Core.askChartScale([0.0289, 0.0260, 0.0228], "late_rate");
assert.strictEqual(summerRate.kind, "rate", "an explicit rate column uses rate units");
assert.strictEqual(summerRate.min, 0, "rate charts must start at zero");
assert.ok(Math.abs(summerRate.max - 0.04) < 1e-12, "summer rates need a 4% top");
assert.deepStrictEqual(
  summerRate.ticks.map(function (tick) { return tick.label; }),
  ["0%", "1%", "2%", "3%", "4%"],
  "rate ticks must be readable percentages, not repeated rounded decimals"
);
assert.strictEqual(new Set(summerRate.ticks.map(function (tick) { return tick.label; })).size,
  summerRate.ticks.length, "rate tick labels must be unique");

const onTimeRate = Core.askChartScale([0.91, 0.93, 0.92], "on_time_share");
assert.strictEqual(onTimeRate.kind, "rate", "share columns are rates");
assert.deepStrictEqual(
  onTimeRate.ticks.map(function (tick) { return tick.label; }),
  ["0%", "25%", "50%", "75%", "100%"],
  "a high rate keeps an honest zero-based percentage scale"
);

const percentRate = Core.askChartScale([3.2, 2.5, 2.9], "late_rate_percent");
assert.strictEqual(percentRate.kind, "rate", "explicit percent-rate columns are rates");
assert.strictEqual(percentRate.unit, "percent", "already-percent values keep their unit");
assert.deepStrictEqual(
  percentRate.ticks.map(function (tick) { return tick.label; }),
  ["0%", "1%", "2%", "3%", "4%"],
  "percent-valued rate ticks stay readable"
);

const scoreScale = Core.askChartScale([0.0289, 0.0260], "average_review_score");
assert.strictEqual(scoreScale.kind, "number", "small scores are not blindly treated as rates");
assert.strictEqual(scoreScale.axisTitle, "", "non-rate charts must not claim rate units");
assert.strictEqual(scoreScale.note, "", "non-rate charts must not add a rate note");
assert.ok(scoreScale.min < 0.0260 && scoreScale.max > 0.0289, "non-rate range keeps padding");
assert.ok(scoreScale.ticks.every(function (tick) { return tick.label.indexOf("%") === -1; }),
  "non-rate ticks must not gain percent units");
assert.strictEqual(new Set(scoreScale.ticks.map(function (tick) { return tick.label; })).size,
  scoreScale.ticks.length, "non-rate tick labels must be unique");
assert.strictEqual(Core.askChartScale([0.02, 0.03], "rate_count").kind, "number",
  "a count whose name merely contains rate is not a rate");
assert.strictEqual(Core.askChartScale([0.5, 0.6], "average_item_price").kind, "money",
  "price columns use money units even when the values are small");
const itemValueScale = Core.askChartScale([340559.35, 120000], "ITEM_VALUE");
assert.strictEqual(itemValueScale.kind, "money", "item value columns are money");
assert.deepStrictEqual(
  itemValueScale.ticks.map(function (tick) { return tick.label; }),
  ["R$ 0", "R$ 100k", "R$ 200k", "R$ 300k", "R$ 400k"],
  "money ticks are compact reais"
);
assert.strictEqual(Core.formatAskMoney(340559.35, true), "R$ 340.6k",
  "compact money matches the Overview formatting");
assert.strictEqual(Core.formatAskMoney(340559.35, false), "R$ 340,559.35",
  "table money keeps exact reais");
assert.strictEqual(Core.moneyUnit("freight_value"), "BRL", "freight columns are money");
assert.strictEqual(Core.moneyUnit("payment_value"), "BRL", "payment columns are money");
assert.strictEqual(Core.moneyUnit("order_count"), null, "counts are not money");
assert.strictEqual(Core.moneyUnit("freight_share"), null, "money shares are not money");
assert.strictEqual(Core.columnLabel("ITEM_VALUE"), "Item value",
  "column headers are plain sentence case");
assert.strictEqual(Core.columnLabel("total_item_value"), "Total item value",
  "underscores become spaces");
assert.strictEqual(Core.columnLabel("item_value", { item_value: "Item value (R$)" }),
  "Item value (R$)", "model-supplied labels win");
assert.strictEqual(Core.formatRateValue(0.0289, "fraction"), "2.89%", "rate tooltips use percent");
assert.strictEqual(Core.formatRateValue(2.89, "percent"), "2.89%", "percent tooltips keep percent");


const AskChats = require("./ask-chats.js");

function memoryStorage() {
  const values = {};
  return {
    getItem: function (key) { return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : null; },
    setItem: function (key, value) { values[key] = String(value); },
    removeItem: function (key) { delete values[key]; },
    values: values
  };
}

function askPayload(text, date) {
  return {
    ok: true,
    as_of: date,
    answer: {
      kind: "query",
      title: "Orders",
      text: text,
      sql: "SELECT COUNT(DISTINCT order_id) AS orders FROM orders WHERE order_purchase_timestamp < :as_of",
      row_count: 1,
      table: { columns: ["orders"], rows: [{ orders: Number(text.split(" ")[0]) }] },
      chart: null,
      provenance: {
        cutoff: date,
        source: "local fixture",
        method: "Distinct orders before the cutoff.",
        privacy: "No identifiers returned."
      }
    }
  };
}

const ratePayload = askPayload("Saved rate answer", "2018-08-22");
ratePayload.answer.table = {
  columns: ["month", "late_rate"],
  rows: [
    { month: "2017-06", late_rate: 0.0289 },
    { month: "2017-07", late_rate: 0.0260 },
    { month: "2017-08", late_rate: 0.0228 }
  ]
};
ratePayload.answer.chart = {
  type: "line", x: "month", y: "late_rate",
  labels: { month: "Month", late_rate: "Late rate" }
};
const storedRate = AskChats.compactPayload(ratePayload);
assert.strictEqual(storedRate.answer.chart.y, "late_rate", "saved chats keep the rate chart spec");
assert.deepStrictEqual(storedRate.answer.chart.labels,
  { month: "Month", late_rate: "Late rate" },
  "saved chats keep model-supplied column labels");
const storedRateValues = storedRate.answer.table.rows.map(function (row) {
  return row[storedRate.answer.chart.y];
});
const replayScale = Core.askChartScale(storedRateValues, storedRate.answer.chart.y);
assert.strictEqual(replayScale.kind, "rate", "saved rate charts replay with rate units");
assert.deepStrictEqual(
  replayScale.ticks.map(function (tick) { return tick.label; }),
  ["0%", "1%", "2%", "3%", "4%"],
  "saved rate charts replay with readable percent ticks"
);

const partialPayload = askPayload("Saved partial answer", "2018-08-22");
partialPayload.answer.table = {
  columns: ["month", "orders"],
  rows: [
    { month: "2018-06", orders: 10 },
    { month: "2018-07", orders: 12 },
    { month: "2018-08", orders: 4 }
  ]
};
partialPayload.answer.chart = { type: "line", x: "month", y: "orders" };
const storedPartial = AskChats.compactPayload(partialPayload);
const storedPartialPoints = storedPartial.answer.table.rows.map(function (row) {
  return { x: row[storedPartial.answer.chart.x], y: row[storedPartial.answer.chart.y] };
});
assert.strictEqual(
  Core.partialBucketIndex(storedPartialPoints, storedPartial.as_of), 2,
  "saved answers replay the partial bucket from their stored as-of date"
);

const pinPayload = askPayload("Saved pin answer", "2018-08-22");
pinPayload.answer.pin = {
  kind: "rolling",
  label: "last 90 days",
  period: "Last 90 days to 21 Aug 2018",
  button_label: "Pin to Overview \u00b7 moves with the date (last 90 days)",
  windows: ["30d", "90d", "12mo", "ytd"],
  window_labels: { "90d": "90 d" },
  default_window: "90d"
};
const storedPin = AskChats.compactPayload(pinPayload).answer.pin;
assert.strictEqual(storedPin.period, "Last 90 days to 21 Aug 2018",
  "saved chats keep the pin period line");
assert.strictEqual(storedPin.button_label,
  "Pin to Overview \u00b7 moves with the date (last 90 days)",
  "saved chats keep the pin button kind");

const oldChat = AskChats.emptyChat("2018-08-22", "chat-old", 1000);
AskChats.addTurn(oldChat, "Orders on the first date?", "turn-old", 1001);
AskChats.commitTurn(
  oldChat,
  "turn-old",
  askPayload("1 order", "2018-08-22"),
  [{ role: "user", content: "Orders on the first date?" }, { role: "assistant", content: "1 order" }],
  1002
);
const askChats = [oldChat];
const dateChange = AskChats.applyDateChange(askChats, "chat-old", "2018-08-23", {
  makeId: function () { return "chat-new"; },
  now: 2000
});
const freshChat = askChats.filter(function (chat) { return chat.id === "chat-new"; })[0];
assert.strictEqual(dateChange.activeChatId, "chat-new", "changing the picker opens a fresh chat");
assert.strictEqual(oldChat.date, "2018-08-22", "the old chat keeps its original simulated date");
assert.strictEqual(oldChat.turns.length, 1, "the old chat keeps its turns");
assert.strictEqual(oldChat.history.length, 2, "the old chat keeps its follow-up context");
assert.strictEqual(freshChat.date, "2018-08-23", "the fresh chat uses the newly selected date");
assert.strictEqual(freshChat.turns.length, 0, "the fresh chat starts empty");
assert.strictEqual(freshChat.history.length, 0, "the fresh chat cannot inherit old follow-up context");
assert.strictEqual(
  AskChats.applyDateChange(askChats, "chat-new", "2018-08-23", { now: 2001 }).activeChatId,
  "chat-new",
  "an empty draft is reused when the same date is selected again"
);

AskChats.addTurn(oldChat, "Follow-up on the first date?", "turn-late", 3000);
AskChats.commitTurn(
  oldChat,
  "turn-late",
  askPayload("2 orders", "2018-08-22"),
  AskChats.nextHistory(oldChat.history, "Follow-up on the first date?", askPayload("2 orders", "2018-08-22")),
  3001
);
assert.strictEqual(oldChat.turns[oldChat.turns.length - 1].payload.answer.text, "2 orders",
  "an in-flight answer is committed to the chat and date that submitted it");
assert.strictEqual(freshChat.turns.length, 0, "an in-flight answer never lands in the newly opened chat");
assert.strictEqual(freshChat.history.length, 0, "an in-flight answer never mixes follow-up context across dates");

const store = memoryStorage();
const saved = AskChats.saveStore(store, askChats, "chat-new");
assert.strictEqual(saved.ok, true, "chat history saves to browser storage");
const reloaded = AskChats.loadStore(store);
assert.strictEqual(reloaded.ok, true, "saved chat history reloads");
const replayed = reloaded.chats.filter(function (chat) { return chat.id === "chat-old"; })[0];
assert.ok(replayed, "the saved conversation is present after reload");
assert.strictEqual(replayed.date, "2018-08-22", "reload restores the conversation date");
assert.strictEqual(replayed.turns.length, 2, "reload restores every turn");
assert.strictEqual(replayed.turns[0].payload.answer.text, "1 order", "reload restores the first rendered answer");
assert.strictEqual(replayed.turns[1].payload.answer.text, "2 orders", "reload restores the last rendered answer");
assert.strictEqual(replayed.turns[1].payload.answer.table.rows[0].orders, 2, "reload restores the answer table");
assert.strictEqual(replayed.turns[1].payload.answer.sql.indexOf(":as_of") !== -1, true, "reload restores exact SQL");
assert.strictEqual(replayed.history[replayed.history.length - 1].content, "2 orders",
  "reload restores the short follow-up context");
assert.strictEqual(reloaded.activeChatId, "chat-new", "reload reopens the active draft at its own date");
assert.strictEqual(
  reloaded.chats.filter(function (chat) { return chat.id === "chat-new"; })[0].date,
  "2018-08-23",
  "reload restores the fresh chat's separate date"
);

const seedChatsFile = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "ask-seed-chats.json"), "utf8"));

const freshSeedStore = memoryStorage();
const freshSeedLoaded = AskChats.loadStore(freshSeedStore);
assert.deepStrictEqual(freshSeedLoaded.seedsAdded, [],
  "a fresh browser has not received any ready-made chats yet");
const freshSeedMerge = AskChats.applySeeds(
  freshSeedLoaded.chats, seedChatsFile, freshSeedLoaded.seedsAdded);
assert.strictEqual(freshSeedMerge.chats.length, 4,
  "a fresh browser receives the four ready-made chats");
assert.deepStrictEqual(freshSeedMerge.seedsAdded,
  ["seed-1", "seed-2", "seed-3", "seed-4"],
  "every ready-made chat id is recorded once");
assert.ok(freshSeedMerge.chats.every(function (chat) {
  return chat.date === "2018-08-22" && chat.turns.length === 1 &&
    chat.turns[0].status === "complete";
}), "ready-made chats open as complete answers dated 22 Aug 2018");
assert.strictEqual(
  AskChats.saveStore(
    freshSeedStore, freshSeedMerge.chats, null,
    { seedsAdded: freshSeedMerge.seedsAdded }
  ).ok,
  true,
  "ready-made chats save like normal chats"
);
const freshSeedReload = AskChats.loadStore(freshSeedStore);
assert.strictEqual(freshSeedReload.chats.length, 4, "ready-made chats survive reload");
assert.deepStrictEqual(freshSeedReload.seedsAdded, freshSeedMerge.seedsAdded,
  "the recorded ready-made chat ids survive reload");

const seedSource = seedChatsFile.chats.filter(function (chat) { return chat.id === "seed-1"; })[0];
const seedReloaded = freshSeedReload.chats.filter(function (chat) { return chat.id === "seed-1"; })[0];
assert.strictEqual(seedReloaded.turns[0].payload.answer.text, seedSource.answer.text,
  "a ready-made answer keeps its saved text exactly");
assert.deepStrictEqual(seedReloaded.turns[0].payload.answer.table, seedSource.answer.table,
  "a ready-made answer keeps its saved table numbers exactly");
assert.deepStrictEqual(seedReloaded.turns[0].payload.answer.chart, seedSource.answer.chart,
  "a ready-made answer keeps its saved chart exactly");
assert.strictEqual(seedReloaded.turns[0].payload.answer.answer_id, seedSource.answer.answer_id,
  "a ready-made answer keeps the answer id its pin needs");
assert.strictEqual(seedReloaded.turns[0].payload.answer.pin.button_label,
  seedSource.answer.pin.button_label, "a ready-made answer keeps its pin button");

const deletedSeedStore = memoryStorage();
AskChats.saveStore(
  deletedSeedStore, freshSeedReload.chats, freshSeedReload.activeChatId,
  { seedsAdded: freshSeedReload.seedsAdded }
);
const withoutSeedTwo = AskChats.removeChat(AskChats.loadStore(deletedSeedStore).chats, "seed-2");
AskChats.saveStore(
  deletedSeedStore, withoutSeedTwo, freshSeedReload.activeChatId,
  { seedsAdded: freshSeedReload.seedsAdded }
);
const afterSeedDelete = AskChats.loadStore(deletedSeedStore);
assert.strictEqual(
  afterSeedDelete.chats.filter(function (chat) { return chat.id === "seed-2"; }).length,
  0,
  "a deleted ready-made chat is gone"
);
const afterSeedDeleteMerge = AskChats.applySeeds(
  afterSeedDelete.chats, seedChatsFile, afterSeedDelete.seedsAdded);
assert.strictEqual(
  afterSeedDeleteMerge.chats.filter(function (chat) { return chat.id === "seed-2"; }).length,
  0,
  "a deleted ready-made chat stays deleted after another merge"
);
assert.strictEqual(afterSeedDeleteMerge.chats.length, 3,
  "the other ready-made chats remain after one is deleted");

const existingAskStore = memoryStorage();
const ownChat = AskChats.emptyChat("2018-08-22", "chat-seed-own", 8000);
AskChats.addTurn(ownChat, "My own saved question", "turn-seed-own", 8001);
AskChats.commitTurn(
  ownChat, "turn-seed-own", askPayload("3 orders", "2018-08-22"),
  AskChats.nextHistory([], "My own saved question", askPayload("3 orders", "2018-08-22")), 8002
);
AskChats.saveStore(existingAskStore, [ownChat], "chat-seed-own");
const existingAskLoaded = AskChats.loadStore(existingAskStore);
const existingAskMerge = AskChats.applySeeds(
  existingAskLoaded.chats, seedChatsFile, existingAskLoaded.seedsAdded);
assert.strictEqual(existingAskMerge.chats.length, 5,
  "a browser with saved chats also receives the ready-made chats");
assert.strictEqual(
  existingAskMerge.chats.filter(function (chat) { return chat.id === "chat-seed-own"; }).length,
  1,
  "receiving ready-made chats keeps the visitor's own chat"
);
const seedBase = Date.parse("2026-11-09T05:19:10+08:00");
const seedSorted = existingAskMerge.chats.slice().sort(function (a, b) {
  return b.updatedAt - a.updatedAt;
});
assert.deepStrictEqual(
  seedSorted.slice(0, 4).map(function (chat) { return chat.id; }),
  ["seed-1", "seed-2", "seed-3", "seed-4"],
  "ready-made chats sort above an older own chat, newest first"
);
assert.strictEqual(seedSorted[0].updatedAt, seedBase + 4000,
  "seed-1 carries the asked time plus the first one-second slot");
assert.strictEqual(seedSorted[3].updatedAt, seedBase + 1000,
  "seed-4 keeps the one-second spacing below seed-1");
assert.ok(seedSorted[0].updatedAt > ownChat.updatedAt,
  "ready-made chats outrank a visitor's older chat");
assert.ok(seedSorted.slice(0, 4).every(function (chat) {
  return chat.date === "2018-08-22";
}), "ready-made chats keep the data as-of date");
AskChats.saveStore(
  existingAskStore, existingAskMerge.chats, existingAskLoaded.activeChatId,
  { seedsAdded: existingAskMerge.seedsAdded }
);
const existingAskReload = AskChats.loadStore(existingAskStore);
const existingAskSecondMerge = AskChats.applySeeds(
  existingAskReload.chats, seedChatsFile, existingAskReload.seedsAdded);
assert.strictEqual(existingAskSecondMerge.chats.length, 5,
  "ready-made chats are added to an existing browser only once");
assert.strictEqual(existingAskSecondMerge.seedsAdded.length, 4,
  "the existing browser records each ready-made chat id once");
assert.strictEqual(
  existingAskSecondMerge.chats.filter(function (chat) { return chat.id === "chat-seed-own"; }).length,
  1,
  "the visitor's own chat survives repeated merges"
);

// A browser that stored the ready-made chats with an older ask time gets the file's time,
// unless the visitor continued or renamed that chat.
const weekOldTime = Date.parse("2026-11-01T00:00:00Z");
const weekOldChat = AskChats.emptyChat("2018-08-22", "chat-week-old", weekOldTime);
AskChats.addTurn(weekOldChat, "A question asked days ago", "turn-week-old", weekOldTime);
AskChats.commitTurn(weekOldChat, "turn-week-old", askPayload("3 orders", "2018-08-22"),
  AskChats.nextHistory([], "A question asked days ago", askPayload("3 orders", "2018-08-22")), weekOldTime);
const staleSeedFile = JSON.parse(JSON.stringify(seedChatsFile));
delete staleSeedFile.asked_at;
const staleMerge = AskChats.applySeeds([weekOldChat], staleSeedFile, []);
const staleSeeds = staleMerge.chats.filter(function (chat) { return /^seed-/.test(chat.id); });
assert.ok(staleSeeds.every(function (chat) { return chat.updatedAt < weekOldChat.updatedAt; }),
  "setup: stale ready-made chats sort below the visitor's chat");
const renamedSeed = staleSeeds[1];
AskChats.renameChat(staleMerge.chats, renamedSeed.id, "My renamed chat");
const renamedTime = renamedSeed.updatedAt;
const refreshed = AskChats.applySeeds(staleMerge.chats, seedChatsFile, staleMerge.seedsAdded);
assert.strictEqual(refreshed.changed, true, "refreshing stale ask times reports a change");
assert.strictEqual(refreshed.chats.length, 5, "refreshing adds no duplicate chats");
refreshed.chats.filter(function (chat) { return /^seed-/.test(chat.id) && chat.id !== renamedSeed.id; })
  .forEach(function (chat) {
    assert.ok(chat.updatedAt > weekOldChat.updatedAt, "untouched ready-made chat takes the file's ask time");
    assert.strictEqual(chat.turns[0].askedAt, chat.updatedAt, "turn time matches the chat time");
  });
assert.strictEqual(renamedSeed.updatedAt, renamedTime, "a renamed ready-made chat keeps its own time");
const oldPinFile = JSON.parse(JSON.stringify(seedChatsFile));
oldPinFile.chats[0].answer.pin = Object.assign({}, oldPinFile.chats[0].answer.pin,
  { kind: "fixed", button_label: "Pin to Overview \u00b7 fixed to July 2018" });
const oldPinMerge = AskChats.applySeeds([], oldPinFile, []);
const newPinMerge = AskChats.applySeeds(oldPinMerge.chats, seedChatsFile, oldPinMerge.seedsAdded);
assert.strictEqual(newPinMerge.changed, true, "a changed ready-made answer reaches a browser that stored the old one");
assert.strictEqual(newPinMerge.chats.filter(function (chat) { return chat.id === "seed-1"; })[0]
  .turns[0].payload.answer.pin.button_label, seedChatsFile.chats[0].answer.pin.button_label,
  "the stored ready-made chat shows the file's pin button");
assert.strictEqual(AskChats.applySeeds(refreshed.chats, seedChatsFile, refreshed.seedsAdded).changed, false,
  "a second merge changes nothing");

const interrupted = AskChats.emptyChat("2018-08-24", "chat-interrupted", 4000);
AskChats.addTurn(interrupted, "Unfinished question?", "turn-loading", 4001);
const interruptedStore = memoryStorage();
AskChats.saveStore(interruptedStore, [interrupted], "chat-interrupted");
const interruptedReload = AskChats.loadStore(interruptedStore);
const interruptedTurn = interruptedReload.chats[0].turns[0];
assert.strictEqual(interruptedTurn.status, "interrupted",
  "a page close during an answer must not leave a permanent loading turn");
assert.strictEqual(interruptedTurn.payload, null, "an unfinished answer has no replayable payload");
assert.ok(interruptedTurn.error, "an unfinished answer explains what happened");

const manyChats = [];
for (let index = 0; index < 35; index += 1) {
  const chat = AskChats.emptyChat("2018-08-22", "chat-many-" + index, index + 1);
  AskChats.addTurn(chat, "Question " + index, "turn-many-" + index, index + 2);
  AskChats.commitTurn(chat, "turn-many-" + index, askPayload(index + " orders", "2018-08-22"), [], index + 3);
  manyChats.push(chat);
}
const boundedStore = memoryStorage();
const bounded = AskChats.saveStore(boundedStore, manyChats, "chat-many-34");
assert.strictEqual(bounded.ok, true, "bounded chat history still saves");
assert.ok(bounded.chats.length <= AskChats.MAX_CHATS, "retained chats must stay within the configured bound");
assert.ok(bounded.chats.some(function (chat) { return chat.id === "chat-many-34"; }),
  "the active chat must survive pruning");

const blockedStorage = {
  getItem: function () { throw new Error("storage blocked"); },
  setItem: function () { throw new Error("storage blocked"); }
};
const blockedLoad = AskChats.loadStore(blockedStorage);
assert.strictEqual(blockedLoad.ok, false, "unavailable storage is reported, not thrown");
assert.deepStrictEqual(blockedLoad.chats, [], "unavailable storage yields no chats");
const blockedChat = AskChats.emptyChat("2018-08-22", "chat-blocked", 5000);
AskChats.addTurn(blockedChat, "Question", "turn-blocked", 5001);
const blockedSave = AskChats.saveStore(blockedStorage, [blockedChat], "chat-blocked");
assert.strictEqual(blockedSave.ok, false, "a full or unavailable store fails gracefully");
assert.ok(blockedSave.error, "a failed save reports its cause to the caller");
assert.ok(blockedSave.chats.length >= 1, "a failed save keeps the in-memory chat usable");

assert.strictEqual(blockedChat.turns.length, 1,
  "a failed save must not delete turns from the in-memory chat");
assert.strictEqual(AskChats.redactIdentifiers("order 0123456789abcdef0123456789abcdef please"),
  "order [identifier removed] please", "stored chat text must not keep identifier-shaped values");

const invalidStore = memoryStorage();
invalidStore.setItem(AskChats.STORAGE_KEY, "{not json");
assert.strictEqual(AskChats.loadStore(invalidStore).ok, false, "corrupt stored JSON is handled as unusable");

const largeRows = [];
for (let index = 0; index < 200; index += 1) largeRows.push({ n: index });
const largePayload = {
  ok: true,
  as_of: "2018-09-01",
  answer: {
    kind: "query",
    title: "Rows",
    text: "The query returned 200 rows.",
    sql: "SELECT n FROM rows WHERE n < :as_of",
    row_count: 200,
    chart: null,
    table: { columns: ["n"], rows: largeRows },
    provenance: {
      as_of: "2018-09-01",
      cutoff: "Exclusive cutoff: events dated before 2018-09-01 are visible; the latest complete day is 2018-08-31.",
      source: "local fixture",
      method: "Enumerate the bounded result rows.",
      privacy: "No identifiers returned."
    }
  }
};
const largeChat = AskChats.emptyChat("2018-09-01", "chat-large", 6000);
AskChats.addTurn(largeChat, "List 200 rows", "turn-large", 6001);
AskChats.commitTurn(largeChat, "turn-large", largePayload, [], 6002);
const largeStore = memoryStorage();
assert.strictEqual(AskChats.saveStore(largeStore, [largeChat], "chat-large").ok, true,
  "a full 200-row answer still saves");
const largeTurn = AskChats.loadStore(largeStore).chats[0].turns[0];
assert.strictEqual(largeTurn.payload.answer.table.rows.length, 200,
  "the complete 200-row API answer must survive storage and replay");
assert.strictEqual(largeTurn.payload.answer.table.rows[199].n, 199,
  "the last permitted API row must survive storage and replay");
assert.strictEqual(largeTurn.payload.answer.table.truncated, undefined,
  "a complete answer is not marked as truncated");
assert.strictEqual(largeTurn.payload.answer.provenance.as_of, "2018-09-01",
  "the server provenance as-of date must be kept for replay");
assert.strictEqual(
  largeTurn.payload.answer.provenance.cutoff.indexOf("Exclusive cutoff") === 0,
  true,
  "the server's descriptive cutoff text must be kept, not treated as an ISO date");

const overflowRows = [];
for (let index = 0; index < 250; index += 1) overflowRows.push({ n: index });
const overflow = AskChats.compactPayload({
  ok: true,
  as_of: "2018-09-01",
  answer: {
    kind: "query",
    title: "Rows",
    text: "The query returned 250 rows.",
    sql: "SELECT n FROM rows WHERE n < :as_of",
    row_count: 250,
    table: { columns: ["n"], rows: overflowRows }
  }
});
assert.strictEqual(overflow.answer.table.rows.length, 200,
  "stored tables keep the 200-row API limit");
assert.strictEqual(overflow.answer.table.truncated, true,
  "a table beyond the stored limit is explicitly marked as truncated");
assert.strictEqual(overflow.answer.table.total_rows, 250,
  "a truncated table keeps the real source row count");

const refusal = AskChats.compactPayload({
  ok: false,
  refused: true,
  as_of: "2018-09-01",
  refusal: { title: "No grounded answer", reason: "unsupported", message: "The data cannot answer that." }
});
assert.strictEqual(refusal.refused, true, "the server refusal shape is stored as a refusal");
assert.strictEqual(refusal.refusal.reason, "unsupported", "the refusal reason is replayed");
const failure = AskChats.compactPayload({
  ok: false,
  error: { code: "provider_error", message: "The provider is not configured." }
});
assert.strictEqual(failure.error.code, "provider_error", "the server error shape is stored as an error");

const stagedFailure = AskChats.compactPayload({
  ok: false,
  as_of: "2018-09-03",
  error: {
    code: "provider_error",
    message: "The provider did not return usable text.",
    stage: "synthesis",
    finish_reason: "stop"
  }
});
assert.strictEqual(stagedFailure.as_of, "2018-09-03",
  "a saved error keeps the selected as-of date");
assert.strictEqual(stagedFailure.error.stage, "synthesis",
  "a saved error keeps the stage that failed");
assert.strictEqual(stagedFailure.error.finish_reason, "stop",
  "a saved error keeps the narrow provider diagnostic");

const errorChat = AskChats.emptyChat("2018-09-03", "chat-error", 7000);
AskChats.addTurn(errorChat, "Question that fails", "turn-error", 7001);
AskChats.commitTurn(errorChat, "turn-error", {
  ok: false,
  as_of: "2018-09-03",
  error: {
    code: "provider_error",
    message: "The provider did not return usable text.",
    stage: "synthesis",
    finish_reason: "stop"
  }
}, [], 7002);
const errorStore = memoryStorage();
assert.strictEqual(AskChats.saveStore(errorStore, [errorChat], "chat-error").ok, true,
  "a chat with a provider error still saves");
const replayedErrorChat = AskChats.loadStore(errorStore).chats[0];
assert.strictEqual(replayedErrorChat.date, "2018-09-03",
  "saved chats keep the selected as-of date after an error");
assert.strictEqual(replayedErrorChat.turns[0].payload.as_of, "2018-09-03",
  "a saved error payload keeps its as-of date");
assert.strictEqual(replayedErrorChat.turns[0].payload.error.stage, "synthesis",
  "saved chats keep the error stage for the evidence wording");

const AskEvidence = require("./ask-evidence.js");

class FakeNode {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.attributes = {};
    this.className = "";
    this._text = "";
  }
  append() {
    for (let index = 0; index < arguments.length; index += 1) {
      if (arguments[index] !== null && arguments[index] !== undefined) {
        this.children.push(arguments[index]);
      }
    }
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  get textContent() {
    if (this._text) return this._text;
    return this.children.map(function (child) {
      return typeof child === "string" ? child : child.textContent;
    }).join("");
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
}

function fakeDocument() {
  return {
    createElement: function (tagName) { return new FakeNode(tagName); },
    createTextNode: function (value) {
      const node = new FakeNode("#text");
      node._text = String(value);
      return node;
    }
  };
}

function findBy(node, predicate, found) {
  if (predicate(node)) found.push(node);
  node.children.forEach(function (child) {
    if (child && typeof child === "object" && Array.isArray(child.children)) {
      findBy(child, predicate, found);
    }
  });
  return found;
}

const screenshotPayload = {
  ok: true,
  as_of: "2018-08-22",
  answer: {
    kind: "query",
    title: "2018 item value by month through July",
    text: "Item value was 1234.00 in 2018 through July, with the highest month in May.",
    sql: "SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month, "
      + "SUM(oi.price) AS item_value, '<b>not html</b>' AS note "
      + "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
      + "WHERE o.order_purchase_timestamp < :as_of GROUP BY month ORDER BY month",
    row_count: 7,
    chart: { type: "line", x: "month", y: "item_value" },
    table: {
      columns: ["month", "item_value", "total_item_value"],
      rows: [{ month: "2018-01", item_value: 100.0, total_item_value: 1234.0 }]
    },
    provenance: {
      as_of: "2018-08-22",
      cutoff: "Exclusive cutoff: events dated before 2018-08-22 are visible; the latest complete day is 2018-08-21.",
      source: "deployment/olist_dashboard/data/analytics.sqlite, built from the public Olist CSVs in data/raw. Local SQLite only; no cloud database. The cutoff is enforced with per-request views over private base tables.",
      method: "The configured model proposed the SQL; the server validated and executed it read-only, then the model summarized only the returned rows. Numeric constant-output projections are rejected, but this lexical guard cannot formally prove that every SQL expression derives from source rows, so values are query output rather than an audited lineage.",
      privacy: "Raw identifier columns, identifier-shaped output values, address columns and review text are blocked by the server. This is a defense-in-depth filter, not a formal guarantee against every possible derived inference."
    }
  }
};

const evidence = AskEvidence.build(screenshotPayload, { date: "2018-08-22", asOf: "2018-08-22" });
assert.deepStrictEqual(evidence.primary, [],
  "a query answer must lead with one plain line, not labelled developer notes");
assert.strictEqual(evidence.plain,
  "Data as of 22 Aug 2018. Counted the rows returned by this query.",
  "older answers without a plain line still get one honest sentence");
assert.deepStrictEqual(
  evidence.audit.map(function (item) { return item[0]; }),
  ["Rows", "Data available", "Source", "Measure"],
  "audit must keep only rows, data availability, source and measure"
);
assert.strictEqual(evidence.audit[1][1],
  "Data through 21 Aug 2018 (latest complete day)",
  "data availability must use the latest complete day, not the exclusive cutoff day");
assert.strictEqual(evidence.audit[3][1],
  "Numeric measures returned by the query: item_value, total_item_value.",
  "measure must name the returned numeric measures without treating the period label as a measure");

const plainEvidence = AskEvidence.build({
  ok: true,
  as_of: "2018-08-22",
  answer: {
    kind: "query",
    title: "Late rate by delivery month",
    text: "Late rate was 3.1%.",
    sql: "SELECT 1",
    chart: null,
    table: { columns: ["month", "late_rate"], rows: [{ month: "2018-07", late_rate: 0.031 }] },
    provenance: {
      as_of: "2018-08-22",
      plain: "Data as of 22 Aug 2018. Counted deliveries completed before the cutoff. Dates use the delivery date."
    }
  }
}, { date: "2018-08-22" });
assert.strictEqual(plainEvidence.plain,
  "Data as of 22 Aug 2018. Counted deliveries completed before the cutoff. Dates use the delivery date.",
  "the server plain evidence line must be shown without rewriting it");
assert.deepStrictEqual(plainEvidence.primary, [],
  "the plain line is the only primary evidence for a query");

const rendered = AskEvidence.render(fakeDocument(), evidence);
assert.strictEqual(rendered.tagName, "details");
assert.strictEqual(rendered.className, "ask-evidence");
assert.strictEqual(rendered.children[0].textContent, "Evidence",
  "the primary disclosure must be labelled Evidence");
const nestedDetails = findBy(rendered, function (node) {
  return node.tagName === "details";
}, []);
assert.strictEqual(nestedDetails.length, 1,
  "Evidence must be one disclosure, not a second nested audit disclosure");
assert.strictEqual(rendered.textContent.indexOf("Audit details"), -1,
  "the nested audit disclosure must be gone");
["Cutoff rule", "Method detail", "Privacy"].forEach(function (label) {
  assert.strictEqual(rendered.textContent.indexOf(label), -1,
    "the dropped evidence item " + label + " must be absent");
});
const codeNodes = findBy(rendered, function (node) { return node.tagName === "code"; }, []);
assert.strictEqual(codeNodes.length, 1, "exact executed SQL must remain accessible");
assert.strictEqual(codeNodes[0].textContent, screenshotPayload.answer.sql,
  "the code node must keep the exact SQL text");
assert.ok(codeNodes[0].textContent.indexOf("<b>not html</b>") !== -1,
  "SQL must render as text, not parsed markup");

const replayedPayload = AskChats.loadStore((function () {
  const store = memoryStorage();
  AskChats.saveStore(store, [largeChat], "chat-large");
  return store;
})()).chats[0].turns[0].payload;
const replayedEvidence = AskEvidence.build(replayedPayload, { date: "2018-09-01" });
assert.strictEqual(replayedEvidence.plain,
  "Data as of 1 Sep 2018. Counted the rows returned by this query.",
  "saved chats must replay one plain evidence line");
assert.ok(replayedEvidence.audit.some(function (item) {
  return item[0] === "Measure";
}) === true, "replayed evidence keeps the measure note");

const oldAnswerEvidence = AskEvidence.build({
  ok: true,
  as_of: "2018-08-22",
  answer: { kind: "query", title: "Orders", text: "Two orders.", chart: null }
}, { date: "2018-08-22" });
assert.strictEqual(oldAnswerEvidence.plain,
  "Data as of 22 Aug 2018. Counted the rows returned by this query.",
  "older saved answers without provenance still replay safely");
assert.strictEqual(oldAnswerEvidence.sql, null, "missing SQL is handled without inventing evidence");

const noNumericEvidence = AskEvidence.build({
  ok: true,
  as_of: "2018-08-22",
  answer: {
    kind: "query",
    title: "No rows",
    text: "The query returned no rows.",
    row_count: 1,
    chart: null,
    table: { columns: ["month", "orders"], rows: [{ month: null, orders: null }] }
  }
}, { date: "2018-08-22" });
const noNumericMeasure = noNumericEvidence.audit.filter(function (item) {
  return item[0] === "Measure";
})[0];
assert.ok(noNumericMeasure, "evidence must still show a measure entry");
assert.strictEqual(noNumericMeasure[1],
  "Returned columns (no numeric values in the result): month, orders.",
  "a result with no numeric values must not describe dimension columns as numeric measures");

function errorSource(payload) {
  const model = AskEvidence.build(payload, { date: "2018-09-03", asOf: "2018-09-03" });
  return model.primary.filter(function (item) { return item[0] === "Source"; })[0];
}

const planningError = errorSource({
  ok: false,
  as_of: "2018-09-03",
  error: {
    code: "provider_error",
    message: "The provider did not return usable text.",
    stage: "planning",
    finish_reason: "length"
  }
});
assert.ok(planningError, "a planning error still shows a source line");
assert.strictEqual(planningError[1],
  "No query result was produced; no SQL was run for this answer.",
  "a planning error must say no query result was produced");
assert.ok(planningError[1].indexOf("No number was calculated") === -1,
  "a planning error must not claim every error failed to calculate a number");

const synthesisError = errorSource({
  ok: false,
  as_of: "2018-09-03",
  error: {
    code: "provider_error",
    message: "The provider did not return usable text.",
    stage: "synthesis",
    finish_reason: "stop"
  }
});
assert.ok(synthesisError, "a synthesis error still shows a source line");
assert.strictEqual(synthesisError[1],
  "SQL ran, but the answer could not be completed.",
  "a synthesis error must say SQL ran before the answer failed");

const unstagedError = errorSource({
  ok: false,
  as_of: "2018-09-03",
  error: { code: "server_error", message: "The server could not complete the request." }
});
assert.strictEqual(unstagedError[1],
  "The local Ask server did not return an answer.",
  "an error without a stage must stay neutral rather than claim a calculation result");
assert.ok(unstagedError[1].indexOf("No number was calculated") === -1,
  "the old blanket calculation claim must be gone");

console.log("PASS core checks (queue, window, hotspot and chart assertions)");
