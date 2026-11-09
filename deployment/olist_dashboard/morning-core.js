/* Pure as-of queue, window, hotspot and pinned-chart logic for the historical overview. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.MorningCore = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DAY = 86400000;
  const DUE_SOON_DAYS = 3;
  const RECENT_OVERDUE_DAYS = 7;
  const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const METRIC_KEYS = ["orders", "sales", "delivered", "onTime", "onTimeShare", "late", "lateShare"];
  const METRIC_LABELS = {
    orders: "Orders placed",
    sales: "Item price total",
    delivered: "Deliveries completed",
    onTime: "On-time deliveries",
    onTimeShare: "On-time share",
    late: "Late deliveries",
    lateShare: "Late-delivery share"
  };
  const METRIC_METHODS = {
    orders: "Distinct orders with a purchase event in the window.",
    sales: "Sum of item price for orders purchased in the window. Freight is excluded and this is not revenue.",
    delivered: "Orders with a customer delivery event in the window.",
    onTime: "Deliveries whose customer delivery date is on or before the recorded promise.",
    onTimeShare: "On-time deliveries divided by completed deliveries in the window.",
    late: "Deliveries whose customer delivery date is after the recorded promise.",
    lateShare: "Late deliveries divided by completed deliveries in the window."
  };
  const ASK_WINDOWS = [14, 30, 60, 90];
  const CHART_WINDOWS = [14, 30, 60];

  function parseIso(value) {
    const parts = String(value).split("-").map(Number);
    return new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  }

  function isValidIso(value) {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const parts = value.split("-").map(Number);
    const date = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
    return (
      date.getUTCFullYear() === parts[0] &&
      date.getUTCMonth() === parts[1] - 1 &&
      date.getUTCDate() === parts[2]
    );
  }

  function isoDate(date) {
    return date.toISOString().slice(0, 10);
  }

  function addDaysIso(value, count) {
    return isoDate(new Date(parseIso(value).getTime() + count * DAY));
  }

  function daysBetween(laterIso, earlierIso) {
    return Math.round((parseIso(laterIso) - parseIso(earlierIso)) / DAY);
  }

  function monthStartIso(value) {
    return String(value).slice(0, 7) + "-01";
  }

  function addMonthsIso(value, count) {
    const date = parseIso(monthStartIso(value));
    return isoDate(new Date(Date.UTC(
      date.getUTCFullYear(),
      date.getUTCMonth() + count,
      1
    )));
  }

  function monthLabel(iso) {
    const date = parseIso(monthStartIso(iso));
    return MONTH_NAMES[date.getUTCMonth()] + " " + date.getUTCFullYear();
  }

  function bump(object, key, amount) {
    object[key] = (object[key] || 0) + amount;
  }

  function minSample(model) {
    const value = model && model.meta ? model.meta.min_sample : null;
    return typeof value === "number" && isFinite(value) && value > 0 ? value : null;
  }

  function createModel(data) {
    if (!data || !Array.isArray(data.orders)) {
      throw new Error("Overview data did not load.");
    }

    const purchases = new Map();
    data.daily_purchases.forEach(function (row) {
      purchases.set(row[0], { orders: row[1], cents: row[2] });
    });

    const deliveries = new Map();
    data.daily_deliveries.forEach(function (row) {
      deliveries.set(row[0], { delivered: row[1], onTime: row[2], lateDays: row[3] || 0 });
    });

    const overdueNotDelivered = new Map();
    (data.daily_overdue_not_delivered || []).forEach(function (row) {
      overdueNotDelivered.set(row[0], { count: row[1], previous: row[2], due: row[3], carrier: row[4] });
    });

    const reviews = new Map();
    (data.daily_reviews || []).forEach(function (row) {
      const day = row[0];
      const record = {
        visibleDay: row[1],
        late: row[2] === 1,
        count: row[3],
        scoreSum: row[4]
      };
      if (!reviews.has(day)) reviews.set(day, []);
      reviews.get(day).push(record);
    });

    const stateDeliveries = Object.create(null);
    Object.keys(data.state_deliveries).forEach(function (state) {
      const map = new Map();
      data.state_deliveries[state].forEach(function (row) {
        map.set(row[0], { delivered: row[1], late: row[2], onTime: row[1] - row[2] });
      });
      stateDeliveries[state] = map;
    });

    const statePurchases = Object.create(null);
    Object.keys(data.state_purchases || {}).forEach(function (state) {
      const map = new Map();
      data.state_purchases[state].forEach(function (row) {
        map.set(row[0], { orders: row[1], cents: row[2] });
      });
      statePurchases[state] = map;
    });

    const lateByDay = new Map();
    Object.keys(stateDeliveries).forEach(function (state) {
      stateDeliveries[state].forEach(function (row, day) {
        lateByDay.set(day, (lateByDay.get(day) || 0) + row.late);
      });
    });

    const stateNames = data.meta.state_names || {};
    const stateCodes = Object.keys(stateNames).sort();

    const orders = data.orders.map(function (row) {
      return {
        orderId: row[0],
        state: row[1] || "",
        purchaseDay: row[2],
        handoverDay: row[3] || null,
        deliveredDay: row[4] || null,
        promiseDay: row[5] || null,
        cents: row[6] || 0
      };
    });

    return {
      meta: data.meta,
      purchases: purchases,
      deliveries: deliveries,
      overdueNotDelivered: overdueNotDelivered,
      reviews: reviews,
      stateDeliveries: stateDeliveries,
      statePurchases: statePurchases,
      lateByDay: lateByDay,
      stateNames: stateNames,
      stateCodes: stateCodes,
      orders: orders
    };
  }

  function bucketRank(bucket) {
    if (bucket === "soon") return 0;
    if (bucket === "overdue") return 1;
    if (bucket === "older") return 2;
    return 3;
  }

  function scan(model, asOfIso, includeItems) {
    const dueEnd = addDaysIso(asOfIso, DUE_SOON_DAYS);
    const result = {
      open: 0,
      overdue: 0,
      dueSoon: 0,
      overdueHandover: 0,
      overdueNoHandover: 0,
      recentOverdue: 0,
      recentOverdueHandover: 0,
      recentOverdueNoHandover: 0,
      olderOverdue: 0,
      olderOverdueHandover: 0,
      olderOverdueNoHandover: 0,
      dueSoonHandover: 0,
      noPromise: 0,
      openByState: Object.create(null),
      overdueByState: Object.create(null),
      dueSoonByState: Object.create(null),
      recentOverdueByState: Object.create(null),
      recentOverdueHandoverByState: Object.create(null),
      olderOverdueByState: Object.create(null),
      olderOverdueHandoverByState: Object.create(null),
      overdueHandoverByState: Object.create(null),
      dueSoonHandoverByState: Object.create(null),
      noPromiseByState: Object.create(null),
      items: includeItems ? [] : null,
      oldestOverdue: null
    };

    model.orders.forEach(function (order) {
      if (order.purchaseDay >= asOfIso) return;
      if (order.deliveredDay !== null && order.deliveredDay < asOfIso) return;

      result.open += 1;
      bump(result.openByState, order.state || "??", 1);

      const stateKey = order.state || "??";
      const handedOver = order.handoverDay !== null && order.handoverDay < asOfIso;
      let bucket = "open";
      let daysOverdue = null;
      let daysToPromise = null;

      if (order.promiseDay !== null && order.promiseDay < asOfIso) {
        daysOverdue = daysBetween(asOfIso, order.promiseDay);
        bucket = daysOverdue <= RECENT_OVERDUE_DAYS ? "overdue" : "older";
        result.overdue += 1;
        bump(result.overdueByState, stateKey, 1);
        if (handedOver) {
          result.overdueHandover += 1;
          bump(result.overdueHandoverByState, stateKey, 1);
        } else {
          result.overdueNoHandover += 1;
        }
        if (bucket === "overdue") {
          result.recentOverdue += 1;
          bump(result.recentOverdueByState, stateKey, 1);
          if (handedOver) {
            result.recentOverdueHandover += 1;
            bump(result.recentOverdueHandoverByState, stateKey, 1);
          } else {
            result.recentOverdueNoHandover += 1;
          }
        } else {
          result.olderOverdue += 1;
          bump(result.olderOverdueByState, stateKey, 1);
          if (handedOver) {
            result.olderOverdueHandover += 1;
            bump(result.olderOverdueHandoverByState, stateKey, 1);
          } else {
            result.olderOverdueNoHandover += 1;
          }
        }
      } else if (
        order.promiseDay !== null &&
        order.promiseDay >= asOfIso &&
        order.promiseDay <= dueEnd
      ) {
        bucket = "soon";
        daysToPromise = daysBetween(order.promiseDay, asOfIso);
        result.dueSoon += 1;
        bump(result.dueSoonByState, stateKey, 1);
        if (handedOver) {
          result.dueSoonHandover += 1;
          bump(result.dueSoonHandoverByState, stateKey, 1);
        }
      } else if (order.promiseDay === null) {
        result.noPromise += 1;
        bump(result.noPromiseByState, stateKey, 1);
      }

      const item = {
        orderId: order.orderId,
        state: order.state,
        purchaseDay: order.purchaseDay,
        handoverDay: order.handoverDay,
        promiseDay: order.promiseDay,
        cents: order.cents,
        handedOver: handedOver,
        bucket: bucket,
        daysOverdue: daysOverdue,
        daysToPromise: daysToPromise,
        handoverAge: handedOver ? daysBetween(asOfIso, order.handoverDay) : null,
        rank: bucketRank(bucket),
        statusLabel: handedOver
          ? "Handed over, no delivery recorded yet"
          : "No handover recorded by cutoff"
      };

      if (includeItems) result.items.push(item);

      if (daysOverdue !== null) {
        if (
          result.oldestOverdue === null ||
          item.promiseDay < result.oldestOverdue.promiseDay ||
          (item.promiseDay === result.oldestOverdue.promiseDay &&
            item.orderId < result.oldestOverdue.orderId)
        ) {
          result.oldestOverdue = item;
        }
      }
    });

    if (includeItems) {
      result.items.sort(function (a, b) {
        if (a.rank !== b.rank) return a.rank - b.rank;
        const aPromise = a.promiseDay || "9999-12-31";
        const bPromise = b.promiseDay || "9999-12-31";
        if (aPromise !== bPromise) {
          if (a.bucket === "soon") return aPromise < bPromise ? -1 : 1;
          return aPromise > bPromise ? -1 : 1;
        }
        if (a.handedOver !== b.handedOver) return a.handedOver ? -1 : 1;
        return a.orderId < b.orderId ? -1 : a.orderId > b.orderId ? 1 : 0;
      });
    }

    return result;
  }

  function sumDaily(map, startIso, endIso, valueKey) {
    let total = 0;
    const rows = [];
    let day = startIso;
    while (day <= endIso) {
      const row = map.get(day) || null;
      const value = row ? row[valueKey] : 0;
      total += value;
      rows.push({ day: day, row: row });
      day = addDaysIso(day, 1);
    }
    return { total: total, rows: rows };
  }

  function collectHotspots(model, startIso, endIso) {
    const results = [];
    Object.keys(model.stateDeliveries).forEach(function (state) {
      let delivered = 0;
      let late = 0;
      model.stateDeliveries[state].forEach(function (row, day) {
        if (day >= startIso && day <= endIso) {
          delivered += row.delivered;
          late += row.late;
        }
      });
      if (delivered > 0) {
        results.push({ state: state, delivered: delivered, late: late });
      }
    });
    results.sort(function (a, b) {
      return (
        b.late - a.late ||
        b.delivered - a.delivered ||
        a.state.localeCompare(b.state)
      );
    });
    return results;
  }

  function topStates(byState) {
    const rows = Object.keys(byState).map(function (state) {
      return { state: state, count: byState[state] };
    });
    rows.sort(function (a, b) {
      return b.count - a.count || a.state.localeCompare(b.state);
    });
    return rows;
  }

  function compute(model, asOfIso, rangeDays) {
    const days = rangeDays === 30 || rangeDays === 14 || rangeDays === 60 ? rangeDays : 30;
    const endIso = addDaysIso(asOfIso, -1);
    const startIso = addDaysIso(endIso, -(days - 1));
    const hotspotStart = addDaysIso(endIso, -89);
    const current = scan(model, asOfIso, true);
    const previous = scan(model, addDaysIso(asOfIso, -1), false);

    const purchaseSum = sumDaily(model.purchases, startIso, endIso, "orders");
    const salesSum = sumDaily(model.purchases, startIso, endIso, "cents");
    const deliverySum = sumDaily(model.deliveries, startIso, endIso, "delivered");
    const onTimeSum = sumDaily(model.deliveries, startIso, endIso, "onTime");

    const series = purchaseSum.rows.map(function (purchaseRow, index) {
      const deliveryRow = deliverySum.rows[index].row;
      return {
        day: purchaseRow.day,
        orders: purchaseRow.row ? purchaseRow.row.orders : 0,
        cents: purchaseRow.row ? purchaseRow.row.cents : 0,
        delivered: deliveryRow ? deliveryRow.delivered : 0,
        onTime: deliveryRow ? deliveryRow.onTime : 0
      };
    });

    const lastDay = endIso;
    const lastPurchase = model.purchases.get(lastDay) || { orders: 0, cents: 0 };
    const lastDelivery = model.deliveries.get(lastDay) || { delivered: 0, onTime: 0 };

    return {
      asOf: asOfIso,
      latestFullDay: endIso,
      rangeDays: days,
      windowStart: startIso,
      windowEnd: endIso,
      hotspotStart: hotspotStart,
      hotspotEnd: endIso,
      counts: {
        open: current.open,
        overdue: current.overdue,
        dueSoon: current.dueSoon,
        overdueHandover: current.overdueHandover,
        overdueNoHandover: current.overdueNoHandover,
        recentOverdue: current.recentOverdue,
        recentOverdueHandover: current.recentOverdueHandover,
        recentOverdueNoHandover: current.recentOverdueNoHandover,
        olderOverdue: current.olderOverdue,
        olderOverdueHandover: current.olderOverdueHandover,
        olderOverdueNoHandover: current.olderOverdueNoHandover,
        dueSoonHandover: current.dueSoonHandover,
        noPromise: current.noPromise
      },
      openByState: current.openByState,
      overdueByState: current.overdueByState,
      dueSoonByState: current.dueSoonByState,
      recentOverdueByState: current.recentOverdueByState,
      recentOverdueHandoverByState: current.recentOverdueHandoverByState,
      olderOverdueByState: current.olderOverdueByState,
      olderOverdueHandoverByState: current.olderOverdueHandoverByState,
      overdueHandoverByState: current.overdueHandoverByState,
      dueSoonHandoverByState: current.dueSoonHandoverByState,
      noPromiseByState: current.noPromiseByState,
      previousCounts: {
        open: previous.open,
        overdue: previous.overdue,
        dueSoon: previous.dueSoon
      },
      previousOpenByState: previous.openByState,
      previousOverdueByState: previous.overdueByState,
      previousDueSoonByState: previous.dueSoonByState,
      queue: current.items.filter(function (item) {
        return item.bucket === "overdue" || item.bucket === "soon" || item.bucket === "older";
      }),
      priorityQueue: current.items.filter(function (item) {
        return item.bucket === "overdue" || item.bucket === "soon";
      }),
      allOpen: current.items,
      oldestOverdue: current.oldestOverdue,
      topOverdueStates: topStates(current.overdueByState),
      context: {
        orders: purchaseSum.total,
        salesCents: salesSum.total,
        delivered: deliverySum.total,
        onTime: onTimeSum.total,
        onTimeShare: deliverySum.total ? onTimeSum.total / deliverySum.total : null
      },
      lastDay: {
        day: lastDay,
        purchased: lastPurchase.orders,
        cents: lastPurchase.cents,
        delivered: lastDelivery.delivered,
        onTime: lastDelivery.onTime
      },
      series: series,
      hotspots: collectHotspots(model, hotspotStart, endIso)
    };
  }

  function metricLabel(metric) {
    return METRIC_LABELS[metric] || metric;
  }

  function metricMethod(metric) {
    return METRIC_METHODS[metric] || "";
  }

  function defaultViz(metric) {
    return metric === "onTimeShare" || metric === "lateShare" ? "line" : "bar";
  }

  function dayPurchaseRow(model, state, day) {
    if (state) {
      const map = model.statePurchases[state];
      return (map && map.get(day)) || null;
    }
    return model.purchases.get(day) || null;
  }

  function dayDeliveryRow(model, state, day) {
    if (state) {
      const map = model.stateDeliveries[state];
      return (map && map.get(day)) || null;
    }
    return model.deliveries.get(day) || null;
  }

  function dayLate(model, state, day) {
    if (state) {
      const map = model.stateDeliveries[state];
      const row = map && map.get(day);
      return row ? row.late : 0;
    }
    return model.lateByDay.get(day) || 0;
  }

  function metricValue(point, metric) {
    if (!point) return null;
    if (metric === "onTimeShare") {
      return point.delivered ? point.onTime / point.delivered : null;
    }
    if (metric === "lateShare") {
      return point.delivered ? point.late / point.delivered : null;
    }
    return point[metric === "sales" ? "cents" : metric];
  }

  function totalsFor(model, startIso, endIso, state, asOfIso) {
    const totals = {
      orders: 0,
      cents: 0,
      delivered: 0,
      onTime: 0,
      late: 0,
      lateDaysSum: 0,
      onTimeReviewCount: 0,
      onTimeReviewScoreSum: 0,
      lateReviewCount: 0,
      lateReviewScoreSum: 0,
      onTimeShare: null,
      lateShare: null,
      lateDaysAvg: null,
      onTimeReviewAvg: null,
      lateReviewAvg: null
    };
    let day = startIso;
    while (day <= endIso) {
      const purchase = dayPurchaseRow(model, state, day);
      const delivery = dayDeliveryRow(model, state, day);
      if (purchase) {
        totals.orders += purchase.orders;
        totals.cents += purchase.cents;
      }
      if (delivery) {
        totals.delivered += delivery.delivered;
        totals.onTime += delivery.onTime;
        if (!state) totals.lateDaysSum += delivery.lateDays || 0;
      }
      totals.late += dayLate(model, state, day);
      if (!state && asOfIso && model.reviews) {
        const records = model.reviews.get(day);
        if (records) {
          records.forEach(function (record) {
            if (record.visibleDay >= asOfIso) return;
            if (record.late) {
              totals.lateReviewCount += record.count;
              totals.lateReviewScoreSum += record.scoreSum;
            } else {
              totals.onTimeReviewCount += record.count;
              totals.onTimeReviewScoreSum += record.scoreSum;
            }
          });
        }
      }
      day = addDaysIso(day, 1);
    }
    totals.onTimeShare = totals.delivered ? totals.onTime / totals.delivered : null;
    totals.lateShare = totals.delivered ? totals.late / totals.delivered : null;
    totals.lateDaysAvg = !state && totals.late ? totals.lateDaysSum / totals.late : null;
    totals.onTimeReviewAvg = totals.onTimeReviewCount
      ? totals.onTimeReviewScoreSum / totals.onTimeReviewCount
      : null;
    totals.lateReviewAvg = totals.lateReviewCount
      ? totals.lateReviewScoreSum / totals.lateReviewCount
      : null;
    return totals;
  }

  function buildWindow(model, asOfIso, days, state) {
    const requested = ASK_WINDOWS.indexOf(Number(days)) !== -1 ? Number(days) : 30;
    const endIso = addDaysIso(asOfIso, -1);
    const startIso = addDaysIso(endIso, -(requested - 1));
    const previousEnd = addDaysIso(startIso, -1);
    const previousStart = addDaysIso(previousEnd, -(requested - 1));
    const seriesStart =
      model.meta && model.meta.series_start && model.meta.series_start.purchases
        ? model.meta.series_start.purchases
        : null;
    const previousAvailable = !seriesStart || previousStart >= seriesStart;
    const points = [];
    let day = startIso;
    while (day <= endIso) {
      const purchase = dayPurchaseRow(model, state, day);
      const delivery = dayDeliveryRow(model, state, day);
      points.push({
        day: day,
        orders: purchase ? purchase.orders : 0,
        cents: purchase ? purchase.cents : 0,
        delivered: delivery ? delivery.delivered : 0,
        onTime: delivery ? delivery.onTime : 0,
        late: dayLate(model, state, day)
      });
      day = addDaysIso(day, 1);
    }
    return {
      asOf: asOfIso,
      days: requested,
      state: state || null,
      start: startIso,
      end: endIso,
      points: points,
      totals: totalsFor(model, startIso, endIso, state, asOfIso),
      previous: previousAvailable ? totalsFor(model, previousStart, previousEnd, state, asOfIso) : null,
      previousAvailable: previousAvailable,
      previousStart: previousStart,
      previousEnd: previousEnd
    };
  }

  function buildMonthlyWindow(model, asOfIso, months, state) {
    const requested = months === 12 ? 12 : 12;
    const endIso = addDaysIso(asOfIso, -1);
    const endMonth = monthStartIso(endIso);
    const seriesStart = (model.meta && model.meta.series_start) || {};
    const purchaseStart = seriesStart.purchases || null;
    const deliveryStart = seriesStart.deliveries || null;
    let cursor = addMonthsIso(endMonth, -(requested - 1));
    const points = [];
    while (cursor <= endMonth) {
      const nextMonth = addMonthsIso(cursor, 1);
      const monthEnd = addDaysIso(nextMonth, -1);
      const lastDay = monthEnd < endIso ? monthEnd : endIso;
      let orders = 0;
      let cents = 0;
      let delivered = 0;
      let onTime = 0;
      let late = 0;
      let day = cursor;
      while (day <= lastDay) {
        const purchase = dayPurchaseRow(model, state, day);
        const delivery = dayDeliveryRow(model, state, day);
        if (purchase) {
          orders += purchase.orders;
          cents += purchase.cents;
        }
        if (delivery) {
          delivered += delivery.delivered;
          onTime += delivery.onTime;
        }
        late += dayLate(model, state, day);
        day = addDaysIso(day, 1);
      }
      points.push({
        day: cursor,
        month: cursor.slice(0, 7),
        label: monthLabel(cursor),
        orders: orders,
        cents: cents,
        delivered: delivered,
        onTime: onTime,
        late: late,
        ordersNoData: purchaseStart !== null && cursor < purchaseStart,
        deliveriesNoData: deliveryStart !== null && cursor < deliveryStart,
        partial: nextMonth > asOfIso && cursor < asOfIso
      });
      cursor = nextMonth;
    }
    return {
      asOf: asOfIso,
      days: null,
      months: requested,
      state: state || null,
      start: points.length ? points[0].day : endMonth,
      end: endIso,
      granularity: "month",
      points: points
    };
  }

  function buildChartWindow(model, asOfIso, selection, state) {
    if (selection === "12mo" || selection === 12) {
      return buildMonthlyWindow(model, asOfIso, 12, state);
    }
    return buildWindow(model, asOfIso, Number(selection), state);
  }

  function buildStateRanking(model, asOfIso, days, metric) {
    const requested = ASK_WINDOWS.indexOf(Number(days)) !== -1 ? Number(days) : 30;
    const endIso = addDaysIso(asOfIso, -1);
    const startIso = addDaysIso(endIso, -(requested - 1));
    const rows = [];
    model.stateCodes.forEach(function (code) {
      const totals = totalsFor(model, startIso, endIso, code);
      const value = metric === "onTimeShare" ? totals.onTimeShare :
        metric === "sales" ? totals.cents :
        metric === "orders" ? totals.orders :
        metric === "delivered" ? totals.delivered :
        metric === "onTime" ? totals.onTime :
        metric === "late" ? totals.late :
        metric === "lateShare" ? totals.lateShare : null;
      if (value === null) return;
      if ((metric === "orders" || metric === "sales") && totals.orders === 0) return;
      if (metric !== "orders" && metric !== "sales" && totals.delivered === 0) return;
      rows.push({ state: code, totals: totals, value: value });
    });
    rows.sort(function (a, b) {
      return b.value - a.value || b.totals.delivered - a.totals.delivered || a.state.localeCompare(b.state);
    });
    return {
      asOf: asOfIso,
      days: requested,
      start: startIso,
      end: endIso,
      metric: metric,
      rows: rows,
      national: totalsFor(model, startIso, endIso, null)
    };
  }

  function hotspotsForRange(model, asOfIso, startIso, endIso) {
    const rows = [];
    const totals = { late: 0, delivered: 0 };
    Object.keys(model.stateDeliveries).forEach(function (state) {
      let delivered = 0;
      let late = 0;
      model.stateDeliveries[state].forEach(function (row, day) {
        if (day >= startIso && day <= endIso) {
          delivered += row.delivered;
          late += row.late;
        }
      });
      if (delivered > 0) {
        rows.push({ state: state, delivered: delivered, late: late });
        totals.delivered += delivered;
        totals.late += late;
      }
    });
    rows.sort(function (a, b) {
      return b.late - a.late || b.delivered - a.delivered || a.state.localeCompare(b.state);
    });
    totals.rate = totals.delivered ? totals.late / totals.delivered : null;
    return { asOf: asOfIso, start: startIso, end: endIso, rows: rows, totals: totals };
  }

  function buildHotspots(model, asOfIso, days) {
    const requested = ASK_WINDOWS.indexOf(Number(days)) !== -1 ? Number(days) : 90;
    const endIso = addDaysIso(asOfIso, -1);
    const startIso = addDaysIso(endIso, -(requested - 1));
    const result = hotspotsForRange(model, asOfIso, startIso, endIso);
    result.days = requested;
    return result;
  }

  function buildHotspotsWindow(model, asOfIso, selection) {
    if (selection === "12mo" || selection === 12) {
      const endIso = addDaysIso(asOfIso, -1);
      let startIso = addMonthsIso(monthStartIso(endIso), -11);
      const seriesStart = model.meta && model.meta.series_start;
      const statesStart = seriesStart ? seriesStart.states : null;
      if (statesStart && startIso < statesStart) startIso = statesStart;
      const result = hotspotsForRange(model, asOfIso, startIso, endIso);
      result.days = null;
      result.months = 12;
      return result;
    }
    const days = Number(selection);
    return buildHotspots(model, asOfIso, days === 30 || days === 90 ? days : 90);
  }

  function parseTemporalX(value) {
    if (typeof value !== "string") return null;
    const text = value.trim();
    let match = /^(\d{4})-(\d{2})(?:-(\d{2}))?(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$/.exec(text);
    if (!match) {
      match = /^(\d{4})$/.exec(text);
      if (!match) return null;
      const yearOnly = Number(match[1]);
      return Date.UTC(yearOnly, 0, 1);
    }
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = match[3] ? Number(match[3]) : 1;
    const hour = match[4] ? Number(match[4]) : 0;
    const minute = match[5] ? Number(match[5]) : 0;
    const second = match[6] ? Number(match[6]) : 0;
    if (month < 1 || month > 12 || day < 1 || day > 31) return null;
    const date = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
    if (
      date.getUTCFullYear() !== year ||
      date.getUTCMonth() !== month - 1 ||
      date.getUTCDate() !== day
    ) {
      return null;
    }
    return date.getTime();
  }

  function bucketEndExclusive(value) {
    const text = typeof value === "string" ? value.trim() : "";
    const month = /^(\d{4})-(\d{2})$/.exec(text);
    if (month) {
      const year = Number(month[1]);
      const monthIndex = Number(month[2]);
      if (monthIndex < 1 || monthIndex > 12) return null;
      return Date.UTC(year, monthIndex, 1);
    }
    const yearOnly = /^(\d{4})$/.exec(text);
    if (yearOnly) {
      return Date.UTC(Number(yearOnly[1]) + 1, 0, 1);
    }
    return null;
  }

  function partialBucketIndex(points, asOf) {
    if (!Array.isArray(points) || !points.length || !isValidIso(asOf)) return -1;
    const cutoff = parseIso(asOf).getTime();
    let index = -1;
    let latest = -Infinity;
    points.forEach(function (point, pointIndex) {
      const end = bucketEndExclusive(point && point.x);
      if (end === null) return;
      const start = parseTemporalX(point.x);
      if (start !== null && start > latest) {
        latest = start;
        index = pointIndex;
      }
    });
    if (index !== -1) {
      const end = bucketEndExclusive(points[index].x);
      return latest < cutoff && end > cutoff ? index : -1;
    }
    const starts = points.map(function (point) { return parseTemporalX(point && point.x); });
    if (starts.some(function (value) { return value === null; })) return -1;
    let latestIndex = 0;
    starts.forEach(function (start, startIndex) {
      if (start > starts[latestIndex]) latestIndex = startIndex;
    });
    const gaps = [];
    for (let step = 1; step < starts.length; step += 1) {
      const gap = starts[step] - starts[step - 1];
      if (gap > 0) gaps.push(gap);
    }
    if (!gaps.length) return -1;
    const bucketSize = Math.min.apply(null, gaps);
    const start = starts[latestIndex];
    return start < cutoff && start + bucketSize > cutoff ? latestIndex : -1;
  }

  function partialBucketNote(points, index, asOf) {
    if (index < 0 || !Array.isArray(points) || !points[index]) return "";
    const label = String(points[index].x == null ? "" : points[index].x).trim();
    if (!label) return "";
    let period = /^\d{4}$/.test(label) ? "year" : "month";
    if (bucketEndExclusive(label) === null) {
      const gaps = [];
      for (let step = 1; step < points.length; step += 1) {
        const gap = parseTemporalX(points[step].x) - parseTemporalX(points[step - 1].x);
        if (gap > 0) gaps.push(gap);
      }
      const bucketSize = gaps.length ? Math.min.apply(null, gaps) : DAY;
      period = bucketSize >= 6 * DAY ? "week" : bucketSize <= DAY ? "day" : "period";
    }
    return label + " is a partial " + period + " because the cutoff is " + asOf + ".";
  }

  function chartXScale(points) {
    const values = points.map(function (point) { return parseTemporalX(point.x); });
    if (values.length && values.every(function (value) { return value !== null; })) {
      const minimum = Math.min.apply(null, values);
      const maximum = Math.max.apply(null, values);
      return {
        temporal: true,
        values: values,
        at: function (index, left, width) {
          if (values.length === 1 || minimum === maximum) return left + width / 2;
          return left + width * ((values[index] - minimum) / (maximum - minimum));
        }
      };
    }
    return {
      temporal: false,
      values: null,
      at: function (index, left, width) {
        return points.length === 1 ? left + width / 2 : left + (width * index) / (points.length - 1);
      }
    };
  }

  const RATE_NAME_RE = /(?:^|[^a-z0-9])(?:rate|share|pct|percent|percentage|ratio|proportion|%)\s*$/i;
  const MONEY_NAME_RE = /(?:^|_)(?:item_value|price|freight(?:_value)?|payment(?:_value)?)$/i;

  function columnLabel(column, labels) {
    const raw = String(column == null ? "" : column);
    if (labels && typeof labels === "object" && Object.prototype.hasOwnProperty.call(labels, raw)) {
      const supplied = labels[raw];
      if (typeof supplied === "string" && supplied.trim()) return supplied.trim();
    }
    const words = raw.replace(/_/g, " ").replace(/\s+/g, " ").trim();
    if (!words) return "";
    return words.charAt(0).toUpperCase() + words.slice(1).toLowerCase();
  }

  function isLabelColumn(column) {
    const name = String(column == null ? "" : column).trim().toLowerCase();
    return name === "category" || name === "product_category_name" ||
      name === "payment_type" || name === "payment_method" ||
      name.slice(-9) === "_category" || name.slice(-13) === "_payment_type";
  }

  function displayValue(column, value) {
    if (typeof value !== "string" || !isLabelColumn(column)) return value;
    const words = value.replace(/_/g, " ").replace(/\s+/g, " ").trim();
    if (!words) return value;
    return words.charAt(0).toUpperCase() + words.slice(1).toLowerCase();
  }

  function moneyUnit(column) {
    return MONEY_NAME_RE.test(String(column == null ? "" : column).trim()) ? "BRL" : null;
  }

  function formatAskMoney(value, compact) {
    if (typeof value !== "number" || !isFinite(value)) return null;
    if (!compact) {
      return "R$ " + value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
    }
    const magnitude = Math.abs(value);
    if (magnitude >= 1000000) {
      return "R$ " + (value / 1000000).toFixed(2).replace(/\.?0+$/, "") + "m";
    }
    if (magnitude >= 1000) {
      return "R$ " + (value / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    }
    if (magnitude >= 1) return "R$ " + Math.round(value).toLocaleString("en-US");
    return "R$ " + value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }

  function finiteNumbers(values) {
    return (values || []).filter(function (value) {
      return typeof value === "number" && isFinite(value);
    });
  }

  function rateUnit(column, values) {
    if (!RATE_NAME_RE.test(String(column || ""))) return null;
    const finite = finiteNumbers(values);
    if (!finite.length) return null;
    if (finite.some(function (value) { return value < 0; })) return null;
    const maximum = Math.max.apply(null, finite);
    if (maximum <= 1.0000001) return "fraction";
    if (maximum <= 100.0000001) return "percent";
    return null;
  }

  function niceStep(value) {
    if (!(value > 0) || !isFinite(value)) return 1;
    const exponent = Math.floor(Math.log10(value));
    const base = Math.pow(10, exponent);
    const fraction = value / base;
    const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
    return nice * base;
  }

  function stepDecimals(step) {
    if (!(step > 0) || !isFinite(step)) return 0;
    const exponent = Math.floor(Math.log10(step));
    const mantissa = step / Math.pow(10, exponent);
    let decimals = Math.max(0, -exponent);
    if (exponent <= 0 && Math.abs(mantissa - 2.5) < 1e-9) decimals += 1;
    return Math.min(6, decimals);
  }

  function formatAxisValue(value, step) {
    const decimals = stepDecimals(step);
    if (Math.abs(value) >= 1000) {
      return value.toLocaleString("en-US", {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
      });
    }
    return value.toFixed(decimals);
  }

  function formatRateValue(value, unit, fixedDecimals) {
    if (typeof value !== "number" || !isFinite(value)) return null;
    const pct = unit === "percent" ? value : value * 100;
    if (fixedDecimals !== undefined) return pct.toFixed(fixedDecimals) + "%";
    const magnitude = Math.abs(pct);
    const decimals = magnitude >= 1 ? 2 : magnitude >= 0.1 ? 3 : 4;
    return pct.toFixed(decimals).replace(/\.?0+$/, "") + "%";
  }

  // One decimal count for every label in a series: the most decimals any value
  // needs, capped at 1. Series with values under 1% keep their own precision.
  function rateLabelDecimals(values, unit) {
    const finite = values.filter(function (v) { return typeof v === "number" && isFinite(v); });
    const pcts = finite.map(function (v) { return unit === "percent" ? v : v * 100; });
    if (!pcts.length || pcts.some(function (p) { return Math.abs(p) < 1; })) return undefined;
    return pcts.some(function (p) { return Math.abs(p - Math.round(p)) > 1e-6; }) ? 1 : 0;
  }

  function askChartScale(values, column) {
    const finite = finiteNumbers(values);
    if (!finite.length) return null;
    const unit = rateUnit(column, finite);
    if (unit) {
      const percents = finite.map(function (value) {
        return unit === "percent" ? value : value * 100;
      });
      const maximum = Math.max.apply(null, percents);
      const step = niceStep(maximum > 0 ? maximum / 4 : 1);
      const decimals = stepDecimals(step);
      const ticks = [];
      for (let index = 0; index <= 4; index += 1) {
        const tickPercent = step * index;
        ticks.push({
          value: unit === "percent" ? tickPercent : tickPercent / 100,
          label: tickPercent.toFixed(decimals) + "%"
        });
      }
      return {
        kind: "rate",
        unit: unit,
        min: 0,
        max: unit === "percent" ? step * 4 : (step * 4) / 100,
        ticks: ticks,
        axisTitle: "Rate (%)",
        note: ""
      };
    }
    const rawMin = Math.min.apply(null, finite);
    const rawMax = Math.max.apply(null, finite);
    if (moneyUnit(column)) {
      const low = Math.min(0, rawMin);
      const high = rawMax > low ? rawMax : low + 1;
      const step = niceStep((high - low) / 4);
      const min = Math.floor(low / step) * step;
      const max = Math.ceil(high / step) * step;
      const count = Math.max(1, Math.round((max - min) / step));
      const ticks = [];
      for (let index = 0; index <= count; index += 1) {
        const value = min + step * index;
        ticks.push({ value: value, label: formatAskMoney(value, true) });
      }
      return {
        kind: "money",
        unit: "BRL",
        min: min,
        max: max,
        ticks: ticks,
        axisTitle: "R$",
        note: ""
      };
    }
    let min = rawMin;
    let max = rawMax;
    if (min === max) {
      const pad = Math.abs(min || 1) * 0.1;
      min -= pad;
      max += pad;
    } else {
      const pad = (max - min) * 0.08;
      min -= pad;
      max += pad;
    }
    const step = (max - min) / 4;
    const ticks = [];
    for (let index = 0; index <= 4; index += 1) {
      const value = min + (max - min) * (index / 4);
      ticks.push({ value: value, label: formatAxisValue(value, step) });
    }
    return {
      kind: "number",
      unit: null,
      min: min,
      max: max,
      ticks: ticks,
      axisTitle: "",
      note: ""
    };
  }

  function formatAskNumber(value) {
    if (typeof value !== "number" || !isFinite(value)) return null;
    if (Number.isInteger(value)) return value.toLocaleString("en-US");
    const magnitude = Math.abs(value);
    if (magnitude > 0 && magnitude < 0.01) {
      return value.toLocaleString("en-US", { maximumSignificantDigits: 4 });
    }
    return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }

  return {
    createModel: createModel,
    compute: compute,
    minSample: minSample,
    buildWindow: buildWindow,
    buildMonthlyWindow: buildMonthlyWindow,
    buildChartWindow: buildChartWindow,
    buildStateRanking: buildStateRanking,
    buildHotspots: buildHotspots,
    buildHotspotsWindow: buildHotspotsWindow,
    monthStartIso: monthStartIso,
    addMonthsIso: addMonthsIso,
    monthLabel: monthLabel,
    metricValue: metricValue,
    metricLabel: metricLabel,
    metricMethod: metricMethod,
    defaultViz: defaultViz,
    METRIC_KEYS: METRIC_KEYS,
    METRIC_LABELS: METRIC_LABELS,
    ASK_WINDOWS: ASK_WINDOWS,
    CHART_WINDOWS: CHART_WINDOWS,
    scan: scan,
    parseIso: parseIso,
    isValidIso: isValidIso,
    isoDate: isoDate,
    addDaysIso: addDaysIso,
    daysBetween: daysBetween,
    parseTemporalX: parseTemporalX,
    partialBucketIndex: partialBucketIndex,
    partialBucketNote: partialBucketNote,
    chartXScale: chartXScale,
    askChartScale: askChartScale,
    columnLabel: columnLabel,
    displayValue: displayValue,
    moneyUnit: moneyUnit,
    formatAskMoney: formatAskMoney,
    rateUnit: rateUnit,
    formatRateValue: formatRateValue,
    rateLabelDecimals: rateLabelDecimals,
    formatAskNumber: formatAskNumber,
    DUE_SOON_DAYS: DUE_SOON_DAYS,
    RECENT_OVERDUE_DAYS: RECENT_OVERDUE_DAYS
  };
});
