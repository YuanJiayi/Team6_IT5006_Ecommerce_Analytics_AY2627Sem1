/* Local Ask chat history: bounded browser storage and replay-safe chat state. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AskChats = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const STORAGE_KEY = "olistOps.askChats.v1";
  const STORE_VERSION = 1;
  const MAX_CHATS = 30;
  const MAX_STORE_CHARS = 1200000;
  const MAX_TURNS = 40;
  const MAX_HISTORY_MESSAGES = 6;
  const MAX_HISTORY_CHARS = 3000;
  const MAX_HISTORY_MESSAGE_CHARS = 600;
  const MAX_QUESTION_CHARS = 500;
  const MAX_TITLE_CHARS = 80;
  const MAX_TEXT_CHARS = 4000;
  const MAX_SQL_CHARS = 12000;
  const MAX_TABLE_ROWS = 200;
  const MAX_TABLE_COLUMNS = 30;
  const MAX_CELL_CHARS = 500;
  const MAX_LABEL_CHARS = 160;
  const ISO_RE = /^\d{4}-\d{2}-\d{2}$/;

  function clip(value, max) {
    if (value === null || value === undefined) return "";
    return String(value).slice(0, max);
  }

  function finiteNumber(value) {
    return typeof value === "number" && isFinite(value) ? value : null;
  }

  function redactIdentifiers(value) {
    return clip(value, 10000).replace(/\b[0-9a-f]{20,}\b/gi, "[identifier removed]");
  }

  function makeId(prefix) {
    return prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
  }

  function cleanSeedIds(ids) {
    if (!Array.isArray(ids)) return [];
    const result = [];
    ids.forEach(function (id) {
      if (typeof id === "string" && id.length <= 120 && result.indexOf(id) === -1) {
        result.push(id);
      }
    });
    return result;
  }

  function emptyChat(date, id, updatedAt) {
    return {
      id: id || makeId("chat"),
      date: ISO_RE.test(String(date || "")) ? date : null,
      title: "New chat",
      customTitle: false,
      turns: [],
      history: [],
      updatedAt: finiteNumber(updatedAt) || Date.now()
    };
  }

  function deriveTitle(question) {
    const clean = String(question || "").replace(/\s+/g, " ").trim();
    if (!clean) return "New chat";
    return clip(clean, 58) + (clean.length > 58 ? "\u2026" : "");
  }

  function cleanHistory(history) {
    if (!Array.isArray(history)) return [];
    const items = [];
    history.slice(-MAX_HISTORY_MESSAGES).forEach(function (item) {
      if (!item || typeof item !== "object") return;
      const role = item.role;
      const content = typeof item.content === "string" ? redactIdentifiers(item.content).trim() : "";
      if (role !== "user" && role !== "assistant") return;
      if (!content) return;
      items.push({ role: role, content: content.slice(0, MAX_HISTORY_MESSAGE_CHARS) });
    });
    let total = items.reduce(function (sum, item) { return sum + item.content.length; }, 0);
    while (items.length > 1 && total > MAX_HISTORY_CHARS) {
      total -= items[0].content.length;
      items.shift();
    }
    return items;
  }

  function cleanCell(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === "number") return isFinite(value) ? value : null;
    if (typeof value === "boolean") return value;
    return clip(value, MAX_CELL_CHARS);
  }

  function cleanTable(table) {
    if (!table || !Array.isArray(table.columns) || !Array.isArray(table.rows)) return null;
    const columns = [];
    table.columns.slice(0, MAX_TABLE_COLUMNS).forEach(function (column) {
      const label = clip(column, MAX_LABEL_CHARS);
      if (label && columns.indexOf(label) === -1) columns.push(label);
    });
    if (!columns.length) return null;
    const rows = [];
    table.rows.slice(0, MAX_TABLE_ROWS).forEach(function (row) {
      if (!row || typeof row !== "object") return;
      const clean = {};
      columns.forEach(function (column) {
        clean[column] = cleanCell(row[column]);
      });
      rows.push(clean);
    });
    const storedTotal = finiteNumber(table.total_rows);
    const sourceTotal = storedTotal !== null ? Math.max(storedTotal, rows.length) : table.rows.length;
    const result = { columns: columns, rows: rows };
    if (table.truncated === true || sourceTotal > rows.length) {
      result.truncated = true;
      result.total_rows = sourceTotal;
    }
    if (Array.isArray(table.incomplete_rows)) {
      result.incomplete_rows = table.incomplete_rows
        .filter(function (index) { return typeof index === "number" && index >= 0 && index < rows.length; })
        .slice(0, MAX_TABLE_ROWS);
    }
    if (typeof table.incomplete_note === "string" && table.incomplete_note) {
      result.incomplete_note = clip(table.incomplete_note, MAX_TEXT_CHARS);
    }
    if (Array.isArray(table.overall_totals)) {
      result.overall_totals = table.overall_totals.slice(0, 10).map(function (item) {
        return {
          column: clip(item && item.column, MAX_LABEL_CHARS),
          value: finiteNumber(item && item.value)
        };
      }).filter(function (item) {
        return item.column && item.value !== null;
      });
    }
    return result;
  }

  function cleanChart(chart) {
    if (!chart || typeof chart !== "object") return null;
    if (chart.type !== "bar" && chart.type !== "line") return null;
    const x = clip(chart.x, MAX_LABEL_CHARS);
    const y = clip(chart.y, MAX_LABEL_CHARS);
    if (!x || !y) return null;
    const result = { type: chart.type, x: x, y: y };
    if (chart.labels && typeof chart.labels === "object") {
      const labels = {};
      Object.keys(chart.labels).forEach(function (column) {
        const label = clip(chart.labels[column], 60);
        if (label) labels[clip(column, MAX_LABEL_CHARS)] = label;
      });
      if (Object.keys(labels).length) result.labels = labels;
    }
    if (Array.isArray(chart.incomplete)) {
      result.incomplete = chart.incomplete.slice(0, MAX_TABLE_ROWS).map(function (value) {
        return typeof value === "number" ? value : clip(value, MAX_LABEL_CHARS);
      });
    }
    return result;
  }

  const PIN_ID_RE = /^[0-9a-f]{32}$/;

  function cleanPin(pin) {
    if (!pin || typeof pin !== "object") return null;
    if (pin.kind !== "rolling" && pin.kind !== "fixed" && pin.kind !== "all") return null;
    const result = {
      kind: pin.kind,
      label: clip(pin.label, MAX_LABEL_CHARS),
      period: clip(pin.period, MAX_LABEL_CHARS),
      button_label: clip(pin.button_label, MAX_LABEL_CHARS),
      windows: Array.isArray(pin.windows)
        ? pin.windows.filter(function (value) {
          return value === "30d" || value === "90d" || value === "12mo" || value === "ytd";
        })
        : [],
      window_labels: {},
      default_window: null,
      period_start: null,
      period_end: null
    };
    if (pin.window_labels && typeof pin.window_labels === "object") {
      ["30d", "90d", "12mo", "ytd"].forEach(function (key) {
        const value = clip(pin.window_labels[key], 40);
        if (value) result.window_labels[key] = value;
      });
    }
    if (pin.default_window) result.default_window = clip(pin.default_window, 10) || null;
    if (ISO_RE.test(String(pin.period_start || ""))) result.period_start = pin.period_start;
    if (ISO_RE.test(String(pin.period_end || ""))) result.period_end = pin.period_end;
    if (!result.period) delete result.period;
    if (!result.button_label) delete result.button_label;
    return result;
  }

  function cleanProvenance(provenance) {
    if (!provenance || typeof provenance !== "object") return null;
    const clean = {
      as_of: ISO_RE.test(String(provenance.as_of || "")) ? provenance.as_of : null,
      plain: clip(provenance.plain, MAX_TEXT_CHARS),
      cutoff: clip(provenance.cutoff, MAX_TEXT_CHARS),
      source: clip(provenance.source, 900),
      method: clip(provenance.method, MAX_TEXT_CHARS),
      privacy: clip(provenance.privacy, MAX_TEXT_CHARS)
    };
    if (!clean.as_of) delete clean.as_of;
    if (!clean.plain) delete clean.plain;
    if (!clean.cutoff && !clean.source && !clean.method && !clean.privacy) return null;
    return clean;
  }

  function compactPayload(payload) {
    if (!payload || typeof payload !== "object") return null;
    const stored = {};
    const asOf = ISO_RE.test(String(payload.as_of || "")) ? payload.as_of : null;
    if (asOf) stored.as_of = asOf;
    if (payload.ok) {
      const answer = payload.answer && typeof payload.answer === "object" ? payload.answer : {};
      stored.ok = true;
      stored.answer = {
        kind: clip(answer.kind, 40),
        title: clip(answer.title, 200),
        text: clip(answer.text, MAX_TEXT_CHARS)
      };
      const rowCount = finiteNumber(answer.row_count);
      if (rowCount !== null) stored.answer.row_count = rowCount;
      if (answer.sql) stored.answer.sql = clip(answer.sql, MAX_SQL_CHARS);
      if (PIN_ID_RE.test(String(answer.answer_id || ""))) {
        stored.answer.answer_id = answer.answer_id;
      }
      const table = cleanTable(answer.table);
      if (table) stored.answer.table = table;
      const chart = cleanChart(answer.chart);
      if (chart) stored.answer.chart = chart;
      const pin = cleanPin(answer.pin);
      if (pin) stored.answer.pin = pin;
      const provenance = cleanProvenance(answer.provenance);
      if (provenance) stored.answer.provenance = provenance;
      return stored;
    }
    if (payload.refused) {
      const refusal = payload.refusal && typeof payload.refusal === "object" ? payload.refusal : {};
      stored.refused = true;
      stored.refusal = {
        title: clip(refusal.title, 200),
        message: clip(refusal.message, MAX_TEXT_CHARS),
        reason: clip(refusal.reason, 80)
      };
      return stored;
    }
    if (payload.error) {
      const error = payload.error && typeof payload.error === "object" ? payload.error : {};
      stored.error = {
        message: clip(error.message, MAX_TEXT_CHARS),
        code: clip(error.code, 80),
        sql: error.sql ? clip(error.sql, MAX_SQL_CHARS) : ""
      };
      if (error.stage) stored.error.stage = clip(error.stage, 40);
      if (error.finish_reason) stored.error.finish_reason = clip(error.finish_reason, 40);
      return stored;
    }
    return null;
  }

  function nextHistory(history, question, payload) {
    const items = cleanHistory(history);
    items.push({ role: "user", content: clip(question, MAX_HISTORY_MESSAGE_CHARS) });
    let reply = "";
    if (payload && payload.ok && payload.answer) {
      reply = payload.answer.text || payload.answer.title || "";
    } else if (payload && payload.refused && payload.refusal) {
      reply = payload.refusal.message || "";
    } else if (payload && payload.error) {
      reply = payload.error.message || "";
    }
    if (reply) items.push({ role: "assistant", content: clip(reply, MAX_HISTORY_MESSAGE_CHARS) });
    return cleanHistory(items);
  }

  function addTurn(chat, question, turnId, askedAt) {
    const cleanQuestion = clip(redactIdentifiers(String(question || "").trim()), MAX_QUESTION_CHARS);
    if (!cleanQuestion) return null;
    if (!Array.isArray(chat.turns)) chat.turns = [];
    const turn = {
      id: turnId || makeId("turn"),
      question: cleanQuestion,
      askedAt: finiteNumber(askedAt) || Date.now(),
      status: "loading",
      payload: null,
      error: ""
    };
    chat.turns.push(turn);
    if (chat.turns.length > MAX_TURNS) chat.turns = chat.turns.slice(-MAX_TURNS);
    if (!chat.customTitle) chat.title = deriveTitle(cleanQuestion);
    chat.updatedAt = turn.askedAt;
    return turn;
  }

  function commitTurn(chat, turnId, payload, history, updatedAt) {
    if (!chat || !Array.isArray(chat.turns)) return null;
    const turn = chat.turns.filter(function (item) { return item.id === turnId; })[0];
    if (!turn) return null;
    const stored = compactPayload(payload);
    turn.status = stored ? "complete" : "interrupted";
    turn.payload = stored;
    turn.error = stored ? "" : "This answer was interrupted. Ask again.";
    chat.history = cleanHistory(history);
    chat.updatedAt = finiteNumber(updatedAt) || Date.now();
    return turn;
  }

  function renameChat(chats, chatId, title) {
    const chat = chats.filter(function (item) { return item.id === chatId; })[0];
    if (!chat) return null;
    const clean = clip(String(title || "").replace(/\s+/g, " ").trim(), MAX_TITLE_CHARS);
    if (!clean) return null;
    chat.title = clean;
    chat.customTitle = true;
    chat.updatedAt = Date.now();
    return chat;
  }

  function removeChat(chats, chatId) {
    return chats.filter(function (chat) { return chat.id !== chatId; });
  }

  function applyDateChange(chats, activeChatId, date, options) {
    const opts = options || {};
    const active = chats.filter(function (chat) { return chat.id === activeChatId; })[0];
    if (!active) return { chats: chats, activeChatId: activeChatId, createdId: null };
    if (active.date === date) return { chats: chats, activeChatId: activeChatId, createdId: null };
    if (!active.turns.length) {
      active.date = date;
      active.updatedAt = finiteNumber(opts.now) || Date.now();
      return { chats: chats, activeChatId: activeChatId, createdId: null };
    }
    const draft = emptyChat(date, (opts.makeId || makeId)("chat"), finiteNumber(opts.now) || Date.now());
    chats.push(draft);
    return { chats: chats, activeChatId: draft.id, createdId: draft.id };
  }

  function hasPending(chat) {
    return chat.turns.some(function (turn) { return turn.status === "loading"; });
  }

  function seedTimestamp(askedAt, asOf, index, total) {
    const asked = Date.parse(String(askedAt || ""));
    const base = isFinite(asked) ? asked : Date.parse(String(asOf || "") + "T00:00:00Z");
    const safeBase = isFinite(base) ? base : 0;
    return safeBase + Math.max(0, total - index) * 1000;
  }

  function chatFromSeed(seed, asOf, index, total, askedAt) {
    if (!seed || typeof seed !== "object") return null;
    const id = typeof seed.id === "string" ? seed.id : null;
    const question = typeof seed.question === "string" ? seed.question : "";
    if (!id || !question.trim()) return null;
    const payload = compactPayload({ ok: true, as_of: asOf, answer: seed.answer });
    if (!payload || !payload.ok) return null;
    const seedAskedAt = seedTimestamp(askedAt, asOf, index, total);
    return sanitizeChat({
      id: id,
      date: asOf,
      title: deriveTitle(question),
      customTitle: false,
      updatedAt: seedAskedAt,
      history: nextHistory([], question, payload),
      turns: [{
        id: id + "-turn",
        question: question,
        askedAt: seedAskedAt,
        status: "complete",
        payload: payload,
        error: ""
      }]
    });
  }

  function applySeeds(chats, seedPayload, seedsAdded) {
    const current = Array.isArray(chats) ? chats : [];
    const added = cleanSeedIds(seedsAdded);
    const seen = Object.create(null);
    added.forEach(function (id) { seen[id] = true; });
    const payload = seedPayload && typeof seedPayload === "object" ? seedPayload : {};
    const asOf = ISO_RE.test(String(payload.as_of || "")) ? payload.as_of : null;
    if (!asOf || !Array.isArray(payload.chats)) {
      return { chats: current, seedsAdded: added };
    }
    const present = Object.create(null);
    current.forEach(function (chat) {
      if (chat && typeof chat.id === "string") present[chat.id] = true;
    });
    const result = current.slice();
    const total = payload.chats.length;
    const askedAt = typeof payload.asked_at === "string" ? payload.asked_at : "";
    let changed = false;
    payload.chats.forEach(function (seed, index) {
      const id = seed && typeof seed.id === "string" ? seed.id : null;
      if (!id) return;
      // A stored ready-made chat nobody has continued or renamed mirrors the current file.
      const storedIndex = present[id] ? result.findIndex(function (chat) { return chat.id === id; }) : -1;
      const stored = storedIndex >= 0 ? result[storedIndex] : null;
      if (stored && stored.turns.length === 1 && !stored.customTitle) {
        const fresh = chatFromSeed(seed, asOf, index, total, askedAt);
        if (fresh && JSON.stringify(fresh) !== JSON.stringify(stored)) {
          result[storedIndex] = fresh;
          changed = true;
        }
      }
      if (seen[id]) return;
      if (present[id]) {
        seen[id] = true;
        added.push(id);
        return;
      }
      const chat = chatFromSeed(seed, asOf, index, total, askedAt);
      if (!chat) return;
      result.push(chat);
      present[id] = true;
      seen[id] = true;
      added.push(id);
      changed = true;
    });
    return { chats: result, seedsAdded: added, changed: changed };
  }

  function serialize(chats, activeChatId, seedsAdded) {
    return JSON.stringify({
      version: STORE_VERSION,
      activeChatId: activeChatId || null,
      seedsAdded: cleanSeedIds(seedsAdded),
      chats: chats
    });
  }

  function pruneChats(chats, activeChatId, maxChats, maxChars, seedsAdded) {
    const limitChats = Math.max(1, maxChats || MAX_CHATS);
    const limitChars = Math.max(1000, maxChars || MAX_STORE_CHARS);
    let keep = chats.slice().sort(function (a, b) { return b.updatedAt - a.updatedAt; });
    if (keep.length > limitChats) {
      const protectedChats = keep.filter(function (chat) {
        return chat.id === activeChatId || hasPending(chat);
      });
      const others = keep.filter(function (chat) {
        return chat.id !== activeChatId && !hasPending(chat);
      });
      const room = Math.max(0, limitChats - protectedChats.length);
      keep = protectedChats.concat(others.slice(0, room))
        .sort(function (a, b) { return b.updatedAt - a.updatedAt; });
    }
    let json = serialize(keep, activeChatId, seedsAdded);
    let pruned = keep.length !== chats.length;
    while (json.length > limitChars && keep.length > 1) {
      let dropIndex = -1;
      for (let index = keep.length - 1; index >= 0; index -= 1) {
        if (keep[index].id !== activeChatId && !hasPending(keep[index])) {
          dropIndex = index;
          break;
        }
      }
      if (dropIndex === -1) break;
      keep.splice(dropIndex, 1);
      pruned = true;
      json = serialize(keep, activeChatId, seedsAdded);
    }
    if (json.length > limitChars && keep.length === 1) {
      const chat = keep[0];
      while (json.length > limitChars && chat.turns.length > 1) {
        chat.turns.shift();
        pruned = true;
        json = serialize(keep, activeChatId, seedsAdded);
      }
    }
    return { chats: keep, pruned: pruned };
  }

  function saveStore(storage, chats, activeChatId, options) {
    const opts = options || {};
    const seedsAdded = cleanSeedIds(opts.seedsAdded);
    const base = chats.filter(function (chat) {
      return chat.turns.length > 0 || chat.id === activeChatId;
    });
    const working = base.map(function (chat) {
      return JSON.parse(JSON.stringify(chat));
    });
    let pruned = pruneChats(working, activeChatId, opts.maxChats, opts.maxChars, seedsAdded);
    let keep = pruned.chats;
    let attempts = 0;
    while (attempts < 6) {
      attempts += 1;
      try {
        storage.setItem(opts.key || STORAGE_KEY, serialize(keep, activeChatId, seedsAdded));
        return { ok: true, chats: keep, seedsAdded: seedsAdded, pruned: pruned.pruned, error: null };
      } catch (error) {
        if (keep.length > 1) {
          let dropIndex = -1;
          for (let index = keep.length - 1; index >= 0; index -= 1) {
            if (keep[index].id !== activeChatId) {
              dropIndex = index;
              break;
            }
          }
          if (dropIndex !== -1) {
            keep.splice(dropIndex, 1);
            pruned = { chats: keep, pruned: true };
            continue;
          }
        }
        if (keep.length === 1 && keep[0].turns.length > 1) {
          keep[0].turns.shift();
          continue;
        }
        return { ok: false, chats: base, seedsAdded: seedsAdded, pruned: true, error: error };
      }
    }
    return { ok: false, chats: base, seedsAdded: seedsAdded, pruned: true, error: new Error("storage retry limit reached") };
  }

  function sanitizeTurn(raw) {
    if (!raw || typeof raw !== "object") return null;
    const id = typeof raw.id === "string" && raw.id.length <= 120 ? raw.id : makeId("turn");
    const question = clip(raw.question, MAX_QUESTION_CHARS);
    if (!question) return null;
    let status = raw.status === "loading" ? "loading" : "complete";
    let payload = compactPayload(raw.payload);
    let error = clip(raw.error, MAX_TEXT_CHARS);
    if (status === "loading") {
      status = "interrupted";
      payload = null;
      error = error || "This answer was interrupted. Ask again.";
    }
    if (!payload && status !== "interrupted") {
      status = "interrupted";
      error = error || "This answer was interrupted. Ask again.";
    }
    return {
      id: id,
      question: question,
      askedAt: finiteNumber(raw.askedAt) || 0,
      status: status,
      payload: payload,
      error: error
    };
  }

  function sanitizeChat(raw) {
    if (!raw || typeof raw !== "object") return null;
    const id = typeof raw.id === "string" && raw.id.length <= 120 ? raw.id : null;
    const date = typeof raw.date === "string" && ISO_RE.test(raw.date) ? raw.date : null;
    if (!id || !date) return null;
    const chat = emptyChat(date, id, finiteNumber(raw.updatedAt) || 0);
    chat.title = clip(raw.title, MAX_TITLE_CHARS) || "New chat";
    chat.customTitle = raw.customTitle === true;
    chat.history = cleanHistory(raw.history);
    if (Array.isArray(raw.turns)) {
      chat.turns = raw.turns.slice(-MAX_TURNS).map(sanitizeTurn).filter(Boolean);
    }
    if (!chat.customTitle && chat.turns.length) chat.title = deriveTitle(chat.turns[0].question);
    return chat;
  }

  function loadStore(storage) {
    let raw = null;
    try {
      raw = storage.getItem(STORAGE_KEY);
    } catch (error) {
      return { ok: false, chats: [], activeChatId: null, seedsAdded: [], error: error };
    }
    if (!raw) return { ok: true, chats: [], activeChatId: null, seedsAdded: [], error: null };
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.chats)) {
        return { ok: false, chats: [], activeChatId: null, seedsAdded: [], error: new Error("invalid chat store") };
      }
      const chats = parsed.chats.map(sanitizeChat).filter(Boolean);
      const active = chats.filter(function (chat) {
        return chat.id === parsed.activeChatId;
      })[0];
      return {
        ok: true,
        chats: chats,
        activeChatId: active ? active.id : (chats.length ? chats[0].id : null),
        seedsAdded: cleanSeedIds(parsed.seedsAdded),
        error: null
      };
    } catch (error) {
      return { ok: false, chats: [], activeChatId: null, seedsAdded: [], error: error };
    }
  }

  return {
    STORAGE_KEY: STORAGE_KEY,
    MAX_CHATS: MAX_CHATS,
    MAX_STORE_CHARS: MAX_STORE_CHARS,
    MAX_HISTORY_MESSAGES: MAX_HISTORY_MESSAGES,
    emptyChat: emptyChat,
    makeId: makeId,
    deriveTitle: deriveTitle,
    cleanHistory: cleanHistory,
    compactPayload: compactPayload,
    nextHistory: nextHistory,
    redactIdentifiers: redactIdentifiers,
    addTurn: addTurn,
    commitTurn: commitTurn,
    renameChat: renameChat,
    removeChat: removeChat,
    applyDateChange: applyDateChange,
    applySeeds: applySeeds,
    pruneChats: pruneChats,
    saveStore: saveStore,
    loadStore: loadStore
  };
});
