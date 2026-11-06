"""Persistent memory store for the Memory Agent (Level 2, v0).

One SQLite file (stdlib, no server). The design target is PostgreSQL + pgvector; the agent only talks to this class,
so the backend can be swapped without touching the read or write path.

  episodes : every observed turn, verbatim and append-only, with its retrieval embedding
  facts    : durable statements derived from salient turns, with status current / superseded / deleted and validity
             dates (v0 has no Tier-2 extraction yet, so a fact's text is its turn's text)
  audit    : every write decision (op, probabilities, which fact it touched)

Everything is partitioned by `scope` (a user, a project, a team): agents share a store and read by scope.
Embeddings are cached in memory per scope in a growable buffer (capacity doubles), so a write appends in amortised O(1)
and a dense search is one matrix-vector product. `version(scope)` changes on every write to that scope, which lets the
agent reuse a packet for identical concurrent requests safely.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY, scope TEXT NOT NULL, session TEXT, speaker TEXT, role TEXT, ts TEXT,
  text TEXT NOT NULL, source_id TEXT, salience REAL, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS episodes_scope ON episodes(scope, id);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, scope TEXT NOT NULL, episode_id INTEGER NOT NULL REFERENCES episodes(id),
  speaker TEXT, text TEXT NOT NULL, status TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
  superseded_by INTEGER, conflict INTEGER DEFAULT 0, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS facts_scope ON facts(scope, status);
CREATE INDEX IF NOT EXISTS facts_episode ON facts(episode_id);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, at REAL, scope TEXT, action TEXT, detail TEXT);
"""


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()     # writes are serialised; reads see committed rows
        self._ep_cache: dict[str, _Matrix] = {}
        self._fact_cache: dict[str, tuple[list[int], np.ndarray]] = {}
        self._version: dict[str, int] = {}

    def _all(self, sql, args=()):
        """Run a read under the store lock: one SQLite connection is shared by every agent thread."""
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _one(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def version(self, scope) -> int:
        return self._version.get(scope, 0)

    def _bump(self, scope):
        self._version[scope] = self._version.get(scope, 0) + 1

    # ---- episodes
    def add_episode(self, scope, text, vec, *, session=None, speaker=None, role="user", ts=None, source_id=None) -> int:
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO episodes(scope, session, speaker, role, ts, text, source_id, vec) VALUES (?,?,?,?,?,?,?,?)",
                                  (scope, session, speaker, role, ts, text, source_id, _blob(vec)))
            eid = cur.lastrowid
            if scope in self._ep_cache:
                self._ep_cache[scope].append(eid, vec)
            self._bump(scope)
        return eid

    def set_salience(self, eid: int, p: float):
        with self.lock, self.db:
            self.db.execute("UPDATE episodes SET salience=? WHERE id=?", (float(p), eid))

    def episodes(self, ids) -> dict[int, sqlite3.Row]:
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        rows = self._all(f"SELECT id, scope, session, speaker, role, ts, text, source_id, salience FROM episodes "
                         f"WHERE id IN ({','.join('?' * len(ids))})", ids)
        return {r["id"]: r for r in rows}

    def last_episode(self, scope):
        return self._one("SELECT id, speaker, ts, text FROM episodes WHERE scope=? ORDER BY id DESC LIMIT 1", (scope,))

    def episode_matrix(self, scope) -> tuple[list[int], np.ndarray]:
        with self.lock:
            if scope not in self._ep_cache:
                m = _Matrix()
                for r in self._all("SELECT id, vec FROM episodes WHERE scope=? ORDER BY id", (scope,)):
                    m.append(r["id"], np.frombuffer(r["vec"], dtype=np.float32))
                self._ep_cache[scope] = m
            return self._ep_cache[scope].view()

    # ---- facts
    def add_fact(self, scope, episode_id, text, vec, *, speaker=None, valid_from=None, conflict=False) -> int:
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO facts(scope, episode_id, speaker, text, status, valid_from, conflict, vec) VALUES (?,?,?,?,?,?,?,?)",
                                  (scope, episode_id, speaker, text, "current", valid_from, int(conflict), _blob(vec)))
            fid = cur.lastrowid
            self._fact_cache.pop(scope, None)
            self._bump(scope)
        return fid

    def close_fact(self, fid: int, status: str, *, valid_to=None, superseded_by=None):
        """status: superseded (a newer fact replaces it) or deleted (the user retracted it). Rows are never removed."""
        with self.lock, self.db:
            row = self._one("SELECT scope FROM facts WHERE id=?", (fid,))
            self.db.execute("UPDATE facts SET status=?, valid_to=?, superseded_by=? WHERE id=?", (status, valid_to, superseded_by, fid))
            if row:
                self._fact_cache.pop(row["scope"], None)
                self._bump(row["scope"])

    def facts(self, ids) -> dict[int, sqlite3.Row]:
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        rows = self._all(f"SELECT * FROM facts WHERE id IN ({','.join('?' * len(ids))})", ids)
        return {r["id"]: r for r in rows}

    def current_fact_matrix(self, scope) -> tuple[list[int], np.ndarray]:
        with self.lock:
            if scope not in self._fact_cache:
                self._fact_cache[scope] = self._matrix(self._all("SELECT id, vec FROM facts WHERE scope=? AND status='current' ORDER BY id", (scope,)))
            return self._fact_cache[scope]

    def facts_of_episodes(self, episode_ids) -> dict[int, sqlite3.Row]:
        ids = [int(i) for i in episode_ids]
        if not ids:
            return {}
        rows = self._all(f"SELECT * FROM facts WHERE episode_id IN ({','.join('?' * len(ids))})", ids)
        return {r["episode_id"]: r for r in rows}

    def all_facts(self, scope) -> list[sqlite3.Row]:
        return self._all("SELECT * FROM facts WHERE scope=? ORDER BY id", (scope,))

    # ---- audit
    def audit(self, scope, action, **detail):
        with self.lock, self.db:
            self.db.execute("INSERT INTO audit(at, scope, action, detail) VALUES (?,?,?,?)", (time.time(), scope, action, json.dumps(detail)))

    def audit_rows(self, scope=None, actions=None) -> list[dict]:
        q, args = "SELECT at, scope, action, detail FROM audit WHERE 1=1", []
        if scope:
            q += " AND scope=?"; args.append(scope)
        if actions:
            q += f" AND action IN ({','.join('?' * len(actions))})"; args += list(actions)
        return [{"scope": r["scope"], "action": r["action"], **json.loads(r["detail"])} for r in self._all(q + " ORDER BY id", args)]

    def audit_counts(self, scope=None) -> dict[str, int]:
        q = "SELECT action, COUNT(*) n FROM audit" + (" WHERE scope=?" if scope else "") + " GROUP BY action"
        return {r["action"]: r["n"] for r in self._all(q, (scope,) if scope else ())}

    # ---- helpers
    @staticmethod
    def _matrix(rows) -> tuple[list[int], np.ndarray]:
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r["id"] for r in rows], np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])



class _Matrix:
    """Row ids + an embedding matrix with spare capacity; view() returns the filled part without copying."""
    def __init__(self):
        self.ids, self.buf, self.n = [], None, 0

    def append(self, rid, vec):
        v = np.asarray(vec, dtype=np.float32)
        if self.buf is None:
            self.buf = np.zeros((64, v.shape[0]), dtype=np.float32)
        elif self.n == len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
        self.buf[self.n] = v
        self.ids.append(rid)
        self.n += 1

    def view(self) -> tuple[list[int], np.ndarray]:
        if self.buf is None:
            return [], np.zeros((0, 0), dtype=np.float32)
        return self.ids[:self.n], self.buf[:self.n]
