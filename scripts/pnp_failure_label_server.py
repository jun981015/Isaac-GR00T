#!/usr/bin/env python3
"""Small browser labeling UI for RoboCasa PnP failure videos.

The server discovers failed PnP rollout videos under an eval output root,
serves them in a browser, and persists labels to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_PNP_TASKS = [
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
]

LABELS = [
    ("pick_fail", "Pick fail / no stable grasp"),
    ("drop_after_pick", "Dropped after pick"),
    ("place_fail", "Picked but place failed"),
    ("other", "Other failure"),
    ("unclear", "Unclear"),
    ("success_mislabeled", "Looks successful"),
    ("no_count", "No count / bad video"),
]

EP_RE = re.compile(r"ep(?P<ep>\d+)_seed(?P<env_seed>-?\d+).*outcome(?P<outcome>[01])\.mp4$")
LABEL_LOCK = threading.Lock()


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def parse_video_name(path: Path) -> dict[str, int | None]:
    match = EP_RE.search(path.name)
    if not match:
        return {"episode_idx": None, "env_seed": None, "outcome": None}
    return {
        "episode_idx": int(match.group("ep")),
        "env_seed": int(match.group("env_seed")),
        "outcome": int(match.group("outcome")),
    }


def discover_items(eval_root: Path, tasks: list[str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    task_set = set(tasks)
    for video in sorted(eval_root.glob("action_seed_*/*/videos/*.mp4")):
        task = video.parents[1].name
        action_seed = video.parents[2].name.replace("action_seed_", "")
        if task not in task_set:
            continue
        if "outcome0" not in video.name:
            continue
        parsed = parse_video_name(video)
        rel = video.relative_to(eval_root).as_posix()
        episode_idx = parsed["episode_idx"]
        hdf5_path = None
        episode_json = None
        if episode_idx is not None:
            hdf5_candidates = sorted(video.parents[1].glob(f"hdf5/ep{episode_idx:03d}_*.hdf5"))
            json_candidate = video.parents[1] / "episodes" / f"ep{episode_idx:03d}.json"
            if hdf5_candidates:
                hdf5_path = hdf5_candidates[0].relative_to(eval_root).as_posix()
            if json_candidate.exists():
                episode_json = json_candidate.relative_to(eval_root).as_posix()
        items.append(
            {
                "id": rel,
                "video": rel,
                "task": task,
                "action_seed": action_seed,
                "episode_idx": episode_idx,
                "env_seed": parsed["env_seed"],
                "hdf5": hdf5_path,
                "episode_json": episode_json,
            }
        )
    return items


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RoboCasa PnP Failure Labeler</title>
  <style>
    :root { --bg:#101113; --panel:#181b1f; --text:#e8e2d6; --muted:#9a9489; --line:#30343a; --accent:#e0a23b; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 14px 18px; border-bottom: 1px solid var(--line); display: flex; gap: 16px; align-items: center; }
    header h1 { font-size: 18px; margin: 0; }
    header span { color: var(--muted); font-size: 13px; }
    main { display: grid; grid-template-columns: 360px 1fr; height: calc(100vh - 54px); }
    aside { border-right: 1px solid var(--line); overflow: auto; background: #121417; }
    .filters { padding: 12px; display: grid; gap: 8px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: #121417; z-index: 2; }
    input, select, textarea { width: 100%; background: #0d0f11; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
    .item { padding: 10px 12px; border-bottom: 1px solid #23262b; cursor: pointer; }
    .item:hover { background: #1a1d22; }
    .item.active { background: #2a2417; border-left: 4px solid var(--accent); }
    .item .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .item .label { color: var(--accent); font-size: 12px; margin-top: 4px; }
    section { padding: 18px; overflow: auto; }
    video { width: 100%; max-height: 70vh; background: #000; border: 1px solid var(--line); border-radius: 12px; }
    .title { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 12px; }
    .title h2 { margin: 0; font-size: 20px; }
    .path { color: var(--muted); font-size: 12px; word-break: break-all; margin-top: 4px; }
    .buttons { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 14px 0; }
    button { background: #22262c; color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; cursor: pointer; font-weight: 700; }
    button:hover { border-color: var(--accent); }
    button.selected { background: var(--accent); color: #1b1304; border-color: var(--accent); }
    .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 12px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
    .card h3 { margin: 0 0 8px 0; font-size: 13px; color: var(--muted); }
    .help { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .stats { color: var(--muted); font-size: 12px; line-height: 1.5; padding: 8px; border: 1px solid var(--line); border-radius: 8px; background: #0d0f11; }
    .taskStats { width: 100%; border-collapse: collapse; margin-top: 8px; }
    .taskStats th, .taskStats td { text-align: left; padding: 3px 4px; border-bottom: 1px solid #23262b; font-size: 11px; }
    .taskStats th { color: var(--text); font-weight: 700; }
    .taskStats td.num { text-align: right; font-variant-numeric: tabular-nums; }
    a { color: #f0b24a; }
  </style>
</head>
<body>
  <header>
    <h1>RoboCasa PnP Failure Labeler</h1>
    <span id="count">loading...</span>
    <span>Keys: 1-7 label, J/K move, Space play/pause</span>
  </header>
  <main>
    <aside>
      <div class="filters">
        <input id="search" placeholder="filter task / ep / label" />
        <select id="taskFilter"><option value="">all tasks</option></select>
        <select id="seedFilter"><option value="">all action seeds</option></select>
        <select id="statusFilter">
          <option value="">all labels</option>
          <option value="unlabeled">unlabeled</option>
        </select>
        <div class="stats" id="filterStats">stats loading...</div>
      </div>
      <div id="list"></div>
    </aside>
    <section>
      <div class="title">
        <div>
          <h2 id="headline">Select a video</h2>
          <div class="path" id="subhead"></div>
        </div>
        <button id="saveBtn">Save note</button>
      </div>
      <video id="video" controls preload="metadata"></video>
      <div class="buttons" id="labelButtons"></div>
      <textarea id="note" rows="4" placeholder="notes: e.g. grasped but slipped at t=5s"></textarea>
      <div class="row">
        <div class="card"><h3>Paths</h3><div class="help" id="paths"></div></div>
        <div class="card"><h3>Current Label</h3><div class="help" id="currentLabel"></div></div>
        <div class="card"><h3>Suggested taxonomy</h3><div class="help">pick_fail: no stable grasp<br/>drop_after_pick: grasped then lost object<br/>place_fail: picked object but final placement failed<br/>success_mislabeled: outcome0 but looks successful<br/>no_count: corrupted / unusable video</div></div>
      </div>
    </section>
  </main>
  <script>
    const labels = __LABELS__;
    let items = [];
    let labelMap = {};
    let filtered = [];
    let idx = 0;

    const el = (id) => document.getElementById(id);
    const itemLabel = (item) => labelMap[item.id]?.label || "";

    async function load() {
      const res = await fetch('/api/items');
      const data = await res.json();
      items = data.items;
      labelMap = data.labels || {};
      labels.forEach(([value, text], i) => {
        const b = document.createElement('button');
        b.textContent = `${i + 1}. ${text}`;
        b.onclick = () => setLabel(value);
        b.dataset.value = value;
        el('labelButtons').appendChild(b);
      });
      const tasks = [...new Set(items.map(x => x.task))].sort();
      tasks.forEach(t => {
        const o = document.createElement('option');
        o.value = t; o.textContent = t; el('taskFilter').appendChild(o);
      });
      const seeds = [...new Set(items.map(x => x.action_seed))].sort((a,b) => Number(a) - Number(b));
      seeds.forEach(s => {
        const o = document.createElement('option');
        o.value = s; o.textContent = `action_seed_${s}`; el('seedFilter').appendChild(o);
      });
      ['search','taskFilter','seedFilter','statusFilter'].forEach(id => el(id).oninput = renderList);
      el('saveBtn').onclick = () => saveCurrent();
      renderList();
      select(0);
    }

    function renderList() {
      const q = el('search').value.toLowerCase();
      const task = el('taskFilter').value;
      const seed = el('seedFilter').value;
      const status = el('statusFilter').value;
      filtered = items.filter(item => {
        const label = itemLabel(item);
        const text = `${item.task} ep${item.episode_idx} seed${item.action_seed} ${label}`.toLowerCase();
        if (task && item.task !== task) return false;
        if (seed && item.action_seed !== seed) return false;
        if (status === 'unlabeled' && label) return false;
        return !q || text.includes(q);
      });
      const list = el('list');
      list.innerHTML = '';
      filtered.forEach((item, i) => {
        const d = document.createElement('div');
        d.className = 'item' + (i === idx ? ' active' : '');
        d.onclick = () => select(i);
        d.innerHTML = `<b>${item.task}</b> ep${String(item.episode_idx).padStart(3,'0')}<div class="meta">action_seed=${item.action_seed} env_seed=${item.env_seed}</div><div class="label">${itemLabel(item) || 'unlabeled'}</div>`;
        list.appendChild(d);
      });
      const labeled = items.filter(x => itemLabel(x)).length;
      el('count').textContent = `${labeled}/${items.length} labeled, ${filtered.length} shown`;
      const filteredLabeled = filtered.filter(x => itemLabel(x)).length;
      const byLabel = {};
      filtered.forEach(x => {
        const label = itemLabel(x) || 'unlabeled';
        byLabel[label] = (byLabel[label] || 0) + 1;
      });
      const labelText = Object.keys(byLabel).sort().map(k => `${k}: ${byLabel[k]}`).join('<br/>');
      const taskStatsItems = items.filter(item => {
        if (seed && item.action_seed !== seed) return false;
        return true;
      });
      const byTask = {};
      taskStatsItems.forEach(item => {
        if (!byTask[item.task]) byTask[item.task] = { total: 0, labeled: 0 };
        byTask[item.task].total += 1;
        if (itemLabel(item)) byTask[item.task].labeled += 1;
      });
      const taskRows = Object.keys(byTask).sort().map(t => {
        const s = byTask[t];
        return `<tr><td>${t}</td><td class="num">${s.labeled}/${s.total}</td><td class="num">${s.total - s.labeled}</td></tr>`;
      }).join('');
      const taskTable = `<table class="taskStats"><thead><tr><th>task</th><th>labeled/fail</th><th>left</th></tr></thead><tbody>${taskRows}</tbody></table>`;
      el('filterStats').innerHTML = `shown labeled: ${filteredLabeled}/${filtered.length}<br/>total labeled: ${labeled}/${items.length}<hr/>${labelText}<hr/>task progress${seed ? ` (action_seed_${seed})` : ''}${taskTable}`;
    }

    function select(i) {
      if (!filtered.length) return;
      idx = Math.max(0, Math.min(i, filtered.length - 1));
      const item = filtered[idx];
      el('headline').textContent = `${item.task} ep${String(item.episode_idx).padStart(3,'0')} outcome0`;
      el('subhead').textContent = `action_seed=${item.action_seed}, env_seed=${item.env_seed}`;
      el('video').src = `/media/${encodeURIComponent(item.id)}`;
      el('note').value = labelMap[item.id]?.note || '';
      el('paths').innerHTML = `video: ${item.video}<br/>hdf5: ${item.hdf5 || ''}<br/>episode: ${item.episode_json || ''}`;
      updateButtons();
      renderList();
    }

    function updateButtons() {
      const item = filtered[idx];
      const cur = item ? itemLabel(item) : "";
      document.querySelectorAll('#labelButtons button').forEach(b => b.classList.toggle('selected', b.dataset.value === cur));
      el('currentLabel').textContent = cur || 'unlabeled';
    }

    async function setLabel(label) {
      const item = filtered[idx];
      if (!item) return;
      labelMap[item.id] = { ...(labelMap[item.id] || {}), label, note: el('note').value };
      await saveCurrent();
      if (idx < filtered.length - 1) select(idx + 1);
    }

    async function saveCurrent() {
      const item = filtered[idx];
      if (!item) return;
      const payload = { id: item.id, label: itemLabel(item), note: el('note').value };
      const res = await fetch('/api/label', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (!res.ok) alert('save failed');
      const data = await res.json();
      labelMap = data.labels;
      updateButtons();
      renderList();
    }

    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
      if (e.key >= '1' && e.key <= '7') setLabel(labels[Number(e.key)-1][0]);
      if (e.key === 'j' || e.key === 'J') select(idx + 1);
      if (e.key === 'k' || e.key === 'K') select(idx - 1);
      if (e.key === ' ') { e.preventDefault(); const v=el('video'); v.paused ? v.play() : v.pause(); }
    });

    load();
  </script>
</body>
</html>
"""


class LabelServer(ThreadingHTTPServer):
    def __init__(self, addr, handler, eval_root: Path, labels_file: Path, tasks: list[str]):
        self.eval_root = eval_root.resolve()
        self.labels_file = labels_file
        self.tasks = tasks
        self.items = discover_items(self.eval_root, tasks)
        super().__init__(addr, handler)


class Handler(BaseHTTPRequestHandler):
    server: LabelServer

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.serve_html()
        if parsed.path == "/api/items":
            labels = load_json(self.server.labels_file, {})
            return json_response(self, {"items": self.server.items, "labels": labels})
        if parsed.path.startswith("/media/"):
            rel = parsed.path[len("/media/") :]
            return self.serve_media(rel)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        item_id = str(payload["id"])
        valid_ids = {str(item["id"]) for item in self.server.items}
        if item_id not in valid_ids:
            return json_response(self, {"error": "unknown id"}, status=400)
        with LABEL_LOCK:
            labels = load_json(self.server.labels_file, {})
            labels[item_id] = {
                "label": str(payload.get("label") or ""),
                "note": str(payload.get("note") or ""),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_write_json(self.server.labels_file, labels)
        json_response(self, {"ok": True, "labels": labels})

    def serve_html(self) -> None:
        labels_js = json.dumps(LABELS, ensure_ascii=False)
        body = HTML.replace("__LABELS__", labels_js).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_media(self, rel: str) -> None:
        from urllib.parse import unquote

        rel = unquote(rel)
        path = (self.server.eval_root / rel).resolve()
        if not str(path).startswith(str(self.server.eval_root)) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\\d+)-(\\d*)", range_header)
            if m:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
        end = min(end, size - 1)
        status = HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18777)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_PNP_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    labels_file = args.labels_file or (eval_root / "pnp_failure_labels.json")
    server = LabelServer((args.host, args.port), Handler, eval_root, labels_file, args.tasks)
    print(f"Serving {len(server.items)} failed PnP videos")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Labels: {labels_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
