#!/usr/bin/env python3
"""Lightweight HTTP server for serving rendered TikZ outputs and plans."""

import json
import signal
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from config import get_host, get_port

CACHE_DIR = Path.home() / ".cache" / "tikz-skill"
RENDERS_DIR = CACHE_DIR / "renders"
STATE_FILE = CACHE_DIR / "server.json"
PLANS_DIR = Path.home() / ".claude" / "plans"
TEMP_MD_DIR = CACHE_DIR / "temp"


def _safe_path(base_dir, name):
    """Join base_dir/name, block '..' only. Follows symlinks."""
    if ".." in Path(name).parts:
        return None
    return base_dir / name


class TikZHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RENDERS_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/healthz":
            self._serve_json(200, {"status": "ok"})
        elif self.path == "/":
            self._serve_index()
        elif self.path == "/settings":
            self._serve_settings()
        elif self.path.startswith("/api/config"):
            self._api_config_get()
        elif self.path.startswith("/api/mtime"):
            self._api_mtime()
        elif self.path.startswith("/plans/"):
            self._serve_plan()
        elif self.path.startswith("/temp"):
            self._serve_temp()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/config":
            self._api_config_post()
        else:
            self.send_error(405)

    def _api_config_get(self):
        from config import get_all_config
        self._serve_json(200, get_all_config())

    def _api_config_post(self):
        from config import save_config, get_all_config
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            save_config(data)
            self._serve_json(200, get_all_config())
        except Exception as e:
            self._serve_json(400, {"error": str(e)})

    def _api_mtime(self):
        qs = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(qs)
        file_path = urllib.parse.unquote(params.get("file", [""])[0])
        mtime = 0
        if file_path.startswith("plans/"):
            p = PLANS_DIR / file_path[len("plans/"):]
            if p.exists():
                mtime = p.stat().st_mtime
        elif file_path.startswith("temp/"):
            p = TEMP_MD_DIR / file_path[len("temp/"):]
            if p.exists():
                mtime = p.stat().st_mtime
        elif file_path:
            p = RENDERS_DIR / file_path
            if p.exists():
                mtime = p.stat().st_mtime
        self._serve_json(200, {"mtime": mtime})

    def _serve_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_plan(self):
        """Serve a plan markdown file rendered as HTML."""
        name = urllib.parse.unquote(self.path[len("/plans/"):])
        self._serve_markdown(PLANS_DIR, name)

    def _serve_temp(self):
        """Serve temp dir: directory index, .md rendering, or static file."""
        raw = self.path[len("/temp"):].lstrip("/")
        name = urllib.parse.unquote(raw).strip("/")
        target = _safe_path(TEMP_MD_DIR, name) if name else TEMP_MD_DIR
        if not target or not target.exists():
            self.send_error(404)
            return
        if target.is_dir():
            self._serve_dir_index(target, name)
        elif target.suffix == ".md":
            self._serve_markdown(target.parent, target.name)
        else:
            # Serve static file
            import mimetypes
            ct = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _serve_dir_index(self, dir_path, rel_prefix):
        """Auto-generate recursive directory index for temp."""
        items = sorted(dir_path.iterdir(), key=lambda f: (f.is_file(), f.name.lower()))
        rows = ""
        for item in items:
            rel = f"{rel_prefix}/{item.name}" if rel_prefix else item.name
            href = f"/temp/{urllib.parse.quote(rel)}"
            if item.is_dir():
                icon, suffix = "\U0001F4C1", "/"
                rows += f'<tr><td>{icon}</td><td><a href="{href}/">{item.name}/</a></td><td>-</td></tr>\n'
            else:
                icon = "\U0001F4D6" if item.suffix == ".md" else "\U0001F4C4"
                size = item.stat().st_size
                size_str = f"{size}" if size < 1024 else f"{size/1024:.1f}K" if size < 1048576 else f"{size/1048576:.1f}M"
                rows += f'<tr><td>{icon}</td><td><a href="{href}">{item.name}</a></td><td>{size_str}</td></tr>\n'

        parent_link = ""
        if rel_prefix:
            parent = "/temp/" + "/".join(rel_prefix.split("/")[:-1]) if "/" in rel_prefix else "/temp/"
            parent_link = f'<a href="{parent}">&larr; 上级目录</a>'

        title = f"/{rel_prefix}" if rel_prefix else "/temp"
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f5f5f5; color: #333; }}
h1 {{ font-size: 1.3rem; margin-bottom: 1rem; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
table {{ border-collapse: collapse; width: 100%; max-width: 700px; }}
td {{ padding: .4rem .8rem; border-bottom: 1px solid #eee; }}
td:first-child {{ width: 2rem; text-align: center; }}
td:last-child {{ width: 5rem; text-align: right; color: #888; font-size: .85rem; }}
.nav {{ margin-bottom: 1rem; font-size: .9rem; }}
</style></head><body>
<div class="nav"><a href="/">&larr; 首页</a> {parent_link}</div>
<h1>{title}</h1>
<table>{rows}</table>
</body></html>"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_markdown(self, base_dir, name):
        """Serve a markdown file from base_dir rendered as HTML."""
        md_file = base_dir / name
        if not md_file.exists() or md_file.suffix != ".md":
            self.send_error(404, "File not found")
            return

        import base64 as _b64
        from config import get_default_view_mode
        md_content = md_file.read_text()
        md_b64 = _b64.b64encode(md_content.encode()).decode()
        title = md_file.stem.replace("-", " ").title()
        default_mode = get_default_view_mode()

        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

/* === E-ink mode (default) === */
body {{ font-family: Georgia, 'Noto Serif SC', serif; background: #f8f8f8; color: #000;
  height: 100vh; overflow: hidden; display: flex; flex-direction: column; }}
.topbar {{ display: flex; align-items: center; justify-content: space-between;
  padding: .6rem 1.5rem; border-bottom: 2px solid #888; background: #e8e8e8;
  font-size: .85rem; color: #333; flex-shrink: 0; }}
.topbar a {{ color: #000; text-decoration: none; font-weight: 600; }}
.topbar a:hover {{ text-decoration: underline; }}
.topbar .title {{ font-weight: bold; color: #000; }}
.mode-btn {{ background: none; border: 2px solid #666; border-radius: 4px;
  padding: .2rem .7rem; cursor: pointer; font-size: .8rem; color: #000; font-weight: 600; }}
.mode-btn:hover {{ background: #d0d0d0; }}
.reader {{ flex: 1; display: flex; overflow: hidden; position: relative; cursor: default;
  -webkit-user-select: none; user-select: none; }}
.page-zone {{ position: absolute; top: 0; height: 100%; z-index: 2; }}
.page-zone.left {{ left: 0; width: 35%; cursor: w-resize; }}
.page-zone.right {{ right: 0; width: 35%; cursor: e-resize; }}
.page-content {{ max-width: 720px; margin: 0 auto; padding: 2.5rem 2rem;
  overflow: hidden; line-height: 2; font-size: 1.35rem; font-weight: 600; letter-spacing: .02em; }}
.page-content h1 {{ font-size: 1.9rem; margin: 1.8rem 0 1rem; border-bottom: 2px solid #666; padding-bottom: .4rem; font-weight: 800; }}
.page-content h2 {{ font-size: 1.6rem; margin: 1.5rem 0 .8rem; font-weight: 700; }}
.page-content h3 {{ font-size: 1.4rem; margin: 1.2rem 0 .6rem; font-weight: 700; }}
.page-content p {{ margin: .8rem 0; }}
.page-content ul, .page-content ol {{ margin: .8rem 0; padding-left: 1.8rem; }}
.page-content li {{ margin: .3rem 0; }}
.page-content code {{ font-family: 'SF Mono', Menlo, monospace; font-size: .9em;
  background: #d0d0d0; padding: .15em .4em; border-radius: 3px; color: #000; }}
.page-content pre {{ background: #d8d8d8; padding: 1rem; border-radius: 4px;
  overflow: hidden; margin: 1rem 0; border: 2px solid #888;
  white-space: pre-wrap; word-break: break-all; }}
.page-content pre code {{ background: none; padding: 0;
  white-space: pre-wrap; word-break: break-all; }}
.table-wrap {{ margin: 1rem 0; }}
.page-content table {{ border-collapse: collapse; width: 100%; display: block; }}
.page-content th, .page-content td {{ word-break: break-word; font-size: .9em; padding: .4rem .5rem;
  border: 2px solid #888; text-align: left; }}
.page-content th {{ background: #d0d0d0; font-weight: 800; }}
.page-content blockquote {{ border-left: 4px solid #444; padding-left: 1rem; color: #222; margin: 1rem 0; font-style: italic; }}
.page-content a {{ color: #000; text-decoration: underline; font-weight: 600; }}
.page-content img {{ display: block; margin: 1rem auto; width: 100%; height: auto; }}

/* === Normal mode === */
body.normal {{ height: auto; overflow: auto; background: #fff;
  font-family: system-ui, -apple-system, sans-serif; }}
body.normal .reader {{ overflow: visible; }}
body.normal .page-zone {{ display: none; }}
body.normal .page-content {{ overflow: visible; padding: 1.5rem 2rem; line-height: 1.7; max-width: 100%; }}
body.normal .page-content code {{ background: #f3f4f6; }}
body.normal .page-content pre {{ background: #f3f4f6; border-color: #e5e7eb; }}
body.normal .reader {{ -webkit-user-select: auto; user-select: auto; cursor: auto; }}
body.normal .page-content a {{ color: #2563eb; }}
body.normal .page-content th {{ background: #f9fafb; }}
body.normal .page-content blockquote {{ border-left-color: #d1d5db; color: #6b7280; }}

/* === Eye-care mode (护眼模式) === */
body.eyecare {{ height: auto; overflow: auto; background: #f5eddc; color: #4a3f35;
  font-family: Georgia, 'Noto Serif SC', serif; }}
body.eyecare .topbar {{ background: #e8dcc8; border-bottom-color: #c8b899; color: #5b4636; }}
body.eyecare .topbar a {{ color: #5b4636; }}
body.eyecare .mode-btn {{ border-color: #b0a08a; color: #5b4636; }}
body.eyecare .mode-btn:hover {{ background: #ddd0b8; }}
body.eyecare .reader {{ overflow: visible; -webkit-user-select: auto; user-select: auto; cursor: auto; }}
body.eyecare .page-zone {{ display: none; }}
body.eyecare .page-content {{ overflow: visible; padding: 1.5rem 2rem; line-height: 1.9;
  max-width: 100%; font-size: 1.2rem; }}
body.eyecare .page-content code {{ background: #e8dcc8; color: #5b4636; }}
body.eyecare .page-content pre {{ background: #ede3d0; border-color: #c8b899; }}
body.eyecare .page-content a {{ color: #8b6914; }}
body.eyecare .page-content th {{ background: #e8dcc8; }}
body.eyecare .page-content td, body.eyecare .page-content th {{ border-color: #c8b899; }}
body.eyecare .page-content blockquote {{ border-left-color: #b0a08a; color: #6b5d4d; }}
body.eyecare .page-content h1 {{ border-bottom-color: #b0a08a; }}
body.eyecare .page-content img {{ filter: brightness(.95) sepia(.1); }}

/* FAB group (normal & eyecare modes) */
.fab-group {{ display: none; }}
body.normal .fab-group, body.eyecare .fab-group {{ display: flex; flex-direction: column; gap: .5rem;
  position: fixed; right: 1.5rem; bottom: 2rem; z-index: 100; }}
.toc-fab {{ display: none; }}
body.normal .toc-fab, body.eyecare .toc-fab {{ display: flex; width: 3rem; height: 3rem;
  border-radius: 50%; background: #2563eb; color: #fff; border: none; cursor: pointer;
  align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 2px 8px rgba(0,0,0,.2);
  transition: background .2s; }}
body.normal .toc-fab:hover {{ background: #1d4ed8; }}
body.eyecare .toc-fab {{ background: #8b6914; }}
body.eyecare .toc-fab:hover {{ background: #735812; }}
.toc-sidebar {{ position: fixed; top: 0; right: -320px; width: 300px; height: 100vh;
  background: #fff; box-shadow: -2px 0 12px rgba(0,0,0,.1); z-index: 99;
  transition: right .25s ease; padding: 1rem 0; overflow-y: auto; }}
.toc-sidebar.open {{ right: 0; }}
.toc-sidebar .toc-header {{ display: flex; align-items: center; justify-content: space-between;
  padding: 0 1.2rem .8rem; border-bottom: 1px solid #eee; margin-bottom: .5rem; }}
.toc-sidebar .toc-header span {{ font-weight: 700; font-size: 1rem; color: #333; }}
.toc-sidebar .toc-close {{ background: none; border: none; font-size: 1.3rem; cursor: pointer; color: #999; }}
.toc-sidebar .toc-close:hover {{ color: #333; }}
.toc-sidebar ul {{ list-style: none; padding: 0; margin: 0; }}
.toc-sidebar li {{ padding: .4rem 1.2rem; font-size: .9rem; }}
.toc-sidebar li.h3 {{ padding-left: 2.2rem; font-size: .85rem; }}
.toc-sidebar li a {{ color: #555; text-decoration: none; display: block; }}
.toc-sidebar li a:hover {{ color: #2563eb; }}
.toc-sidebar li.active a {{ color: #2563eb; font-weight: 600; }}
.toc-overlay {{ display: none; position: fixed; inset: 0; z-index: 98; }}
.toc-overlay.open {{ display: block; }}

/* PDF export button */
.pdf-fab {{ display: none; }}
body.normal .pdf-fab, body.eyecare .pdf-fab {{ display: flex; width: 3rem; height: 3rem;
  border-radius: 50%; background: #16a34a; color: #fff; border: none; cursor: pointer;
  align-items: center; justify-content: center; font-size: .75rem; font-weight: 700;
  box-shadow: 0 2px 8px rgba(0,0,0,.2); transition: background .2s; letter-spacing: .05em; }}
body.normal .pdf-fab:hover {{ background: #15803d; }}

/* Print styles */
@media print {{
  .topbar, .fab-group, .toc-sidebar, .toc-overlay {{ display: none !important; }}
  body, body.normal {{ height: auto; overflow: visible; background: #fff; margin: 0; }}
  .reader {{ overflow: visible; display: block; }}
  .page-content {{ max-width: 100%; padding: 0; overflow: visible; font-size: 12pt; line-height: 1.6; }}
  .page-content pre {{ white-space: pre-wrap; word-break: break-all; border: 1px solid #ccc; }}
  .page-content table {{ page-break-inside: auto; }}
  .page-content tr {{ page-break-inside: avoid; }}
  .page-content h1, .page-content h2, .page-content h3 {{ page-break-after: avoid; }}
  .page-content pre, .page-content blockquote {{ page-break-inside: avoid; }}
  .page-content img {{ max-width: 100%; page-break-inside: avoid; }}
}}
</style>
</head><body>

<div class="topbar">
  <a href="/">&larr; 首页</a>
  <span class="title">{title}</span>
  <span style="display:flex;align-items:center;gap:.8rem">
    <span id="pageinfo"></span>
    <span id="progress"></span>
    <button class="mode-btn" id="mode-toggle"></button>
  </span>
</div>

<div class="reader" id="reader">
  <div class="page-zone left" id="zone-left"></div>
  <div class="page-zone right" id="zone-right"></div>
  <div class="page-content" id="content"></div>
</div>

<div class="fab-group" id="fab-group">
  <button class="pdf-fab" id="btn-pdf" title="导出 PDF">PDF</button>
  <button class="toc-fab" id="btn-top" title="回到顶部">&#8679;</button>
  <button class="toc-fab" id="btn-bottom" title="去到底部">&#8681;</button>
  <button class="toc-fab" id="toc-fab" title="目录">&#9776;</button>
</div>
<div class="toc-overlay" id="toc-overlay"></div>
<div class="toc-sidebar" id="toc-sidebar">
  <div class="toc-header"><span>目录</span><button class="toc-close" id="toc-close">&times;</button></div>
  <ul id="toc-list"></ul>
</div>


<script>
const raw = marked.parse(new TextDecoder().decode(Uint8Array.from(atob("{md_b64}"), c => c.charCodeAt(0))));
const content = document.getElementById('content');
const pageinfo = document.getElementById('pageinfo');
const progress = document.getElementById('progress');
const modeToggle = document.getElementById('mode-toggle');
const reader = document.getElementById('reader');

let pages = [];
let page = 0;

function wrapTables(container) {{
  container.querySelectorAll('table').forEach(t => {{
    if (!t.parentElement.classList.contains('table-wrap')) {{
      const w = document.createElement('div');
      w.className = 'table-wrap';
      t.parentNode.insertBefore(w, t);
      w.appendChild(t);
    }}
  }});
}}

// Split a tall table into row-chunked sub-tables, each fitting within maxH.
// Measures in the real container for accuracy.
function splitTableRows(table, maxH) {{
  const thead = table.querySelector('thead');
  const allRows = Array.from(table.querySelectorAll('tbody tr'));
  if (allRows.length === 0) allRows.push(...Array.from(table.querySelectorAll('tr')));
  const fragments = [];
  let start = 0;

  while (start < allRows.length) {{
    // Binary-search: find how many rows fit
    let lo = 1, hi = allRows.length - start, best = 1;
    while (lo <= hi) {{
      const mid = (lo + hi) >> 1;
      const t = document.createElement('table');
      if (thead) t.appendChild(thead.cloneNode(true));
      const tb = document.createElement('tbody');
      for (let i = start; i < start + mid; i++) tb.appendChild(allRows[i].cloneNode(true));
      t.appendChild(tb);
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      wrap.appendChild(t);
      // Measure in real container
      content.innerHTML = '';
      content.appendChild(wrap);
      if (outerH(wrap) <= maxH) {{ best = mid; lo = mid + 1; }}
      else {{ hi = mid - 1; }}
    }}
    const t = document.createElement('table');
    if (thead) t.appendChild(thead.cloneNode(true));
    const tb = document.createElement('tbody');
    for (let i = start; i < start + best; i++) tb.appendChild(allRows[i].cloneNode(true));
    t.appendChild(tb);
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.appendChild(t);
    fragments.push(wrap);
    start += best;
  }}
  return fragments;
}}

// Get outer height (border-box + margins)
function outerH(el) {{
  const s = getComputedStyle(el);
  return el.getBoundingClientRect().height + parseFloat(s.marginTop) + parseFloat(s.marginBottom);
}}

// Split a tall <pre> into line-chunked sub-blocks, each fitting within maxH.
function splitPreLines(pre, maxH) {{
  const lines = pre.textContent.split('\\n');
  const fragments = [];
  let start = 0;

  while (start < lines.length) {{
    let lo = 1, hi = lines.length - start, best = 1;
    while (lo <= hi) {{
      const mid = (lo + hi) >> 1;
      const p = document.createElement('pre');
      const c = document.createElement('code');
      c.textContent = lines.slice(start, start + mid).join('\\n');
      p.appendChild(c);
      content.innerHTML = '';
      content.appendChild(p);
      if (outerH(p) <= maxH) {{ best = mid; lo = mid + 1; }}
      else {{ hi = mid - 1; }}
    }}
    const p = document.createElement('pre');
    const c = document.createElement('code');
    c.textContent = lines.slice(start, start + best).join('\\n');
    p.appendChild(c);
    fragments.push(p);
    start += best;
  }}
  return fragments;
}}

// Recursively flatten oversized elements — find nested pre/table and split them
function flattenOversized(el, maxH) {{
  const results = [];
  if (outerH(el) <= maxH) {{
    results.push(el.cloneNode(true));
    return results;
  }}
  // Direct pre
  if (el.tagName === 'PRE') {{
    splitPreLines(el, maxH).forEach(f => results.push(f));
    return results;
  }}
  // Direct table or table-wrap
  const table = el.classList.contains('table-wrap') ? el.querySelector('table')
               : el.tagName === 'TABLE' ? el : null;
  if (table) {{
    splitTableRows(table, maxH).forEach(f => results.push(f));
    return results;
  }}
  // Container (ol, ul, blockquote, div, li, etc.) — check if it has oversized children
  const hasOversizedChild = Array.from(el.querySelectorAll('pre, table')).some(
    inner => outerH(inner) > maxH
  );
  if (hasOversizedChild) {{
    // Unwrap: emit each child separately, recursing as needed
    for (const child of Array.from(el.children)) {{
      flattenOversized(child, maxH).forEach(f => results.push(f));
    }}
  }} else {{
    // No oversized inner blocks — keep as-is (will be paginated by second pass)
    results.push(el.cloneNode(true));
  }}
  return results;
}}

function paginate() {{
  // Lock content to its actual display size
  content.style.overflow = 'hidden';
  content.style.height = '100%';
  const cs = getComputedStyle(content);
  const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  const maxH = content.clientHeight;
  const fitH = maxH - padY; // usable height for content
  pages = [];

  // First pass: render all content, recursively split oversized elements
  content.style.overflow = 'visible';
  content.style.height = 'auto';
  content.innerHTML = raw;
  wrapTables(content);

  const expanded = [];
  for (const kid of Array.from(content.children)) {{
    flattenOversized(kid, fitH).forEach(f => expanded.push(f));
  }}

  // Second pass: add elements one by one, check real overflow
  content.style.overflow = 'hidden';
  content.style.height = '100%';
  content.innerHTML = '';

  let pageElems = [];

  for (const el of expanded) {{
    content.appendChild(el.cloneNode(true));

    if (content.scrollHeight > maxH) {{
      // Overflowed — remove last child, flush page
      content.removeChild(content.lastChild);
      if (pageElems.length > 0) {{
        pages.push(content.innerHTML);
      }}
      // Start new page with this element
      content.innerHTML = '';
      content.appendChild(el.cloneNode(true));
      pageElems = [el];

      // Fallback: if single element STILL overflows, force-split it
      if (content.scrollHeight > maxH) {{
        const inner = content.firstChild;
        if (inner && inner.tagName === 'PRE') {{
          content.innerHTML = '';
          const chunks = splitPreLines(inner, fitH);
          // Re-add chunks through normal pagination
          for (const chunk of chunks) {{
            content.appendChild(chunk.cloneNode(true));
            if (content.scrollHeight > maxH) {{
              content.removeChild(content.lastChild);
              if (content.innerHTML) pages.push(content.innerHTML);
              content.innerHTML = '';
              content.appendChild(chunk.cloneNode(true));
            }}
          }}
          pageElems = [content.lastChild];
        }}
      }}
    }} else {{
      pageElems.push(el);
    }}
  }}
  // Last page
  if (content.innerHTML) {{
    pages.push(content.innerHTML);
  }}

  if (pages.length === 0) pages = [raw];
  if (page >= pages.length) page = pages.length - 1;
}}

function render() {{
  content.innerHTML = pages[page] || '';
  content.scrollTop = 0;
  const total = pages.length;
  pageinfo.textContent = total > 1 ? `${{page + 1}} / ${{total}}` : '';
  progress.textContent = total > 1 ? `${{Math.round((page + 1) / total * 100)}}%` : '';
}}

function go(delta) {{
  const next = page + delta;
  if (next >= 0 && next < pages.length) {{ page = next; render(); }}
}}

// Click zones
document.getElementById('zone-left').addEventListener('click', () => go(-1));
document.getElementById('zone-right').addEventListener('click', () => go(1));

// Keyboard
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') go(-1);
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {{ e.preventDefault(); go(1); }}
}});

// Buttons

// Mode toggle (persisted in localStorage)
let mode = localStorage.getItem('plan-view-mode') || '{default_mode}';
const modeLabels = {{ eink: '墨水屏', normal: '普通', eyecare: '护眼' }};
const modeCycle = ['eink', 'normal', 'eyecare'];

function applyMode() {{
  document.body.classList.remove('normal', 'eyecare');
  if (mode !== 'eink') document.body.classList.add(mode);
  const nextIdx = (modeCycle.indexOf(mode) + 1) % modeCycle.length;
  modeToggle.textContent = modeLabels[modeCycle[nextIdx]];
  if (mode === 'eink') {{
    paginate();
    render();
  }} else {{
    content.style.overflow = '';
    content.style.height = 'auto';
    content.innerHTML = raw;
    content.querySelectorAll('table').forEach(t => {{
      if (!t.parentElement.classList.contains('table-wrap')) {{
        const w = document.createElement('div');
        w.className = 'table-wrap';
        t.parentNode.insertBefore(w, t);
        w.appendChild(t);
      }}
    }});
  }}
}}

modeToggle.addEventListener('click', () => {{
  const idx = modeCycle.indexOf(mode);
  mode = modeCycle[(idx + 1) % modeCycle.length];
  localStorage.setItem('plan-view-mode', mode);
  applyMode();
}});

// Re-paginate on resize
window.addEventListener('resize', () => {{
  if (mode === 'eink') {{ paginate(); render(); }}
}});

applyMode();

// TOC logic (normal mode only)
const tocFab = document.getElementById('toc-fab');
const tocSidebar = document.getElementById('toc-sidebar');
const tocOverlay = document.getElementById('toc-overlay');
const tocList = document.getElementById('toc-list');
const tocClose = document.getElementById('toc-close');

function buildToc() {{
  tocList.innerHTML = '';
  const headings = content.querySelectorAll('h1, h2, h3');
  if (headings.length === 0) {{ tocFab.style.display = 'none'; return; }}
  headings.forEach((h, i) => {{
    if (!h.id) h.id = 'heading-' + i;
    const li = document.createElement('li');
    li.className = h.tagName.toLowerCase();
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.addEventListener('click', e => {{
      e.preventDefault();
      h.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      closeToc();
    }});
    li.appendChild(a);
    tocList.appendChild(li);
  }});
}}

function openToc() {{
  tocSidebar.classList.add('open');
  tocOverlay.classList.add('open');
  highlightToc();
}}

function closeToc() {{
  tocSidebar.classList.remove('open');
  tocOverlay.classList.remove('open');
}}

function highlightToc() {{
  const headings = content.querySelectorAll('h1, h2, h3');
  const items = tocList.querySelectorAll('li');
  let activeIdx = 0;
  headings.forEach((h, i) => {{
    if (h.getBoundingClientRect().top <= 100) activeIdx = i;
  }});
  items.forEach((li, i) => li.classList.toggle('active', i === activeIdx));
}}

document.getElementById('btn-top').addEventListener('click', () => window.scrollTo({{ top: 0, behavior: 'smooth' }}));
document.getElementById('btn-bottom').addEventListener('click', () => window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }}));
document.getElementById('btn-pdf').addEventListener('click', () => window.print());
tocFab.addEventListener('click', () => tocSidebar.classList.contains('open') ? closeToc() : openToc());
tocClose.addEventListener('click', closeToc);
tocOverlay.addEventListener('click', closeToc);
window.addEventListener('scroll', () => {{
  if (tocSidebar.classList.contains('open')) highlightToc();
}});

// Rebuild TOC when mode changes
const origApply = applyMode;
applyMode = function() {{
  origApply();
  if (mode !== 'eink') buildToc();
}};
applyMode();

// Auto-refresh: poll server for file changes
let refreshTimer = null;
let lastMtime = 0;
const planFile = decodeURIComponent(location.pathname.substring(1));

function startAutoRefresh() {{
  fetch('/api/config').then(r => r.json()).then(cfg => {{
    const refreshMode = localStorage.getItem('auto_refresh') || cfg.auto_refresh || 'eink-off';
    const interval = parseInt(localStorage.getItem('refresh_interval') || cfg.refresh_interval || 3) * 1000;

    if (refreshTimer) clearInterval(refreshTimer);
    if (refreshMode === 'off') return;
    if (refreshMode === 'eink-off' && mode === 'eink') return;

    // Get initial mtime
    fetch('/api/mtime?file=' + encodeURIComponent(planFile))
      .then(r => r.json()).then(d => {{ lastMtime = d.mtime; }});

    refreshTimer = setInterval(() => {{
      fetch('/api/mtime?file=' + encodeURIComponent(planFile))
        .then(r => r.json()).then(d => {{
          if (lastMtime > 0 && d.mtime > lastMtime) {{
            location.reload();
          }}
          lastMtime = d.mtime;
        }}).catch(() => {{}});
    }}, interval);
  }}).catch(() => {{}});
}}
startAutoRefresh();
</script>
</body></html>"""

        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_settings(self):
        html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>设置</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; color: #333; }
.container { max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
h1 { margin-bottom: 1.5rem; }
.back { display: inline-block; margin-bottom: 1rem; color: #2563eb; text-decoration: none; font-size: .9rem; }
.back:hover { text-decoration: underline; }
.card { background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,.1); }
.card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #333; }
.field { margin-bottom: 1rem; }
.field label { display: block; font-size: .9rem; font-weight: 600; margin-bottom: .3rem; color: #555; }
.field .hint { font-size: .8rem; color: #888; margin-top: .2rem; }
.field input, .field select { width: 100%; padding: .5rem; border: 1px solid #ddd; border-radius: 4px;
  font-size: .95rem; }
.field input:focus, .field select:focus { outline: none; border-color: #2563eb; }
.btn { background: #2563eb; color: #fff; border: none; padding: .6rem 1.5rem; border-radius: 4px;
  font-size: .95rem; cursor: pointer; }
.btn:hover { background: #1d4ed8; }
.btn-secondary { background: #6b7280; margin-left: .5rem; }
.btn-secondary:hover { background: #4b5563; }
.status { margin-top: 1rem; font-size: .9rem; color: #16a34a; display: none; }
.actions { display: flex; align-items: center; gap: .5rem; margin-top: 1rem; }
.divider { border: none; border-top: 1px solid #eee; margin: 1rem 0; }
</style>
</head><body>
<div class="container">
<a class="back" href="/">&larr; 首页</a>
<h1>设置</h1>

<div class="card">
  <h2>服务器配置</h2>
  <div class="field">
    <label>展示地址 (Host)</label>
    <input type="text" id="cfg-host" placeholder="127.0.0.1">
    <div class="hint">用于生成预览 URL 的地址。服务器始终监听 0.0.0.0。修改后需重启服务生效。</div>
  </div>
  <div class="field">
    <label>端口 (Port)</label>
    <input type="number" id="cfg-port" placeholder="8073">
    <div class="hint">预览服务监听端口。修改后需重启服务生效。</div>
  </div>
  <hr class="divider">
  <div class="field">
    <label>默认阅读模式</label>
    <select id="cfg-mode">
      <option value="normal">普通模式</option>
      <option value="eink">墨水屏模式</option>
      <option value="eyecare">护眼模式</option>
    </select>
    <div class="hint">新用户首次打开时的默认模式。用户切换后会记住选择。</div>
  </div>
</div>

<div class="card">
  <h2>自动刷新</h2>
  <div class="field">
    <label>刷新模式</label>
    <select id="cfg-refresh">
      <option value="off">全部关闭</option>
      <option value="on">全部开启</option>
      <option value="eink-off">墨水屏关闭</option>
    </select>
    <div class="hint">off = 不自动刷新；on = 所有模式都刷新；eink-off = 仅普通模式刷新（推荐）</div>
  </div>
  <div class="field">
    <label>刷新间隔（秒）</label>
    <input type="number" id="cfg-interval" min="1" max="60" placeholder="3">
    <div class="hint">检查文件变化的时间间隔</div>
  </div>
</div>

<div class="actions">
  <button class="btn" id="btn-save">保存</button>
  <button class="btn btn-secondary" id="btn-reset">重置为默认</button>
</div>
<div class="status" id="status">已保存</div>

</div>
<script>
const fields = {
  host: document.getElementById('cfg-host'),
  port: document.getElementById('cfg-port'),
  default_view_mode: document.getElementById('cfg-mode'),
  auto_refresh: document.getElementById('cfg-refresh'),
  refresh_interval: document.getElementById('cfg-interval'),
};
const status = document.getElementById('status');

function loadConfig() {
  fetch('/api/config').then(r => r.json()).then(cfg => {
    fields.host.value = cfg.host || '';
    fields.port.value = cfg.port || '';
    fields.default_view_mode.value = cfg.default_view_mode || 'normal';
    fields.auto_refresh.value = cfg.auto_refresh || 'eink-off';
    fields.refresh_interval.value = cfg.refresh_interval || 3;
  });
}

function saveConfig() {
  const data = {
    host: fields.host.value,
    port: parseInt(fields.port.value) || 8073,
    default_view_mode: fields.default_view_mode.value,
    auto_refresh: fields.auto_refresh.value,
    refresh_interval: parseInt(fields.refresh_interval.value) || 3,
  };
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data),
  }).then(r => r.json()).then(() => {
    status.style.display = 'block';
    status.textContent = '已保存';
    setTimeout(() => status.style.display = 'none', 2000);
  });
}

function resetConfig() {
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      host: '127.0.0.1', port: 8073,
      default_view_mode: 'normal',
      auto_refresh: 'eink-off', refresh_interval: 3,
    }),
  }).then(() => {
    loadConfig();
    status.style.display = 'block';
    status.textContent = '已重置为默认';
    setTimeout(() => status.style.display = 'none', 2000);
  });
}

document.getElementById('btn-save').addEventListener('click', saveConfig);
document.getElementById('btn-reset').addEventListener('click', resetConfig);
loadConfig();
</script>
</body></html>"""

        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        # TikZ renders
        svgs = sorted(
            RENDERS_DIR.glob("*.svg"),
            key=lambda f: f.stat().st_birthtime,
            reverse=True,
        )
        render_cards = ""
        for svg in svgs:
            name = svg.stem
            pdf_path = RENDERS_DIR / f"{name}.pdf"
            pdf_link = ""
            if pdf_path.exists():
                pdf_link = f' | <a href="/{name}.pdf">PDF</a>'
            render_cards += f"""
            <div class="card">
                <a href="/{svg.name}"><img src="/{svg.name}" alt="{name}"></a>
                <div class="name">{name}{pdf_link}</div>
            </div>"""
        if not render_cards:
            render_cards = '<p class="empty">暂无渲染图</p>'

        # Plans (sorted by creation time, newest first)
        plan_cards = ""
        if PLANS_DIR.exists():
            plans = sorted(
                PLANS_DIR.glob("*.md"),
                key=lambda f: f.stat().st_birthtime,
                reverse=True,
            )
            for plan in plans:
                name = plan.stem
                display = name.replace("-", " ").title()
                # Read first non-empty, non-frontmatter line as summary
                summary = ""
                for line in plan.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("---"):
                        summary = line[:120]
                        if len(line) > 120:
                            summary += "..."
                        break
                plan_url = f"/plans/{urllib.parse.quote(plan.name)}"
                plan_cards += f"""
                <div class="plan-card">
                    <a href="{plan_url}">{display}</a>
                    <div class="summary">{summary}</div>
                </div>"""
        if not plan_cards:
            plan_cards = '<p class="empty">暂无计划</p>'

        # Temp directory: count items
        temp_count = len(list(TEMP_MD_DIR.iterdir())) if TEMP_MD_DIR.exists() else 0

        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>预览服务</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; color: #333;
  transition: background .3s, color .3s; }}
h1 {{ margin-bottom: .5rem; }}
h2 {{ margin-top: 2rem; margin-bottom: 1rem; color: #555; border-bottom: 1px solid #ddd; padding-bottom: .5rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }}
.card {{ background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
.card img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
.name {{ margin-top: .5rem; font-size: .9rem; color: #555; word-break: break-all; }}
.plan-card {{ background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
.plan-card a {{ font-weight: 600; font-size: 1rem; }}
.summary {{ margin-top: .4rem; font-size: .85rem; color: #666; }}
.empty {{ color: #888; text-align: center; grid-column: 1/-1; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.mode-btn {{ background: none; border: 2px solid #999; border-radius: 4px;
  padding: .2rem .7rem; cursor: pointer; font-size: .8rem; font-weight: 600; }}
.mode-btn:hover {{ background: #e0e0e0; }}
/* Index eyecare mode */
body.eyecare {{ background: #f5eddc; color: #4a3f35; }}
body.eyecare h2 {{ color: #6b5d4d; border-bottom-color: #c8b899; }}
body.eyecare .card, body.eyecare .plan-card {{ background: #ede3d0; box-shadow: 0 2px 4px rgba(90,70,40,.12); }}
body.eyecare .card img {{ border-color: #c8b899; }}
body.eyecare .name {{ color: #6b5d4d; }}
body.eyecare .summary {{ color: #7a6c5a; }}
body.eyecare a {{ color: #8b6914; }}
body.eyecare .mode-btn {{ border-color: #b0a08a; color: #5b4636; }}
body.eyecare .mode-btn:hover {{ background: #ddd0b8; }}
</style>
</head><body>
<h1 style="display:flex;justify-content:space-between;align-items:center;">预览服务
<span style="display:flex;align-items:center;gap:.8rem;">
  <button class="mode-btn" id="idx-mode"></button>
  <a href="/settings" style="font-size:.9rem;font-weight:400;">设置</a>
</span></h1>

<h2><a href="/temp/">临时文档 ({temp_count})</a></h2>

<h2>计划 ({len(list(PLANS_DIR.glob("*.md"))) if PLANS_DIR.exists() else 0})</h2>
<div class="grid">{plan_cards}</div>

<h2>TikZ 渲染 ({len(svgs)})</h2>
<div class="grid">{render_cards}</div>

<script>
const btn = document.getElementById('idx-mode');
const modes = ['normal', 'eyecare'];
const labels = {{ normal: '护眼', eyecare: '普通' }};
let idxMode = localStorage.getItem('index-view-mode') || 'normal';
function applyIdx() {{
  document.body.classList.remove('eyecare');
  if (idxMode === 'eyecare') document.body.classList.add('eyecare');
  const next = modes[(modes.indexOf(idxMode) + 1) % modes.length];
  btn.textContent = labels[idxMode];
}}
btn.addEventListener('click', () => {{
  idxMode = modes[(modes.indexOf(idxMode) + 1) % modes.length];
  localStorage.setItem('index-view-mode', idxMode);
  applyIdx();
}});
applyIdx();
</script>
</body></html>"""

        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    port = get_port()
    host = get_host()

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_MD_DIR.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("0.0.0.0", port), TikZHandler)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "pid": __import__("os").getpid(),
        "host": host,
        "port": port,
    }))

    def shutdown(signum, frame):
        server.shutdown()
        STATE_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"Preview server listening on 0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
