#!/usr/bin/env python
"""Single-host static server for the label UI (serves the repo root over HTTP).

Why this exists instead of `python -m http.server`: the stdlib server sets
`allow_reuse_address = True` (SO_REUSEADDR), so on Windows a SECOND launcher on the same port
can also bind it, and requests then split nondeterministically between the two processes. If the
two have different working directories you get the "left/right image inconsistent" bug — the
browser randomly gets a crop from one root and its vivid companion from the other.

This server binds EXCLUSIVELY (SO_EXCLUSIVEADDRUSE on win32, allow_reuse_address=False), so a
second launcher on the same port fails to bind instead of silently co-hosting. If the requested
port is already held, it advances to the next free port (up to --max-scan) and prints the port it
actually bound — so you always get exactly one clean host and know its URL.

Relocated label-corpus crops
----------------------------
The label-corpus crop trees (``data/label_corpus/batches/*/{crops,vivid}/``) were moved OUT of
the working tree behind ``tools/corpus/artifacts.py`` (see docs/design/label_corpus_relocation.md),
so a plain repo-root static server would 404 them. The labeler page (``corpus_label.html``) still
builds the in-tree URL ``data/label_corpus/batches/<id>/crops/<img>.jpg`` — a client-side string a
Python seam cannot reach — so this server resolves it **transparently**: any request whose path is
a relocated crop/vivid file is served from ``artifacts.resolve`` instead of the served root. A
per-request ``?crops=<dir>`` override serves the crop's file from ``<dir>`` instead (point the page
at an alternate crops folder without moving files or setting ``FRACTAL_ARTIFACTS_ROOT``). Everything
else — the page, ``images.jsonl``, ``batch.json`` (all still in-tree) — is served unchanged.

Run (from anywhere):
  uv run python tools/viz/serve.py [--port 8010] [--bind 127.0.0.1] [--root <dir>]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import socket
import socketserver
import sys
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "corpus"))
import artifacts as _artifacts  # noqa: E402  (the single relocated-artifact resolver)

# data/label_corpus/batches/<batch_id>/(crops|vivid)/<rest...> — the relocated crop families.
_CROP_RE = re.compile(
    r"^data/label_corpus/batches/[^/]+/(?:crops|vivid)/(?P<rest>.+)$")


class CorpusHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler that transparently serves relocated label-corpus crops from the
    artifacts root (and honours a ``?crops=<dir>`` per-request override), leaving every
    other path — page, images.jsonl, batch.json — to the default in-tree mapping."""

    def translate_path(self, path):
        split = urlsplit(path)
        rel = unquote(split.path).lstrip("/").replace("\\", "/")
        m = _CROP_RE.match(rel)
        if m is None:
            return super().translate_path(path)         # normal in-tree file
        override = parse_qs(split.query).get("crops", [None])[0]
        if override:
            # serve the crop file straight out of the override dir (basename join is safe:
            # `rest` is a single filename in this store, but normpath-guard anyway).
            fname = os.path.basename(m.group("rest"))
            return os.path.normpath(os.path.join(override, fname))
        if _artifacts.is_relocated(rel):
            return str(_artifacts.resolve(rel))          # relocated -> artifacts root
        return super().translate_path(path)              # (not yet moved) in-tree fallback


class ExclusiveTCPServer(socketserver.TCPServer):
    # do NOT reuse an address already in use — a second launcher must fail, not co-host.
    allow_reuse_address = False

    def server_bind(self):
        # On Windows SO_REUSEADDR lets a socket STEAL a port already bound by another process;
        # SO_EXCLUSIVEADDRUSE forbids that, so our port cannot be double-bound. No-op elsewhere
        # (POSIX already refuses a second bind without SO_REUSEADDR).
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--root", default=ROOT, help="document root (default: repo root)")
    ap.add_argument("--max-scan", type=int, default=20,
                    help="if --port is busy, try this many ports upward (0 = fail fast)")
    a = ap.parse_args()

    handler = functools.partial(CorpusHandler, directory=str(a.root))
    last_err = None
    for port in range(a.port, a.port + max(a.max_scan, 0) + 1):
        try:
            httpd = ExclusiveTCPServer((a.bind, port), handler)
        except OSError as e:
            last_err = e
            print(f"port {port} unavailable ({e.__class__.__name__}: {e}); trying next ...",
                  flush=True)
            continue
        print(f"serving {a.root}", flush=True)
        print(f"  -> http://{a.bind}:{port}/  (EXCLUSIVE bind — a second host on this port "
              f"will fail, not co-host)", flush=True)
        print(f"  label UI: http://{a.bind}:{port}/tools/viz/corpus_label.html"
              f"?batch=<id>&manifest=blind.jsonl", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down", flush=True)
            httpd.shutdown()
        return
    sys.exit(f"no free exclusive port in [{a.port}, {a.port + a.max_scan}]: {last_err}")


if __name__ == "__main__":
    main()
