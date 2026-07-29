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

Run (from anywhere):
  uv run python tools/viz/serve.py [--port 8010] [--bind 127.0.0.1] [--root <dir>]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import socket
import socketserver
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


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

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(a.root))
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
