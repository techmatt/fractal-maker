#!/usr/bin/env python
r"""verify_sheet_http.py — a built label sheet, verified THROUGH the running server.

`build_combined_label_sheet.py check` proves the built BYTES: the manifest, the route map,
the ±1 order, and every crop resolved through `serve.py`'s `translate_path`. This proves the
next layer down, and it is a different failure class: a sheet whose bytes are perfect still
does not serve if the server is on another port, was launched from another root, is a second
co-hosting process, or the crop relocation resolves inside the Python seam but 404s over the
socket. Every assertion here goes over HTTP.

WHY THIS IS A TOOL AND NOT A SCRATCH SCRIPT. It is the only thing that can answer "does the
sitting actually load" before a labeler sits down, and CLAUDE.md is explicit that the one
load-bearing script left in `scratchpad/` vanished and cost a sweep to recover. It writes its
log to `scratch/`, which stays disposable.

NOT IN THE PYTEST SUITE, deliberately: it needs a live server on a known port, so as a test it
would be skipped-by-default — a green that means nothing (`verification_practice.md` §1). Run
it by hand beside the server, the way you run `-m slow`.

  uv run python tools/viz/serve.py --port 8010 &
  uv run python tools/corpus/verify_sheet_http.py --spec steady_state_uncal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "viz"),
           str(ROOT / "tools" / "scoring"), str(ROOT / "tools" / "mining")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_combined_label_sheet as B      # noqa: E402
import corpus_common as cc                  # noqa: E402
import merge_scores as ms                   # noqa: E402
import paths                                # noqa: E402
from partitions import partition_of_row     # noqa: E402
from tools.v7 import build_manifest as bm   # noqa: E402

# Provenance keys a labeler must never see, beyond the whole-block absence asserted below.
# Named individually because "arm", "fate" and "score" are three different giveaways and a
# regression usually reintroduces exactly one of them.
ARM_FATE_SCORE_KEYS = ("selection_role", "original_score", "decoded_class", "focus_score",
                       "fate", "filter_score", "mix_source", "batch_id", "rank_tier",
                       "rank_score", "p_good", "p_notbad")


def run(spec: B.SheetSpec, base: str) -> int:
    out, ok = [], True

    def emit(s=""):
        out.append(s)
        print(s, flush=True)

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    def get(path, method="GET"):
        req = urllib.request.Request(base + path, method=method)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, (r.read() if method == "GET" else b""), r.headers

    sheet = spec.sheet_id
    emit(f"=== {spec.name} sheet — HTTP verification against {base} ===")
    emit(f"[{sheet}]")

    # ---- 1. the served manifest, as the browser receives it -------------------------------
    st, body, _ = get(f"/data/label_corpus/batches/{sheet}/{B.SHEET_MANIFEST}")
    text = body.decode("utf-8")
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    sel = B.load_sources(spec)
    by_src = Counter(b for b, _ in sel)

    emit("\n[1. row count, opaque ids, leak keys — on the bytes the browser receives]")
    check("manifest served over HTTP", st == 200, f"HTTP {st}")
    # RELATIONAL, not a literal: the selection is re-derived from the source batches on every
    # run, so a filter that moved and a build that dropped rows both go red here.
    route = ms.load_route(str(B.sheet_dir(spec) / B.ROUTE_FILE))
    per_src = Counter(b for b, _ in route.values())
    check("served row count == the re-derived selection", len(rows) == len(sel),
          f"{len(rows)} served vs {len(sel)} selected")
    for b in spec.sources:
        n_tot = len(cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")))
        check(f"{b}: {per_src[b]} of {n_tot} rows served, == the re-derived selection",
              per_src[b] == by_src[b], f"route says {per_src[b]}, filter says {by_src[b]}")
    check("every route target is inside the selection",
          {(b, i) for b, i in route.values()} == {(b, r["image_id"]) for b, r in sel})
    p = len(spec.id_prefix)
    check(f"every id is opaque `{spec.id_prefix}<slot>_<hash>` and unique",
          len({r["image_id"] for r in rows}) == len(rows)
          and all(r["image_id"].startswith(spec.id_prefix)
                  and len(r["image_id"]) == p + 13 for r in rows))
    src_ids = {r["image_id"] for b in spec.sources
               for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl"))}
    check("no served id IS a source image_id", not ({r["image_id"] for r in rows} & src_ids))
    leaked = [k for k in B.SHEET_LEAK_KEYS if f'"{k}"' in text]
    check("no leak key in the served bytes", not leaked, str(leaked))
    check("no source batch id in the served bytes",
          not any(b in text for b in spec.sources))
    check("every served row is {image_id, render, label} with a null label",
          all(set(r) == {"image_id", "render", "label"} and r["label"]["score"] is None
              for r in rows))
    disk = cc.read_jsonl(str(B.sheet_dir(spec) / B.SHEET_MANIFEST))
    check("the HTTP manifest is the built manifest, in the built order",
          [r["image_id"] for r in rows] == [r["image_id"] for r in disk])

    # ---- 2. both crops, per row, over the socket -------------------------------------------
    emit("\n[2. every row resolves to BOTH crops over HTTP]")
    bad, served_bytes = [], 0
    for r in rows:
        for kind in ("crops", "vivid"):
            url = f"/data/label_corpus/batches/{sheet}/{kind}/{r['image_id']}.jpg"
            try:
                s, _b, h = get(url, method="HEAD")
                n = int(h.get("Content-Length", 0))
                if s != 200 or n <= 0:
                    bad.append((kind, r["image_id"], s, n))
                else:
                    served_bytes += n
            except Exception as e:                       # a socket failure is a FAIL, not a stop
                bad.append((kind, r["image_id"], type(e).__name__))
    check(f"all {2 * len(rows)} crop fetches returned 200 with a non-empty body", not bad,
          f"{len(bad)} bad, e.g. {bad[:3]}")
    emit(f"       {2 * len(rows)} objects, {served_bytes / 1e6:.1f} MB")

    # ---- 3. the source registrations ------------------------------------------------------
    emit("\n[3. every source batch registered, at its own size]")
    for b in spec.sources:
        n = len(cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")))
        split, biased, source = bm.assign_split({"batch": b, "ft": "mandelbrot"})
        check(f"{b}: registered ({source}), {n} rows", source != "unregistered",
              f"{(split, biased, source)}")
    check("the sheet did not become a batch (no images.jsonl)",
          not (B.sheet_dir(spec) / "images.jsonl").exists())

    # ---- 4. order=file ---------------------------------------------------------------------
    emit("\n[4. order=file presents the manifest order]")
    st_b, body_b, _ = get(f"/data/label_corpus/batches/{sheet}/batch.json")
    bj = json.loads(body_b)
    check("batch.json served over HTTP and pins presentation_order=file",
          st_b == 200 and bj.get("presentation_order") == "file")
    st_p, page_b, _ = get("/tools/viz/corpus_label.html")
    page = page_b.decode("utf-8")
    check("the page served over HTTP short-circuits to the file order, first branch, no "
          "fallthrough", st_p == 200 and "if(ORDER_MODE==='file') return rows.slice();" in page)
    check("the URL parameter and batch.json both feed ORDER_MODE",
          "QP.get('order')" in page and "bj.presentation_order" in page)

    # ---- 5. what the page can paint --------------------------------------------------------
    emit("\n[5. the page cannot show score, arm, fate or batch of origin]")
    check("the page fetches ONLY batch.json + the manifest (never route.json)",
          "route.json" not in page and page.count("fetch(") == 2)
    check("nothing from batch.json is rendered — only presentation_seed/order are read",
          "bj.presentation_seed" in page and "bj.presentation_order" in page
          and "bj.source_batches" not in page and "bj.counts" not in page
          and "bj.selection" not in page)
    check("no served row carries a provenance block at all (blinding by ABSENCE)",
          all("provenance" not in r for r in rows))
    absent = [k for k in ARM_FATE_SCORE_KEYS if f'"{k}"' in text]
    check("no arm / fate / score / origin key in the served bytes", not absent, str(absent))
    check("every served label.score is null — the page reads 'unlabeled' for every row",
          all(r["label"]["score"] is None for r in rows))
    st_r, _b, _h = get(f"/data/label_corpus/batches/{sheet}/{B.ROUTE_FILE}", method="HEAD")
    emit(f"       note: route.json is merge-side and is static-served (HTTP {st_r}); the page "
         f"never fetches it, so reaching it takes a hand-typed URL.")

    # ---- the realized subset, per leg x partition -------------------------------------------
    by_id = {b: {r["image_id"]: r for r in
                 cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl"))}
             for b in spec.sources}
    cells = Counter()
    for r in rows:
        b, i = route[r["image_id"]]
        cells[(b, partition_of_row(by_id[b][i]["render"]))] += 1
    emit("\n[realized subset]")
    emit(f"       {len(rows)} rows over {len(spec.sources)} batches: {dict(by_src)}")
    for (b, part), n in sorted(cells.items()):
        emit(f"       {b:38s} {part:20s} {n}")

    emit(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    log = paths.scratch(f"{spec.name}_sheet", "http_verify.txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  -> {log}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", choices=sorted(B.SPECS), required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8010",
                    help="the RUNNING server (serve.py prints the port it actually bound)")
    a = ap.parse_args(argv)
    return run(B.SPECS[a.spec], a.base.rstrip("/"))


if __name__ == "__main__":
    raise SystemExit(main())
