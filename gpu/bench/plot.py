#!/usr/bin/env python3
"""Plot one or more bench/run.py results json.

Two figures, written as png next to the first json:
  a) recall@k vs throughput, one line per baseline, one panel per batch
  b) throughput vs batch size, graph indexes pinned to the knob that
     lands nearest 0.95 recall

Throughput is the wall-clock number — kernel-only time is a separate
column in the json and is never plotted as if it were end-to-end.

    python gpu/bench/plot.py gpu/bench/results/20260901-101500.json
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.dirname(BENCH_DIR)
REPO_DIR = os.path.dirname(GPU_DIR)
# nothing under gpu/ is pip-installed
for _p in (BENCH_DIR, GPU_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET_RECALL = 0.95
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def load(paths):
    """Flatten every results file into one row list. With more than one
    file the run's stem joins the series name, so two gpus or two dates
    do not silently overplot each other."""
    rows, ks = [], set()
    multi = len(paths) > 1
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        stem = os.path.splitext(os.path.basename(path))[0]
        ks.add(payload.get("k", 10))
        for row in payload["rows"]:
            if row.get("qps") is None:
                continue
            row = dict(row)
            row["series"] = f"{row['baseline']} [{stem}]" if multi else row["baseline"]
            rows.append(row)
    if not rows:
        raise SystemExit("no timed rows in those results")
    return rows, sorted(ks)[0]


def styles(series_names):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        name: (cycle[i % len(cycle)], MARKERS[i % len(MARKERS)])
        for i, name in enumerate(series_names)
    }


def label_for(row):
    if row["knob"] is None:
        return None
    # ef / nprobe / itopk, whichever the adapter swept
    return f"{row['knob']}={row['knob_value']}"


def plot_recall_qps(rows, k, style, path):
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(
        1, len(batches), figsize=(4.2 * len(batches), 4.4), sharey=True, squeeze=False
    )
    handles = {}
    for ax, batch in zip(axes[0], batches):
        for name, (color, marker) in style.items():
            pts = [r for r in rows if r["series"] == name and r["batch"] == batch]
            if not pts:
                continue
            # sweep order, not measurement order: the line traces the
            # tradeoff curve, so it has to follow the knob
            pts.sort(key=lambda r: (r["knob_value"] is None, r["knob_value"], r["qps"]))
            line, = ax.plot(
                [r["qps"] for r in pts],
                [r["recall"] for r in pts],
                marker=marker,
                color=color,
                label=name,
            )
            handles.setdefault(name, line)
            for r in pts:
                text = label_for(r)
                if text:
                    ax.annotate(
                        text,
                        (r["qps"], r["recall"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=6,
                        color=color,
                    )
        ax.set_xscale("log")
        # the knob labels sit to the upper right of their point and get
        # clipped at the axis edge without the margin
        ax.margins(x=0.15, y=0.1)
        ax.set_xlabel("queries / second (wall, log scale)")
        ax.set_title(f"batch {batch}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(f"recall@{k}")
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
    )
    fig.suptitle(f"recall@{k} vs throughput")
    # leave room for the legend strip under the panels
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def pick_operating_point(pts):
    """The row a graph index would actually be shipped at: knob nearest
    0.95 recall, faster one wins a tie. Knob-free exact indexes have a
    single row and it is returned unchanged."""
    return min(pts, key=lambda r: (abs(r["recall"] - TARGET_RECALL), -r["qps"]))


def plot_qps_batch(rows, k, style, path):
    batches = sorted({r["batch"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, (color, marker) in style.items():
        pts = []
        for batch in batches:
            cell = [r for r in rows if r["series"] == name and r["batch"] == batch]
            if cell:
                pts.append(pick_operating_point(cell))
        if not pts:
            continue
        ax.plot(
            [r["batch"] for r in pts],
            [r["qps"] for r in pts],
            marker=marker,
            color=color,
            label=name,
        )
        for r in pts:
            text = label_for(r)
            if text:
                ax.annotate(
                    text,
                    (r["batch"], r["qps"]),
                    textcoords="offset points",
                    xytext=(4, -9),
                    fontsize=6,
                    color=color,
                )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.margins(x=0.12, y=0.15)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("batch size")
    ax.set_ylabel("queries / second (wall, log scale)")
    ax.set_title(
        f"throughput vs batch, graph indexes at the knob nearest "
        f"recall@{k} {TARGET_RECALL:.2f}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="one or more results json")
    parser.add_argument("--out-prefix", help="default: next to the first json")
    args = parser.parse_args(argv)

    rows, k = load(args.results)
    prefix = args.out_prefix
    if not prefix:
        first = os.path.abspath(args.results[0])
        stem = os.path.splitext(os.path.basename(first))[0]
        if len(args.results) > 1:
            stem = f"{stem}+{len(args.results) - 1}"
        prefix = os.path.join(os.path.dirname(first), stem)

    # one color/marker per series, shared by both figures
    order = []
    for row in rows:
        if row["series"] not in order:
            order.append(row["series"])
    style = styles(order)

    plot_recall_qps(rows, k, style, prefix + "-recall_vs_qps.png")
    plot_qps_batch(rows, k, style, prefix + "-qps_vs_batch.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
