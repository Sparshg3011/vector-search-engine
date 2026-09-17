"""Download an ann-benchmarks dataset and save its test queries as .npy,
for bench/run.py --queries. Needs network: on a cluster whose compute
nodes are offline, run it on a login node.

    python3 gpu/bench/fetch_queries.py sift-128-euclidean
"""

import argparse
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from vecstore import datasets  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=sorted(datasets.DATASETS))
    parser.add_argument("--data-dir", default=os.path.join(REPO_DIR, "data"))
    args = parser.parse_args(argv)

    out = os.path.join(args.data_dir, f"{args.name}-queries.npy")
    if os.path.exists(out):
        print(f"{out} already exists")
        return
    _, queries, _ = datasets.load(args.name, data_dir=args.data_dir)
    np.save(out, np.ascontiguousarray(queries, dtype=np.float32))
    print(f"wrote {out}: {queries.shape[0]} queries, dim {queries.shape[1]}")


if __name__ == "__main__":
    main()
