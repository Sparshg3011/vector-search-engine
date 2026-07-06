"""Where the bytes go: vectors vs graph, python objects vs a packed
layout, and both indexes on disk."""

import os
import sys
import tempfile

import faiss

from vecstore import HNSWIndex
from vecstore.datasets import load

M = 16
EF_CONSTRUCTION = 100


def graph_bytes(index):
    """Containers plus link int objects, each object counted once."""
    seen = set()
    total = 0
    for layer in index._layers:
        total += sys.getsizeof(layer)
        for node, links in layer.items():
            total += sys.getsizeof(links)
            for x in links:
                if id(x) not in seen:
                    seen.add(id(x))
                    total += sys.getsizeof(x)
    return total


def main(name="fashion-mnist-784-euclidean"):
    cache = os.path.join("data", f"{name}-vecstore.npz")
    index = HNSWIndex.load(cache)

    n_links = sum(len(l) for layer in index._layers for l in layer.values())
    vec_mb = index.vectors.nbytes / 2**20
    graph_mb = graph_bytes(index) / 2**20
    packed_mb = n_links * 4 / 2**20  # a uint32 per link is all a C layout pays

    print(f"{len(index)} vectors, dim {index.dim}, {n_links} links")
    print(f"vectors in ram:      {vec_mb:8.1f} MB")
    print(f"graph in ram:        {graph_mb:8.1f} MB")
    print(f"graph if packed:     {packed_mb:8.1f} MB")
    print(f"python object tax:   {graph_mb / packed_mb:8.1f}x")

    train, _, _ = load(name)
    theirs = faiss.IndexHNSWFlat(train.shape[1], M)
    theirs.hnsw.efConstruction = EF_CONSTRUCTION
    theirs.add(train)
    with tempfile.NamedTemporaryFile(suffix=".faiss") as tmp:
        faiss.write_index(theirs, tmp.name)
        faiss_mb = os.path.getsize(tmp.name) / 2**20
    ours_disk_mb = os.path.getsize(cache) / 2**20

    print(f"ours on disk:        {ours_disk_mb:8.1f} MB (compressed npz)")
    print(f"faiss on disk:       {faiss_mb:8.1f} MB (raw)")


if __name__ == "__main__":
    main()
