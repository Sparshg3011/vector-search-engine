import argparse
import json
import os

import numpy as np


def _load_source(src):
    """Accepts a saved HNSWIndex .npz path or a live HNSWIndex and
    returns the raw pieces the gpu format needs. Paths are parsed
    directly instead of via HNSWIndex.load so this file runs anywhere,
    without vecstore on sys.path."""
    if isinstance(src, (str, os.PathLike)):
        # savez appends .npz; accept the bare path the index was saved under
        path = os.fspath(src)
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path += ".npz"
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].item())
        layers = [
            {int(n): list(nbrs) for n, nbrs in layer.items()}
            for layer in meta["layers"]
        ]
        return (
            data["vectors"],
            layers,
            meta["entry"],
            meta["M"],
            meta["dim"],
            meta["metric"],
        )
    return src.vectors, list(src._layers), src._entry, src.M, src.dim, src.metric


def export_index(src, out_path, normalize=False):
    """Saved (or live) HNSWIndex -> .gpu.npz per the SPEC format table:
    fp16 row-padded vectors, int32 layer-0 adjacency padded with -1,
    upper layers 1..top as json. Normalization happens in fp32 before
    the fp16 cast, so unit norms survive quantization as well as they
    can."""
    vectors, layers, entry, M, dim, metric = _load_source(src)
    if entry is None or len(vectors) == 0:
        raise ValueError("cannot export an empty index")
    if metric == "cosine" and not normalize:
        # the gpu only does l2 and ip; on unit vectors both reproduce
        # the cosine ranking, so angular data has to be normalized
        raise ValueError(
            "cosine index requires normalize=True — angular data must be "
            "L2-normalized at export (rerun with --normalize)"
        )

    vectors = np.asarray(vectors, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)
    if metric == "cosine":
        # normalized rows make ip rank identically to cosine
        metric = "ip"

    n = len(vectors)
    # stride is a multiple of 8 fp16 values = 16-byte rows
    stride = (dim + 7) // 8 * 8
    padded = np.zeros((n, stride), dtype=np.float16)
    padded[:, :dim] = vectors.astype(np.float16)

    cap = 2 * M
    adjacency = np.full((n, cap), -1, dtype=np.int32)
    degrees = np.zeros(n, dtype=np.int32)
    for node, nbrs in layers[0].items():
        # _prune guarantees the layer-0 budget; assert anyway, the gpu
        # format has no room for an overflow
        assert len(nbrs) <= cap, f"node {node}: {len(nbrs)} layer-0 links > cap {cap}"
        adjacency[node, : len(nbrs)] = nbrs
        degrees[node] = len(nbrs)

    upper = json.dumps(
        [
            {str(node): [int(nb) for nb in nbrs] for node, nbrs in layer.items()}
            for layer in layers[1:]
        ]
    )
    np.savez_compressed(
        out_path,
        vectors=padded,
        dim=np.int64(dim),
        stride=np.int64(stride),
        adjacency=adjacency,
        degrees=degrees,
        M=np.int64(M),
        entry=np.int64(entry),
        metric=metric,
        normalized=np.bool_(normalize),
        upper_layers=upper,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert a saved HNSWIndex .npz to the .gpu.npz device format"
    )
    parser.add_argument("src", help="saved HNSWIndex .npz")
    parser.add_argument("out", help="output .gpu.npz path")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize rows before the fp16 cast (required for cosine indexes)",
    )
    args = parser.parse_args()
    export_index(args.src, args.out, normalize=args.normalize)
    print(f"wrote {args.out}")
