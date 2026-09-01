import json
import os

import numpy as np


class GpuIndex:
    """Plain bag of one .gpu.npz's contents. Attribute names match the
    npz keys; upper_layers is json-decoded to a list of {node: [nbrs]}
    dicts for layers 1..top."""

    def __init__(
        self,
        vectors,
        dim,
        stride,
        adjacency,
        degrees,
        M,
        entry,
        metric,
        normalized,
        upper_layers,
    ):
        self.vectors = vectors
        self.dim = dim
        self.stride = stride
        self.adjacency = adjacency
        self.degrees = degrees
        self.M = M
        self.entry = entry
        self.metric = metric
        self.normalized = normalized
        self.upper_layers = upper_layers

    def validate(self):
        """Every structural invariant the kernels rely on, checked once
        on the host — the gpu will happily read out of bounds instead
        of raising."""
        if self.stride % 8 != 0 or self.stride < self.dim:
            raise ValueError(
                f"stride {self.stride} must be a multiple of 8 and >= dim {self.dim}"
            )
        n, width = self.adjacency.shape
        if self.vectors.shape != (n, self.stride):
            raise ValueError(
                f"vectors shape {self.vectors.shape} != ({n}, {self.stride}) "
                "implied by adjacency and stride"
            )
        if self.degrees.shape != (n,):
            raise ValueError(
                f"degrees shape {self.degrees.shape} != ({n},) adjacency rows"
            )
        if width != 2 * self.M:
            raise ValueError(f"adjacency width {width} != 2M = {2 * self.M}")
        if not 0 <= self.entry < n:
            raise ValueError(f"entry {self.entry} out of range for {n} nodes")
        if self.degrees.min() < 0 or self.degrees.max() > width:
            raise ValueError(
                f"degrees must lie in [0, {width}], found "
                f"[{self.degrees.min()}, {self.degrees.max()}]"
            )
        # -1 marks padding and nothing else: real ids fill the first
        # degrees[i] slots of each row, -1 fills the rest
        live = np.arange(width)[None, :] < self.degrees[:, None]
        bad = np.flatnonzero(((self.adjacency == -1) & live).any(axis=1))
        if len(bad):
            raise ValueError(
                f"adjacency row {bad[0]}: -1 inside the first "
                f"{self.degrees[bad[0]]} slots"
            )
        bad = np.flatnonzero(((self.adjacency != -1) & ~live).any(axis=1))
        if len(bad):
            raise ValueError(
                f"adjacency row {bad[0]}: live id past degree {self.degrees[bad[0]]}"
            )
        ids = self.adjacency[live]
        if len(ids) and (ids.min() < 0 or ids.max() >= n):
            raise ValueError(
                f"neighbor ids must lie in [0, {n}), found "
                f"[{ids.min()}, {ids.max()}]"
            )
        if self.stride > self.dim and np.any(self.vectors[:, self.dim :] != 0):
            raise ValueError(
                f"vector padding columns {self.dim}..{self.stride} must be all zero"
            )


def load_gpu_index(path):
    """Read a .gpu.npz written by export.py: unwrap the 0-d scalars,
    decode the upper-layer json back to int-keyed dicts. Call
    .validate() before shipping anything to a device."""
    path = os.fspath(path)
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path += ".npz"
    data = np.load(path, allow_pickle=False)
    upper_layers = [
        {int(node): [int(nb) for nb in nbrs] for node, nbrs in layer.items()}
        for layer in json.loads(data["upper_layers"].item())
    ]
    return GpuIndex(
        vectors=data["vectors"],
        dim=int(data["dim"]),
        stride=int(data["stride"]),
        adjacency=data["adjacency"],
        degrees=data["degrees"],
        M=int(data["M"]),
        entry=int(data["entry"]),
        metric=data["metric"].item(),
        normalized=bool(data["normalized"]),
        upper_layers=upper_layers,
    )
