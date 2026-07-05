import os
import urllib.request

import numpy as np

# ann-benchmarks datasets: hdf5 files with train vectors, test queries
# and the true top-100 neighbors for every query
DATASETS = {
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-25-angular": "http://ann-benchmarks.com/glove-25-angular.hdf5",
}


def load(name, data_dir="data"):
    """Download a dataset on first use, then return
    (train, queries, true_neighbors)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    import h5py  # only needed here, core package stays numpy-only

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name}.hdf5")
    if not os.path.exists(path):
        print(f"downloading {name}, this can take a few minutes...")
        urllib.request.urlretrieve(DATASETS[name], path)
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        queries = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"])
    return train, queries, neighbors
