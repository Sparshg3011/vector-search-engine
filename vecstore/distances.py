import numpy as np


def l2(query, vectors):
    """Squared L2 distance from query to each row. Skips the sqrt since
    it doesn't change the ranking."""
    diff = vectors - query
    return np.einsum("ij,ij->i", diff, diff)


def inner_product(query, vectors):
    # negated so lower = closer, consistent with l2
    return -(vectors @ query)


def cosine(query, vectors):
    qn = np.linalg.norm(query)
    vn = np.linalg.norm(vectors, axis=1)
    sims = (vectors @ query) / (qn * vn + 1e-12)
    return 1.0 - sims


METRICS = {"l2": l2, "ip": inner_product, "cosine": cosine}
