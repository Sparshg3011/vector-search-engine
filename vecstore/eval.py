import numpy as np


def recall(true_ids, got_ids):
    """Fraction of the true neighbors that made it into the results.
    Order doesn't matter — recall@k only asks 'did you find them'."""
    true_ids = np.asarray(true_ids).ravel()
    got_ids = np.asarray(got_ids).ravel()
    if len(true_ids) == 0:
        return 1.0
    return len(set(true_ids) & set(got_ids)) / len(true_ids)
