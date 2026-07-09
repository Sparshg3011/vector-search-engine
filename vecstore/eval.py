import numpy as np


def recall(true_ids, got_ids):
    """Fraction of the true neighbors that made it into the results.
    Order doesn't matter — recall@k only asks 'did you find them'.
    Dedupe both sides so the count and the denominator agree, and cast
    to python ints so numpy vs python ids compare equal."""
    true = set(np.asarray(true_ids).ravel().tolist())
    if not true:
        return 1.0
    got = set(np.asarray(got_ids).ravel().tolist())
    return len(true & got) / len(true)
