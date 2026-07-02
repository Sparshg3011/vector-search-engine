import math


def random_level(m_l, rng):
    """Coin-flip layer assignment: most nodes stay at layer 0, each
    level up is ~M times rarer. This is what makes the graph a
    hierarchy without any explicit balancing."""
    return int(-math.log(1.0 - rng.random()) * m_l)
