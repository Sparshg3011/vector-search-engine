from vecstore.eval import recall


def test_perfect_recall():
    assert recall([1, 2, 3], [3, 2, 1]) == 1.0


def test_zero_recall():
    assert recall([1, 2, 3], [4, 5, 6]) == 0.0


def test_partial_recall():
    assert recall([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5


def test_extra_results_do_not_inflate_recall():
    assert recall([1, 2], [1, 2, 3, 4, 5]) == 1.0
