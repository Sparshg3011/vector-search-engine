import pytest

from vecstore import datasets


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        datasets.load("not-a-real-dataset")
