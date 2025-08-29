import pickle
import pytest


def qualname(x):
    return ":".join((x.__module__, x.__name__))


def dumps_or_skip(obj, *, protocol: int):
    try:
        return pickle.dumps(obj, protocol=protocol)
    except TypeError as exc:
        if "object: a class that defines __slots__" in str(exc) and protocol < 2:
            pytest.skip()
