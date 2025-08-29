import pickle
import pytest


def qualname(x):
    return ":".join((x.__module__, x.__name__))


def dumps_or_skip(obj, *, protocol: int):
    try:
        return pickle.dumps(obj, protocol=protocol)
    except TypeError as exc:
        # Python 3.11+ changed the error message for __slots__ pickling
        if ("__slots__" in str(exc) and protocol < 2):
            pytest.skip(f"Skipping due to slots pickling issue: {exc}")
