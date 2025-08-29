from collections import OrderedDict

import pytest

from fickle import DefaultFirewall, Handler, Literal, UseDefault

from .conftest import dumps_or_skip, qualname


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return type(self) is type(other) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((type(self), self.x, self.y))


class SlottedPoint(Point):
    __slots__ = ("x", "y")


class SetStatePoint(Point):
    def __getstate__(self):
        return {"px": self.x, "py": self.y}

    def __setstate__(self, d):
        self.x = d["px"]
        self.y = d["py"]


class PointInstanceHandler(Handler):
    call_setstate = UseDefault()
    call_setstate_from_dict = UseDefault()


class PointHandler(Handler):
    imports = {
        qualname(x): Literal(x, type_handler=PointInstanceHandler)
        for x in (Point, SlottedPoint, SetStatePoint)
    }

    call_copyreg_reconstructor = UseDefault()
    call_new = UseDefault()
    call_unreduce = UseDefault()


@pytest.mark.parametrize("protocol", range(0, 5))
@pytest.mark.parametrize(
    "obj",
    [
        Point(3, 5),
        SlottedPoint(0.1, 0.2 + 5j),
        (SetStatePoint(1.0, 2.0), ["test", b"poop"]),
        {"hello": Point(1, 2), "set": set([1, 2]), "frozenset": frozenset("abc")},
        {"tuple": (0, 1)},
        ("ordered", OrderedDict(((b"hello", 2), (4, 5), (2, 3)))),
    ],
)
def test_roundtrip(protocol, obj):
    fw = DefaultFirewall()
    PointHandler().autoregister(fw)
    PointInstanceHandler().autoregister(fw)

    pickled = dumps_or_skip(obj, protocol=protocol)
    unpickled = fw.loads(pickled)

    assert obj == unpickled
