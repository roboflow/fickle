from collections.abc import Mapping


class IdWrap:
    __slots__ = ("obj", "hash")

    def __init__(self, obj):
        self.obj = obj
        self.hash = hash(id(obj))

    def __eq__(self, other):
        return self.obj is other.obj

    def __hash__(self):
        return self.hash


class IdMap(Mapping):
    # TODO: avoid costly id() for PyPy

    def __init__(self):
        self.mapping = {}

    def __getitem__(self, key):
        return self.mapping[IdWrap(key)]

    def __setitem__(self, key, value):
        self.mapping[IdWrap(key)] = value

    def get(self, key, default=None):
        return self.mapping.get(IdWrap(key), default)

    def __iter__(self):
        return (idwrap.obj for idwrap in self.mapping)

    def __len__(self):
        return len(self.mapping)
