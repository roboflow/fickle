"""
PyTorch seems to store checkpoints in a weird zipfile that contains
a pickle file and a bunch of raw arrays. The pickle makes it a big security
risk to load other people's model weights, and unfortunately a lot of the
ML community uses such unsafe formats.

This is a module implementing loading for a decent subset of the format.
"""

import dataclasses
import functools
import operator
import pathlib
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Tuple

import attr
import numpy as np
from marshmallow import fields

from .. import DefaultFirewall, Handler, Unpickler
from . import marshmallow as mext

storage_manager_var = ContextVar("storage_manager")


@contextmanager
def setting(contextvar: ContextVar, value: object):
    token = contextvar.set(value)
    try:
        yield
    finally:
        contextvar.reset(token)


class InvalidDTypeWarning(UserWarning):
    pass


def trivial_strides_for_tensor_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    reversed_strides = []
    current = 1
    for dim in shape[::-1]:
        reversed_strides.append(current)
        current *= dim if dim else 1
    return tuple(reversed_strides[::-1])


@dataclasses.dataclass
class StoredTensor:
    storage: "StorageInfo"
    storage_offset: int
    size: Tuple[int]
    stride: Tuple[int]
    requires_grad: bool
    backward_hooks: object
    storage_manager: "StorageManager" = dataclasses.field(
        default_factory=lambda: storage_manager_var.get(None)
    )

    PYTORCH_DTYPE_TO_ITEMSIZE = {"bfloat16": 2, "bool": 1}
    PYTORCH_DTYPE_TO_ITEMSIZE.update((f"float{n * 8}", n) for n in (2, 4, 8))
    PYTORCH_DTYPE_TO_ITEMSIZE.update((f"complex{n * 8}", n) for n in (8, 16))
    PYTORCH_DTYPE_TO_ITEMSIZE.update((f"int{n * 8}", n) for n in (1, 2, 4, 8))
    PYTORCH_DTYPE_TO_ITEMSIZE.update((f"uint{n * 8}", n) for n in (1, 2, 4, 8))

    @functools.cached_property
    def numpy_dtype(self):
        dtype = self.storage.dtype
        if dtype not in self.PYTORCH_DTYPE_TO_ITEMSIZE:
            raise ValueError("invalid dtype: {dtype!r}")

        return np.dtype(dtype).newbyteorder("<")

    @functools.cached_property
    def has_trivial_layout(self) -> bool:
        """
        Does this tensor do any funny stride (i.e. memory layout) tricks?
        """
        if 0 in self.size:
            return True  # Empty tensors have no memory, so there's no memory layout.

        # If a dimension has length 1, the stride won't matter for it because the only
        # valid index is 0.
        return all(
            dim == 1 or trivial_stride == actual_stride
            for dim, actual_stride, trivial_stride in zip(
                self.size, self.stride, trivial_strides_for_tensor_shape(self.size)
            )
        )

    @property
    def size_in_bytes(self) -> int:
        itemsize = self.PYTORCH_DTYPE_TO_ITEMSIZE[self.storage.dtype]
        return functools.reduce(operator.mul, self.size, itemsize)

    def _load_from_file(self, fileobj):
        byte_size = self.size_in_bytes
        fileobj.seek(self.storage_offset)
        buf: bytes = fileobj.read(byte_size)
        if len(buf) != byte_size:
            raise ValueError(f"expected {len(buf)} bytes, got {byte_size}")
        return buf

    def _buffer_to_numpy(self, buffer):
        return np.frombuffer(buffer, self.numpy_dtype).reshape(self.size)

    def open_storage(self):
        return self.storage_manager.open(f"data/{self.storage.key}")

    def _load_from_zipfile(self):
        with self.open_storage() as f:
            return self._load_from_file(f)

    @property
    def array(self):
        if self.numpy_dtype is None:
            raise ValueError(f"invalid dtype {self.storage.dtype!r}")

        return self._buffer_to_numpy(self.buffer)

    @property
    def buffer(self):
        return self._load_from_zipfile()


@attr.s
class StorageInfo:
    dtype: str = attr.ib()
    key: str = attr.ib()


class StorageManager:
    def open(self, key: str):
        raise NotImplementedError


@attr.s
class StorageManagerZip(StorageManager):
    zip_file: zipfile.ZipFile = attr.ib()
    prefix: str = attr.ib(repr=False)

    def open(self, path: str):
        # ZipFile.open doesn't support 'b'...
        return self.zip_file.open(self.prefix + path, "r")


class StorageInfoField(mext.XTuple):
    def __init__(self, **kwargs):
        super().__init__((mext.XString(), mext.XString()), **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if type(value) is not tuple:
            raise fields.ValidationError

        value = value[1:3]
        value = super()._deserialize(value, attr, data, **kwargs)

        return StorageInfo(*value)


class StoredTensorHandler(mext.BaseMarshmallowHandler):
    imports = {"torch._utils:_rebuild_tensor_v2": StoredTensor}

    class Schema(mext.BaseMarshmallowHandler.Schema):
        storage = StorageInfoField()
        storage_offset = fields.Integer()
        size = mext.XListAsPyTuple(fields.Integer())
        stride = mext.XListAsPyTuple(fields.Integer())
        requires_grad = fields.Boolean()
        backward_hooks = fields.Raw()


class StorageHandler(Handler):
    imports = {
        "torch:BFloat16Storage": "bfloat16",
        "torch:HalfStorage": "float16",
        "torch:FloatStorage": "float32",
        "torch:DoubleStorage": "float64",
        "torch:ComplexFloatStorage": "complex64",
        "torch:ComplexDoubleStorage": "complex128",
        "torch:BoolStorage": "bool",
        "torch:CharStorage": "int8",
        "torch:ShortStorage": "int16",
        "torch:IntStorage": "int32",
        "torch:LongStorage": "int64",
        "torch:ByteStorage": "uint8",
    }
    imports_register_instance = False


class PyTorchFirewall(DefaultFirewall):
    def get_autoregister_handlers(self):
        handlers = super().get_autoregister_handlers()
        handlers.append(StorageHandler())
        handlers.append(StoredTensorHandler())
        return handlers


class MyUnpickler(Unpickler):
    def persistent_load(self, value):
        return value


def fake_torch_load_zipped(
    zip_file: zipfile.ZipFile,
    firewall: DefaultFirewall = None,
    Unpickler=MyUnpickler,
):
    """
    from zipfile import ZipFile

    zf = ZipFile("/data/ml/sd-v1-4.ckpt")
    zipped = fake_torch_load_zipped(zf)
    """

    if firewall is None:
        firewall = PyTorchFirewall(unknown=True)

    entries = [pathlib.PurePosixPath(entry.filename) for entry in zip_file.infolist()]
    entries = [entry for entry in entries if entry.name == "data.pkl"]
    if len(entries) != 1:
        raise ValueError(f"expected one data.pkl entry, found {len(entries)}")

    prefix = str(entries[0].parent) + "/"
    sm = StorageManagerZip(zip_file=zip_file, prefix=prefix)

    with setting(storage_manager_var, sm), sm.open("data.pkl") as pf:
        return Unpickler(file=pf, firewall=firewall).load()
