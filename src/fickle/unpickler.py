import dataclasses
import pickletools
from typing import Dict, List

from .exc import UnpickleStopped, UnpickleError
from .firewall import Firewall


__all__ = ["Unpickler"]


class OpcodeMethodMapping:
    literal_opcodes = """
INT
BININT BININT1 BININT2
LONG LONG1 LONG4

STRING BINSTRING SHORT_BINSTRING
BINBYTES SHORT_BINBYTES BINBYTES8
BYTEARRAY8

UNICODE
SHORT_BINUNICODE BINUNICODE BINUNICODE8

FLOAT
BINFLOAT
""".split()

    aliased_opcodes = [
        "PUT BINPUT LONG_BINPUT",
        "GET BINGET LONG_BINGET",
        "EXT1 EXT2 EXT4",
    ]

    def __init__(self):
        self.processor_mapping = {}
        self.update_mapping()

    def update_mapping(self):
        self.processor_mapping.update(
            (op.name, "process_op_" + op.name) for op in pickletools.code2op.values()
        )

        for k in self.literal_opcodes:
            self.processor_mapping[k] = "process_literal"

        for line in self.aliased_opcodes:
            opcodes = line.split()
            method = "process_op_" + opcodes[0]
            for opcode in opcodes:
                self.processor_mapping[opcode] = method


@dataclasses.dataclass
class Unpickler:
    """
    This class implements a subset of the Pickle stack machine as described in
    :py:mod:`pickletools`.
    """

    file: object = dataclasses.field()
    firewall: Firewall = dataclasses.field()
    stack: list = dataclasses.field(default_factory=list, init=False)
    metastack: List[list] = dataclasses.field(default_factory=list, init=False)
    memo: Dict[int, object] = dataclasses.field(default_factory=dict, init=False)
    opcode_to_method_mapping = OpcodeMethodMapping()

    def load(self) -> object:
        process_methods = self.opcode_to_method_mapping.processor_mapping
        iterator = pickletools.genops(self.file)
        unpickle_stopped = False
        try:
            for opcode, arg, pos in iterator:
                method = getattr(self, process_methods[opcode.name])
                if opcode.arg is None:
                    method()
                else:
                    method(arg)
        except UnpickleStopped:
            unpickle_stopped = True

        if not unpickle_stopped:
            raise UnpickleError("did not find STOP opcode")

        for remaining in iterator:
            raise UnpickleError("instructions after STOP opcode")

        return self.result

    def pop(self):
        return self.stack.pop()

    def push(self, item):
        self.stack.append(item)

    def pop_many(self, count: int):
        xs = self.stack[-count:]
        if len(xs) != count:
            raise IndexError("stack underflow")
        del self.stack[-count:]
        return xs

    def pop_until_mark(self):
        result = self.stack
        self.stack = self.metastack.pop()
        return result

    def push_mark(self):
        self.metastack.append(self.stack)
        self.stack = []

    @property
    def top(self):
        return self.stack[-1]

    @property
    def stack_is_nonempty(self):
        """Return whether the stack has anything in it."""
        return bool(self.metastack or self.stack)

    def get_extension(self, code):
        """
        Return the extension object by integer code.

        The default implementation just raises IndexError.
        """
        raise IndexError(code)  # unknown ext code

    def find_class(self, module: str, name: str) -> object:
        return self.firewall.load_object(module, name)

    def persistent_load(self, persistent_id: str) -> object:
        raise NotImplementedError

    def process_literal(self, arg):
        self.push(arg)

    def process_op_APPEND(self):
        obj = self.pop()
        top = self.top
        self.firewall.call(top, "call_append", obj)

    def process_op_APPENDS(self):
        lst = self.pop_until_mark()
        top = self.top
        self.firewall.call(top, "call_extend", lst)

    def process_op_NONE(self):
        self.push(None)

    def process_op_NEWTRUE(self):
        self.push(True)

    def process_op_NEWFALSE(self):
        self.push(False)

    def process_op_EMPTY_LIST(self):
        self.push([])

    def process_op_EMPTY_TUPLE(self):
        self.push(())

    def process_op_EMPTY_DICT(self):
        self.push({})

    def process_op_EMPTY_SET(self):
        self.push(set())

    def process_op_TUPLE(self):
        self.push(tuple(self.pop_until_mark()))

    def process_op_TUPLE1(self):
        self.push(tuple(self.pop_many(1)))

    def process_op_TUPLE2(self):
        self.push(tuple(self.pop_many(2)))

    def process_op_TUPLE3(self):
        self.push(tuple(self.pop_many(3)))

    def process_op_LIST(self):
        self.push(self.pop_until_mark())

    def process_op_SETITEM(self):
        k, v = self.pop_many(2)
        d = self.top
        self.firewall.call(d, "call_setitem", k, v)

    def process_op_SETITEMS(self):
        kvs = self.pop_until_mark()
        if len(kvs) % 2 != 0:
            raise IndexError("must have an even number of arguments")

        d = self.top
        for i in range(0, len(kvs), 2):
            self.firewall.call(d, "call_setitem", *kvs[i : i + 2])

    def process_op_DICT(self):
        kvs = self.pop_until_mark()
        if len(kvs) % 2 != 0:
            raise IndexError("must have an even number of arguments")

        d = {}
        for i in range(0, len(kvs), 2):
            d[kvs[i]] = d[kvs[i + 1]]

        self.push(d)

    def process_op_ADDITEMS(self):
        items = self.pop_until_mark()
        self.firewall.call(self.top, "call_additems", items)

    def process_op_FROZENSET(self):
        self.push(frozenset(self.pop_until_mark()))

    def process_op_MARK(self):
        self.push_mark()

    def process_op_POP(self):
        self.pop()

    def process_op_DUP(self):
        self.push(self.top)

    def process_op_POP_MARK(self):
        self.pop_until_mark()

    def process_op_GET(self, arg):
        self.push(self.memo[arg])

    def process_op_PUT(self, arg):
        self.memo[arg] = self.top

    def process_op_MEMOIZE(self):
        self.memo[len(self.memo)] = self.top

    def process_op_EXT(self, arg):
        self.push(self.get_extension(arg))

    def process_op_GLOBAL(self, arg):
        module, _, attribute = arg.partition(" ")
        # note: import_object validates the type of its arguments
        self.push(self.firewall.import_object(module, attribute))

    def process_op_STACK_GLOBAL(self):
        module, attribute = self.pop_many(2)
        # note: import_object validates the type of its arguments
        self.push(self.firewall.import_object(module, attribute))

    def process_op_REDUCE(self):
        func, args = self.pop_many(2)
        self.push(self.firewall.call(func, "danger_call_unreduce", args))

    def process_op_INST(self):
        raise NotImplementedError  # TODO

    def process_op_OBJ(self):
        raise NotImplementedError  # TODO

    def process_op_NEWOBJ(self):
        cls, args = self.pop_many(2)
        self.push(self.firewall.call(cls, "danger_call_new", args, {}))

    def process_op_NEWOBJ_EX(self):
        cls, args, kwargs = self.pop_many(3)
        self.push(self.firewall.call(cls, "danger_call_new", args, kwargs))

    def setstate(self, instance, state):
        setstate = getattr(instance, "__setstate__", None)
        if setstate is not None:
            self.firewall.call(instance, "call_setstate", state)
            return

        slotstate = {}
        if type(state) is tuple and len(state) == 2:
            state, slotstate = state

        if state is None:
            state = {}
        if slotstate is None:
            slotstate = {}

        self.firewall.call(instance, "danger_call_setstate_from_dict", state, slotstate)

    def process_op_BUILD(self):
        state = self.pop()
        inst = self.stack[-1]

        self.setstate(inst, state)

    def process_op_PROTO(self, arg):
        pass

    def process_op_STOP(self):
        self.result = self.pop()
        if self.stack_is_nonempty:
            raise UnpickleError("stack should be empty after STOP")
        raise UnpickleStopped

    def process_op_FRAME(self, arg):
        pass

    def process_op_PERSID(self, arg):
        self.push(self.persistent_load(arg))

    def process_op_BINPERSID(self):
        self.push(self.persistent_load(self.pop()))
