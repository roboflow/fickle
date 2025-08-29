import importlib
import io
import sys
from collections import defaultdict
import threading
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import weakref

import attr

from .util import IdMap
from .exc import FirewallError, UnpickleTypeError, UnpickleError

__all__ = [
    "Firewall",
    "DefaultFirewall",
    "ImportRequest",
    "BaseHandler",
    "Handler",
    "Import",
    "Literal",
    "Factory",
    "Method",
    "Unknown",
    "UnknownImport",
    "UnknownHandler",
    "UseDefault",
    "ListHandler",
    "DictHandler",
    "SetHandler",
    "OrderedDictHandler",
    "CodecsEncodeHandler",
    "ComplexHandler",
]


_COPYREG_RECONSTRUCTOR = object()


@attr.s
class _HandlerCache:
    """
    This exists because the user can pass in Handler subclasses rather than instances,
    and we need to instantiate those subclasses but recycle them as much as possible.
    """

    handlers = attr.ib(factory=weakref.WeakValueDictionary)
    lock = attr.ib(factory=threading.Lock)

    def __getitem__(
        self, handler: "Union[BaseHandler, Type[BaseHandler]]"
    ) -> "BaseHandler":
        with self.lock:
            # if it's an instance, remember its type
            if isinstance(handler, BaseHandler):
                self.handlers.setdefault(type(handler), handler)
                return handler

            # if it's a subclass, then create an instance or return cached one
            handler_type = handler
            handler = self.handlers.get(handler_type)
            if handler is None:
                self.handlers[handler_type] = handler = handler_type()

            return handler


@attr.s
class Firewall:
    """
    The firewall determines what calls are allowed and how they should be handled.
    The actual calls are handled by :class:`Handler` objects which need to have
    been registered using :meth:`register`.

    To determine the appropriate handler, the firewall does the following:

    1. The object is checked against instance-registered handlers.
    2. The object type is checked against type-registered handlers.
    3. The object's superclasses are checked against subtype-registered handlers.

    The first one to match wins. If none apply, an error is raised.

    Parameters
    ----------
    unknown: bool
        Register default handlers which produce :class:`Unknown` objects.
    """

    unknown: bool = attr.ib(default=False)
    instance_handlers: IdMap = attr.ib(factory=IdMap, init=False)
    type_handlers: IdMap = attr.ib(factory=IdMap, init=False)
    subtype_handlers: IdMap = attr.ib(factory=IdMap, init=False)
    importers: "Dict[Tuple[str, ...], List[Tuple[BaseHandler, str]]]" = attr.ib(
        factory=lambda: defaultdict(list), init=False
    )
    handler_cache = attr.ib(factory=_HandlerCache, init=False)

    def __attrs_post_init__(self):
        for h in self.get_autoregister_handlers():
            h.autoregister(self)

    def load(self, file) -> object:
        """
        Load pickled data from ``file``. Return loaded data.

        Convenience method to avoid constructing the Unpickler object explicitly.
        """
        from .unpickler import Unpickler

        return Unpickler(file=file, firewall=self).load()

    def loads(self, s: bytes) -> object:
        """
        Load pickled data from provided bytestring. Return loaded data.
        """
        return self.load(io.BytesIO(s))

    def _assert_identifier(self, s):
        if not s.isidentifier():
            raise UnpickleError(f"bad identifier {s!r}")

    def _object_path_to_tuple(self, path: str, name: str = None):
        if name is None:
            module, sep, name = path.partition(":")
        else:
            module = path
            sep = ":"

        key = module.split(".") if module else []

        for s in key:
            self._assert_identifier(s)

        if sep:
            if name:
                if not key:
                    raise AssertionError("cannot have empty module but nonempty name")
                self._assert_identifier(name)
                key.append("")
                key.append(name)
            else:
                key.append("")

        return tuple(key)

    def get_autoregister_handlers(self):
        if self.unknown:
            return [UnknownHandler()]
        else:
            return []

    def register_instance(
        self, handler: "Union[BaseHandler, Type[BaseHandler]]", instance: object
    ):
        """
        Register a method handler for a specific object instance.

        If the object is precisely ``instance``, then this handler applies. If there
        already exists a handler, then it is replaced.
        """
        self.instance_handlers[instance] = self.handler_cache[handler]

    def register_type(
        self, handler: "Union[BaseHandler, Type[BaseHandler]]", object_type: type
    ):
        """
        Register a method handler for all objects of a specific type.

        If there already exists a handler, then it is replaced.
        """
        self.type_handlers[object_type] = self.handler_cache[handler]

    def register_subtype(
        self, handler: "Union[BaseHandler, Type[BaseHandler]]", object_subtype: object
    ):
        """
        Register a method handler for all objects whose class inherits from this type.

        If there already exists a handler, then it is replaced.
        """
        self.subtype_handlers[object_subtype] = self.handler_cache[handler]

    def register_importer(
        self, handler: "Union[BaseHandler, Type[BaseHandler]]", path: str
    ):
        """
        Register a handler when a subpath of ``path`` is imported.

        If the pickle file asks for "pkg.subpkg.MyClass", then the following
        paths will be queried:

        - "pkg.subpkg:MyClass"
        - "pkg.subpkg:"
        - "pkg.subpkg"
        - "pkg"
        - ""

        If the handler is registered using one of these tuples, then it will
        be called with the requested module and name ("pkg.subpkg" and
        "MyClass"). If it raises NotImplementedError, then the next handler
        will be called if there is one.
        """
        assert type(path) is str

        self.importers[self._object_path_to_tuple(path)].append(
            (self.handler_cache[handler], path)
        )

    def get_handler(self, instance: object, method: str) -> "BaseHandler":
        h = self.instance_handlers.get(instance)
        if h is not None:
            return h

        h = self.type_handlers.get(type(instance))
        if h is not None:
            return h

        for cls in type(instance).mro():
            h = self.subtype_handlers.get(cls)
            if h is not None:
                return h

        raise FirewallError(f"cannot find handler for {instance!r}.{method:s}")

    @staticmethod
    def _assert_nonempty_str(x):
        if type(x) is not str or not x:
            raise UnpickleTypeError("must be nonempty string")

    def call_exception(self, exc, handler, method, instance, args, kwargs):
        """
        This method will get called if a handler call_* method (e.g.,
        :meth:`BaseHandler.call_new`) raises an exception.

        By default this method raises a :meth:`FirewallError`, but you can
        make it log instead (for example) and re-raise the same exception.
        """

        raise FirewallError(
            f"handler {handler!r} failed to handle {method:s} "
            f"on instance={instance!r} with "
            f"args={args!r} and kwargs={kwargs!r}"
        ) from exc

    def call(self, instance, method, *args, **kwargs):
        handler = self.get_handler(instance, method)
        try:
            return getattr(handler, method)(self, instance, *args, **kwargs)
        except Exception as exc:
            return self.call_exception(
                exc=exc,
                handler=handler,
                method=method,
                instance=instance,
                args=args,
                kwargs=kwargs,
            )

    def import_object(self, module: str, name: str) -> object:
        self._assert_nonempty_str(module)
        self._assert_nonempty_str(name)

        # "pkg.subpkg:MyClass" becomes ("pkg", "subpkg", "", "MyClass")
        importers = self.importers
        path = self._object_path_to_tuple(module, name)

        # look up, in this order:
        # - ("pkg", "subpkg", "", "MyClass")
        # - ("pkg", "subpkg", "")
        # - ("pkg", "subpkg")
        # - ("pkg",)
        # - ()
        for i in range(len(path), -1, -1):
            subpath = path[:i]
            for handler, rule_path in reversed(importers.get(subpath, ())):
                try:
                    return handler.import_object(
                        ImportRequest(handler, self, rule_path, module, name)
                    )
                except NotImplementedError:
                    pass

        module_and_name = f"{module}:{name}"
        raise FirewallError(f"access denied to {module_and_name!r}")


@attr.s
class ImportRequest:
    handler: "Handler" = attr.ib()
    firewall: "Firewall" = attr.ib()
    registered_path: str = attr.ib()
    module: str = attr.ib()
    name: str = attr.ib()


class BaseHandler:
    """
    Unpickle method call handler for an object instance.

    This lacks all the automatic registration magic in :class:`Handler`, so you
    probably want that one instead.
    """

    @staticmethod
    def _check_kwargs(d):
        if type(d) is not dict:
            raise UnpickleTypeError("invalid kwargs - not a dict")

        if not all(type(k) is str for k in d):
            raise UnpickleTypeError("invalid kwargs - keys not strings")

    @staticmethod
    def _check_args(x):
        t = type(x)
        if t is not list and t is not tuple:
            raise UnpickleTypeError("invalid args - not a tuple or list")

    def import_object(self, request: ImportRequest) -> object:
        raise NotImplementedError

    def autoregister(self, firewall: Firewall) -> None:
        pass

    def call_new(
        self, firewall: Firewall, instance, args: list, kwargs: Dict[str, object]
    ):
        """
        Called by NEWOBJ and NEWOBJ_EX opcodes.
        """
        raise NotImplementedError

    def call_unreduce(self, firewall: Firewall, instance, args):
        """
        Called by REDUCE opcode.
        """
        raise NotImplementedError

    def call_copyreg_reconstructor(self, firewall: Firewall, instance, base, state):
        """
        Used in older versions of the pickle protocol.
        """
        raise NotImplementedError

    def call_setstate(self, firewall: Firewall, instance, state):
        """
        Called by BUILD opcode when :attr:`instance` has a ``__setstate__`` method.
        """
        raise NotImplementedError

    def call_setstate_from_dict(
        self, firewall: Firewall, instance, state: dict, slotstate: dict
    ):
        """
        Called by BUILD opcode when :attr:`instance` does not have a ``__setstate__``
        method.
        """
        raise NotImplementedError

    def call_append(self, firewall: Firewall, instance, item):
        """
        Called by APPEND opcode to append to a list.
        """
        raise NotImplementedError

    def call_extend(self, firewall: Firewall, instance, sequence: list):
        """
        Called by EXTEND opcode to append to a list.
        """
        raise NotImplementedError

    def call_setitem(self, firewall: Firewall, instance, key, value):
        """
        Called by SETITEM and SETITEMS opcode to add to a dict.
        """
        raise NotImplementedError

    def call_additems(self, firewall: Firewall, instance, sequence: list):
        """
        Called by ADDITEMS opcode to add to a set.
        """
        raise NotImplementedError

    def danger_call_new(self, firewall: Firewall, instance, args, kwargs):
        """
        Wrapper which does extra validation before :meth:`call_new`.
        """
        self._check_args(args)
        self._check_kwargs(kwargs)
        return self.call_new(firewall, instance, args, kwargs)

    def danger_call_unreduce(self, firewall: Firewall, instance, args):
        """
        Wrapper which does extra validation before :meth:`call_unreduce`.
        """
        self._check_args(args)
        return self.call_unreduce(firewall, instance, args)

    def danger_call_setstate_from_dict(
        self, firewall: Firewall, instance, state, slotstate
    ):
        """
        Wrapper which does extra validation before :meth:`call_setstate_from_dict`.
        """
        t = type(state)
        if t is not dict:
            raise UnpickleTypeError(f"state must be dict, found {t!r}")

        t = type(slotstate)
        if t is not dict:
            raise UnpickleTypeError("slotstate must be dict, found {t!r}")

        return self.call_setstate_from_dict(firewall, instance, state, slotstate)

    def default_new(self, firewall, instance, args, kwargs):
        return instance.__new__(instance, *args, **kwargs)

    def default_unreduce(self, firewall, instance, args):
        return instance(*args)

    def default_setstate(self, firewall, instance, state):
        return instance.__setstate__(state)

    def default_setstate_from_dict(
        self, firewall, instance, state: dict, slotstate: dict
    ):
        inst_dict = instance.__dict__
        intern = sys.intern
        for k, v in state.items():
            if type(k) is str:
                k = intern(k)
            inst_dict[k] = v

        for k, v in slotstate.items():
            setattr(instance, k, v)

    def default_append(self, firewall, instance, item):
        instance.append(item)

    def default_extend(self, firewall, instance, sequence):
        instance.extend(sequence)

    def default_setitem(self, firewall, instance, key, value):
        instance[key] = value

    def default_additems(self, firewall, instance, sequence):
        instance.update(sequence)

    def default_copyreg_reconstructor(self, firewall, instance, base, state):
        # NOTE: copy-paste from CPython 3.11 module copyreg
        if base is object:
            obj = object.__new__(instance)
        else:
            obj = base.__new__(instance, state)
            if base.__init__ != object.__init__:
                base.__init__(obj, state)
        return obj


class UseDefault:
    """
    Call the default method directly. This may be dangerous, as no extra checking on the
    arguments is performed!
    """

    def __set_name__(self, owner, name):
        before, sep, after = name.partition("call_")
        if not sep or before:
            raise AssertionError(f"name {name!r} must start with call_")

        self.name = f"default_{after}"

    def __get__(self, instance, owner):
        return getattr(instance, self.name)


@attr.s
class _ImportHandler:
    """
    Base class for import handlers.
    """

    _attribute_doc = """
    instance_handler: Union[bool, BaseHandler, Type[BaseHandler]], optional
        If :attr:`instance_handler` is True, then also register the handler for the
        imported object. If it's None, then check
        :attr:`Handler.imports_register_instance`. If it is a Handler object
        or subclass, then register that handler instead.
    type_handler: Union[BaseHandler, Type[BaseHandler]], optional
        Register this handler for objects whose type is the imported object.
    subtype_handler: Union[BaseHandler, Type[BaseHandler]], optional
        Register this handler for objects whose type inherits from the imported object.
    """.rstrip()

    instance_handler: "Optional[Union[bool, Handler, Type[Handler]]]" = attr.ib(
        default=None, kw_only=True
    )
    type_handler: "Union[Handler, Type[Handler]]" = attr.ib(default=False, kw_only=True)
    subtype_handler: "Union[Handler, Type[Handler]]" = attr.ib(
        default=None, kw_only=True
    )

    def do(self, request: "ImportRequest") -> object:
        instance = self._do(request)

        instance_handler = self.instance_handler
        type_handler = self.type_handler
        subtype_handler = self.subtype_handler

        if instance_handler is None:
            instance_handler = request.handler.imports_register_instance

        if instance_handler is True:
            instance_handler = request.handler

        if instance_handler:
            request.firewall.register_instance(instance_handler, instance)

        if type_handler:
            request.firewall.register_type(type_handler, instance)

        if subtype_handler:
            request.firewall.register_instance(subtype_handler, instance)

        return instance

    def _do(self, request: "ImportRequest") -> object:
        raise NotImplementedError


@attr.s
class Literal(_ImportHandler):
    """
    Handle import by returning a user-provided object.

    Parameters
    ----------
    value: object
        Object that will be returned on import.
    """

    __doc__ += _ImportHandler._attribute_doc

    value: "object" = attr.ib()

    def _do(self, request: "ImportRequest"):
        return self.value


@attr.s
class Factory(_ImportHandler):
    """
    Handle import by calling a user-provided function.

    Parameters
    ----------
    factory: callable
        Callable which will be called on import. The callable must
        accept one argument (an instance of :class:`ImportRequest`).
    """

    __doc__ += _ImportHandler._attribute_doc

    factory: "Callable[[ImportRequest], object]" = attr.ib()

    def _do(self, request: "ImportRequest"):
        return self.factory(request)


@attr.s
class Method(_ImportHandler):
    """
    Handle import by calling a method on the :class:`Handler` object.

    Parameters
    ----------
    factory: callable
        Handler method which will be called on import. The method must
        accept one argument (an instance of :class:`ImportRequest`).
    """

    __doc__ += _ImportHandler._attribute_doc

    name: str = attr.ib()

    def _do(self, request: "ImportRequest"):
        return getattr(request.handler, self.name)(request)


@attr.s
class Import(_ImportHandler):
    """
    Handle import by importing a specific name.

    Parameters
    ----------
    module: str
        Name of the module from which the object will be imported. If empty,
        then the module requested by the pickled file will be imported instead.
    name: str
        Handler method which will be called on import. The method must
        accept one argument (an instance of :class:`ImportRequest`).
    """

    __doc__ += _ImportHandler._attribute_doc

    module: str = attr.ib(default=None)
    name: str = attr.ib(default=None)

    def __attrs_post_init__(self):
        if self.module and self.name is None:
            self.module, sep, name = self.module.rpartition(":")
            if sep:
                self.name = name

    def _do(self, request: "ImportRequest"):
        return getattr(
            importlib.import_module(self.module or request.module),
            self.name or request.name,
        )


class Handler(BaseHandler):
    """
    Unpickle method call handler for an object instance.

    Attributes
    ----------
    imports: dict, optional
        TODO
    imports_register_instance: bool, optional
        If True, then imported objects should be instance-registered with
        :meth:`Firewall.register_instance`. True by default.
    instances: Iterable[object], optional
        See :meth:`Firewall.register_instance`.
    types: Iterable[object], optional
        See :meth:`Firewall.register_type`.
    subtypes: Iterable[type], optional
        See :meth:`Firewall.register_subtype`.

    """

    imports: Dict[str, Union[_ImportHandler, object]] = {}
    imports_register_instance: bool = True

    instances: Iterable[object] = ()
    types: Iterable[type] = ()
    subtypes: Iterable[type] = ()

    def import_object(self, request: ImportRequest) -> object:
        value = self.imports.get(request.registered_path, None)

        if value is None:
            raise NotImplementedError

        if isinstance(value, _ImportHandler):
            return value.do(request)
        else:
            if self.imports_register_instance:
                request.firewall.register_instance(self, value)
            return value

    def autoregister(self, firewall: "Firewall"):
        """
        Automatically register this handler class for instance
        :attr:`AUTOREGISTER_INSTANCE`, for class :attr:`AUTOREGISTER_TYPE`, and for
        superclasses of :attr:`AUTOREGISTER_SUBTYPE`.

        Also register this handler to be consulted when importing from import paths
        given in :attr:`AUTOREGISTER_OBJECT_PATHS`.
        """

        for x in self.instances:
            firewall.register_instance(self, x)

        for x in self.types:
            firewall.register_type(self, x)

        for x in self.subtypes:
            firewall.register_subtype(self, x)

        for path in self.imports.keys():
            firewall.register_importer(self, path)


class CopyregReconstructorHandler(Handler):
    _COPYREG_RECONSTRUCTOR = object()
    imports = {"copy_reg:_reconstructor": _COPYREG_RECONSTRUCTOR}

    def call_unreduce(self, firewall, instance, args):
        cls, base, state = args
        return firewall.call(cls, "call_copyreg_reconstructor", base, state)


class ObjectHandler(Handler):
    imports = {"__builtin__:object": object}

    def call_unreduce(self, firewall, instance, args):
        if args:
            raise ValueError("do not support args")

        return self.default_unreduce(firewall, instance, ())


class ListHandler(Handler):
    types = [list]

    call_append = UseDefault()
    call_extend = UseDefault()


class DictHandler(Handler):
    types = [dict]

    call_setitem = UseDefault()


class SetInstanceHandler(Handler):
    types = [set]

    call_additems = UseDefault()


class SetHandler(Handler):
    imports = {}
    imports["__builtin__:set"] = imports["builtins:set"] = set
    imports["__builtin__:frozenset"] = imports["builtins:frozenset"] = frozenset

    def call_unreduce(self, firewall, instance, args):
        if args:
            [iterable] = args
            if type(iterable) is not tuple and type(iterable) is not list:
                raise TypeError("must be tuple or list")
            return instance(iterable)
        else:
            return instance()


class OrderedDictHandler(Handler):
    imports = {
        "collections:OrderedDict": Import(type_handler=DictHandler)
    }

    def call_unreduce(self, firewall, instance, args):
        if args:
            raise NotImplementedError("ordereddict args not supported")
        # TODO: support initializing from a sequence or mapping
        return self.default_unreduce(firewall, instance, ())


class ComplexHandler(Handler):
    imports = {"builtins:complex": complex, "__builtin__:complex": complex}

    def call_unreduce(self, firewall, instance, args):
        a, b = args
        if type(a) is not float or type(b) is not float:
            raise TypeError(f"bad complex {args!r}")
        return instance(a, b)


@attr.s(eq=False)
class UnknownImport:
    module: str = attr.ib()
    name: str = attr.ib()

    def __repr__(self):
        return f"U[{self.module}:{self.name}]"

    def __call__(self, args, kwargs={}):
        return Unknown(type=self, args=args, kwargs=kwargs)


@attr.s(eq=False)
class Unknown:
    type: object = attr.ib()
    args: list = attr.ib(factory=list)
    kwargs: dict = attr.ib(factory=dict)
    state: object = attr.ib(default=None)

    def __setstate__(self, state):
        raise NotImplementedError

    def __call__(self, args, kwargs={}):
        return Unknown(type=self, args=args, kwargs=kwargs)

    def __repr__(self):
        lst = [repr(x) for x in self.args]
        if self.kwargs:
            lst.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        if self.state is not None:
            lst.append(f"[state]={self.state!r}")
        args = ", ".join(lst)
        return f"{self.type!r}({args})"


class UnknownHandler(Handler):
    types = (UnknownImport, Unknown)
    imports_register_instance = False
    imports = {"": Method("_import")}

    def _import(self, request: "ImportRequest") -> object:
        return UnknownImport(request.module, request.name)

    def call_setstate(self, firewall, instance, state):
        instance.state = state

    def call_unreduce(self, firewall, instance, args):
        return instance(args)

    def call_new(self, firewall, instance, args, kwargs):
        return instance(args, kwargs)


class CodecsEncodeHandler(Handler):
    imports = {"_codecs:encode": object()}

    ALLOWED_CODECS = {"latin1"}

    def call_unreduce(self, firewall, instance, args):
        string, codec = args
        if type(string) is not str:
            raise TypeError
        if type(codec) is not str:
            raise TypeError
        if codec not in self.ALLOWED_CODECS:
            raise AssertionError(f"bad codec {codec}")

        return string.encode(codec)


class DefaultFirewall(Firewall):
    def get_autoregister_handlers(self):
        lst = super().get_autoregister_handlers()
        lst.extend(
            (
                CopyregReconstructorHandler(),
                ListHandler(),
                DictHandler(),
                SetHandler(),
                SetInstanceHandler(),
                OrderedDictHandler(),
                ObjectHandler(),
                CodecsEncodeHandler(),
                ComplexHandler(),
            )
        )
        return lst
