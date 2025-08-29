"""
Marshmallow is a deserialization and validation library. It seems natural
that some programmers may wish to use it to validate class constructor
parameters.
"""

from functools import cached_property
from typing import Dict, Tuple, Union

from marshmallow import RAISE, Schema, fields

from .. import Handler


class ExactTypeField(fields.Field):
    """
    A marshmallow field which requires an exact match on the input type.
    """

    required_type = None

    def __init__(self, *args, required_type: Union[type, Tuple[type]] = None, **kwargs):
        if required_type is None:
            required_type = self.required_type

        if type(required_type) is not tuple:
            required_type = (required_type,)
        self.required_type = required_type

        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        t = type(value)
        required_type = self.required_type

        if all(t is not req_t for req_t in required_type):
            raise fields.ValidationError(
                "{value!r} has type {t!r} but one of {required_type!r} is required"
            )

        return super()._deserialize(value, attr, data, **kwargs)


class XList(ExactTypeField, fields.List):
    """
    Like :py:class:`fields.List`, but requires the object to be an actual list.
    """

    required_type = list


class XListAsPyTuple(ExactTypeField, fields.List):
    """
    Like :py:class:`fields.List`, but requires the object to be a tuple, and
    also returns a tuple upon deserialization.
    """

    required_type = tuple

    def _deserialize(self, *args, **kwargs):
        # convert list to tuple
        value = super()._deserialize(*args, **kwargs)
        return tuple(value)


class XTuple(ExactTypeField, fields.Tuple):
    """
    Like :py:class:`fields.Tuple`, but requires the object to be an actual
    tuple.
    """

    required_type = tuple


class XTupleAsPyList(ExactTypeField, fields.Tuple):
    """
    Like :py:class:`fields.Tuple`, but requires the object to be a list, and
    also returns a list upon deserialization.
    """

    required_type = list

    def _deserialize(self, *args, **kwargs):
        # convert tuple to list
        value = super()._deserialize(*args, **kwargs)
        return list(value)


class XDict(ExactTypeField, fields.Dict):
    """
    Like :py:class:`fields.Dict`, but requires the object to be an actual
    dict.
    """

    required_type = dict


class XString(ExactTypeField, fields.String):
    """
    Like :py:class:`fields.String`, but requires the object to be an actual
    string or bytestring (which will then be decoded as utf-8).
    """

    required_type = (str, bytes)


class XRaw(ExactTypeField, fields.Raw):
    """
    Like :py:class:`fields.Raw`, but requires the object to be of a
    user-specified type.
    """


def map_arguments(positional_arguments, args, kwargs) -> Dict[str, object]:
    """
    Map positional and keyword arguments, and return a dictionary.
    Unexpected arguments are added to the dictionary as-is. We do not check
    that the returned argument names are a subset of the
    ``positional_arguments``.

    Parameters
    ----------
    argument_names: list
        List of expected argument names.
    args: list
        Arguments to be mapped.
    kwargs: dict
        Keyword arguments to be mapped.

    Raises
    ------
    TypeError
        Will be raised if an argument is specified both in ``args`` and
        ``kwargs``.

    Returns
    -------
    result: dict
        A dictionary where keys are argument names and values are the argument
        values.
    """
    result = {}

    # go through positional arguments
    for name, value in zip(positional_arguments, args):
        result[name] = value

    # now map the keyword arguments
    for name, value in kwargs.items():
        if name in result:
            raise TypeError(f"duplicate argument {name}")
        result[name] = value

    return result


class OrderedSchema(Schema):
    class Meta:
        ordered = True


class BaseMarshmallowHandler(Handler):
    """
    This handler uses a marshmallow schema to validate the arguments to the
    :py:meth:`call_unreduce` and :py:meth:`call_new` methods.

    Instead of the default marshmallow field classes :py:class:`fields.List`,
    :py:class:`fields.Dict` and so on, we recommend that you use subclasses of
    `ExactTypeField` such as :py:class:`XList` and :py:class:`XDict` because
    they will check that the type of the object is precisely ``list`` or
    ``dict`` and not just subclasses or types that happen to implement the
    right subset of methods.

    Examples
    --------

    The simplest possible case::

        @dataclasses.dataclass
        class MyPoint:
            x: int
            y: int

        class PointHandler(BaseMarshmallowHandler):
            AUTOREGISTER_INSTANCE = MyPoint
            AUTOREGISTER_OBJECT_PATHS = ("another_package.vectors:Point",)

            class Schema(OrderedSchema):
                x = fields.Integer()
                y = fields.Integer()

    """

    # TODO: also implement call_copyreg_reconstructor for old pickle protocol versions

    class Schema(OrderedSchema):
        pass

    @cached_property
    def unreduce_schema(self):
        """By default it's just :py:attr:`Schema`."""
        return self.Schema(unknown=RAISE, partial=False)

    @cached_property
    def new_schema(self):
        """By default it's just :py:attr:`Schema`."""
        return self.Schema(unknown=RAISE, partial=False)

    def _marshmallow_map_args(self, schema, args, kwargs):
        if not schema.opts.ordered:
            raise AssertionError("schema must be ordered")
        names = schema._declared_fields.keys()
        mapped_kwargs = map_arguments(names, args, kwargs)
        checked_kwargs = schema.load(mapped_kwargs)
        return checked_kwargs.values()

    def _marshmallow_map_kwargs(self, schema, args, kwargs):
        names = schema._declared_fields.keys()
        mapped_kwargs = map_arguments(names, args, kwargs)
        checked_kwargs = schema.load(mapped_kwargs)
        return checked_kwargs

    def call_unreduce(self, firewall, instance, args):
        args = self._marshmallow_map_args(self.unreduce_schema, args, {})
        return self.default_unreduce(firewall, instance, args)

    def call_new(self, firewall, instance, args, kwargs):
        kwargs = self._marshmallow_map_kwargs(self.new_schema, args, kwargs)
        return self.default_new(firewall, instance, (), kwargs)
