import pytest

from fickle import DefaultFirewall, Handler

from .conftest import dumps_or_skip, qualname


class ProtectionFailed(AssertionError):
    pass


class RefuseException(Exception):
    pass


class Dangerous:
    def __init__(self, break_my_security: bool):
        self.break_my_security = break_my_security

    def __getstate__(self):
        return self.break_my_security

    def __setstate__(self, state):
        self.break_my_security = state
        if state:
            raise ProtectionFailed


class DangerousHandler(Handler):
    imports = {qualname(Dangerous): Dangerous}

    def call_new(self, firewall, instance, args, kwargs):
        if args or kwargs:
            raise RefuseException
        return self.default_new(firewall, instance, args, kwargs)


class DangerousInstanceHandler(Handler):
    types = [Dangerous]

    def call_setstate(self, firewall, instance, state):
        if state:
            raise RefuseException
        self.default_setstate(firewall, instance, state)


class NoReraiseFirewall(DefaultFirewall):
    def call_exception(self, exc, **kw):
        raise exc


@pytest.mark.parametrize("protocol", range(2, 5))
def test_refuse(protocol):
    fw = NoReraiseFirewall()
    DangerousHandler().autoregister(fw)
    DangerousInstanceHandler().autoregister(fw)

    assert (
        fw.loads(dumps_or_skip(Dangerous(False), protocol=protocol)).break_my_security
        is False
    )

    with pytest.raises(RefuseException):
        fw.loads(dumps_or_skip(Dangerous(True), protocol=protocol))
