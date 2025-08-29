class UnpickleStopped(BaseException):
    pass


class UnpickleError(ValueError):
    pass


class UnpickleTypeError(UnpickleError):
    pass


class FirewallError(UnpickleError):
    pass
