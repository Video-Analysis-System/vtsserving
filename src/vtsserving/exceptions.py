from __future__ import annotations

from http import HTTPStatus


class VtsServingException(Exception):
    """
    Base class for all VtsServing's errors.
    Each custom exception should be derived from this class
    """

    error_code = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class StateException(Exception):
    """
    Raise when the state of an object is not valid
    """

    error_code = HTTPStatus.BAD_REQUEST


class RemoteException(VtsServingException):
    """
    A special exception that is used to wrap the exception from remote server
    """

    def __init__(self, message: str, payload: VtsServingException | None = None):
        self.payload = payload
        super().__init__(message)


class InvalidArgument(VtsServingException):
    """
    Raise when VtsServing received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters
    """

    error_code = HTTPStatus.BAD_REQUEST


class InternalServerError(VtsServingException):
    """
    Raise when VtsServing received valid arguments from CLI arguments, HTTP
    Request, or python API function parameters, but got internal issues while
    processing.
    * Note to VtsServing org developers: raise this exception only when exceptions happend
    in the users' code (runner or service) and want to surface it to the user.
    """


class APIDeprecated(VtsServingException):
    """
    Raise when trying to use deprecated APIs of VtsServing
    """


class BadInput(InvalidArgument):
    """Raise when API server receiving bad input request"""

    error_code = HTTPStatus.BAD_REQUEST


class NotFound(VtsServingException):
    """
    Raise when specified resource or name not found
    """

    error_code = HTTPStatus.NOT_FOUND


class UnprocessableEntity(VtsServingException):
    """
    Raise when API server receiving unprocessable entity request
    """

    error_code = HTTPStatus.UNPROCESSABLE_ENTITY


class ServiceUnavailable(VtsServingException):
    """
    Raise when incoming requests exceeds the capacity of a server
    """

    error_code = HTTPStatus.SERVICE_UNAVAILABLE


class VtsServingConfigException(VtsServingException):
    """Raise when VtsServing is mis-configured or when required configuration is missing"""


class MissingDependencyException(VtsServingException):
    """
    Raise when VtsServing component failed to load required dependency - some VtsServing
    components has dependency that is optional to the library itself. For example,
    when using SklearnModel, the scikit-learn module is required although
    VtsServing does not require scikit-learn to be a dependency when installed
    """


class CLIException(VtsServingException):
    """Raise when CLI encounters an issue"""


class YataiRESTApiClientError(VtsServingException):
    pass


class ImportServiceError(VtsServingException):
    pass
