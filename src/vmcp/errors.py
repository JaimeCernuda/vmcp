"""
vMCP error types and codes following JSON-RPC 2.0 conventions.

This module defines comprehensive error handling for the vMCP system,
including standard JSON-RPC errors and vMCP-specific error codes.
"""

from enum import IntEnum
from typing import Any


class VMCPErrorCode(IntEnum):
    """vMCP error codes following JSON-RPC conventions."""

    # JSON-RPC standard errors (-32768 to -32000)
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Transport errors (1xxx)
    TRANSPORT_ERROR = 1000
    CONNECTION_FAILED = 1001
    CONNECTION_TIMEOUT = 1002
    TRANSPORT_CLOSED = 1003
    MESSAGE_TOO_LARGE = 1004
    PROTOCOL_VERSION_MISMATCH = 1005

    # Routing errors (2xxx)
    NO_SERVER_FOUND = 2000
    SERVER_UNAVAILABLE = 2001
    ROUTING_FAILED = 2002
    ALL_SERVERS_DOWN = 2003
    LOAD_BALANCER_ERROR = 2004
    CIRCUIT_BREAKER_OPEN = 2005

    # Permission errors (3xxx)
    UNAUTHORIZED = 3000
    PERSONA_NOT_FOUND = 3001
    SERVER_NOT_ALLOWED = 3002
    INSUFFICIENT_PERMISSIONS = 3003
    ACCESS_DENIED = 3004
    AUTHENTICATION_REQUIRED = 3005

    # Protocol errors (4xxx)
    PROTOCOL_ERROR = 4000
    UNSUPPORTED_VERSION = 4001
    INVALID_MESSAGE = 4002
    HANDSHAKE_FAILED = 4003
    CAPABILITY_MISMATCH = 4004

    # Server errors (5xxx)
    GATEWAY_ERROR = 5000
    BACKEND_ERROR = 5001
    CACHE_ERROR = 5002
    CONFIGURATION_ERROR = 5003
    REGISTRY_ERROR = 5004
    HEALTH_CHECK_FAILED = 5005

    # Resource errors (6xxx)
    RESOURCE_NOT_FOUND = 6000
    RESOURCE_UNAVAILABLE = 6001
    RESOURCE_LOCKED = 6002
    QUOTA_EXCEEDED = 6003
    RATE_LIMIT_EXCEEDED = 6004

    # Repository errors (7xxx)
    REPOSITORY_ERROR = 7000
    PACKAGE_NOT_FOUND = 7001
    INSTALLATION_FAILED = 7002
    VERSION_CONFLICT = 7003
    DEPENDENCY_ERROR = 7004
    REPOSITORY_SYNC_FAILED = 7005

    # Extension errors (8xxx)
    EXTENSION_ERROR = 8000
    EXTENSION_NOT_FOUND = 8001
    EXTENSION_INSTALL_FAILED = 8002
    EXTENSION_ENABLE_FAILED = 8003
    EXTENSION_INVALID_MANIFEST = 8004


class VMCPError(Exception):
    """Base exception for vMCP errors with JSON-RPC compatibility."""

    def __init__(
        self,
        code: VMCPErrorCode,
        message: str,
        data: dict[str, Any] | None = None,
        server_id: str | None = None,
        request_id: Any | None = None,
    ) -> None:
        """
        Initialize vMCP error.

        Args:
            code: Error code from VMCPErrorCode enum
            message: Human-readable error message
            data: Additional error data (optional)
            server_id: ID of server that caused the error (optional)
            request_id: ID of request that caused the error (optional)
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}
        self.server_id = server_id
        self.request_id = request_id

        # Add server context to data if provided
        if server_id:
            self.data["server_id"] = server_id

    def to_json_rpc_error(self, request_id: Any = None) -> dict[str, Any]:
        """
        Convert to JSON-RPC error response.

        Args:
            request_id: Request ID to include in response

        Returns:
            JSON-RPC error response dictionary
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id or self.request_id,
            "error": {
                "code": int(self.code),
                "message": self.message,
                "data": self.data,
            },
        }

    def __repr__(self) -> str:
        """String representation of error."""
        return f"VMCPError({self.code}, {self.message!r}, data={self.data})"


# Convenience error classes for specific error categories


class TransportError(VMCPError):
    """Transport-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.TRANSPORT_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class VMCPConnectionError(TransportError):
    """Connection-related transport errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, VMCPErrorCode.CONNECTION_FAILED, **kwargs)


class ConnectionTimeoutError(TransportError):
    """Connection timeout errors."""

    def __init__(self, message: str, timeout: float, **kwargs: Any) -> None:
        kwargs["timeout"] = timeout
        super().__init__(message, VMCPErrorCode.CONNECTION_TIMEOUT, **kwargs)


class RoutingError(VMCPError):
    """Routing-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.ROUTING_FAILED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class ServerNotFoundError(RoutingError):
    """Server not found error."""

    def __init__(self, server_id: str, method: str | None = None) -> None:
        message = f"Server '{server_id}' not found"
        if method:
            message += f" for method '{method}'"
        super().__init__(
            message, VMCPErrorCode.NO_SERVER_FOUND, server_id=server_id, method=method
        )


class ServerUnavailableError(RoutingError):
    """Server unavailable error."""

    def __init__(
        self, server_id: str, reason: str | None = None, **kwargs: Any
    ) -> None:
        message = f"Server '{server_id}' is unavailable"
        if reason:
            message += f": {reason}"
        kwargs["reason"] = reason
        super().__init__(
            message, VMCPErrorCode.SERVER_UNAVAILABLE, server_id=server_id, **kwargs
        )


class CircuitBreakerOpenError(RoutingError):
    """Circuit breaker open error."""

    def __init__(
        self,
        server_id: str,
        failure_count: int,
        last_failure_time: float | None = None,
        **kwargs: Any,
    ) -> None:
        message = (
            f"Circuit breaker open for server '{server_id}' (failures: {failure_count})"
        )
        kwargs.update(
            {
                "failure_count": failure_count,
                "last_failure_time": last_failure_time,
            }
        )
        super().__init__(
            message, VMCPErrorCode.CIRCUIT_BREAKER_OPEN, server_id=server_id, **kwargs
        )


class VMCPPermissionError(VMCPError):
    """Permission-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.UNAUTHORIZED,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class PersonaNotFoundError(PermissionError):
    """Persona not found error."""

    def __init__(self, persona_name: str) -> None:
        super().__init__(
            f"Persona '{persona_name}' not found",
            VMCPErrorCode.PERSONA_NOT_FOUND,
            persona=persona_name,
        )


class ServerNotAllowedError(PermissionError):
    """Server not allowed for persona error."""

    def __init__(
        self, persona_name: str, server_id: str, method: str | None = None
    ) -> None:
        message = f"Persona '{persona_name}' not allowed to access server '{server_id}'"
        if method:
            message += f" for method '{method}'"
        super().__init__(
            message,
            VMCPErrorCode.SERVER_NOT_ALLOWED,
            persona=persona_name,
            server_id=server_id,
            method=method,
        )


class ProtocolError(VMCPError):
    """Protocol-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.PROTOCOL_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class UnsupportedVersionError(ProtocolError):
    """Unsupported protocol version error."""

    def __init__(self, version: str, supported_versions: list[str]) -> None:
        super().__init__(
            f"Unsupported protocol version '{version}'. Supported: {', '.join(supported_versions)}",
            VMCPErrorCode.UNSUPPORTED_VERSION,
            version=version,
            supported_versions=supported_versions,
        )


class InvalidMessageError(ProtocolError):
    """Invalid message format error."""

    def __init__(
        self, message: str, validation_errors: list[str] | None = None
    ) -> None:
        super().__init__(
            f"Invalid message format: {message}",
            VMCPErrorCode.INVALID_MESSAGE,
            validation_errors=validation_errors or [],
        )


class ConfigurationError(VMCPError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(VMCPErrorCode.CONFIGURATION_ERROR, message, kwargs)


class RegistryError(VMCPError):
    """Registry-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(VMCPErrorCode.REGISTRY_ERROR, message, kwargs)


class CacheError(VMCPError):
    """Cache-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(VMCPErrorCode.CACHE_ERROR, message, kwargs)


class RepositoryError(VMCPError):
    """Repository-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.REPOSITORY_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class PackageNotFoundError(RepositoryError):
    """Package not found in repository error."""

    def __init__(self, package_name: str, repository: str | None = None) -> None:
        message = f"Package '{package_name}' not found"
        if repository:
            message += f" in repository '{repository}'"
        super().__init__(
            message,
            VMCPErrorCode.PACKAGE_NOT_FOUND,
            package=package_name,
            repository=repository,
        )


class InstallationFailedError(RepositoryError):
    """Package installation failed error."""

    def __init__(self, package_name: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            f"Failed to install package '{package_name}': {reason}",
            VMCPErrorCode.INSTALLATION_FAILED,
            package=package_name,
            reason=reason,
            **kwargs,
        )


class RateLimitExceededError(VMCPError):
    """Rate limit exceeded error."""

    def __init__(
        self, limit: int, window: int, retry_after: int | None = None, **kwargs: Any
    ) -> None:
        message = f"Rate limit exceeded: {limit} requests per {window}s"
        if retry_after:
            message += f". Retry after {retry_after}s"
        super().__init__(
            VMCPErrorCode.RATE_LIMIT_EXCEEDED,
            message,
            {"limit": limit, "window": window, "retry_after": retry_after, **kwargs},
        )


class ExtensionError(VMCPError):
    """Extension-related errors."""

    def __init__(
        self,
        message: str,
        code: VMCPErrorCode = VMCPErrorCode.EXTENSION_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(code, message, kwargs)


class ExtensionNotFoundError(ExtensionError):
    """Extension not found error."""

    def __init__(self, extension_id: str, **kwargs: Any) -> None:
        super().__init__(
            f"Extension '{extension_id}' not found",
            VMCPErrorCode.EXTENSION_NOT_FOUND,
            extension_id=extension_id,
            **kwargs,
        )


def error_from_json_rpc(
    error_dict: dict[str, Any], request_id: Any = None
) -> VMCPError:
    """
    Create VMCPError from JSON-RPC error dictionary.

    Args:
        error_dict: JSON-RPC error dictionary
        request_id: Associated request ID

    Returns:
        VMCPError instance
    """
    code = error_dict.get("code", VMCPErrorCode.INTERNAL_ERROR)
    message = error_dict.get("message", "Unknown error")
    data = error_dict.get("data", {})

    # Try to convert to specific error type
    try:
        error_code = VMCPErrorCode(code)
    except ValueError:
        error_code = VMCPErrorCode.INTERNAL_ERROR
        data["original_code"] = code

    return VMCPError(code=error_code, message=message, data=data, request_id=request_id)


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    if isinstance(error, VMCPError):
        # Retryable error codes
        retryable_codes = {
            VMCPErrorCode.CONNECTION_TIMEOUT,
            VMCPErrorCode.SERVER_UNAVAILABLE,
            VMCPErrorCode.TRANSPORT_ERROR,
            VMCPErrorCode.BACKEND_ERROR,
            VMCPErrorCode.HEALTH_CHECK_FAILED,
            VMCPErrorCode.RATE_LIMIT_EXCEEDED,
        }
        return error.code in retryable_codes

    # Standard Python exceptions that are retryable
    retryable_types = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    return isinstance(error, retryable_types)
