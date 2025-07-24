"""
Unit tests for vMCP error handling system.
"""

import pytest

from vmcp.errors import (
    CircuitBreakerOpenError,
    ConfigurationError,
    RoutingError,
    ServerNotFoundError,
    ServerUnavailableError,
    TransportError,
    VMCPError,
    VMCPErrorCode,
    VMCPPermissionError,
    error_from_json_rpc,
)


class TestVMCPErrorCode:
    """Test error code enumeration."""

    def test_error_codes_exist(self):
        """Test that all expected error codes exist."""
        assert VMCPErrorCode.INVALID_REQUEST == -32600
        assert VMCPErrorCode.METHOD_NOT_FOUND == -32601
        assert VMCPErrorCode.INVALID_PARAMS == -32602
        assert VMCPErrorCode.INTERNAL_ERROR == -32603
        assert VMCPErrorCode.PARSE_ERROR == -32700

        # vMCP specific codes
        assert VMCPErrorCode.TRANSPORT_ERROR == 1000
        assert VMCPErrorCode.ROUTING_FAILED == 2002
        assert VMCPErrorCode.SERVER_UNAVAILABLE == 2001
        assert VMCPErrorCode.NO_SERVER_FOUND == 2000
        assert VMCPErrorCode.ALL_SERVERS_DOWN == 2003
        assert VMCPErrorCode.CONFIGURATION_ERROR == 5003
        assert VMCPErrorCode.UNAUTHORIZED == 3000
        assert VMCPErrorCode.GATEWAY_ERROR == 5000

    def test_error_code_values_unique(self):
        """Test that all error codes have unique values."""
        codes = [code.value for code in VMCPErrorCode]
        assert len(codes) == len(set(codes)), "Duplicate error code values found"


class TestVMCPError:
    """Test base VMCPError class."""

    def test_basic_error_creation(self):
        """Test creating a basic VMCP error."""
        error = VMCPError(VMCPErrorCode.INTERNAL_ERROR, "Test error")

        assert error.code == VMCPErrorCode.INTERNAL_ERROR
        assert error.message == "Test error"
        assert error.data == {}
        assert error.server_id is None

    def test_error_with_details(self):
        """Test creating error with details."""
        data = {"key": "value", "count": 42}
        error = VMCPError(
            VMCPErrorCode.ROUTING_FAILED,
            "Routing failed",
            data=data,
            server_id="test-server",
        )

        assert error.code == VMCPErrorCode.ROUTING_FAILED
        assert error.message == "Routing failed"
        assert error.data["key"] == "value"
        assert error.data["count"] == 42
        assert error.server_id == "test-server"

    def test_error_string_representation(self):
        """Test error string representation."""
        error = VMCPError(VMCPErrorCode.TRANSPORT_ERROR, "Connection failed")
        error_str = str(error)

        assert "Connection failed" in error_str

    def test_error_with_server_id_in_string(self):
        """Test error string includes server ID when present."""
        error = VMCPError(
            VMCPErrorCode.SERVER_UNAVAILABLE, "Server down", server_id="test-server"
        )
        error_str = str(error)

        assert "Server down" in error_str


class TestSpecificErrorTypes:
    """Test specific error type classes."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config file")

        assert error.code == VMCPErrorCode.CONFIGURATION_ERROR
        assert error.message == "Invalid config file"
        assert isinstance(error, VMCPError)

    def test_transport_error(self):
        """Test TransportError."""
        error = TransportError("Connection timeout")

        assert error.code == VMCPErrorCode.TRANSPORT_ERROR
        assert error.message == "Connection timeout"

    def test_routing_error(self):
        """Test RoutingError."""
        error = RoutingError("No route found")

        assert error.code == VMCPErrorCode.ROUTING_FAILED
        assert error.message == "No route found"

    def test_server_unavailable_error(self):
        """Test ServerUnavailableError."""
        error = ServerUnavailableError("server-1", reason="Connection refused")

        assert error.code == VMCPErrorCode.SERVER_UNAVAILABLE
        assert error.data["server_id"] == "server-1"
        assert "Connection refused" in error.message
        assert error.data["reason"] == "Connection refused"

    def test_no_server_found_error(self):
        """Test ServerNotFoundError."""
        error = ServerNotFoundError("server-1")

        assert error.code == VMCPErrorCode.NO_SERVER_FOUND
        assert error.data["server_id"] == "server-1"
        assert "Server 'server-1' not found" in error.message

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError."""
        error = CircuitBreakerOpenError("server-1", 5, 123456.0)

        assert error.code == VMCPErrorCode.CIRCUIT_BREAKER_OPEN
        assert error.data["server_id"] == "server-1"
        assert "Circuit breaker open" in error.message
        assert error.data["failure_count"] == 5
        assert error.data["last_failure_time"] == 123456.0

    def test_vmcp_permission_error(self):
        """Test VMCPPermissionError."""
        error = VMCPPermissionError("Access denied")

        assert error.code == VMCPErrorCode.UNAUTHORIZED
        assert error.message == "Access denied"

    def test_permission_denied_error(self):
        """Test PermissionDeniedError."""
        error = VMCPPermissionError("Access denied")

        assert error.code == VMCPErrorCode.UNAUTHORIZED
        assert error.message == "Access denied"


class TestJSONRPCErrorResponse:
    """Test JSON-RPC error response creation."""

    def test_create_basic_error_response(self):
        """Test creating basic JSON-RPC error response."""
        error = VMCPError(VMCPErrorCode.METHOD_NOT_FOUND, "Method not found")
        response = error.to_json_rpc_error(request_id=1)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method not found"
        assert response["error"]["data"] == {}

    def test_create_error_response_with_data(self):
        """Test creating JSON-RPC error response with data."""
        error_data = {"method": "unknown/method", "available": ["tools/list", "ping"]}
        error = VMCPError(
            VMCPErrorCode.METHOD_NOT_FOUND,
            "Method not found",
            data=error_data
        )
        response = error.to_json_rpc_error(request_id="test-id")

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id"
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method not found"
        assert response["error"]["data"] == error_data

    def test_create_error_response_null_id(self):
        """Test creating JSON-RPC error response with null ID."""
        error = VMCPError(VMCPErrorCode.PARSE_ERROR, "Parse error")
        response = error.to_json_rpc_error(request_id=None)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"

    def test_create_error_response_with_vmcp_error(self):
        """Test creating JSON-RPC error response from VMCPError."""
        vmcp_error = ServerUnavailableError("test-server", reason="Connection timeout")
        response = vmcp_error.to_json_rpc_error(request_id=42)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert response["error"]["code"] == VMCPErrorCode.SERVER_UNAVAILABLE.value
        assert "test-server" in response["error"]["message"]
        assert response["error"]["data"]["server_id"] == "test-server"
        assert response["error"]["data"]["reason"] == "Connection timeout"


class TestErrorInheritance:
    """Test error class inheritance relationships."""

    def test_specific_errors_inherit_from_vmcp_error(self):
        """Test that all specific error types inherit from VMCPError."""
        config_error = ConfigurationError("test")
        transport_error = TransportError("test")
        routing_error = RoutingError("test")
        server_error = ServerUnavailableError("test-server")
        permission_error = VMCPPermissionError("test")

        assert isinstance(config_error, VMCPError)
        assert isinstance(transport_error, VMCPError)
        assert isinstance(routing_error, VMCPError)
        assert isinstance(server_error, VMCPError)
        assert isinstance(permission_error, VMCPError)

    def test_vmcp_error_inherits_from_exception(self):
        """Test that VMCPError inherits from Exception."""
        error = VMCPError(VMCPErrorCode.INTERNAL_ERROR, "test")
        assert isinstance(error, Exception)

    def test_errors_can_be_raised_and_caught(self):
        """Test that errors can be raised and caught properly."""
        with pytest.raises(VMCPError) as exc_info:
            raise ConfigurationError("Invalid configuration")

        assert exc_info.value.code == VMCPErrorCode.CONFIGURATION_ERROR
        assert "Invalid configuration" in str(exc_info.value)

        # Test catching specific error type
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test")

        # Test catching base error type
        with pytest.raises(VMCPError):
            raise TransportError("Test")


class TestErrorFromJSONRPC:
    """Test error_from_json_rpc function."""

    def test_error_from_json_rpc(self):
        """Test creating VMCPError from JSON-RPC error dict."""
        error_dict = {
            "code": VMCPErrorCode.TRANSPORT_ERROR,
            "message": "Connection failed",
            "data": {"host": "localhost", "port": 3000}
        }

        error = error_from_json_rpc(error_dict, request_id="test-123")

        assert error.code == VMCPErrorCode.TRANSPORT_ERROR
        assert error.message == "Connection failed"
        assert error.data["host"] == "localhost"
        assert error.data["port"] == 3000
        assert error.request_id == "test-123"

    def test_error_from_json_rpc_invalid_code(self):
        """Test creating VMCPError from JSON-RPC error dict with invalid code."""
        error_dict = {
            "code": 99999,  # Invalid code
            "message": "Unknown error",
            "data": {}
        }

        error = error_from_json_rpc(error_dict)

        assert error.code == VMCPErrorCode.INTERNAL_ERROR
        assert error.message == "Unknown error"
        assert error.data["original_code"] == 99999


class TestErrorCodeMapping:
    """Test error code mapping and consistency."""

    def test_error_classes_use_correct_codes(self):
        """Test that error classes use their expected error codes."""
        test_cases = [
            (ConfigurationError("test"), VMCPErrorCode.CONFIGURATION_ERROR),
            (TransportError("test"), VMCPErrorCode.TRANSPORT_ERROR),
            (RoutingError("test"), VMCPErrorCode.ROUTING_FAILED),
            (ServerUnavailableError("server"), VMCPErrorCode.SERVER_UNAVAILABLE),
            (ServerNotFoundError("server"), VMCPErrorCode.NO_SERVER_FOUND),
            (CircuitBreakerOpenError("server", 5), VMCPErrorCode.CIRCUIT_BREAKER_OPEN),
            (VMCPPermissionError("test"), VMCPErrorCode.UNAUTHORIZED),
        ]

        for error, expected_code in test_cases:
            assert error.code == expected_code, (
                f"Error {type(error).__name__} has wrong code"
            )
