"""
Unit tests for vMCP error handling system.
"""

import pytest

from src.vmcp.errors import (
    AllServersDownError,
    ConfigurationError,
    GatewayError,
    NoServerFoundError,
    PermissionDeniedError,
    RoutingError,
    ServerUnavailableError,
    TransportError,
    VMCPError,
    VMCPErrorCode,
    create_json_rpc_error_response,
)


class TestVMCPErrorCode:
    """Test error code enumeration."""

    def test_error_codes_exist(self):
        """Test that all expected error codes exist."""
        assert VMCPErrorCode.SUCCESS == 0
        assert VMCPErrorCode.INVALID_REQUEST == -32600
        assert VMCPErrorCode.METHOD_NOT_FOUND == -32601
        assert VMCPErrorCode.INVALID_PARAMS == -32602
        assert VMCPErrorCode.INTERNAL_ERROR == -32603
        assert VMCPErrorCode.PARSE_ERROR == -32700

        # vMCP specific codes
        assert VMCPErrorCode.TRANSPORT_ERROR == -32001
        assert VMCPErrorCode.ROUTING_FAILED == -32002
        assert VMCPErrorCode.SERVER_UNAVAILABLE == -32003
        assert VMCPErrorCode.NO_SERVER_FOUND == -32004
        assert VMCPErrorCode.ALL_SERVERS_DOWN == -32005
        assert VMCPErrorCode.CONFIGURATION_ERROR == -32006
        assert VMCPErrorCode.PERMISSION_DENIED == -32007
        assert VMCPErrorCode.GATEWAY_ERROR == -32008

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
        assert error.details is None
        assert error.server_id is None

    def test_error_with_details(self):
        """Test creating error with details."""
        details = {"key": "value", "count": 42}
        error = VMCPError(
            VMCPErrorCode.ROUTING_FAILED,
            "Routing failed",
            details=details,
            server_id="test-server"
        )

        assert error.code == VMCPErrorCode.ROUTING_FAILED
        assert error.message == "Routing failed"
        assert error.details == details
        assert error.server_id == "test-server"

    def test_error_string_representation(self):
        """Test error string representation."""
        error = VMCPError(VMCPErrorCode.TRANSPORT_ERROR, "Connection failed")
        error_str = str(error)

        assert "TRANSPORT_ERROR" in error_str
        assert "Connection failed" in error_str
        assert "-32001" in error_str

    def test_error_with_server_id_in_string(self):
        """Test error string includes server ID when present."""
        error = VMCPError(
            VMCPErrorCode.SERVER_UNAVAILABLE,
            "Server down",
            server_id="test-server"
        )
        error_str = str(error)

        assert "test-server" in error_str
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
        error = TransportError("Connection timeout", transport_type="stdio")

        assert error.code == VMCPErrorCode.TRANSPORT_ERROR
        assert error.message == "Connection timeout"
        assert error.transport_type == "stdio"

    def test_routing_error(self):
        """Test RoutingError."""
        error = RoutingError("No route found", method="tools/list")

        assert error.code == VMCPErrorCode.ROUTING_FAILED
        assert error.message == "No route found"
        assert error.method == "tools/list"

    def test_server_unavailable_error(self):
        """Test ServerUnavailableError."""
        error = ServerUnavailableError("server-1", reason="Connection refused")

        assert error.code == VMCPErrorCode.SERVER_UNAVAILABLE
        assert error.server_id == "server-1"
        assert "Connection refused" in error.message
        assert error.reason == "Connection refused"

    def test_no_server_found_error(self):
        """Test NoServerFoundError."""
        error = NoServerFoundError("tools/list", method="tools/list")

        assert error.code == VMCPErrorCode.NO_SERVER_FOUND
        assert error.server_id == "tools/list"
        assert error.method == "tools/list"
        assert "No server found" in error.message

    def test_all_servers_down_error(self):
        """Test AllServersDownError."""
        server_ids = ["server-1", "server-2", "server-3"]
        error = AllServersDownError(server_ids)

        assert error.code == VMCPErrorCode.ALL_SERVERS_DOWN
        assert error.server_ids == server_ids
        assert "All servers are down" in error.message
        assert "server-1, server-2, server-3" in error.message

    def test_gateway_error(self):
        """Test GatewayError."""
        error = GatewayError("Failed to start gateway")

        assert error.code == VMCPErrorCode.GATEWAY_ERROR
        assert error.message == "Failed to start gateway"

    def test_permission_denied_error(self):
        """Test PermissionDeniedError."""
        error = PermissionDeniedError("Access denied", resource="protected-resource")

        assert error.code == VMCPErrorCode.PERMISSION_DENIED
        assert error.message == "Access denied"
        assert error.resource == "protected-resource"


class TestJSONRPCErrorResponse:
    """Test JSON-RPC error response creation."""

    def test_create_basic_error_response(self):
        """Test creating basic JSON-RPC error response."""
        response = create_json_rpc_error_response(
            request_id=1,
            code=VMCPErrorCode.METHOD_NOT_FOUND,
            message="Method not found"
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method not found"
        assert "data" not in response["error"]

    def test_create_error_response_with_data(self):
        """Test creating JSON-RPC error response with data."""
        error_data = {"method": "unknown/method", "available": ["tools/list", "ping"]}
        response = create_json_rpc_error_response(
            request_id="test-id",
            code=VMCPErrorCode.METHOD_NOT_FOUND,
            message="Method not found",
            data=error_data
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id"
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method not found"
        assert response["error"]["data"] == error_data

    def test_create_error_response_null_id(self):
        """Test creating JSON-RPC error response with null ID."""
        response = create_json_rpc_error_response(
            request_id=None,
            code=VMCPErrorCode.PARSE_ERROR,
            message="Parse error"
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] is None
        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"

    def test_create_error_response_with_vmcp_error(self):
        """Test creating JSON-RPC error response from VMCPError."""
        vmcp_error = ServerUnavailableError("test-server", reason="Connection timeout")

        response = create_json_rpc_error_response(
            request_id=42,
            code=vmcp_error.code,
            message=vmcp_error.message,
            data={
                "server_id": vmcp_error.server_id,
                "reason": vmcp_error.reason,
                "details": vmcp_error.details
            } if vmcp_error.details else {
                "server_id": vmcp_error.server_id,
                "reason": vmcp_error.reason
            }
        )

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
        gateway_error = GatewayError("test")

        assert isinstance(config_error, VMCPError)
        assert isinstance(transport_error, VMCPError)
        assert isinstance(routing_error, VMCPError)
        assert isinstance(server_error, VMCPError)
        assert isinstance(gateway_error, VMCPError)

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


class TestErrorCodeMapping:
    """Test error code mapping and consistency."""

    def test_error_classes_use_correct_codes(self):
        """Test that error classes use their expected error codes."""
        test_cases = [
            (ConfigurationError("test"), VMCPErrorCode.CONFIGURATION_ERROR),
            (TransportError("test"), VMCPErrorCode.TRANSPORT_ERROR),
            (RoutingError("test"), VMCPErrorCode.ROUTING_FAILED),
            (ServerUnavailableError("server"), VMCPErrorCode.SERVER_UNAVAILABLE),
            (NoServerFoundError("server"), VMCPErrorCode.NO_SERVER_FOUND),
            (AllServersDownError([]), VMCPErrorCode.ALL_SERVERS_DOWN),
            (GatewayError("test"), VMCPErrorCode.GATEWAY_ERROR),
            (PermissionDeniedError("test"), VMCPErrorCode.PERMISSION_DENIED),
        ]

        for error, expected_code in test_cases:
            assert error.code == expected_code, f"Error {type(error).__name__} has wrong code"
