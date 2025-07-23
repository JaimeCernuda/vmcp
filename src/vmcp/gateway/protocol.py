"""
Protocol handling for vMCP gateway.

This module provides JSON-RPC 2.0 message parsing, validation, and serialization
for the vMCP gateway system.
"""

import json
import logging
from typing import Any

from jsonschema import ValidationError, validate

from ..errors import InvalidMessageError, ProtocolError

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 schemas
REQUEST_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "number", "null"]},
        "method": {"type": "string", "minLength": 1},
        "params": {"type": ["object", "array"]},
    },
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "id"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "number", "null"]},
        "result": {},
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "integer"},
                "message": {"type": "string"},
                "data": {},
            },
            "additionalProperties": False,
        },
    },
    "oneOf": [{"required": ["result"]}, {"required": ["error"]}],
    "additionalProperties": False,
}

NOTIFICATION_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "method": {"type": "string", "minLength": 1},
        "params": {"type": ["object", "array"]},
    },
    "not": {"required": ["id"]},
    "additionalProperties": False,
}


class ProtocolHandler:
    """Handles JSON-RPC 2.0 protocol parsing and validation."""

    def __init__(self) -> None:
        """Initialize protocol handler."""
        self.supported_version = "2.0"

    def parse_message(self, raw_message: str | bytes) -> dict[str, Any]:
        """
        Parse raw message into JSON-RPC structure.

        Args:
            raw_message: Raw message string or bytes

        Returns:
            Parsed JSON-RPC message

        Raises:
            InvalidMessageError: If message is invalid
        """
        try:
            # Handle bytes input
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            # Parse JSON
            message = json.loads(raw_message)

            # Validate structure
            self.validate_message(message)

            return message

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            raise InvalidMessageError(f"Invalid JSON: {e}", validation_errors=[str(e)]) from e
        except Exception as e:
            logger.error(f"Message parse error: {e}")
            raise InvalidMessageError(f"Failed to parse message: {e}") from e

    def validate_message(self, message: dict[str, Any]) -> None:
        """
        Validate JSON-RPC message structure.

        Args:
            message: Message to validate

        Raises:
            InvalidMessageError: If message is invalid
        """
        try:
            # Check if it's a notification (no id field)
            if "id" not in message:
                validate(message, NOTIFICATION_SCHEMA)
                return

            # Check if it's a request or response
            if "method" in message:
                # It's a request
                validate(message, REQUEST_SCHEMA)
            else:
                # It's a response
                validate(message, RESPONSE_SCHEMA)

        except ValidationError as e:
            logger.warning(f"Message validation error: {e.message}")
            raise InvalidMessageError(
                "Invalid message structure", validation_errors=[e.message]
            ) from e

    def serialize_message(self, message: dict[str, Any]) -> str:
        """
        Serialize message to JSON-RPC format.

        Args:
            message: Message to serialize

        Returns:
            Serialized JSON string

        Raises:
            ProtocolError: If serialization fails
        """
        try:
            # Ensure jsonrpc version is set
            if "jsonrpc" not in message:
                message["jsonrpc"] = self.supported_version

            # Validate before serializing
            self.validate_message(message)

            # Serialize with consistent formatting
            return json.dumps(message, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            logger.error(f"Message serialization error: {e}")
            raise ProtocolError(f"Failed to serialize message: {e}") from e

    def create_request(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        request_id: str | int | None = None,
    ) -> dict[str, Any]:
        """
        Create a JSON-RPC request message.

        Args:
            method: Method name
            params: Parameters (optional)
            request_id: Request ID (optional, creates notification if None)

        Returns:
            JSON-RPC request message
        """
        message = {"jsonrpc": self.supported_version, "method": method}

        if params is not None:
            message["params"] = params

        if request_id is not None:
            message["id"] = request_id

        return message

    def create_response(
        self,
        request_id: str | int | None,
        result: Any | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a JSON-RPC response message.

        Args:
            request_id: Request ID from original request
            result: Result data (mutually exclusive with error)
            error: Error data (mutually exclusive with result)

        Returns:
            JSON-RPC response message

        Raises:
            ProtocolError: If both result and error are provided
        """
        if result is not None and error is not None:
            raise ProtocolError("Cannot provide both result and error")

        if result is None and error is None:
            raise ProtocolError("Must provide either result or error")

        message = {"jsonrpc": self.supported_version, "id": request_id}

        if result is not None:
            message["result"] = result
        else:
            message["error"] = error

        return message

    def create_error_response(
        self,
        request_id: str | int | None,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a JSON-RPC error response.

        Args:
            request_id: Request ID from original request
            code: Error code
            message: Error message
            data: Additional error data (optional)

        Returns:
            JSON-RPC error response
        """
        error = {"code": code, "message": message}

        if data is not None:
            error["data"] = data

        return self.create_response(request_id, error=error)

    def extract_method(self, message: dict[str, Any]) -> str:
        """
        Extract method name from request message.

        Args:
            message: JSON-RPC message

        Returns:
            Method name

        Raises:
            ProtocolError: If message has no method
        """
        method = message.get("method")
        if not method:
            raise ProtocolError("Message has no method field")
        return method

    def extract_params(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | list[Any] | None:
        """
        Extract parameters from request message.

        Args:
            message: JSON-RPC message

        Returns:
            Parameters or None if not present
        """
        return message.get("params")

    def extract_id(self, message: dict[str, Any]) -> str | int | None:
        """
        Extract ID from message.

        Args:
            message: JSON-RPC message

        Returns:
            Message ID or None if not present (notification)
        """
        return message.get("id")

    def is_request(self, message: dict[str, Any]) -> bool:
        """
        Check if message is a request.

        Args:
            message: JSON-RPC message

        Returns:
            True if message is a request
        """
        return "method" in message and "id" in message

    def is_notification(self, message: dict[str, Any]) -> bool:
        """
        Check if message is a notification.

        Args:
            message: JSON-RPC message

        Returns:
            True if message is a notification
        """
        return "method" in message and "id" not in message

    def is_response(self, message: dict[str, Any]) -> bool:
        """
        Check if message is a response.

        Args:
            message: JSON-RPC message

        Returns:
            True if message is a response
        """
        return "id" in message and "method" not in message

    def is_error_response(self, message: dict[str, Any]) -> bool:
        """
        Check if message is an error response.

        Args:
            message: JSON-RPC message

        Returns:
            True if message is an error response
        """
        return self.is_response(message) and "error" in message

    def is_success_response(self, message: dict[str, Any]) -> bool:
        """
        Check if message is a success response.

        Args:
            message: JSON-RPC message

        Returns:
            True if message is a success response
        """
        return self.is_response(message) and "result" in message
