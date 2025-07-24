"""
Base transport interface for vMCP gateway.

This module defines the abstract base class for all transport implementations
in the vMCP gateway system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ...errors import TransportError

logger = logging.getLogger(__name__)


class Transport(ABC):
    """Abstract base class for vMCP transports."""

    def __init__(
        self, message_handler: Callable[[dict[str, Any]], Any], name: str = "unknown"
    ) -> None:
        """
        Initialize transport.

        Args:
            message_handler: Function to handle incoming messages
            name: Transport name for logging
        """
        self.message_handler = message_handler
        self.name = name
        self._running = False
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
            "connections": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
        }

    @abstractmethod
    async def start(self) -> None:
        """
        Start the transport.

        Raises:
            TransportError: If transport fails to start
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport gracefully."""
        pass

    @abstractmethod
    async def send_message(
        self, message: dict[str, Any], connection_id: str | None = None
    ) -> None:
        """
        Send a message through the transport.

        Args:
            message: Message to send
            connection_id: Optional connection identifier

        Raises:
            TransportError: If message cannot be sent
        """
        pass

    def is_running(self) -> bool:
        """Check if transport is running."""
        return self._running

    def get_stats(self) -> dict[str, Any]:
        """Get transport statistics."""
        return {
            **self._stats,
            "running": self._running,
            "name": self.name,
        }

    def _record_message_received(self, size: int) -> None:
        """Record received message statistics."""
        self._stats["messages_received"] += 1
        self._stats["bytes_received"] += size

    def _record_message_sent(self, size: int) -> None:
        """Record sent message statistics."""
        self._stats["messages_sent"] += 1
        self._stats["bytes_sent"] += size

    def _record_error(self) -> None:
        """Record error statistics."""
        self._stats["errors"] += 1

    def _record_connection(self) -> None:
        """Record new connection."""
        self._stats["connections"] += 1

    async def _handle_message_safe(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Safely handle message with error catching.

        Args:
            message: Message to handle

        Returns:
            Response message or None
        """
        try:
            response = await self.message_handler(message)
            return response  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(
                f"Error handling message in {self.name} transport: {e}", exc_info=True
            )
            self._record_error()

            # Create error response if message has ID
            if "id" in message:
                return {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": {"transport": self.name},
                    },
                }
            return None


class ConnectionManager:
    """Manages connections for transports that support multiple connections."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.connections: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def add_connection(self, connection_id: str, connection: Any) -> None:
        """
        Add a new connection.

        Args:
            connection_id: Unique connection identifier
            connection: Connection object
        """
        async with self._lock:
            self.connections[connection_id] = connection
            logger.debug(f"Added connection {connection_id}")

    async def remove_connection(self, connection_id: str) -> Any | None:
        """
        Remove a connection.

        Args:
            connection_id: Connection identifier to remove

        Returns:
            Removed connection object or None if not found
        """
        async with self._lock:
            connection = self.connections.pop(connection_id, None)
            if connection:
                logger.debug(f"Removed connection {connection_id}")
            return connection

    async def get_connection(self, connection_id: str) -> Any | None:
        """
        Get a connection by ID.

        Args:
            connection_id: Connection identifier

        Returns:
            Connection object or None if not found
        """
        async with self._lock:
            return self.connections.get(connection_id)

    async def get_all_connections(self) -> dict[str, Any]:
        """Get all active connections."""
        async with self._lock:
            return dict(self.connections)

    async def close_all_connections(self) -> None:
        """Close all connections."""
        async with self._lock:
            for connection_id, connection in list(self.connections.items()):
                try:
                    if hasattr(connection, "close"):
                        await connection.close()
                    elif hasattr(connection, "disconnect"):
                        await connection.disconnect()
                except Exception as e:
                    logger.warning(f"Error closing connection {connection_id}: {e}")

            self.connections.clear()
            logger.info("Closed all connections")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)

    def get_connection_ids(self) -> list[str]:
        """Get list of active connection IDs."""
        return list(self.connections.keys())


class MessageFraming:
    """Utility class for message framing over stream transports."""

    @staticmethod
    def frame_message(message: str) -> bytes:
        """
        Frame a message for transmission using length prefix.

        Args:
            message: Message string to frame

        Returns:
            Framed message bytes
        """
        content = message.encode("utf-8")
        length = len(content)
        return f"{length}\n".encode() + content + b"\n"

    @staticmethod
    def frame_message_fastmcp(message: str) -> bytes:
        """
        Frame a message for FastMCP (newline-delimited JSON).

        Args:
            message: Message string to frame

        Returns:
            Framed message bytes
        """
        return message.encode("utf-8") + b"\n"

    @staticmethod
    def frame_message_http(message: str) -> bytes:
        """
        Frame a message using HTTP-style Content-Length header.

        Args:
            message: Message string to frame

        Returns:
            Framed message bytes
        """
        content = message.encode("utf-8")
        length = len(content)
        return f"Content-Length: {length}\r\n\r\n".encode() + content

    @staticmethod
    async def read_framed_message(reader: asyncio.StreamReader) -> str | None:
        """
        Read a framed message from stream reader.
        Supports both length-prefixed and FastMCP newline-delimited formats.

        Args:
            reader: Stream reader

        Returns:
            Message string or None if stream ended

        Raises:
            TransportError: If framing is invalid
        """
        try:
            # Read first line
            first_line = await reader.readline()
            if not first_line:
                return None

            first_line_str = first_line.decode("utf-8").strip()

            # Check if it's a length prefix (integer) or direct JSON
            try:
                length = int(first_line_str)
                # It's a length prefix - use original framing

                # Validate length
                if length < 0 or length > 10 * 1024 * 1024:  # 10MB limit
                    raise TransportError(f"Invalid message length: {length}")

                # Read message content
                content = await reader.readexactly(length)

                # Read trailing newline
                await reader.readline()

                return content.decode("utf-8")

            except ValueError as e:
                # Not a length prefix - assume it's FastMCP format (direct JSON)
                if first_line_str.startswith("{") and first_line_str.endswith("}"):
                    # It's a complete JSON message
                    return first_line_str
                else:
                    # Invalid format
                    raise TransportError(
                        f"Invalid message format: {first_line_str[:100]}..."
                    ) from e

        except asyncio.IncompleteReadError as e:
            raise TransportError(f"Incomplete message: {e}") from e
        except UnicodeDecodeError as e:
            raise TransportError(f"Invalid UTF-8 encoding: {e}") from e

    @staticmethod
    async def read_http_framed_message(reader: asyncio.StreamReader) -> str | None:
        """
        Read an HTTP-style framed message from stream reader.

        Args:
            reader: Stream reader

        Returns:
            Message string or None if stream ended

        Raises:
            TransportError: If framing is invalid
        """
        try:
            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if not line:
                    return None

                line_str = (
                    line.decode("utf-8").strip()
                    if isinstance(line, bytes)
                    else str(line).strip()
                )
                if not line_str:  # Empty line indicates end of headers
                    break

                if ":" in line_str:
                    key, value = line_str.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Get content length
            content_length_str = headers.get("content-length")
            if not content_length_str:
                raise TransportError("Missing Content-Length header")
            content_length = int(content_length_str)

            length = content_length

            # Validate length
            if length < 0 or length > 10 * 1024 * 1024:  # 10MB limit
                raise TransportError(f"Invalid content length: {length}")

            # Read message content
            content = await reader.readexactly(length)

            return content.decode("utf-8")

        except asyncio.IncompleteReadError as e:
            raise TransportError(f"Incomplete HTTP message: {e}") from e
        except UnicodeDecodeError as e:
            raise TransportError(f"Invalid UTF-8 encoding: {e}") from e
