"""
Standard I/O transport implementation for vMCP gateway.

This module implements the stdio transport that handles JSON-RPC 2.0 messages
over stdin/stdout with proper framing, as specified in the technical specification.
"""

import asyncio
import contextlib
import logging
import sys
from typing import Any

from ...errors import ConnectionTimeoutError, TransportError, VMCPConnectionError
from ..protocol import ProtocolHandler
from .base import MessageFraming, Transport

logger = logging.getLogger(__name__)


class StdioTransport(Transport):
    """Standard I/O transport implementation."""

    def __init__(
        self, message_handler, protocol_handler: ProtocolHandler | None = None
    ) -> None:
        """
        Initialize stdio transport.

        Args:
            message_handler: Function to handle incoming messages
            protocol_handler: Protocol handler (creates new one if None)
        """
        super().__init__(message_handler, "stdio")
        self.protocol_handler = protocol_handler or ProtocolHandler()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the stdio transport."""
        if self._running:
            logger.warning("Stdio transport already running")
            return

        logger.info("Starting stdio transport")

        try:
            # Create async streams for stdin/stdout
            loop = asyncio.get_event_loop()

            # Setup stdin reader
            self._reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(self._reader)

            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            # Setup stdout writer
            transport, _ = await loop.connect_write_pipe(
                lambda: asyncio.Protocol(), sys.stdout
            )
            self._writer = asyncio.StreamWriter(transport, None, self._reader, loop)

            self._running = True
            self._record_connection()

            # Start reading messages
            self._read_task = asyncio.create_task(self._read_loop())

            logger.info("Stdio transport started successfully")

        except Exception as e:
            logger.error(f"Failed to start stdio transport: {e}")
            raise TransportError(f"Failed to start stdio transport: {e}") from e

    async def stop(self) -> None:
        """Stop the stdio transport."""
        if not self._running:
            return

        logger.info("Stopping stdio transport")
        self._running = False

        # Cancel read task
        if self._read_task:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None

        # Close writer
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing stdio writer: {e}")
            finally:
                self._writer = None

        # Reader doesn't need explicit closing
        self._reader = None

        logger.info("Stdio transport stopped")

    async def send_message(
        self, message: dict[str, Any], connection_id: str | None = None
    ) -> None:
        """
        Send a message through stdio.

        Args:
            message: Message to send
            connection_id: Ignored for stdio transport

        Raises:
            TransportError: If message cannot be sent
        """
        if not self._running or not self._writer:
            raise TransportError("Stdio transport not running")

        try:
            # Serialize message
            message_str = self.protocol_handler.serialize_message(message)

            # Frame message
            framed_data = MessageFraming.frame_message(message_str)

            # Send with write lock to prevent interleaving
            async with self._write_lock:
                self._writer.write(framed_data)
                await self._writer.drain()

            self._record_message_sent(len(framed_data))
            logger.debug(
                f"Sent message: {message.get('method', 'response')} ({len(framed_data)} bytes)"
            )

        except Exception as e:
            self._record_error()
            logger.error(f"Failed to send message: {e}")
            raise TransportError(f"Failed to send message: {e}") from e

    async def _read_loop(self) -> None:
        """Read messages from stdin in a loop."""
        logger.debug("Starting stdio read loop")

        while self._running:
            try:
                # Read framed message
                message_str = await self._read_message()
                if message_str is None:
                    # End of stream
                    logger.info("Stdin closed, stopping transport")
                    break

                self._record_message_received(len(message_str))
                logger.debug(f"Received message ({len(message_str)} bytes)")

                # Parse message
                try:
                    message = self.protocol_handler.parse_message(message_str)
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
                    self._record_error()

                    # Send parse error response if possible
                    error_response = self.protocol_handler.create_error_response(
                        None, -32700, "Parse error", {"details": str(e)}
                    )
                    await self.send_message(error_response)
                    continue

                # Handle message
                response = await self._handle_message_safe(message)
                if response:
                    await self.send_message(response)

            except asyncio.CancelledError:
                logger.debug("Stdio read loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in stdio read loop: {e}", exc_info=True)
                self._record_error()

                # Brief pause before retrying
                await asyncio.sleep(0.1)

        logger.debug("Stdio read loop ended")

    async def _read_message(self) -> str | None:
        """
        Read a single framed message from stdin.

        Returns:
            Message string or None if stream ended

        Raises:
            TransportError: If message framing is invalid
        """
        if not self._reader:
            return None

        try:
            return await MessageFraming.read_framed_message(self._reader)
        except Exception as e:
            logger.error(f"Failed to read framed message: {e}")
            raise TransportError(f"Failed to read message: {e}") from e


class StdioServerConnection:
    """Connection to an MCP server over stdio."""

    def __init__(self, server_config) -> None:
        """
        Initialize stdio server connection.

        Args:
            server_config: Server configuration object
        """
        self.server_config = server_config
        self.process: asyncio.subprocess.Process | None = None
        self.protocol_handler = ProtocolHandler()
        self._connected = False
        self._request_id = 0
        self._pending_requests: dict[Any, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._is_fastmcp = False  # Auto-detect FastMCP servers

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return

        try:
            logger.info(f"Connecting to server {self.server_config.id}")

            # Build command
            command = [self.server_config.command] + self.server_config.args
            env = (
                {**self.server_config.environment}
                if self.server_config.environment
                else None
            )

            # Start server process
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._connected = True

            # Start reading responses
            self._read_task = asyncio.create_task(self._read_responses())

            # Perform initialization handshake
            await self._initialize_connection()

            logger.info(f"Connected to server {self.server_config.id}")

        except Exception as e:
            await self.disconnect()
            raise VMCPConnectionError(
                f"Failed to connect to server {self.server_config.id}: {e}"
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return

        logger.info(f"Disconnecting from server {self.server_config.id}")
        self._connected = False

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Cancel read task
        if self._read_task:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None

        # Terminate process
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Server {self.server_config.id} did not terminate gracefully, killing"
                )
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.warning(f"Error terminating server {self.server_config.id}: {e}")
            finally:
                self.process = None

        logger.info(f"Disconnected from server {self.server_config.id}")

    async def send_request(
        self, message: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """
        Send a request to the server and wait for response.

        Args:
            message: Request message
            timeout: Timeout in seconds

        Returns:
            Response message

        Raises:
            VMCPConnectionError: If not connected
            ConnectionTimeoutError: If request times out
            TransportError: If send fails
        """
        if not self._connected or not self.process:
            raise VMCPConnectionError(
                f"Not connected to server {self.server_config.id}"
            )

        # Generate request ID if not present
        if "id" not in message:
            self._request_id += 1
            message["id"] = self._request_id

        request_id = message["id"]

        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future

        try:
            # Send message
            await self._send_message(message)

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError as e:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            raise ConnectionTimeoutError(
                f"Request to server {self.server_config.id} timed out after {timeout}s",
                timeout=timeout,
                server_id=self.server_config.id,
                request_id=request_id,
            ) from e
        except Exception as e:
            # Clean up pending request
            self._pending_requests.pop(request_id, None)
            raise TransportError(f"Failed to send request: {e}") from e

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self.process or not self.process.stdin:
            raise TransportError("No stdin available")

        try:
            # Serialize message
            message_str = self.protocol_handler.serialize_message(message)

            # Use appropriate framing based on server type
            if self._is_fastmcp:
                framed_data = MessageFraming.frame_message_fastmcp(message_str)
            else:
                framed_data = MessageFraming.frame_message(message_str)

            # Send with write lock
            async with self._write_lock:
                self.process.stdin.write(framed_data)
                await self.process.stdin.drain()

            logger.debug(
                f"Sent to {self.server_config.id}: {message.get('method', 'response')}"
            )

        except Exception as e:
            raise TransportError(f"Failed to send message to server: {e}") from e

    async def _read_responses(self) -> None:
        """Read responses from server in a loop."""
        logger.debug(f"Starting response reader for {self.server_config.id}")

        while self._connected and self.process and self.process.stdout:
            try:
                # Read framed message
                message_str = await MessageFraming.read_framed_message(
                    self.process.stdout
                )
                if message_str is None:
                    logger.info(f"Server {self.server_config.id} closed stdout")
                    break

                # Parse message
                try:
                    message = self.protocol_handler.parse_message(message_str)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse response from {self.server_config.id}: {e}"
                    )
                    continue

                # Handle response
                if "id" in message:
                    request_id = message["id"]
                    future = self._pending_requests.pop(request_id, None)
                    if future and not future.done():
                        future.set_result(message)
                    else:
                        logger.warning(
                            f"Unexpected response from {self.server_config.id}: {request_id}"
                        )
                else:
                    # Handle notification
                    logger.debug(
                        f"Received notification from {self.server_config.id}: {message.get('method')}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading from {self.server_config.id}: {e}")
                break

        logger.debug(f"Response reader ended for {self.server_config.id}")

    async def _initialize_connection(self) -> None:
        """Perform initialization handshake with server."""
        # Try FastMCP format first for iowarp-mcps servers
        if "iowarp-mcps" in str(self.server_config.command) or "iowarp-mcps" in str(
            self.server_config.args
        ):
            self._is_fastmcp = True
            logger.debug(f"Detected FastMCP server: {self.server_config.id}")

        # Send initialize request
        init_request = self.protocol_handler.create_request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",  # Use FastMCP protocol version for iowarp servers
                "capabilities": {},
                "clientInfo": {"name": "vMCP Gateway", "version": "1.0.0"},
            },
        )

        try:
            response = await self.send_request(init_request, timeout=10.0)

            if "error" in response:
                raise VMCPConnectionError(
                    f"Server initialization failed: {response['error']}"
                )

            logger.debug(f"Server {self.server_config.id} initialized successfully")

            # Send initialized notification
            initialized_notification = self.protocol_handler.create_request(
                "initialized"
            )
            await self._send_message(initialized_notification)

        except Exception as e:
            raise VMCPConnectionError(
                f"Failed to initialize server {self.server_config.id}: {e}"
            ) from e

    async def ping(self) -> bool:
        """
        Ping the server to check if it's alive.

        Returns:
            True if server responds, False otherwise
        """
        try:
            ping_request = self.protocol_handler.create_request("ping")
            response = await self.send_request(ping_request, timeout=5.0)
            return "error" not in response
        except Exception:
            return False
