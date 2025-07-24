"""
Mock MCP server implementation for testing.

This module provides a comprehensive mock MCP server that can be used
for testing vMCP gateway functionality without requiring actual MCP servers.
"""

import asyncio
import json
import logging
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


class MockMCPServer:
    """Mock MCP server for testing vMCP gateway functionality."""

    def __init__(
        self,
        server_id: str = "mock-server",
        capabilities: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        prompts: list[dict[str, Any]] | None = None,
        simulate_errors: bool = False,
        error_rate: float = 0.0,
        response_delay: float = 0.0,
    ) -> None:
        """
        Initialize mock MCP server.

        Args:
            server_id: Unique identifier for this server
            capabilities: Server capabilities to advertise
            tools: Available tools
            resources: Available resources
            prompts: Available prompts
            simulate_errors: Whether to simulate random errors
            error_rate: Probability of errors (0.0-1.0)
            response_delay: Artificial delay in seconds
        """
        self.server_id = server_id
        self.simulate_errors = simulate_errors
        self.error_rate = error_rate
        self.response_delay = response_delay
        self.initialized = False

        # Default capabilities
        self.capabilities = capabilities or {
            "tools": {"list_changed": True},
            "resources": {"subscribe": True, "list_changed": True},
            "prompts": {"list_changed": True},
            "completion": {"supports_progress": True},
        }

        # Default tools
        self.tools = tools or [
            {
                "name": "echo",
                "description": "Echo back the input",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"}
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "calculator",
                "description": "Perform basic arithmetic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "file_read",
                "description": "Read file contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                },
            },
        ]

        # Default resources
        self.resources = resources or [
            {
                "uri": "mock://files/example.txt",
                "name": "Example File",
                "description": "An example text file",
                "mimeType": "text/plain",
            },
            {
                "uri": "mock://data/users.json",
                "name": "User Data",
                "description": "JSON data about users",
                "mimeType": "application/json",
            },
        ]

        # Default prompts
        self.prompts = prompts or [
            {
                "name": "summarize",
                "description": "Summarize the given text",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text to summarize",
                        "required": True,
                    },
                    {
                        "name": "max_words",
                        "description": "Maximum words in summary",
                        "required": False,
                    },
                ],
            }
        ]

        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle incoming JSON-RPC message.

        Args:
            message: JSON-RPC message

        Returns:
            JSON-RPC response or None for notifications
        """
        self.request_count += 1

        # Add artificial delay if configured
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        # Simulate random errors
        if self.simulate_errors and self.error_rate > 0:
            import random

            if random.random() < self.error_rate:
                self.error_count += 1
                return self._create_error_response(
                    message.get("id"), -32603, "Simulated server error"
                )

        method = message.get("method")
        params = message.get("params", {})
        request_id = message.get("id")

        # Handle notifications (no response)
        if request_id is None:
            if method is not None:
                await self._handle_notification(method, params)
            return None

        try:
            # Route to appropriate handler
            if method == "initialize":
                return await self._handle_initialize(request_id, params)
            elif method == "initialized":
                return None  # Notification
            elif method == "ping":
                return await self._handle_ping(request_id)
            elif method == "tools/list":
                return await self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tools_call(request_id, params)
            elif method == "resources/list":
                return await self._handle_resources_list(request_id)
            elif method == "resources/read":
                return await self._handle_resources_read(request_id, params)
            elif method == "resources/subscribe":
                return await self._handle_resources_subscribe(request_id, params)
            elif method == "resources/unsubscribe":
                return await self._handle_resources_unsubscribe(request_id, params)
            elif method == "prompts/list":
                return await self._handle_prompts_list(request_id)
            elif method == "prompts/get":
                return await self._handle_prompts_get(request_id, params)
            elif method == "completion/complete":
                return await self._handle_completion_complete(request_id, params)
            elif method is not None and method.startswith("mock/"):
                return await self._handle_mock_method(request_id, method, params)
            else:
                return self._create_error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

        except Exception as e:
            logger.error(f"Error handling method {method}: {e}", exc_info=True)
            self.error_count += 1
            return self._create_error_response(
                request_id, -32603, f"Internal error: {e}"
            )

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle notification message."""
        logger.debug(f"Received notification: {method}")

        if method == "initialized":
            self.initialized = True
            logger.info("Server initialized")

    async def _handle_initialize(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {})
        logger.info(f"Initializing with client: {client_info}")

        return self._create_response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": f"Mock MCP Server ({self.server_id})",
                    "version": "1.0.0",
                },
            },
        )

    async def _handle_ping(self, request_id: Any) -> dict[str, Any]:
        """Handle ping request."""
        return self._create_response(request_id, {})

    async def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        return self._create_response(request_id, {"tools": self.tools})

    async def _handle_tools_call(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return self._create_error_response(request_id, -32602, "Missing tool name")

        # Find the tool
        tool = next((t for t in self.tools if t["name"] == tool_name), None)
        if not tool:
            return self._create_error_response(
                request_id, -32602, f"Unknown tool: {tool_name}"
            )

        # Execute mock tool
        result = await self._execute_mock_tool(tool_name, arguments)

        return self._create_response(
            request_id, {"content": [{"type": "text", "text": result}]}
        )

    async def _execute_mock_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a mock tool and return result."""
        if tool_name == "echo":
            message = arguments.get("message", "")
            return f"Echo: {message}"

        elif tool_name == "calculator":
            expression = arguments.get("expression", "")
            try:
                # Simple evaluation (unsafe in production!)
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"

        elif tool_name == "file_read":
            path = arguments.get("path", "")
            return f"Mock file contents from {path}"

        else:
            return f"Mock result from {tool_name}"

    async def _handle_resources_list(self, request_id: Any) -> dict[str, Any]:
        """Handle resources/list request."""
        return self._create_response(request_id, {"resources": self.resources})

    async def _handle_resources_read(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if not uri:
            return self._create_error_response(
                request_id, -32602, "Missing resource URI"
            )

        # Find the resource
        resource = next((r for r in self.resources if r["uri"] == uri), None)
        if not resource:
            return self._create_error_response(
                request_id, -32602, f"Unknown resource: {uri}"
            )

        # Return mock content
        if "example.txt" in uri:
            content = "This is example file content from the mock server."
        elif "users.json" in uri:
            content = json.dumps(
                [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"},
                ],
                indent=2,
            )
        else:
            content = f"Mock content for resource: {uri}"

        return self._create_response(
            request_id,
            {
                "contents": [
                    {"uri": uri, "mimeType": resource["mimeType"], "text": content}
                ]
            },
        )

    async def _handle_resources_subscribe(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle resources/subscribe request."""
        uri = params.get("uri")
        return self._create_response(request_id, {"subscribed": True, "uri": uri})

    async def _handle_resources_unsubscribe(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle resources/unsubscribe request."""
        uri = params.get("uri")
        return self._create_response(request_id, {"unsubscribed": True, "uri": uri})

    async def _handle_prompts_list(self, request_id: Any) -> dict[str, Any]:
        """Handle prompts/list request."""
        return self._create_response(request_id, {"prompts": self.prompts})

    async def _handle_prompts_get(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            return self._create_error_response(
                request_id, -32602, "Missing prompt name"
            )

        # Find the prompt
        prompt = next((p for p in self.prompts if p["name"] == name), None)
        if not prompt:
            return self._create_error_response(
                request_id, -32602, f"Unknown prompt: {name}"
            )

        # Generate mock prompt content
        if name == "summarize":
            text = arguments.get("text", "No text provided")
            max_words = arguments.get("max_words", 50)

            prompt_content = f"Please summarize the following text in no more than {max_words} words:\n\n{text}"
        else:
            prompt_content = f"Mock prompt content for: {name}"

        return self._create_response(
            request_id,
            {
                "description": prompt["description"],
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": prompt_content},
                    }
                ],
            },
        )

    async def _handle_completion_complete(
        self, request_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle completion/complete request."""
        params.get("ref", {})
        params.get("argument", {})

        # Mock completion suggestions
        completions = [
            "completion_option_1",
            "completion_option_2",
            "completion_option_3",
        ]

        return self._create_response(
            request_id,
            {
                "completion": {
                    "values": completions,
                    "total": len(completions),
                    "hasMore": False,
                }
            },
        )

    async def _handle_mock_method(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle custom mock methods."""
        if method == "mock/stats":
            uptime = time.time() - self.start_time
            return self._create_response(
                request_id,
                {
                    "server_id": self.server_id,
                    "uptime": uptime,
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(1, self.request_count),
                    "initialized": self.initialized,
                },
            )

        elif method == "mock/config":
            return self._create_response(
                request_id,
                {
                    "server_id": self.server_id,
                    "capabilities": self.capabilities,
                    "tools_count": len(self.tools),
                    "resources_count": len(self.resources),
                    "prompts_count": len(self.prompts),
                    "simulate_errors": self.simulate_errors,
                    "error_rate": self.error_rate,
                    "response_delay": self.response_delay,
                },
            )

        elif method == "mock/reset":
            self.request_count = 0
            self.error_count = 0
            self.start_time = time.time()
            return self._create_response(request_id, {"reset": True})

        else:
            return self._create_error_response(
                request_id, -32601, f"Unknown mock method: {method}"
            )

    def _create_response(self, request_id: Any, result: Any) -> dict[str, Any]:
        """Create JSON-RPC success response."""
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _create_error_response(
        self, request_id: Any, code: int, message: str
    ) -> dict[str, Any]:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        uptime = time.time() - self.start_time
        return {
            "server_id": self.server_id,
            "uptime": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "initialized": self.initialized,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "prompts_count": len(self.prompts),
        }


class StdioMockServer:
    """Mock MCP server that communicates over stdio."""

    def __init__(self, mock_server: MockMCPServer) -> None:
        """
        Initialize stdio mock server.

        Args:
            mock_server: Mock server instance
        """
        self.mock_server = mock_server
        self.running = False

    async def run(self) -> None:
        """Run the mock server over stdio."""
        self.running = True
        logger.info(f"Starting stdio mock server: {self.mock_server.server_id}")

        try:
            # Read from stdin line by line
            loop = asyncio.get_event_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            # Write to stdout
            writer_transport, writer_protocol = await loop.connect_write_pipe(
                asyncio.streams.FlowControlMixin, sys.stdout
            )
            writer = asyncio.StreamWriter(
                writer_transport, writer_protocol, reader, loop
            )

            while self.running:
                try:
                    # Read JSON-RPC message
                    line = await reader.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    # Parse JSON
                    try:
                        message = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        continue

                    # Handle message
                    response = await self.mock_server.handle_message(message)

                    # Send response if needed
                    if response is not None:
                        response_line = json.dumps(response) + "\n"
                        writer.write(response_line.encode("utf-8"))
                        await writer.drain()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in stdio loop: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error starting stdio server: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Stdio mock server stopped")

    def stop(self) -> None:
        """Stop the mock server."""
        self.running = False


async def run_mock_server_cli() -> None:
    """CLI entry point for running mock server."""
    import argparse

    parser = argparse.ArgumentParser(description="Mock MCP Server for testing")
    parser.add_argument("--server-id", default="mock-server", help="Server ID")
    parser.add_argument(
        "--simulate-errors", action="store_true", help="Simulate random errors"
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.1, help="Error rate (0.0-1.0)"
    )
    parser.add_argument(
        "--response-delay", type=float, default=0.0, help="Response delay in seconds"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create mock server
    mock_server = MockMCPServer(
        server_id=args.server_id,
        simulate_errors=args.simulate_errors,
        error_rate=args.error_rate,
        response_delay=args.response_delay,
    )

    # Run stdio server
    stdio_server = StdioMockServer(mock_server)

    try:
        await stdio_server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stdio_server.stop()


if __name__ == "__main__":
    asyncio.run(run_mock_server_cli())
