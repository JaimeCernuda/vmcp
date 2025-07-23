"""
End-to-end integration tests for vMCP Gateway.

This module tests the complete flow from request ingestion through
routing, server communication, and response delivery.
"""

import asyncio
from pathlib import Path

import pytest

from src.vmcp.config.loader import ConfigLoader
from src.vmcp.gateway.server import GatewayConfig, VMCPGateway
from src.vmcp.registry.registry import MCPServerConfig, Registry


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end request processing flow."""

    @pytest.fixture
    async def mock_server_registry(self, temp_dir: Path) -> Registry:
        """Create registry with mock server configuration."""
        registry = Registry(str(temp_dir / "registry"))
        await registry.initialize()

        # Add mock server configuration
        server_config = MCPServerConfig(
            id="mock-server",
            name="Mock MCP Server",
            transport="stdio",
            command=["python", "-m", "src.vmcp.testing.mock_server", "--server-id", "mock-server"],
            capabilities={
                "tools": {"list_changed": True},
                "resources": {"subscribe": True},
                "prompts": {"list_changed": True}
            },
            enabled=True
        )

        await registry.register_server(server_config)
        return registry

    @pytest.fixture
    async def gateway_with_mock_server(self, temp_dir: Path, mock_server_registry: Registry) -> VMCPGateway:
        """Create gateway with mock server registered."""
        config = GatewayConfig(
            registry_path=str(temp_dir / "registry"),
            log_level="DEBUG",
            max_concurrent_requests=5,
            request_timeout=10
        )

        gateway = VMCPGateway(config)
        gateway.registry = mock_server_registry  # Override with our configured registry

        await gateway.initialize()
        return gateway

    async def test_basic_request_flow(self, gateway_with_mock_server: VMCPGateway):
        """Test basic request processing flow."""
        gateway = gateway_with_mock_server

        # Start gateway
        await gateway.start()

        try:
            # Test tools/list request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }

            response = await gateway.handle_request(request)

            # Verify response structure
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response

            # Verify tools are returned
            tools = response["result"]["tools"]
            assert isinstance(tools, list)
            assert len(tools) > 0

            # Check for expected mock tools
            tool_names = [tool["name"] for tool in tools]
            assert "echo" in tool_names
            assert "calculator" in tool_names

        finally:
            await gateway.stop()

    async def test_tool_execution_flow(self, gateway_with_mock_server: VMCPGateway):
        """Test complete tool execution flow."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            # Test tools/call request
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {
                        "message": "Hello from integration test!"
                    }
                }
            }

            response = await gateway.handle_request(request)

            # Verify response
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 2
            assert "result" in response

            # Verify tool result
            content = response["result"]["content"]
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]["type"] == "text"
            assert "Hello from integration test!" in content[0]["text"]

        finally:
            await gateway.stop()

    async def test_resource_access_flow(self, gateway_with_mock_server: VMCPGateway):
        """Test resource access flow."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            # First, list resources
            list_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/list",
                "params": {}
            }

            list_response = await gateway.handle_request(list_request)

            assert "result" in list_response
            resources = list_response["result"]["resources"]
            assert len(resources) > 0

            # Get URI of first resource
            resource_uri = resources[0]["uri"]

            # Read the resource
            read_request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {
                    "uri": resource_uri
                }
            }

            read_response = await gateway.handle_request(read_request)

            assert "result" in read_response
            contents = read_response["result"]["contents"]
            assert len(contents) > 0
            assert "text" in contents[0]

        finally:
            await gateway.stop()

    async def test_routing_with_multiple_servers(self, temp_dir: Path):
        """Test routing between multiple mock servers."""
        # Create registry with multiple servers
        registry = Registry(str(temp_dir / "registry"))
        await registry.initialize()

        # Add multiple mock servers
        for i in range(3):
            server_config = MCPServerConfig(
                id=f"mock-server-{i}",
                name=f"Mock Server {i}",
                transport="stdio",
                command=["python", "-m", "src.vmcp.testing.mock_server", "--server-id", f"mock-server-{i}"],
                capabilities={
                    "tools": {"list_changed": True},
                    "resources": {"subscribe": True}
                },
                enabled=True
            )
            await registry.register_server(server_config)

        # Create gateway
        config = GatewayConfig(
            registry_path=str(temp_dir / "registry"),
            log_level="DEBUG"
        )

        gateway = VMCPGateway(config)
        gateway.registry = registry

        await gateway.initialize()
        await gateway.start()

        try:
            # Send multiple requests and verify they get routed
            for i in range(5):
                request = {
                    "jsonrpc": "2.0",
                    "id": i + 10,
                    "method": "tools/list",
                    "params": {}
                }

                response = await gateway.handle_request(request)

                assert response["jsonrpc"] == "2.0"
                assert response["id"] == i + 10
                assert "result" in response

        finally:
            await gateway.stop()
            await registry.shutdown()

    async def test_error_handling_flow(self, gateway_with_mock_server: VMCPGateway):
        """Test error handling in complete flow."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            # Test invalid method
            request = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "invalid/method",
                "params": {}
            }

            response = await gateway.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 5
            assert "error" in response
            assert response["error"]["code"] == -32601  # Method not found

            # Test invalid tool call
            request = {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "nonexistent_tool",
                    "arguments": {}
                }
            }

            response = await gateway.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 6
            assert "error" in response

        finally:
            await gateway.stop()

    async def test_concurrent_requests(self, gateway_with_mock_server: VMCPGateway):
        """Test handling concurrent requests."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            # Create multiple concurrent requests
            requests = []
            for i in range(10):
                request = {
                    "jsonrpc": "2.0",
                    "id": i + 100,
                    "method": "tools/call",
                    "params": {
                        "name": "echo",
                        "arguments": {
                            "message": f"Concurrent message {i}"
                        }
                    }
                }
                requests.append(gateway.handle_request(request))

            # Execute all requests concurrently
            responses = await asyncio.gather(*requests)

            # Verify all responses
            assert len(responses) == 10

            for i, response in enumerate(responses):
                assert response["jsonrpc"] == "2.0"
                assert response["id"] == i + 100
                assert "result" in response

                content = response["result"]["content"]
                assert f"Concurrent message {i}" in content[0]["text"]

        finally:
            await gateway.stop()

    async def test_vmcp_extension_methods(self, gateway_with_mock_server: VMCPGateway):
        """Test vMCP extension methods."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            # Test vmcp/servers/list
            request = {
                "jsonrpc": "2.0",
                "id": 200,
                "method": "vmcp/servers/list",
                "params": {}
            }

            response = await gateway.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 200
            assert "result" in response

            servers = response["result"]["servers"]
            assert isinstance(servers, list)
            assert len(servers) > 0

            # Check server structure
            server = servers[0]
            assert "id" in server
            assert "name" in server
            assert "transport" in server
            assert "enabled" in server
            assert "healthy" in server
            assert "capabilities" in server

            # Test vmcp/metrics
            request = {
                "jsonrpc": "2.0",
                "id": 201,
                "method": "vmcp/metrics",
                "params": {}
            }

            response = await gateway.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 201
            assert "result" in response

            metrics = response["result"]
            assert "requests" in metrics
            assert "routing" in metrics
            assert "registry" in metrics

        finally:
            await gateway.stop()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics of the complete system."""

    async def test_throughput_under_load(self, gateway_with_mock_server: VMCPGateway, performance_monitor):
        """Test system throughput under load."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            performance_monitor.start()

            # Send many requests rapidly
            num_requests = 100
            requests = []

            for i in range(num_requests):
                request = {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {
                        "name": "echo",
                        "arguments": {"message": f"Load test {i}"}
                    }
                }
                requests.append(gateway.handle_request(request))

            responses = await asyncio.gather(*requests)

            performance_stats = performance_monitor.stop()

            # Verify all requests succeeded
            assert len(responses) == num_requests

            for i, response in enumerate(responses):
                assert response["id"] == i
                assert "result" in response

            # Check performance metrics
            throughput = num_requests / performance_stats["duration"]
            print(f"Throughput: {throughput:.2f} requests/second")
            print(f"Duration: {performance_stats['duration']:.2f} seconds")
            print(f"Memory delta: {performance_stats['memory_delta']} bytes")

            # Basic performance assertions
            assert throughput > 10, "Throughput too low"
            assert performance_stats["duration"] < 30, "Processing took too long"

        finally:
            await gateway.stop()

    async def test_memory_usage_stability(self, gateway_with_mock_server: VMCPGateway):
        """Test that memory usage remains stable under sustained load."""
        gateway = gateway_with_mock_server

        await gateway.start()

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Send requests in batches to simulate sustained load
            for batch in range(5):
                requests = []

                for i in range(20):
                    request = {
                        "jsonrpc": "2.0",
                        "id": batch * 20 + i,
                        "method": "ping",
                        "params": {}
                    }
                    requests.append(gateway.handle_request(request))

                await asyncio.gather(*requests)

                # Brief pause between batches
                await asyncio.sleep(0.1)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
            print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
            print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")

            # Memory should not increase dramatically
            max_allowed_increase = 50 * 1024 * 1024  # 50 MB
            assert memory_increase < max_allowed_increase, "Memory usage increased too much"

        finally:
            await gateway.stop()


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test integration with configuration system."""

    async def test_configuration_loading_and_application(self, temp_dir: Path):
        """Test loading configuration and applying it to gateway."""
        # Create configuration file
        config_file = temp_dir / "config.toml"
        config_data = {
            "version": "0.1.0",
            "gateway": {
                "registry_path": str(temp_dir / "registry"),
                "log_level": "DEBUG",
                "cache_enabled": True,
                "cache_ttl": 120,
                "max_connections": 50,
                "request_timeout": 15
            },
            "transports": {
                "stdio": {"enabled": True}
            },
            "routing": {
                "default_strategy": "capability",
                "load_balancer": "adaptive"
            },
            "servers": {
                "test-server": {
                    "id": "test-server",
                    "name": "Test Server",
                    "transport": "stdio",
                    "command": ["python", "-m", "src.vmcp.testing.mock_server"],
                    "enabled": True,
                    "capabilities": {
                        "tools": {"list_changed": True}
                    }
                }
            }
        }

        import toml
        with open(config_file, 'w') as f:
            toml.dump(config_data, f)

        # Load configuration
        config_loader = ConfigLoader()
        loaded_config = config_loader.load_from_file(config_file)

        # Create gateway with loaded configuration
        gateway_config = GatewayConfig(**loaded_config["gateway"])
        gateway = VMCPGateway(gateway_config)

        await gateway.initialize()

        try:
            # Verify configuration was applied
            assert gateway.config.cache_ttl == 120
            assert gateway.config.max_connections == 50
            assert gateway.config.request_timeout == 15

            # Verify registry was initialized with correct path
            assert str(gateway.registry.registry_path) == str(temp_dir / "registry")

        finally:
            await gateway.stop()

    async def test_environment_variable_substitution(self, temp_dir: Path):
        """Test environment variable substitution in configuration."""
        import os

        # Set environment variables
        os.environ["VMCP_REGISTRY_PATH"] = str(temp_dir / "custom_registry")
        os.environ["VMCP_LOG_LEVEL"] = "WARNING"
        os.environ["VMCP_CACHE_TTL"] = "180"

        try:
            # Create configuration with environment variables
            config_file = temp_dir / "config_with_env.toml"
            config_data = {
                "gateway": {
                    "registry_path": "${VMCP_REGISTRY_PATH}",
                    "log_level": "${VMCP_LOG_LEVEL}",
                    "cache_ttl": "${VMCP_CACHE_TTL}"
                }
            }

            import toml
            with open(config_file, 'w') as f:
                toml.dump(config_data, f)

            # Load configuration
            config_loader = ConfigLoader()
            loaded_config = config_loader.load_from_file(config_file)

            # Verify substitution occurred
            assert loaded_config["gateway"]["registry_path"] == str(temp_dir / "custom_registry")
            assert loaded_config["gateway"]["log_level"] == "WARNING"
            assert loaded_config["gateway"]["cache_ttl"] == "180"

        finally:
            # Clean up environment variables
            os.environ.pop("VMCP_REGISTRY_PATH", None)
            os.environ.pop("VMCP_LOG_LEVEL", None)
            os.environ.pop("VMCP_CACHE_TTL", None)
