"""
Pytest configuration and shared fixtures for vMCP tests.

This module provides common test fixtures, configuration, and utilities
used across all vMCP test modules.
"""

import asyncio
import logging
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from src.vmcp.config.loader import ConfigLoader
from src.vmcp.gateway.server import GatewayConfig, VMCPGateway
from src.vmcp.registry.registry import Registry
from src.vmcp.routing.router import Router
from src.vmcp.testing.mock_server import MockMCPServer

# Configure pytest-asyncio - auto mode is configured in pyproject.toml


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def config_loader() -> ConfigLoader:
    """Create a configuration loader instance."""
    return ConfigLoader()


@pytest.fixture
def default_config(config_loader: ConfigLoader) -> dict[str, Any]:
    """Get default configuration for tests."""
    return config_loader.load_defaults()


@pytest.fixture
def test_config(temp_dir: Path) -> dict[str, Any]:
    """Create test configuration with temporary paths."""
    return {
        "version": "0.1.0",
        "gateway": {
            "registry_path": str(temp_dir / "registry"),
            "log_level": "DEBUG",
            "cache_enabled": True,
            "cache_ttl": 60,
            "max_connections": 10,
            "request_timeout": 5,
            "health_check_interval": 5,
            "max_concurrent_requests": 5
        },
        "transports": {
            "stdio": {"enabled": True},
            "http": {"enabled": False, "port": 3000, "host": "127.0.0.1"}
        },
        "routing": {
            "default_strategy": "hybrid",
            "load_balancer": "round_robin",
            "cache_enabled": True,
            "cache_ttl": 60
        },
        "servers": {
            "test-server": {
                "id": "test-server",
                "name": "Test MCP Server",
                "transport": "stdio",
                "command": ["python", "-m", "src.vmcp.testing.mock_server"],
                "enabled": True,
                "capabilities": {
                    "tools": {"list_changed": True},
                    "resources": {"subscribe": True}
                }
            }
        }
    }


@pytest.fixture
def gateway_config(test_config: dict[str, Any]) -> GatewayConfig:
    """Create gateway configuration for tests."""
    return GatewayConfig(**test_config["gateway"])


@pytest.fixture
async def registry(temp_dir: Path) -> AsyncGenerator[Registry, None]:
    """Create and initialize a registry for tests."""
    registry = Registry(str(temp_dir / "registry"))
    await registry.initialize()
    yield registry
    await registry.shutdown()


@pytest.fixture
def router(registry: Registry) -> Router:
    """Create a router instance for tests."""
    return Router(registry)


@pytest.fixture
async def gateway(gateway_config: GatewayConfig) -> AsyncGenerator[VMCPGateway, None]:
    """Create and initialize a gateway for tests."""
    gateway = VMCPGateway(gateway_config)
    await gateway.initialize()
    yield gateway
    await gateway.stop()


@pytest.fixture
def mock_server() -> MockMCPServer:
    """Create a mock MCP server for tests."""
    return MockMCPServer(
        server_id="test-mock-server",
        simulate_errors=False,
        error_rate=0.0,
        response_delay=0.0
    )


@pytest.fixture
def mock_server_with_errors() -> MockMCPServer:
    """Create a mock MCP server that simulates errors."""
    return MockMCPServer(
        server_id="error-mock-server",
        simulate_errors=True,
        error_rate=0.2,
        response_delay=0.1
    )


@pytest.fixture
def sample_json_rpc_request() -> dict[str, Any]:
    """Sample JSON-RPC request for tests."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }


@pytest.fixture
def sample_json_rpc_notification() -> dict[str, Any]:
    """Sample JSON-RPC notification for tests."""
    return {
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {}
    }


@pytest.fixture
def sample_initialize_request() -> dict[str, Any]:
    """Sample initialize request for tests."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"list_changed": True},
                "resources": {"subscribe": True}
            },
            "clientInfo": {
                "name": "vMCP Test Client",
                "version": "1.0.0"
            }
        }
    }


@pytest.fixture
def sample_tools_call_request() -> dict[str, Any]:
    """Sample tools/call request for tests."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {
                "message": "Hello, World!"
            }
        }
    }


@pytest.fixture
def caplog_debug(caplog):
    """Configure caplog for debug level logging."""
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Async fixtures for pytest-asyncio
@pytest_asyncio.fixture
async def async_mock_server() -> AsyncGenerator[MockMCPServer, None]:
    """Async fixture for mock server."""
    server = MockMCPServer(server_id="async-test-server")
    yield server
    # Cleanup if needed


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_network: mark test as requiring network access")


# Skip conditions
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit test marker to test files in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration test marker to test files in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# Custom assertion helpers
class VMCPTestHelpers:
    """Helper methods for vMCP testing."""

    @staticmethod
    def assert_json_rpc_response(response: dict[str, Any], expected_id: Any) -> None:
        """Assert that response is a valid JSON-RPC response."""
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == expected_id
        assert "result" in response or "error" in response

    @staticmethod
    def assert_json_rpc_error(response: dict[str, Any], expected_code: int) -> None:
        """Assert that response is a JSON-RPC error with expected code."""
        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == expected_code

    @staticmethod
    def assert_mcp_capabilities(capabilities: dict[str, Any]) -> None:
        """Assert that capabilities object has expected MCP structure."""
        assert isinstance(capabilities, dict)
        # Add more specific capability assertions as needed


@pytest.fixture
def test_helpers() -> VMCPTestHelpers:
    """Provide test helper methods."""
    return VMCPTestHelpers()


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import os
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss

        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss

            return {
                "duration": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "peak_memory": self.process.memory_info().peak_wss if hasattr(self.process.memory_info(), 'peak_wss') else None
            }

    return PerformanceMonitor()


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_async_tasks():
    """Cleanup any remaining async tasks after each test."""
    yield
    # Simplified cleanup - let pytest-asyncio handle task cleanup
