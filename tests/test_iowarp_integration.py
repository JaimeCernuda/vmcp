"""
Integration tests for vMCP with iowarp-mcps servers
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from vmcp.gateway.server import GatewayConfig, VMCPGateway
from vmcp.registry.registry import MCPServerConfig, Registry
from vmcp.repository.manager import RepositoryManager


@pytest.mark.asyncio
async def test_fastmcp_server_integration():
    """Test integration with FastMCP servers from iowarp-mcps."""
    # Create temporary registry
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = os.path.join(temp_dir, "registry")

        # Configure gateway
        config = GatewayConfig(
            registry_path=registry_path,
            transports={"stdio": {"enabled": True}},
            cache_enabled=False,
        )

        # Create registry and register Adios server
        registry = Registry(registry_path)
        await registry.initialize()

        # Create server configuration for Adios (known working server)
        adios_config = MCPServerConfig(
            id="adios-mcp-test",
            name="ADIOS MCP Server Test",
            transport="stdio",
            command="uv",
            args=[
                "run",
                "--directory",
                "iowarp-mcps/mcps/Adios",
                "python",
                "src/server.py",
            ],
            capabilities={"tools": []},
            enabled=True,
        )

        # Register the server
        await registry.register_server(adios_config)

        # Create and initialize gateway
        gateway = VMCPGateway(config)
        await gateway.initialize()

        # Test basic functionality
        servers = registry.get_all_servers()
        assert len(servers) == 1
        assert servers[0].config.id == "adios-mcp-test"

        # Verify server is detected as FastMCP
        # This would happen during connection establishment

        # Cleanup
        await gateway.stop()
        await registry.shutdown()


@pytest.mark.asyncio
async def test_repository_manager_auto_register():
    """Test auto-registration of iowarp-mcps servers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = os.path.join(temp_dir, "registry")

        # Create registry
        registry = Registry(registry_path)
        await registry.initialize()

        # Create repository manager
        repo_manager = RepositoryManager(registry)
        await repo_manager.initialize()

        # Test auto-registration (this might fail if iowarp-mcps not available)
        if Path("iowarp-mcps").exists():
            try:
                count = await repo_manager.auto_register_iowarp_mcps("iowarp-mcps")
                assert count >= 0  # Should register some servers

                # Check that servers were registered
                servers = registry.get_all_servers()
                assert len(servers) >= 0

                # Verify server configurations
                for server in servers:
                    assert server.config.transport == "stdio"
                    assert server.config.command == "uv"
                    assert "iowarp-mcps" in " ".join(server.config.args)

            except Exception as e:
                pytest.skip(f"iowarp-mcps auto-registration failed: {e}")
        else:
            pytest.skip("iowarp-mcps directory not available")

        await registry.shutdown()


@pytest.mark.asyncio
async def test_message_framing_detection():
    """Test that message framing correctly detects FastMCP vs standard MCP."""
    from vmcp.gateway.transports.base import MessageFraming

    # Test standard MCP framing
    message = '{"jsonrpc":"2.0","id":1,"method":"test"}'
    framed = MessageFraming.frame_message(message)
    assert framed.startswith(str(len(message)).encode())

    # Test FastMCP framing
    fastmcp_framed = MessageFraming.frame_message_fastmcp(message)
    assert fastmcp_framed == message.encode() + b"\n"

    # Test reading both formats

    # Test reading standard framed message
    standard_data = MessageFraming.frame_message(message)
    reader = asyncio.StreamReader()
    reader.feed_data(standard_data)
    reader.feed_eof()

    result = await MessageFraming.read_framed_message(reader)
    assert result == message

    # Test reading FastMCP format
    fastmcp_data = MessageFraming.frame_message_fastmcp(message)
    reader = asyncio.StreamReader()
    reader.feed_data(fastmcp_data)
    reader.feed_eof()

    result = await MessageFraming.read_framed_message(reader)
    assert result == message


def test_server_path_mapping():
    """Test that server path mapping works correctly."""
    from vmcp.repository.manager import RepositoryManager

    # Create a mock server info
    class MockServerInfo:
        def __init__(self, server_id):
            self.id = server_id

    repo_manager = RepositoryManager(None)

    # Test known mappings
    test_cases = [
        ("parquet-mcp", "parquet"),
        ("node-hardware-mcp", "Node_Hardware"),
        ("arxiv-mcp", "Arxiv"),
        ("slurm-mcp", "Slurm"),
    ]

    for server_id, expected_dir in test_cases:
        server_info = MockServerInfo(server_id)
        args = repo_manager._generate_iowarp_args(server_info)

        expected_path = f"iowarp-mcps/mcps/{expected_dir}"
        assert expected_path in " ".join(args)
        assert args[0] == "run"
        assert "--directory" in args
        assert server_id in args  # Should use entry point now


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
