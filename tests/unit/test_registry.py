"""Tests for server registry functionality."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from vmcp.errors import RegistryError, ServerNotFoundError
from vmcp.registry.registry import MCPServerConfig, Registry, MCPServerState


class TestMCPServerConfig:
    def test_server_config_creation(self):
        """Test creating server config."""
        config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        assert config.id == "test-server"
        assert config.name == "Test Server"
        assert config.transport == "stdio"
        assert config.enabled is True  # default
        assert config.command == "python"
        assert config.args == ["server.py"]

    def test_server_config_environment_expansion(self):
        """Test environment variable expansion."""
        config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            environment={"HOME": "$HOME", "USER": "testuser"}
        )
        
        expanded = config.expand_environment()
        assert "HOME" in expanded
        assert "USER" in expanded
        assert expanded["USER"] == "testuser"


class TestMCPServerState:
    def test_server_state_creation(self):
        """Test creating server state."""
        config = MCPServerConfig(id="test-server", name="Test Server")
        state = MCPServerState(config=config)

        assert state.is_healthy is False
        assert state.connection_count == 0
        assert state.total_requests == 0
        assert state.failed_requests == 0
        assert state.last_error is None
        assert state.last_error_time is None
        assert state.uptime_start is None

    def test_server_state_update_health(self):
        """Test health update."""
        config = MCPServerConfig(id="test-server", name="Test Server")
        state = MCPServerState(config=config)
        
        state.update_health(True)
        assert state.is_healthy is True
        assert state.last_health_check is not None
        assert state.uptime_start is not None
        
        state.update_health(False, "Connection failed")
        assert state.is_healthy is False
        assert state.last_error == "Connection failed"
        assert state.last_error_time is not None
        assert state.failed_requests == 1

    def test_server_state_record_request(self):
        """Test request recording."""
        config = MCPServerConfig(id="test-server", name="Test Server")
        state = MCPServerState(config=config)
        
        state.record_request(True)
        assert state.total_requests == 1
        assert state.failed_requests == 0
        
        state.record_request(False)
        assert state.total_requests == 2
        assert state.failed_requests == 1
        
        assert state.get_error_rate() == 50.0


class TestRegistry:
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create server registry with temporary path."""
        return Registry(registry_path=str(temp_registry_path))

    def test_registry_initialization(self, registry, temp_registry_path):
        """Test registry initialization."""
        assert registry.registry_path == temp_registry_path
        assert registry._servers == {}
        # Registry doesn't create the file until first save
        assert registry._config_file == temp_registry_path / "servers.json"

    @pytest.mark.asyncio
    async def test_add_server(self, registry):
        """Test adding a server to registry."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)
        assert "test-server" in registry._servers
        assert registry._servers["test-server"].config.id == "test-server"

    @pytest.mark.asyncio
    async def test_add_duplicate_server(self, registry):
        """Test adding duplicate server."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)

        # Adding same server again should fail
        with pytest.raises(RegistryError, match="Server .* already registered"):
            await registry.register_server(server_config)

    @pytest.mark.asyncio
    async def test_remove_server(self, registry):
        """Test removing a server."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)
        assert "test-server" in registry._servers

        await registry.unregister_server("test-server")
        assert "test-server" not in registry._servers

    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self, registry):
        """Test removing non-existent server."""
        with pytest.raises(ServerNotFoundError):
            await registry.unregister_server("nonexistent")

    @pytest.mark.asyncio
    async def test_get_server(self, registry):
        """Test getting server information."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)

        retrieved = registry.get_server("test-server")
        assert retrieved is not None
        assert retrieved.config.id == "test-server"

        # Non-existent server
        assert registry.get_server("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_servers(self, registry):
        """Test listing all servers."""
        server1 = MCPServerConfig(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = MCPServerConfig(
            id="server2", name="Server 2", transport="stdio", command="node", args=[]
        )

        await registry.register_server(server1)
        await registry.register_server(server2)

        servers = registry.get_all_servers()
        assert len(servers) == 2
        assert any(s.config.id == "server1" for s in servers)
        assert any(s.config.id == "server2" for s in servers)

    @pytest.mark.asyncio
    async def test_list_servers_by_status(self, registry):
        """Test listing servers by status."""
        server1 = MCPServerConfig(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = MCPServerConfig(
            id="server2", name="Server 2", transport="stdio", command="python", args=[]
        )

        await registry.register_server(server1)
        await registry.register_server(server2)
        
        # Update health status
        state1 = registry.get_server("server1")
        state2 = registry.get_server("server2")
        state1.update_health(True)
        state2.update_health(False)

        healthy_servers = registry.get_healthy_servers()
        assert len(healthy_servers) == 1
        assert healthy_servers[0].config.id == "server1"

    @pytest.mark.asyncio
    async def test_update_server_status(self, registry):
        """Test updating server status."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)

        state = registry.get_server("test-server")
        state.update_health(True)
        assert state.is_healthy is True

    @pytest.mark.asyncio
    async def test_update_nonexistent_server_status(self, registry):
        """Test updating status of non-existent server."""
        with pytest.raises(ServerNotFoundError):
            await registry.update_server_config("nonexistent", enabled=True)

    @pytest.mark.asyncio
    async def test_get_server_stats(self, registry):
        """Test getting server statistics."""
        server1 = MCPServerConfig(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = MCPServerConfig(
            id="server2", name="Server 2", transport="stdio", command="python", args=[]
        )

        await registry.register_server(server1)
        await registry.register_server(server2)
        
        # Update health status
        state1 = registry.get_server("server1")
        state2 = registry.get_server("server2")
        state1.update_health(True)
        state2.update_health(False)

        stats = registry.get_registry_stats()
        assert stats["total_servers"] == 2
        assert stats["healthy_servers"] == 1
        assert stats["unhealthy_servers"] == 1

    @pytest.mark.asyncio
    async def test_persistence_save_load(self, registry, temp_registry_path):
        """Test saving and loading registry data."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)
        await registry.save_config()

        # Create new registry instance and load
        new_registry = Registry(registry_path=str(temp_registry_path))
        await new_registry.load_servers()

        retrieved = new_registry.get_server("test-server")
        assert retrieved is not None
        assert retrieved.config.id == "test-server"
        assert retrieved.config.name == "Test Server"

    @pytest.mark.asyncio
    async def test_load_invalid_json(self, registry, temp_registry_path):
        """Test loading with invalid JSON file."""
        # Create invalid JSON file
        servers_file = temp_registry_path / "servers.json"
        servers_file.write_text("invalid json content")

        # Should handle gracefully and initialize empty registry
        with pytest.raises(RegistryError):
            await registry.load_servers()

    @pytest.mark.asyncio
    async def test_enable_disable_server(self, registry):
        """Test enabling and disabling servers."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
            enabled=False,
        )

        await registry.register_server(server_config)

        # Enable server
        await registry.update_server_config("test-server", enabled=True)
        assert registry.get_server("test-server").config.enabled is True

        # Disable server
        await registry.update_server_config("test-server", enabled=False)
        assert registry.get_server("test-server").config.enabled is False

    @pytest.mark.asyncio
    async def test_get_enabled_servers(self, registry):
        """Test getting only enabled servers."""
        server1 = MCPServerConfig(
            id="server1",
            name="Server 1",
            transport="stdio",
            command="python",
            args=[],
            enabled=True,
        )
        server2 = MCPServerConfig(
            id="server2",
            name="Server 2",
            transport="stdio",
            command="python",
            args=[],
            enabled=False,
        )

        await registry.register_server(server1)
        await registry.register_server(server2)

        enabled_servers = registry.get_enabled_servers()
        assert len(enabled_servers) == 1
        assert enabled_servers[0].config.id == "server1"

    @pytest.mark.asyncio
    async def test_health_check_update(self, registry):
        """Test health check updates."""
        server_config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        await registry.register_server(server_config)

        # Update health check
        server = registry.get_server("test-server")
        server.update_health(True)
        assert server.is_healthy is True
        assert server.failed_requests == 0

        # Failed health check
        server.update_health(False, "Connection failed")
        assert server.is_healthy is False
        assert server.failed_requests == 1
        assert server.last_error == "Connection failed"
