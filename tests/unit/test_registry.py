"""Tests for server registry functionality."""

import tempfile
from pathlib import Path

import pytest

from vmcp.errors import RegistryError
from vmcp.registry.registry import ServerInfo, ServerRegistry, ServerState


class TestServerState:
    def test_server_state_creation(self):
        """Test creating server state."""
        state = ServerState()

        assert state.status == "unknown"
        assert state.error_count == 0
        assert state.consecutive_failures == 0
        assert state.last_error_time is None
        assert state.uptime_start is not None

    def test_server_state_uptime(self):
        """Test uptime calculation."""
        state = ServerState()
        uptime = state.uptime

        assert isinstance(uptime, float)
        assert uptime >= 0


class TestServerInfo:
    def test_server_info_creation(self):
        """Test creating server info."""
        info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        assert info.id == "test-server"
        assert info.name == "Test Server"
        assert info.transport == "stdio"
        assert info.enabled is True  # default
        assert info.state.status == "unknown"

    def test_server_info_with_state(self):
        """Test server info with custom state."""
        state = ServerState()
        state.status = "healthy"

        info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
            state=state,
        )

        assert info.state.status == "healthy"


class TestServerRegistry:
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create server registry with temporary path."""
        return ServerRegistry(registry_path=str(temp_registry_path))

    def test_registry_initialization(self, registry, temp_registry_path):
        """Test registry initialization."""
        assert registry.registry_path == str(temp_registry_path)
        assert registry._servers == {}
        assert (temp_registry_path / "servers.json").exists()

    def test_add_server(self, registry):
        """Test adding a server to registry."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        result = registry.add_server(server_info)
        assert result is True
        assert "test-server" in registry._servers
        assert registry._servers["test-server"] == server_info

    def test_add_duplicate_server(self, registry):
        """Test adding duplicate server."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)

        # Adding same server again should fail
        with pytest.raises(RegistryError, match="Server .* already exists"):
            registry.add_server(server_info)

    def test_remove_server(self, registry):
        """Test removing a server."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)
        assert "test-server" in registry._servers

        result = registry.remove_server("test-server")
        assert result is True
        assert "test-server" not in registry._servers

    def test_remove_nonexistent_server(self, registry):
        """Test removing non-existent server."""
        result = registry.remove_server("nonexistent")
        assert result is False

    def test_get_server(self, registry):
        """Test getting server information."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)

        retrieved = registry.get_server("test-server")
        assert retrieved == server_info

        # Non-existent server
        assert registry.get_server("nonexistent") is None

    def test_list_servers(self, registry):
        """Test listing all servers."""
        server1 = ServerInfo(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = ServerInfo(
            id="server2", name="Server 2", transport="http", command="node", args=[]
        )

        registry.add_server(server1)
        registry.add_server(server2)

        servers = registry.list_servers()
        assert len(servers) == 2
        assert server1 in servers
        assert server2 in servers

    def test_list_servers_by_status(self, registry):
        """Test listing servers by status."""
        server1 = ServerInfo(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = ServerInfo(
            id="server2", name="Server 2", transport="stdio", command="python", args=[]
        )

        server1.state.status = "healthy"
        server2.state.status = "unhealthy"

        registry.add_server(server1)
        registry.add_server(server2)

        healthy_servers = registry.list_servers(status="healthy")
        assert len(healthy_servers) == 1
        assert healthy_servers[0] == server1

        unhealthy_servers = registry.list_servers(status="unhealthy")
        assert len(unhealthy_servers) == 1
        assert unhealthy_servers[0] == server2

    def test_update_server_status(self, registry):
        """Test updating server status."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)

        registry.update_server_status("test-server", "healthy")
        updated = registry.get_server("test-server")
        assert updated.state.status == "healthy"

    def test_update_nonexistent_server_status(self, registry):
        """Test updating status of non-existent server."""
        result = registry.update_server_status("nonexistent", "healthy")
        assert result is False

    def test_get_server_stats(self, registry):
        """Test getting server statistics."""
        server1 = ServerInfo(
            id="server1", name="Server 1", transport="stdio", command="python", args=[]
        )
        server2 = ServerInfo(
            id="server2", name="Server 2", transport="stdio", command="python", args=[]
        )

        server1.state.status = "healthy"
        server2.state.status = "unhealthy"

        registry.add_server(server1)
        registry.add_server(server2)

        stats = registry.get_stats()
        assert stats["total_servers"] == 2
        assert stats["healthy_servers"] == 1
        assert stats["unhealthy_servers"] == 1

    def test_persistence_save_load(self, registry, temp_registry_path):
        """Test saving and loading registry data."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)
        registry.save()

        # Create new registry instance and load
        new_registry = ServerRegistry(registry_path=str(temp_registry_path))
        new_registry.load()

        retrieved = new_registry.get_server("test-server")
        assert retrieved is not None
        assert retrieved.id == "test-server"
        assert retrieved.name == "Test Server"

    def test_load_invalid_json(self, registry, temp_registry_path):
        """Test loading with invalid JSON file."""
        # Create invalid JSON file
        servers_file = temp_registry_path / "servers.json"
        servers_file.write_text("invalid json content")

        # Should handle gracefully and initialize empty registry
        registry.load()
        assert len(registry._servers) == 0

    def test_enable_disable_server(self, registry):
        """Test enabling and disabling servers."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
            enabled=False,
        )

        registry.add_server(server_info)

        # Enable server
        result = registry.enable_server("test-server")
        assert result is True
        assert registry.get_server("test-server").enabled is True

        # Disable server
        result = registry.disable_server("test-server")
        assert result is True
        assert registry.get_server("test-server").enabled is False

    def test_get_enabled_servers(self, registry):
        """Test getting only enabled servers."""
        server1 = ServerInfo(
            id="server1",
            name="Server 1",
            transport="stdio",
            command="python",
            args=[],
            enabled=True,
        )
        server2 = ServerInfo(
            id="server2",
            name="Server 2",
            transport="stdio",
            command="python",
            args=[],
            enabled=False,
        )

        registry.add_server(server1)
        registry.add_server(server2)

        enabled_servers = registry.get_enabled_servers()
        assert len(enabled_servers) == 1
        assert enabled_servers[0] == server1

    def test_health_check_update(self, registry):
        """Test health check updates."""
        server_info = ServerInfo(
            id="test-server",
            name="Test Server",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        registry.add_server(server_info)

        # Update health check
        registry.update_health_check("test-server", True)
        server = registry.get_server("test-server")
        assert server.state.status == "healthy"
        assert server.state.consecutive_failures == 0

        # Failed health check
        registry.update_health_check("test-server", False, error="Connection failed")
        server = registry.get_server("test-server")
        assert server.state.status == "unhealthy"
        assert server.state.consecutive_failures == 1
        assert server.state.last_error == "Connection failed"
