"""
Server registry for managing MCP server configurations.

This module provides comprehensive server management including configuration loading,
health monitoring, lifecycle management, and dynamic server mounting/unmounting.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..errors import ConfigurationError, RegistryError, ServerNotFoundError
from ..gateway.transports.stdio import StdioServerConnection
from ..routing.connection_pool import ConnectionPool, ConnectionPoolManager, PoolConfig

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    id: str = Field(..., description="Unique server identifier")
    name: str = Field(..., description="Human-readable server name")
    transport: str = Field(default="stdio", description="Transport type")
    command: str | None = Field(None, description="Command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    url: str | None = Field(None, description="Server URL for HTTP/WebSocket")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    capabilities: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict, description="Server capabilities"
    )
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    max_retries: int = Field(default=3, description="Maximum connection retries")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    enabled: bool = Field(default=True, description="Whether server is enabled")
    pool_config: dict[str, Any] | None = Field(None, description="Connection pool configuration")

    def get_pool_config(self) -> PoolConfig:
        """Get connection pool configuration."""
        if self.pool_config:
            return PoolConfig(**self.pool_config)
        return PoolConfig()

    def expand_environment(self) -> dict[str, str]:
        """Expand environment variables in configuration."""
        expanded = {}
        for key, value in self.environment.items():
            expanded[key] = os.path.expandvars(value)
        return expanded


class MCPServerState(BaseModel):
    """Runtime state of an MCP server."""
    config: MCPServerConfig
    is_healthy: bool = False
    last_health_check: datetime | None = None
    connection_count: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: str | None = None
    last_error_time: datetime | None = None
    uptime_start: datetime | None = None

    class Config:
        arbitrary_types_allowed = True

    def update_health(self, healthy: bool, error: str | None = None) -> None:
        """Update health status."""
        self.is_healthy = healthy
        self.last_health_check = datetime.now()

        if not healthy and error:
            self.last_error = error
            self.last_error_time = datetime.now()
            self.failed_requests += 1
        elif healthy and not self.uptime_start:
            self.uptime_start = datetime.now()

    def record_request(self, success: bool = True) -> None:
        """Record request statistics."""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

    def get_uptime(self) -> float | None:
        """Get server uptime in seconds."""
        if self.uptime_start:
            return (datetime.now() - self.uptime_start).total_seconds()
        return None

    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100


class Registry:
    """Server registry with health monitoring and discovery."""

    def __init__(self, registry_path: str = "~/.vmcp/registry") -> None:
        """
        Initialize registry.
        
        Args:
            registry_path: Path to registry configuration directory
        """
        self.registry_path = Path(registry_path).expanduser()
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self._config_file = self.registry_path / "servers.json"
        self._servers: dict[str, MCPServerState] = {}
        self._pool_manager = ConnectionPoolManager()
        self._health_monitor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        logger.info(f"Registry initialized with path: {self.registry_path}")

    async def initialize(self) -> None:
        """Initialize the registry."""
        await self.load_servers()
        await self.start_health_monitoring()

    async def shutdown(self) -> None:
        """Shutdown the registry."""
        await self.stop_health_monitoring()
        await self._pool_manager.close_all_pools()

    async def load_servers(self) -> None:
        """Load server configurations from disk."""
        if not self._config_file.exists():
            logger.info("No server configuration file found, starting with empty registry")
            return

        try:
            with open(self._config_file, encoding='utf-8') as f:
                config_data = json.load(f)

            servers_data = config_data.get("servers", [])
            loaded_count = 0

            async with self._lock:
                for server_data in servers_data:
                    try:
                        config = MCPServerConfig(**server_data)
                        state = MCPServerState(config=config)
                        self._servers[config.id] = state
                        loaded_count += 1

                        logger.debug(f"Loaded server configuration: {config.id}")

                    except ValidationError as e:
                        logger.error(f"Invalid server configuration: {e}")
                        continue

            logger.info(f"Loaded {loaded_count} server configurations from registry")

        except Exception as e:
            logger.error(f"Failed to load server configurations: {e}")
            raise RegistryError(f"Failed to load servers: {e}") from e

    async def save_config(self) -> None:
        """Save current configuration to disk."""
        try:
            config_data = {
                "version": "1.0.0",
                "servers": [
                    state.config.dict() for state in self._servers.values()
                ]
            }

            # Write to temporary file first
            temp_file = self._config_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)

            # Atomic move
            temp_file.replace(self._config_file)

            logger.debug("Saved server configurations to registry")

        except Exception as e:
            logger.error(f"Failed to save server configurations: {e}")
            raise RegistryError(f"Failed to save servers: {e}") from e

    async def register_server(self, config: MCPServerConfig) -> None:
        """
        Register a new MCP server.
        
        Args:
            config: Server configuration
            
        Raises:
            RegistryError: If server already exists or registration fails
        """
        async with self._lock:
            if config.id in self._servers:
                raise RegistryError(f"Server {config.id} already registered")

            # Validate configuration
            if config.transport == "stdio" and not config.command:
                raise ConfigurationError(f"Server {config.id}: stdio transport requires command")

            if config.transport in ["http", "websocket"] and not config.url:
                raise ConfigurationError(f"Server {config.id}: {config.transport} transport requires url")

            # Create server state
            state = MCPServerState(config=config)
            self._servers[config.id] = state

            logger.info(f"Registered server: {config.id} ({config.transport})")

        # Save configuration
        await self.save_config()

        # Initialize connection pool if enabled
        if config.enabled:
            await self._initialize_server_pool(config.id)

        # Perform initial health check
        await self._check_server_health(config.id)

    async def unregister_server(self, server_id: str) -> None:
        """
        Unregister an MCP server.
        
        Args:
            server_id: Server ID to unregister
            
        Raises:
            ServerNotFoundError: If server not found
        """
        async with self._lock:
            if server_id not in self._servers:
                raise ServerNotFoundError(server_id)

            del self._servers[server_id]
            logger.info(f"Unregistered server: {server_id}")

        # Remove connection pool
        await self._pool_manager.remove_pool(server_id)

        # Save configuration
        await self.save_config()

    async def update_server_config(
        self,
        server_id: str,
        **updates: Any
    ) -> None:
        """
        Update server configuration.
        
        Args:
            server_id: Server ID to update
            **updates: Configuration updates
            
        Raises:
            ServerNotFoundError: If server not found
        """
        async with self._lock:
            if server_id not in self._servers:
                raise ServerNotFoundError(server_id)

            state = self._servers[server_id]

            # Create updated configuration
            config_dict = state.config.dict()
            config_dict.update(updates)

            try:
                new_config = MCPServerConfig(**config_dict)
                state.config = new_config
                logger.info(f"Updated server configuration: {server_id}")
            except ValidationError as e:
                raise ConfigurationError(f"Invalid configuration update: {e}") from e

        # Save configuration
        await self.save_config()

        # Reinitialize pool if needed
        if self._servers[server_id].config.enabled:
            await self._pool_manager.remove_pool(server_id)
            await self._initialize_server_pool(server_id)

    def get_server(self, server_id: str) -> MCPServerState | None:
        """Get server by ID."""
        return self._servers.get(server_id)

    def get_server_config(self, server_id: str) -> MCPServerConfig | None:
        """Get server configuration by ID."""
        state = self._servers.get(server_id)
        return state.config if state else None

    def get_all_servers(self) -> list[MCPServerState]:
        """Get all registered servers."""
        return list(self._servers.values())

    def get_enabled_servers(self) -> list[MCPServerState]:
        """Get all enabled servers."""
        return [state for state in self._servers.values() if state.config.enabled]

    def get_healthy_servers(self) -> list[MCPServerState]:
        """Get all healthy enabled servers."""
        return [
            state for state in self._servers.values()
            if state.config.enabled and state.is_healthy
        ]

    def get_servers_by_capability(self, capability: str) -> list[MCPServerState]:
        """
        Get servers that support a specific capability.
        
        Args:
            capability: Capability name (e.g., "tools", "resources")
            
        Returns:
            List of servers supporting the capability
        """
        return [
            state for state in self._servers.values()
            if (state.config.enabled and
                state.is_healthy and
                capability in state.config.capabilities)
        ]

    async def get_connection_pool(self, server_id: str) -> ConnectionPool | None:
        """Get connection pool for server."""
        if server_id not in self._servers:
            return None

        pools = await self._pool_manager.get_all_stats()
        if server_id in pools:
            return self._pool_manager._pools.get(server_id)
        return None

    async def start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        if self._health_monitor_task:
            return

        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring task."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None
            logger.info("Stopped health monitoring")

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        logger.debug("Starting health monitor loop")

        while True:
            try:
                await self._check_all_servers_health()

                # Use minimum interval from all servers
                intervals = [
                    state.config.health_check_interval
                    for state in self._servers.values()
                    if state.config.enabled
                ]
                interval = min(intervals) if intervals else 30

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Default interval on error

        logger.debug("Health monitor loop ended")

    async def _check_all_servers_health(self) -> None:
        """Check health of all enabled servers."""
        enabled_servers = self.get_enabled_servers()
        if not enabled_servers:
            return

        # Check health in parallel
        tasks = [
            self._check_server_health(state.config.id)
            for state in enabled_servers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                server_id = enabled_servers[i].config.id
                logger.error(f"Health check failed for {server_id}: {result}")

    async def _check_server_health(self, server_id: str) -> None:
        """Check health of individual server."""
        state = self._servers.get(server_id)
        if not state or not state.config.enabled:
            return

        try:
            # Get connection pool
            pool = await self._pool_manager.get_pool(
                server_id,
                lambda: self._create_server_connection(server_id),
                state.config.get_pool_config()
            )

            # Perform health check using connection pool
            async with pool.acquire(timeout=5.0) as connection:
                if hasattr(connection, 'ping'):
                    healthy = await connection.ping()
                else:
                    # Fallback: assume healthy if connection exists
                    healthy = connection is not None

            state.update_health(healthy)

            if healthy:
                logger.debug(f"Health check passed for {server_id}")
            else:
                logger.warning(f"Health check failed for {server_id}")

        except Exception as e:
            error_msg = f"Health check error: {e}"
            state.update_health(False, error_msg)
            logger.warning(f"Health check failed for {server_id}: {e}")

    async def _initialize_server_pool(self, server_id: str) -> None:
        """Initialize connection pool for server."""
        state = self._servers.get(server_id)
        if not state:
            return

        try:
            await self._pool_manager.get_pool(
                server_id,
                lambda: self._create_server_connection(server_id),
                state.config.get_pool_config()
            )
            logger.debug(f"Initialized connection pool for {server_id}")
        except Exception as e:
            logger.error(f"Failed to initialize pool for {server_id}: {e}")

    async def _create_server_connection(self, server_id: str):
        """Create connection to server."""
        state = self._servers.get(server_id)
        if not state:
            raise ServerNotFoundError(server_id)

        config = state.config

        if config.transport == "stdio":
            connection = StdioServerConnection(config)
            await connection.connect()
            return connection
        else:
            raise ConfigurationError(f"Unsupported transport: {config.transport}")

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        total_servers = len(self._servers)
        enabled_servers = len(self.get_enabled_servers())
        healthy_servers = len(self.get_healthy_servers())

        # Calculate overall health
        if total_servers == 0:
            overall_health = "no_servers"
        elif healthy_servers == enabled_servers:
            overall_health = "healthy"
        elif healthy_servers > enabled_servers // 2:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"

        # Get transport breakdown
        transport_counts = {}
        for state in self._servers.values():
            transport = state.config.transport
            transport_counts[transport] = transport_counts.get(transport, 0) + 1

        # Get capability breakdown
        capability_counts = {}
        for state in self._servers.values():
            for capability in state.config.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1

        return {
            "total_servers": total_servers,
            "enabled_servers": enabled_servers,
            "disabled_servers": total_servers - enabled_servers,
            "healthy_servers": healthy_servers,
            "unhealthy_servers": enabled_servers - healthy_servers,
            "overall_health": overall_health,
            "transports": transport_counts,
            "capabilities": capability_counts,
            "connection_pools": len(self._pool_manager.list_pools()),
            "registry_path": str(self.registry_path),
            "config_file_exists": self._config_file.exists(),
        }

    def get_server_details(self, server_id: str) -> dict[str, Any] | None:
        """Get detailed information about a server."""
        state = self._servers.get(server_id)
        if not state:
            return None

        pool_stats = None
        pools = self._pool_manager.get_all_stats()
        if server_id in pools:
            pool_stats = pools[server_id]

        return {
            "config": state.config.dict(),
            "state": {
                "is_healthy": state.is_healthy,
                "last_health_check": state.last_health_check.isoformat() if state.last_health_check else None,
                "connection_count": state.connection_count,
                "total_requests": state.total_requests,
                "failed_requests": state.failed_requests,
                "error_rate": state.get_error_rate(),
                "last_error": state.last_error,
                "last_error_time": state.last_error_time.isoformat() if state.last_error_time else None,
                "uptime": state.get_uptime(),
                "uptime_start": state.uptime_start.isoformat() if state.uptime_start else None,
            },
            "pool_stats": pool_stats,
        }
