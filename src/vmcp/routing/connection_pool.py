"""
Production-grade connection pool implementation for vMCP.

This module provides connection pooling with health checking, automatic recovery,
and comprehensive monitoring for efficient resource management.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConnectionTimeoutError, VMCPConnectionError

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Connection pool configuration."""

    min_size: int = 1
    max_size: int = 10
    idle_timeout: int = 300  # Seconds
    acquire_timeout: int = 5  # Seconds
    validation_interval: int = 60  # Seconds
    max_lifetime: int = 3600  # Seconds
    retry_attempts: int = 3
    retry_delay: float = 0.1
    health_check_interval: int = 30


@dataclass
class PooledConnection:
    """Wrapper for pooled connections with metadata."""

    connection: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False
    healthy: bool = True
    connection_id: str = ""

    def is_expired(self, max_lifetime: int) -> bool:
        """Check if connection has exceeded maximum lifetime."""
        return time.time() - self.created_at > max_lifetime

    def is_idle_timeout(self, idle_timeout: int) -> bool:
        """Check if connection has been idle too long."""
        return not self.in_use and (time.time() - self.last_used > idle_timeout)

    def needs_validation(self, validation_interval: int) -> bool:
        """Check if connection needs health validation."""
        return time.time() - self.last_validated > validation_interval

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()
        self.use_count += 1


class ConnectionPool:
    """Production-grade connection pool implementation."""

    def __init__(
        self,
        server_id: str,
        connection_factory: Callable,
        config: PoolConfig | None = None,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            server_id: Unique server identifier
            connection_factory: Async function to create new connections
            config: Pool configuration
        """
        self.server_id = server_id
        self.connection_factory = connection_factory
        self.config = config or PoolConfig()

        self.connections: list[PooledConnection] = []
        self._waiters: list[asyncio.Future] = []
        self._lock = asyncio.Lock()
        self._closing = False
        self._maintenance_task: asyncio.Task | None = None

        # Statistics
        self._stats = {
            "acquired": 0,
            "released": 0,
            "created": 0,
            "destroyed": 0,
            "timeouts": 0,
            "errors": 0,
            "validation_failures": 0,
            "idle_evictions": 0,
            "lifetime_evictions": 0,
        }

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        logger.info(f"Initializing connection pool for {self.server_id}")

        try:
            # Create minimum connections
            create_tasks = []
            for i in range(self.config.min_size):
                create_tasks.append(self._create_connection(f"init-{i}"))

            connections = await asyncio.gather(*create_tasks, return_exceptions=True)

            async with self._lock:
                for i, conn in enumerate(connections):
                    if isinstance(conn, PooledConnection):
                        self.connections.append(conn)
                        logger.debug(
                            f"Created initial connection {i} for {self.server_id}"
                        )
                    else:
                        logger.error(f"Failed to create initial connection {i}: {conn}")

            # Start maintenance task
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())

            logger.info(
                f"Pool initialized for {self.server_id} with "
                f"{len(self.connections)}/{self.config.min_size} connections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize pool for {self.server_id}: {e}")
            raise VMCPConnectionError(f"Pool initialization failed: {e}") from e

    async def close(self) -> None:
        """Close all connections in pool."""
        logger.info(f"Closing connection pool for {self.server_id}")
        self._closing = True

        # Stop maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None

        # Cancel all waiters
        async with self._lock:
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.cancel()
            self._waiters.clear()

            # Close all connections
            close_tasks = []
            for conn in self.connections:
                close_tasks.append(self._destroy_connection(conn))

            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.connections.clear()

        logger.info(f"Connection pool closed for {self.server_id}")

    @asynccontextmanager
    async def acquire(self, timeout: float | None = None) -> Any:
        """
        Acquire connection from pool with timeout.

        Args:
            timeout: Timeout in seconds (uses config default if None)

        Yields:
            Connection object

        Raises:
            ConnectionTimeoutError: If timeout exceeded
            VMCPConnectionError: If pool is closing or other errors
        """
        if self._closing:
            raise VMCPConnectionError("Connection pool is closing")

        timeout = timeout or self.config.acquire_timeout
        start_time = time.time()
        connection = None

        try:
            while time.time() - start_time < timeout:
                async with self._lock:
                    # Find available healthy connection
                    for conn in self.connections:
                        if not conn.in_use and conn.healthy:
                            # Validate if needed
                            if conn.needs_validation(
                                self.config.validation_interval
                            ) and not await self._validate_connection(conn):
                                continue

                            # Check if connection is too old
                            if conn.is_expired(self.config.max_lifetime):
                                asyncio.create_task(self._retire_connection(conn))
                                continue

                            conn.in_use = True
                            conn.touch()
                            connection = conn
                            self._stats["acquired"] += 1
                            break

                    # Create new connection if under limit
                    if not connection and len(self.connections) < self.config.max_size:
                        try:
                            new_conn = await self._create_connection(
                                f"demand-{int(time.time())}"
                            )
                            new_conn.in_use = True
                            new_conn.touch()
                            self.connections.append(new_conn)
                            connection = new_conn
                            self._stats["acquired"] += 1
                        except Exception as e:
                            logger.error(f"Failed to create connection on demand: {e}")
                            self._stats["errors"] += 1

                if connection:
                    break

                # Wait for available connection
                waiter: asyncio.Future[None] = asyncio.Future()
                self._waiters.append(waiter)

                try:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break

                    await asyncio.wait_for(waiter, timeout=min(1.0, remaining_time))
                except asyncio.TimeoutError:
                    pass
                finally:
                    async with self._lock:
                        if waiter in self._waiters:
                            self._waiters.remove(waiter)

            if not connection:
                self._stats["timeouts"] += 1
                elapsed = time.time() - start_time
                raise ConnectionTimeoutError(
                    f"Failed to acquire connection within {timeout}s (waited {elapsed:.2f}s)",
                    timeout=timeout,
                    server_id=self.server_id,
                )

            try:
                yield connection.connection
            except Exception:
                # Mark connection as unhealthy on error
                connection.healthy = False
                raise

        finally:
            if connection:
                await self._release_connection(connection)

    async def _create_connection(self, connection_id: str) -> PooledConnection:
        """Create new pooled connection with retries."""
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"Creating connection {connection_id} (attempt {attempt + 1})"
                )
                connection = await self.connection_factory()

                pooled = PooledConnection(
                    connection=connection, connection_id=connection_id
                )

                self._stats["created"] += 1
                logger.debug(f"Created connection {connection_id} for {self.server_id}")
                return pooled

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to create connection {connection_id} "
                    f"(attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )

                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (
                        2**attempt
                    )  # Exponential backoff
                    await asyncio.sleep(delay)

        raise VMCPConnectionError(
            f"Failed to create connection after {self.config.retry_attempts} attempts: {last_error}"
        ) from last_error

    async def _release_connection(self, conn: PooledConnection) -> None:
        """Release connection back to pool."""
        async with self._lock:
            conn.in_use = False
            conn.last_used = time.time()
            self._stats["released"] += 1

            # Remove unhealthy or expired connections
            if not conn.healthy or conn.is_expired(self.config.max_lifetime):
                asyncio.create_task(self._retire_connection(conn))
            else:
                # Notify waiters
                self._notify_waiters()

    async def _retire_connection(self, conn: PooledConnection) -> None:
        """Retire a connection from the pool."""
        async with self._lock:
            if conn in self.connections:
                self.connections.remove(conn)

        await self._destroy_connection(conn)

        # Create replacement if needed
        async with self._lock:
            if not self._closing and len(self.connections) < self.config.min_size:
                try:
                    new_conn = await self._create_connection("replacement")
                    self.connections.append(new_conn)
                except Exception as e:
                    logger.error(f"Failed to create replacement connection: {e}")

    def _notify_waiters(self) -> None:
        """Notify waiting acquirers."""
        for waiter in self._waiters[:]:
            if not waiter.done():
                waiter.set_result(None)
                break

    async def _validate_connection(self, conn: PooledConnection) -> bool:
        """Validate connection health."""
        try:
            # Basic validation - check if connection object exists
            if conn.connection is None:
                return False

            # Attempt to ping if method exists
            if hasattr(conn.connection, "ping"):
                result = await conn.connection.ping()
                conn.last_validated = time.time()
                conn.healthy = bool(result)
                return conn.healthy

            # If no ping method, assume healthy
            conn.last_validated = time.time()
            conn.healthy = True
            return True

        except Exception as e:
            logger.debug(f"Connection validation failed for {conn.connection_id}: {e}")
            conn.healthy = False
            self._stats["validation_failures"] += 1
            return False

    async def _destroy_connection(self, conn: PooledConnection) -> None:
        """Destroy a connection."""
        try:
            if hasattr(conn.connection, "disconnect"):
                await conn.connection.disconnect()
            elif hasattr(conn.connection, "close"):
                await conn.connection.close()

            self._stats["destroyed"] += 1
            logger.debug(
                f"Destroyed connection {conn.connection_id} for {self.server_id}"
            )

        except Exception as e:
            logger.warning(f"Error destroying connection {conn.connection_id}: {e}")

    async def _maintenance_loop(self) -> None:
        """Background maintenance task."""
        logger.debug(f"Starting maintenance loop for {self.server_id}")

        while not self._closing:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop for {self.server_id}: {e}")

        logger.debug(f"Maintenance loop ended for {self.server_id}")

    async def _perform_maintenance(self) -> None:
        """Perform pool maintenance tasks."""
        async with self._lock:
            connections_to_retire = []

            for conn in self.connections:
                if conn.in_use:
                    continue

                # Check for idle timeout
                if (
                    conn.is_idle_timeout(self.config.idle_timeout)
                    and len(self.connections) > self.config.min_size
                ):
                    connections_to_retire.append(conn)
                    self._stats["idle_evictions"] += 1
                    continue

                # Check for lifetime expiration
                if conn.is_expired(self.config.max_lifetime):
                    connections_to_retire.append(conn)
                    self._stats["lifetime_evictions"] += 1
                    continue

                # Validate connection if needed
                if conn.needs_validation(
                    self.config.validation_interval
                ) and not await self._validate_connection(conn):
                    connections_to_retire.append(conn)

            # Retire connections outside the lock
            for conn in connections_to_retire:
                asyncio.create_task(self._retire_connection(conn))

        # Log maintenance stats
        if connections_to_retire:
            logger.debug(
                f"Pool maintenance for {self.server_id}: retired {len(connections_to_retire)} connections"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        active_connections = sum(1 for c in self.connections if c.in_use)
        healthy_connections = sum(1 for c in self.connections if c.healthy)

        return {
            **self._stats,
            "server_id": self.server_id,
            "total_connections": len(self.connections),
            "active_connections": active_connections,
            "idle_connections": len(self.connections) - active_connections,
            "healthy_connections": healthy_connections,
            "unhealthy_connections": len(self.connections) - healthy_connections,
            "waiting_acquirers": len(self._waiters),
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "idle_timeout": self.config.idle_timeout,
                "max_lifetime": self.config.max_lifetime,
            },
            "is_closing": self._closing,
        }

    def get_connection_details(self) -> list[dict[str, Any]]:
        """Get detailed information about each connection."""
        now = time.time()
        return [
            {
                "connection_id": conn.connection_id,
                "created_at": conn.created_at,
                "last_used": conn.last_used,
                "last_validated": conn.last_validated,
                "age": now - conn.created_at,
                "idle_time": now - conn.last_used if not conn.in_use else 0,
                "use_count": conn.use_count,
                "in_use": conn.in_use,
                "healthy": conn.healthy,
            }
            for conn in self.connections
        ]


class ConnectionPoolManager:
    """Manages multiple connection pools."""

    def __init__(self) -> None:
        """Initialize connection pool manager."""
        self._pools: dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()

    async def get_pool(
        self,
        server_id: str,
        connection_factory: Callable,
        config: PoolConfig | None = None,
    ) -> ConnectionPool:
        """
        Get or create connection pool for server.

        Args:
            server_id: Server identifier
            connection_factory: Function to create connections
            config: Pool configuration

        Returns:
            Connection pool instance
        """
        async with self._lock:
            if server_id not in self._pools:
                pool = ConnectionPool(server_id, connection_factory, config)
                await pool.initialize()
                self._pools[server_id] = pool
                logger.info(f"Created connection pool for server {server_id}")

            return self._pools[server_id]

    async def remove_pool(self, server_id: str) -> bool:
        """
        Remove and close connection pool.

        Args:
            server_id: Server identifier

        Returns:
            True if pool was removed, False if not found
        """
        async with self._lock:
            pool = self._pools.pop(server_id, None)
            if pool:
                await pool.close()
                logger.info(f"Removed connection pool for server {server_id}")
                return True
            return False

    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        async with self._lock:
            close_tasks = []
            for pool in self._pools.values():
                close_tasks.append(pool.close())

            await asyncio.gather(*close_tasks, return_exceptions=True)
            self._pools.clear()

        logger.info("Closed all connection pools")

    def list_pools(self) -> list[str]:
        """Get list of all pool server IDs."""
        return list(self._pools.keys())

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all pools."""
        return {server_id: pool.get_stats() for server_id, pool in self._pools.items()}

    def get_pool_count(self) -> int:
        """Get number of managed pools."""
        return len(self._pools)
