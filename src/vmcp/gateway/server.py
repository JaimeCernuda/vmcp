"""
Main vMCP Gateway server implementation.

This module implements the production-grade gateway that orchestrates all
components including transports, routing, registry, and monitoring.
"""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from ..errors import ConfigurationError, TransportError, VMCPError, VMCPErrorCode
from ..monitoring.health import HealthChecker
from ..monitoring.metrics import MetricsCollector
from ..registry.registry import Registry
from ..routing.router import Router
from .protocol import ProtocolHandler
from .transports.base import Transport
from .transports.stdio import StdioTransport

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(),
    wrapper_class=structlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@dataclass
class GatewayConfig:
    """Configuration for the vMCP Gateway."""

    registry_path: str = "~/.vmcp/registry"

    transports: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "stdio": {"enabled": True},
            "http": {"enabled": False, "port": 3000, "host": "127.0.0.1"},
            "websocket": {"enabled": False, "port": 3001, "host": "127.0.0.1"},
        }
    )

    # Cache configuration
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    cache_max_size: int = 1000

    # Performance configuration
    max_connections: int = 1000
    request_timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB

    # Health monitoring
    health_check_interval: int = 30
    health_check_timeout: int = 5

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "structured"  # "structured" or "standard"
    log_file: str | None = None

    # Security configuration
    enable_request_validation: bool = True
    max_concurrent_requests: int = 100
    rate_limit_requests: int = 1000  # per minute

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.registry_path = str(Path(self.registry_path).expanduser())

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")

        # Validate transport configuration
        for transport_name, config in self.transports.items():
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Invalid transport config for {transport_name}"
                )
            if "enabled" not in config:
                config["enabled"] = False


class VMCPGateway:
    """Main vMCP Gateway server implementation."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        """
        Initialize vMCP Gateway.

        Args:
            config: Gateway configuration
        """
        self.config = config or GatewayConfig()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        # Configure logging
        self._configure_logging()

        # Core components
        self.protocol_handler = ProtocolHandler()
        self.registry = Registry(self.config.registry_path)
        self.router = Router(self.registry)
        self.metrics_collector = MetricsCollector()

        # Transports
        self.transports: dict[str, Transport] = {}

        # Health monitoring
        self.health_checker: HealthChecker | None = None

        # Request limiting
        self._active_requests = 0
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        logger.info("vMCP Gateway initialized", config=self.config.__dict__)

    def _configure_logging(self) -> None:
        """Configure logging based on configuration."""
        level = getattr(logging, self.config.log_level.upper())

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        if self.config.log_format == "structured":
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )

        # Add console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if configured
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    async def initialize(self) -> None:
        """Initialize the gateway and all components."""
        if self._running:
            logger.warning("Gateway already initialized")
            return

        logger.info("Initializing vMCP Gateway")

        try:
            # Initialize registry
            await self.registry.initialize()

            # Initialize health checker
            components = {
                "gateway": self,
                "registry": self.registry,
                "router": self.router,
                "metrics": self.metrics_collector,
            }
            self.health_checker = HealthChecker(components)

            # Initialize transports
            await self._initialize_transports()

            logger.info("vMCP Gateway initialization completed")

        except Exception as e:
            logger.error("Failed to initialize gateway", error=str(e), exc_info=True)
            raise VMCPError(
                VMCPErrorCode.INTERNAL_ERROR, f"Initialization failed: {e}"
            ) from e

    async def _initialize_transports(self) -> None:
        """Initialize configured transports."""
        for transport_name, transport_config in self.config.transports.items():
            if not transport_config.get("enabled", False):
                continue

            try:
                if transport_name == "stdio":
                    transport = StdioTransport(
                        self.handle_request, self.protocol_handler
                    )
                    self.transports[transport_name] = transport
                    logger.info("Initialized stdio transport")

                elif transport_name == "http":
                    # HTTP transport would be implemented here
                    logger.warning("HTTP transport not yet implemented")

                elif transport_name == "websocket":
                    # WebSocket transport would be implemented here
                    logger.warning("WebSocket transport not yet implemented")

                else:
                    logger.warning("Unknown transport type", transport=transport_name)

            except Exception as e:
                logger.error(
                    "Failed to initialize transport",
                    transport=transport_name,
                    error=str(e),
                    exc_info=True,
                )
                raise TransportError(
                    f"Failed to initialize {transport_name} transport: {e}"
                ) from e

    async def start(self) -> None:
        """Start the gateway server."""
        if self._running:
            logger.warning("Gateway already running")
            return

        logger.info("Starting vMCP Gateway")

        try:
            # Initialize if not done
            if not self.health_checker:
                await self.initialize()

            self._running = True

            # Install signal handlers
            self._install_signal_handlers()

            # Start all transports
            transport_tasks = []
            for name, transport in self.transports.items():
                logger.info("Starting transport", transport=name)
                task = asyncio.create_task(transport.start(), name=f"transport-{name}")
                transport_tasks.append(task)
                self._tasks.append(task)

            # Wait for transports to start
            if transport_tasks:
                await asyncio.gather(*transport_tasks)

            # Start background tasks
            self._start_background_tasks()

            logger.info("vMCP Gateway started successfully")

        except Exception as e:
            logger.error("Failed to start gateway", error=str(e), exc_info=True)
            self._running = False
            raise VMCPError(VMCPErrorCode.INTERNAL_ERROR, f"Start failed: {e}") from e

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()

            def signal_handler() -> None:
                logger.info("Received shutdown signal")
                asyncio.create_task(self.stop())

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health monitoring task
        if self.health_checker:
            task = asyncio.create_task(
                self._health_monitor_loop(), name="health-monitor"
            )
            self._tasks.append(task)

        # Metrics collection task
        task = asyncio.create_task(
            self._metrics_collection_loop(), name="metrics-collector"
        )
        self._tasks.append(task)

        # Cleanup task
        task = asyncio.create_task(self._cleanup_loop(), name="cleanup")
        self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the gateway server gracefully."""
        if not self._running:
            logger.info("Gateway already stopped")
            return

        logger.info("Stopping vMCP Gateway")
        self._running = False
        self._shutdown_event.set()

        try:
            # Stop all transports
            stop_tasks = []
            for name, transport in self.transports.items():
                logger.info("Stopping transport", transport=name)
                task = asyncio.create_task(transport.stop())
                stop_tasks.append(task)

            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            # Cancel all background tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            # Shutdown registry
            await self.registry.shutdown()

            logger.info("vMCP Gateway stopped successfully")

        except Exception as e:
            logger.error("Error during gateway shutdown", error=str(e), exc_info=True)

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle incoming request from transport.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response or None for notifications
        """
        request_id = request.get("id")
        method = request.get("method", "unknown")

        # Check if it's a notification (no response needed)
        if request_id is None:
            logger.debug("Received notification", method=method)
            await self._handle_notification(request)
            return None

        # Rate limiting
        if not await self._check_rate_limit():
            return self.protocol_handler.create_error_response(
                request_id,
                429,  # Too Many Requests
                "Rate limit exceeded",
            )

        # Request validation
        if self.config.enable_request_validation:
            try:
                self.protocol_handler.validate_message(request)
            except Exception as e:
                logger.warning("Invalid request", error=str(e))
                return self.protocol_handler.create_error_response(
                    request_id,
                    -32600,  # Invalid Request
                    f"Invalid request: {e}",
                )

        # Handle vMCP extension methods
        if method.startswith("vmcp/"):
            return await self._handle_vmcp_method(request)

        # Use semaphore to limit concurrent requests
        async with self._request_semaphore:
            self._active_requests += 1

            try:
                # Record request start
                await self.metrics_collector.record_request(
                    method,
                    success=True,  # Will be updated on completion
                    cached=False,  # Will be updated if cached
                )

                # Route request
                response = await self.router.route(request)

                # Record successful completion
                logger.debug("Request completed", method=method, request_id=request_id)

                return response

            except Exception as e:
                logger.error(
                    "Request handling failed",
                    method=method,
                    request_id=request_id,
                    error=str(e),
                    exc_info=True,
                )

                # Record failed request
                await self.metrics_collector.record_request(method, success=False)

                return self.protocol_handler.create_error_response(
                    request_id,
                    -32603,  # Internal Error
                    "Internal server error",
                )

            finally:
                self._active_requests -= 1

    async def _handle_notification(self, request: dict[str, Any]) -> None:
        """Handle notification message (no response)."""
        method = request.get("method", "unknown")
        logger.debug("Handling notification", method=method)

        # Process notification asynchronously
        if method == "initialized":
            # Server initialization complete
            pass
        elif method.startswith("notifications/"):
            # Handle MCP notifications
            pass

    async def _handle_vmcp_method(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle vMCP extension methods."""
        method = request.get("method", "")
        request_id = request.get("id")

        try:
            if method == "vmcp/servers/list":
                servers = [
                    {
                        "id": state.config.id,
                        "name": state.config.name,
                        "transport": state.config.transport,
                        "enabled": state.config.enabled,
                        "healthy": state.is_healthy,
                        "capabilities": list(state.config.capabilities.keys()),
                    }
                    for state in self.registry.get_all_servers()
                ]

                return self.protocol_handler.create_response(
                    request_id, result={"servers": servers}
                )

            elif method == "vmcp/servers/info":
                params = request.get("params", {})
                server_id = params.get("id")

                if not server_id:
                    return self.protocol_handler.create_error_response(
                        request_id, -32602, "Missing server id parameter"
                    )

                server_info = self.registry.get_server_details(server_id)
                if not server_info:
                    return self.protocol_handler.create_error_response(
                        request_id, -32602, f"Server not found: {server_id}"
                    )

                return self.protocol_handler.create_response(
                    request_id, result=server_info
                )

            elif method == "vmcp/servers/health":
                health_data = (
                    await self.health_checker.check_health()
                    if self.health_checker
                    else {}
                )
                return self.protocol_handler.create_response(
                    request_id, result=health_data
                )

            elif method == "vmcp/cache/clear":
                self.router.clear_route_cache()
                return self.protocol_handler.create_response(
                    request_id, result={"status": "cache cleared"}
                )

            elif method == "vmcp/metrics":
                metrics = await self.metrics_collector.get_metrics()
                routing_stats = self.router.get_routing_stats()
                registry_stats = self.registry.get_registry_stats()

                combined_metrics = {
                    **metrics,
                    "routing": routing_stats,
                    "registry": registry_stats,
                    "active_requests": self._active_requests,
                }

                return self.protocol_handler.create_response(
                    request_id, result=combined_metrics
                )

            else:
                return self.protocol_handler.create_error_response(
                    request_id, -32601, f"Unknown vMCP method: {method}"
                )

        except Exception as e:
            logger.error(
                "vMCP method error", method=method, error=str(e), exc_info=True
            )
            return self.protocol_handler.create_error_response(
                request_id, -32603, f"vMCP method error: {e}"
            )

    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        # Simple implementation - could be enhanced with proper rate limiting
        return self._active_requests < self.config.max_concurrent_requests

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        logger.debug("Starting health monitor loop")

        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self.health_checker:
                    health_data = await self.health_checker.check_health()

                    # Log health issues
                    if health_data["status"] != "healthy":
                        logger.warning(
                            "System health degraded", health=health_data["status"]
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e), exc_info=True)

        logger.debug("Health monitor loop ended")

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        logger.debug("Starting metrics collection loop")

        while self._running:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute

                # Update gateway metrics
                await self.metrics_collector.update_gauge(
                    "active_requests", self._active_requests
                )
                await self.metrics_collector.update_gauge(
                    "transport_count", len(self.transports)
                )

                # Update server metrics
                healthy_servers = len(self.registry.get_healthy_servers())
                total_servers = len(self.registry.get_all_servers())

                await self.metrics_collector.update_gauge(
                    "servers_healthy", healthy_servers
                )
                await self.metrics_collector.update_gauge(
                    "servers_total", total_servers
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e), exc_info=True)

        logger.debug("Metrics collection loop ended")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.debug("Starting cleanup loop")

        while self._running:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                # Cleanup route cache
                self.router.clear_route_cache()

                logger.debug("Performed periodic cleanup")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error", error=str(e), exc_info=True)

        logger.debug("Cleanup loop ended")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def is_running(self) -> bool:
        """Check if gateway is running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get gateway status information."""
        return {
            "running": self._running,
            "active_requests": self._active_requests,
            "transports": {
                name: transport.get_stats()
                for name, transport in self.transports.items()
            },
            "registry": self.registry.get_registry_stats(),
            "routing": self.router.get_routing_stats(),
            "config": {
                "registry_path": self.config.registry_path,
                "max_connections": self.config.max_connections,
                "request_timeout": self.config.request_timeout,
                "cache_enabled": self.config.cache_enabled,
            },
        }
