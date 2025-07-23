"""
Health checking system for vMCP components.

This module provides comprehensive health checking for all vMCP components
including the gateway, registry, servers, and other subsystems.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str | None = None
    details: dict[str, Any] | None = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {},
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self, components: dict[str, Any]) -> None:
        """
        Initialize health checker.
        
        Args:
            components: Dictionary of components to check
        """
        self.components = components
        self.checks: list[HealthCheck] = []
        self._last_check_time: float | None = None

    async def check_health(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        start_time = time.time()
        checks = []

        # Check gateway
        if "gateway" in self.components:
            checks.append(await self._check_gateway())

        # Check registry
        if "registry" in self.components:
            checks.append(await self._check_registry())

        # Check servers
        if "registry" in self.components:
            server_checks = await self._check_servers()
            checks.extend(server_checks)

        # Check router
        if "router" in self.components:
            checks.append(await self._check_router())

        # Check metrics collector
        if "metrics" in self.components:
            checks.append(await self._check_metrics())

        # Determine overall status
        overall_status = self._determine_overall_status(checks)

        self._last_check_time = time.time()
        self.checks = checks

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "checks": [check.to_dict() for check in checks],
            "summary": self._get_status_summary(checks),
        }

    async def _check_gateway(self) -> HealthCheck:
        """Check gateway health."""
        start = time.time()

        try:
            gateway = self.components.get("gateway")
            if not gateway:
                return HealthCheck(
                    name="gateway",
                    status=HealthStatus.UNHEALTHY,
                    message="Gateway component not found",
                    duration=time.time() - start
                )

            if hasattr(gateway, "is_running") and gateway.is_running():
                # Check active requests and transports
                details = {}
                if hasattr(gateway, "_active_requests"):
                    details["active_requests"] = gateway._active_requests
                if hasattr(gateway, "transports"):
                    details["active_transports"] = len(gateway.transports)

                return HealthCheck(
                    name="gateway",
                    status=HealthStatus.HEALTHY,
                    message="Gateway is running",
                    details=details,
                    duration=time.time() - start
                )
            else:
                return HealthCheck(
                    name="gateway",
                    status=HealthStatus.UNHEALTHY,
                    message="Gateway is not running",
                    duration=time.time() - start
                )

        except Exception as e:
            return HealthCheck(
                name="gateway",
                status=HealthStatus.UNHEALTHY,
                message=f"Gateway check failed: {e}",
                details={"error": str(e)},
                duration=time.time() - start
            )

    async def _check_registry(self) -> HealthCheck:
        """Check registry health."""
        start = time.time()

        try:
            registry = self.components.get("registry")
            if not registry:
                return HealthCheck(
                    name="registry",
                    status=HealthStatus.UNHEALTHY,
                    message="Registry component not found",
                    duration=time.time() - start
                )

            # Get registry statistics
            stats = registry.get_registry_stats()

            # Determine health based on server status
            total_servers = stats.get("total_servers", 0)
            healthy_servers = stats.get("healthy_servers", 0)

            if total_servers == 0:
                status = HealthStatus.DEGRADED
                message = "No servers registered"
            elif healthy_servers == 0:
                status = HealthStatus.UNHEALTHY
                message = "No healthy servers available"
            elif healthy_servers < total_servers // 2:
                status = HealthStatus.DEGRADED
                message = f"Only {healthy_servers}/{total_servers} servers healthy"
            else:
                status = HealthStatus.HEALTHY
                message = f"{healthy_servers}/{total_servers} servers healthy"

            return HealthCheck(
                name="registry",
                status=status,
                message=message,
                details=stats,
                duration=time.time() - start
            )

        except Exception as e:
            return HealthCheck(
                name="registry",
                status=HealthStatus.UNHEALTHY,
                message=f"Registry check failed: {e}",
                details={"error": str(e)},
                duration=time.time() - start
            )

    async def _check_servers(self) -> list[HealthCheck]:
        """Check all registered servers."""
        checks = []
        registry = self.components.get("registry")

        if not registry:
            return checks

        try:
            servers = registry.get_all_servers()

            for server_state in servers:
                start = time.time()

                try:
                    server_id = server_state.config.id

                    if not server_state.config.enabled:
                        # Skip disabled servers
                        continue

                    # Check server health status from registry
                    if server_state.is_healthy:
                        status = HealthStatus.HEALTHY
                        message = "Server is healthy"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = server_state.last_error or "Server is unhealthy"

                    details = {
                        "server_id": server_id,
                        "transport": server_state.config.transport,
                        "total_requests": server_state.total_requests,
                        "failed_requests": server_state.failed_requests,
                        "error_rate": server_state.get_error_rate(),
                        "uptime": server_state.get_uptime(),
                        "capabilities": list(server_state.config.capabilities.keys()),
                    }

                    checks.append(HealthCheck(
                        name=f"server:{server_id}",
                        status=status,
                        message=message,
                        details=details,
                        duration=time.time() - start
                    ))

                except Exception as e:
                    checks.append(HealthCheck(
                        name=f"server:{getattr(server_state, 'config', {}).get('id', 'unknown')}",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Server check failed: {e}",
                        details={"error": str(e)},
                        duration=time.time() - start
                    ))

        except Exception as e:
            logger.error(f"Failed to check servers: {e}")

        return checks

    async def _check_router(self) -> HealthCheck:
        """Check router health."""
        start = time.time()

        try:
            router = self.components.get("router")
            if not router:
                return HealthCheck(
                    name="router",
                    status=HealthStatus.UNHEALTHY,
                    message="Router component not found",
                    duration=time.time() - start
                )

            # Get routing statistics
            stats = router.get_routing_stats()

            # Check for routing errors
            total_requests = stats.get("total_requests", 0)
            error_count = stats.get("routing_breakdown", {}).get("errors", 0)

            if total_requests == 0:
                status = HealthStatus.HEALTHY
                message = "Router ready (no requests processed)"
            else:
                error_rate = (error_count / total_requests) * 100

                if error_rate > 50:
                    status = HealthStatus.UNHEALTHY
                    message = f"High error rate: {error_rate:.1f}%"
                elif error_rate > 10:
                    status = HealthStatus.DEGRADED
                    message = f"Elevated error rate: {error_rate:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Router healthy (error rate: {error_rate:.1f}%)"

            return HealthCheck(
                name="router",
                status=status,
                message=message,
                details=stats,
                duration=time.time() - start
            )

        except Exception as e:
            return HealthCheck(
                name="router",
                status=HealthStatus.UNHEALTHY,
                message=f"Router check failed: {e}",
                details={"error": str(e)},
                duration=time.time() - start
            )

    async def _check_metrics(self) -> HealthCheck:
        """Check metrics collector health."""
        start = time.time()

        try:
            metrics = self.components.get("metrics")
            if not metrics:
                return HealthCheck(
                    name="metrics",
                    status=HealthStatus.DEGRADED,
                    message="Metrics collector not found",
                    duration=time.time() - start
                )

            # Try to get metrics
            metrics_data = await metrics.get_metrics()

            return HealthCheck(
                name="metrics",
                status=HealthStatus.HEALTHY,
                message="Metrics collector operational",
                details={
                    "entries": len(metrics_data.get("counters", {})) + len(metrics_data.get("gauges", {}))
                },
                duration=time.time() - start
            )

        except Exception as e:
            return HealthCheck(
                name="metrics",
                status=HealthStatus.DEGRADED,
                message=f"Metrics check failed: {e}",
                details={"error": str(e)},
                duration=time.time() - start
            )

    def _determine_overall_status(self, checks: list[HealthCheck]) -> HealthStatus:
        """Determine overall system health."""
        if not checks:
            return HealthStatus.UNHEALTHY

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
        }

        for check in checks:
            status_counts[check.status] += 1

        total_checks = len(checks)
        unhealthy_count = status_counts[HealthStatus.UNHEALTHY]
        degraded_count = status_counts[HealthStatus.DEGRADED]

        # Determine overall status
        if unhealthy_count > total_checks // 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > total_checks // 2:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _get_status_summary(self, checks: list[HealthCheck]) -> dict[str, int]:
        """Get summary of health check statuses."""
        summary = {
            "total": len(checks),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
        }

        for check in checks:
            if check.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif check.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            else:
                summary["unhealthy"] += 1

        return summary

    async def get_liveness(self) -> dict[str, Any]:
        """Simple liveness check."""
        return {
            "status": "alive",
            "timestamp": time.time(),
            "uptime": time.time() - (self._last_check_time or time.time()),
        }

    async def get_readiness(self) -> dict[str, Any]:
        """Readiness check for load balancers."""
        # Quick check of critical components
        gateway = self.components.get("gateway")
        registry = self.components.get("registry")

        ready = True
        reasons = []

        if gateway and hasattr(gateway, "is_running"):
            if not gateway.is_running():
                ready = False
                reasons.append("Gateway not running")
        else:
            ready = False
            reasons.append("Gateway not available")

        if registry:
            try:
                stats = registry.get_registry_stats()
                if stats.get("healthy_servers", 0) == 0:
                    ready = False
                    reasons.append("No healthy servers")
            except Exception:
                ready = False
                reasons.append("Registry not accessible")
        else:
            ready = False
            reasons.append("Registry not available")

        return {
            "status": "ready" if ready else "not_ready",
            "timestamp": time.time(),
            "reasons": reasons if not ready else [],
        }

    def get_last_check(self) -> dict[str, Any] | None:
        """Get results from last health check."""
        if not self.checks:
            return None

        return {
            "timestamp": self._last_check_time,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self._get_status_summary(self.checks),
        }
