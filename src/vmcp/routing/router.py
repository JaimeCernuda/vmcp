"""
Production-grade request router for vMCP.

This module implements the main router that coordinates all routing strategies
including path-based, content-based, and capability-based routing with
load balancing, caching, and circuit breaker integration.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..errors import ServerUnavailableError, VMCPError, VMCPErrorCode
from .algorithms import (
    CapabilityRouter,
    ContentBasedRouter,
    HybridRouter,
    PathBasedRouter,
)
from .circuit_breaker import CircuitBreakerRegistry
from .loadbalancer import LoadBalancerFactory, RoundRobinBalancer

logger = logging.getLogger(__name__)


@dataclass
class RoutingContext:
    """Context for routing decisions."""

    request: dict[str, Any]
    session_id: str | None = None
    persona: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def get_cache_key(self) -> str:
        """Generate cache key for this routing context."""
        method = self.request.get("method", "unknown")
        persona = self.persona or "default"
        return f"{method}:{persona}"


@dataclass
class RoutingResult:
    """Result of routing operation."""

    server_id: str | None
    source: str  # "path", "content", "capability", "cache", "fallback"
    duration: float
    cached: bool = False
    error: str | None = None


class Router:
    """Production-grade request router for vMCP."""

    def __init__(self, registry: Any) -> None:
        """
        Initialize router.

        Args:
            registry: Server registry instance
        """
        self.registry = registry

        # Routing strategies
        self.path_router = PathBasedRouter()
        self.content_router = ContentBasedRouter()
        self.hybrid_router = HybridRouter()
        self.capability_router = CapabilityRouter(registry)

        # Load balancing
        self.load_balancer = LoadBalancerFactory.create("round_robin")

        # Circuit breakers
        self.circuit_breakers = CircuitBreakerRegistry()

        # Route caching
        self._route_cache: dict[str, str] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_max_size = 1000

        # Statistics
        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._routing_stats = {
            "path": 0,
            "content": 0,
            "capability": 0,
            "cache": 0,
            "fallback": 0,
            "errors": 0,
        }

    async def route(
        self, request: dict[str, Any], context: RoutingContext | None = None
    ) -> dict[str, Any]:
        """
        Route request to appropriate MCP server.

        Args:
            request: JSON-RPC request
            context: Routing context

        Returns:
            JSON-RPC response

        Raises:
            NoServerFoundError: If no server can handle request
            RoutingError: If routing fails
        """
        if context is None:
            context = RoutingContext(request=request)

        self._request_count += 1

        try:
            # Find target server
            routing_result = await self._find_server(context)

            if not routing_result.server_id:
                raise VMCPError(
                    500,  # type: ignore[arg-type]
                    f"No server found for method: {request.get('method', 'unknown')}",
                )

            server_id = routing_result.server_id

            # Get circuit breaker for server
            breaker = await self.circuit_breakers.get_breaker(server_id)

            # Execute request with circuit breaker protection
            response = await breaker.call(
                self._forward_request, server_id, request, context
            )

            # Update routing cache on success
            if not routing_result.cached:
                self._update_route_cache(context.get_cache_key(), server_id)

            return response  # type: ignore[return-value]

        except Exception as e:
            self._routing_stats["errors"] += 1
            logger.error(
                f"Routing error for {request.get('method')}: {e}", exc_info=True
            )

            # Return JSON-RPC error response
            return self._create_error_response(
                request.get("id"), VMCPErrorCode.ROUTING_FAILED, f"Routing failed: {e}"
            )

    async def _find_server(self, context: RoutingContext) -> RoutingResult:
        """Find appropriate server for request."""
        start_time = time.time()
        method = context.request.get("method", "")
        cache_key = context.get_cache_key()

        # Check route cache first
        if cache_key in self._route_cache:
            cached_server = self._route_cache[cache_key]
            cache_time = self._cache_timestamps.get(cache_key, 0)

            # Validate cache entry
            if time.time() - cache_time < self._cache_ttl and self._is_server_available(
                cached_server
            ):
                self._cache_hits += 1
                self._routing_stats["cache"] += 1

                return RoutingResult(
                    server_id=cached_server,
                    source="cache",
                    duration=time.time() - start_time,
                    cached=True,
                )
            else:
                # Remove stale cache entry
                self._remove_from_cache(cache_key)

        self._cache_misses += 1

        # Try path-based routing
        server_id = self.path_router.route(context.request)
        if server_id and self._is_server_available(server_id):
            self._routing_stats["path"] += 1
            return RoutingResult(
                server_id=server_id, source="path", duration=time.time() - start_time
            )

        # Try content-based routing
        server_id = self.content_router.route(context.request)
        if server_id and self._is_server_available(server_id):
            self._routing_stats["content"] += 1
            return RoutingResult(
                server_id=server_id, source="content", duration=time.time() - start_time
            )

        # Try capability-based routing
        required_capability = self.capability_router.get_method_capability(method)
        if required_capability:
            server_id = self.capability_router.route_by_capability(
                context.request, required_capability
            )
            if server_id and self._is_server_available(server_id):
                self._routing_stats["capability"] += 1
                return RoutingResult(
                    server_id=server_id,
                    source="capability",
                    duration=time.time() - start_time,
                )

        # Fallback: find any server that supports the method
        server_id = await self._find_fallback_server(method)
        if server_id:
            self._routing_stats["fallback"] += 1
            return RoutingResult(
                server_id=server_id,
                source="fallback",
                duration=time.time() - start_time,
            )

        # No server found
        return RoutingResult(
            server_id=None,
            source="none",
            duration=time.time() - start_time,
            error="No capable server found",
        )

    async def _find_fallback_server(self, method: str) -> str | None:
        """Find any server that can handle the method."""
        capable_servers = []

        for server_state in self.registry.get_healthy_servers():
            if self._server_supports_method(server_state, method):
                capable_servers.append(server_state)

        if not capable_servers:
            return None

        # Use load balancer to select from capable servers
        try:
            selected = await self.load_balancer.select(capable_servers)
            return str(selected.config.id)
        except Exception as e:
            logger.error(f"Load balancer error: {e}")
            # Fallback to first server
            return str(capable_servers[0].config.id)

    def _server_supports_method(self, server_state: Any, method: str) -> bool:
        """Check if server supports the given method."""
        # Standard MCP methods
        if method in ["initialize", "initialized", "ping"]:
            return True

        # Check capability-based methods
        if method.startswith("tools/"):
            return "tools" in server_state.config.capabilities
        elif method.startswith("resources/"):
            return "resources" in server_state.config.capabilities
        elif method.startswith("prompts/"):
            return "prompts" in server_state.config.capabilities
        elif method.startswith("completion/"):
            return "completion" in server_state.config.capabilities

        # Unknown method - assume supported for now
        return True

    def _is_server_available(self, server_id: str) -> bool:
        """Check if server is available for routing."""
        server_state = self.registry.get_server(server_id)
        if not server_state:
            return False

        return bool(server_state.config.enabled and server_state.is_healthy)

    async def _forward_request(
        self, server_id: str, request: dict[str, Any], context: RoutingContext
    ) -> dict[str, Any]:
        """Forward request to MCP server."""
        logger.debug(f"Forwarding request to server {server_id}")

        start_time = time.time()

        try:
            # Get connection pool for server
            pool = await self.registry.get_connection_pool(server_id)
            if not pool:
                raise ServerUnavailableError(
                    server_id, reason="No connection pool available"
                )

            # Execute request using connection pool
            async with pool.acquire(timeout=30.0) as connection:
                response = await connection.send_request(request, timeout=30.0)

            # Record successful request
            server_state = self.registry.get_server(server_id)
            if server_state:
                server_state.record_request(success=True)

            duration = time.time() - start_time
            logger.debug(f"Request completed in {duration:.3f}s")

            return response  # type: ignore[no-any-return]

        except Exception as e:
            # Record failed request
            server_state = self.registry.get_server(server_id)
            if server_state:
                server_state.record_request(success=False)

            logger.error(f"Server {server_id} error: {e}")
            raise

    def _update_route_cache(self, cache_key: str, server_id: str) -> None:
        """Update route cache with new mapping."""
        # Enforce cache size limit
        if len(self._route_cache) >= self._cache_max_size:
            self._evict_oldest_cache_entry()

        self._route_cache[cache_key] = server_id
        self._cache_timestamps[cache_key] = time.time()

    def _remove_from_cache(self, cache_key: str) -> None:
        """Remove entry from route cache."""
        self._route_cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)

    def _evict_oldest_cache_entry(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache_timestamps:
            return

        oldest_key = min(
            self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k]
        )
        self._remove_from_cache(oldest_key)

    def _create_error_response(
        self, request_id: Any, code: int, message: str
    ) -> dict[str, Any]:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def add_path_rule(self, pattern: str, server_id: str, priority: int = 0) -> None:
        """Add path-based routing rule."""
        from vmcp.routing.algorithms import RouteRule
        rule = RouteRule(pattern=pattern, server_id=server_id, priority=priority)
        self.path_router.add_rule(rule)
        self.clear_route_cache()

    def add_content_rule(
        self,
        server_id: str,
        tool_name: str | None = None,
        resource_pattern: str | None = None,
        priority: int = 0,
    ) -> None:
        """Add content-based routing rule."""
        if tool_name:
            self.content_router.add_tool_rule(tool_name, server_id, priority)
        elif resource_pattern:
            self.content_router.add_resource_rule(resource_pattern, server_id, priority)

        self.clear_route_cache()

    def remove_rules_for_server(self, server_id: str) -> None:
        """Remove all routing rules for a server."""
        # Remove path rules
        self.path_router.rules = [
            rule for rule in self.path_router.rules if rule.server_id != server_id
        ]

        # Remove content rules
        self.content_router.rules = [
            rule for rule in self.content_router.rules if rule.server_id != server_id
        ]

        self.clear_route_cache()
        logger.info(f"Removed all routing rules for server {server_id}")

    def clear_route_cache(self) -> None:
        """Clear route cache."""
        self._route_cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Cleared route cache")

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        total_requests = self._request_count
        cache_total = self._cache_hits + self._cache_misses

        return {
            "total_requests": total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, cache_total),
            "routing_breakdown": dict(self._routing_stats),
            "cache_size": len(self._route_cache),
            "cache_max_size": self._cache_max_size,
            "cache_ttl": self._cache_ttl,
        }

    def get_routing_rules(self) -> dict[str, list[dict[str, Any]]]:
        """Get all routing rules."""
        return {
            "path_based": self.path_router.get_rules(),
            "content_based": self.content_router.get_rules(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform router health check."""
        healthy_servers = len(self.registry.get_healthy_servers())
        total_servers = len(self.registry.get_enabled_servers())

        # Get circuit breaker health
        breaker_health = await self.circuit_breakers.get_health_summary()

        # Determine overall health
        if total_servers == 0:
            health_status = "no_servers"
        elif healthy_servers == 0:
            health_status = "unhealthy"
        elif healthy_servers < total_servers // 2:
            health_status = "degraded"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "healthy_servers": healthy_servers,
            "total_servers": total_servers,
            "circuit_breakers": breaker_health,
            "routing_stats": self.get_routing_stats(),
            "timestamp": datetime.now().isoformat(),
        }

    async def validate_routing_config(self) -> list[dict[str, Any]]:
        """Validate routing configuration and return issues."""
        issues = []

        # Check for rules pointing to non-existent servers
        all_server_ids = {state.config.id for state in self.registry.get_all_servers()}

        for rule in self.path_router.get_rules():
            if rule["server_id"] not in all_server_ids:
                issues.append(
                    {
                        "type": "missing_server",
                        "rule_type": "path",
                        "rule": rule,
                        "message": f"Path rule references non-existent server: {rule['server_id']}",
                    }
                )

        for rule in self.content_router.get_rules():
            if rule["server_id"] not in all_server_ids:
                issues.append(
                    {
                        "type": "missing_server",
                        "rule_type": "content",
                        "rule": rule,
                        "message": f"Content rule references non-existent server: {rule['server_id']}",
                    }
                )

        # Check for conflicting rules
        path_patterns: dict[str, dict[str, Any]] = {}
        for rule in self.path_router.get_rules():
            pattern = rule["pattern"]
            if pattern in path_patterns:
                issues.append(
                    {
                        "type": "conflicting_rules",
                        "rule_type": "path",
                        "message": f"Duplicate path pattern: {pattern}",
                        "rules": [str(path_patterns[pattern]), str(rule)],
                    }
                )
            else:
                path_patterns[pattern] = rule

        return issues
