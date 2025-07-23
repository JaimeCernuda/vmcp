"""
Load balancing strategies for vMCP routing.

This module implements various load balancing algorithms for distributing
requests across multiple MCP servers efficiently.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServerMetrics:
    """Metrics for load balancing decisions."""

    server_id: str
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_error_time: float | None = None
    weight: int = 1
    last_selected: float = 0.0

    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    @property
    def error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100

    def record_request(self, response_time: float, success: bool = True) -> None:
        """Record request metrics."""
        self.total_requests += 1
        self.response_times.append(response_time)

        if not success:
            self.error_count += 1
            self.last_error_time = time.time()

    def is_healthy(self, error_threshold: float = 10.0) -> bool:
        """Check if server is healthy based on error rate."""
        return self.error_rate < error_threshold


class LoadBalancer(ABC):
    """Abstract base class for load balancers."""

    @abstractmethod
    async def select(self, servers: list[Any]) -> Any:
        """
        Select a server from the list.

        Args:
            servers: List of available servers

        Returns:
            Selected server

        Raises:
            ValueError: If no servers available
        """
        pass

    @abstractmethod
    def record_request(
        self, server_id: str, response_time: float, success: bool = True
    ) -> None:
        """
        Record request metrics for load balancing decisions.

        Args:
            server_id: Server identifier
            response_time: Request response time in seconds
            success: Whether request was successful
        """
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""

    def __init__(self) -> None:
        """Initialize round-robin balancer."""
        self._index = 0
        self._lock = asyncio.Lock()

    async def select(self, servers: list[Any]) -> Any:
        """Select next server in round-robin fashion."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            server = servers[self._index % len(servers)]
            self._index += 1
            return server


class RandomBalancer(LoadBalancer):
    """Random load balancer."""

    async def select(self, servers: list[Any]) -> Any:
        """Select random server."""
        if not servers:
            raise ValueError("No servers available")

        return random.choice(servers)


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer."""

    def __init__(self) -> None:
        """Initialize least connections balancer."""
        self.metrics: dict[str, ServerMetrics] = {}
        self._lock = asyncio.Lock()

    async def select(self, servers: list[Any]) -> Any:
        """Select server with least active connections."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            # Initialize metrics for new servers
            for server in servers:
                server_id = getattr(server, "id", str(server))
                if server_id not in self.metrics:
                    self.metrics[server_id] = ServerMetrics(server_id=server_id)

            # Find server with least connections
            min_connections = float("inf")
            selected_server = None

            for server in servers:
                server_id = getattr(server, "id", str(server))
                metrics = self.metrics[server_id]

                if metrics.active_connections < min_connections:
                    min_connections = metrics.active_connections
                    selected_server = server

            # Increment connection count
            if selected_server:
                server_id = getattr(selected_server, "id", str(selected_server))
                self.metrics[server_id].active_connections += 1

            return selected_server

    def release_connection(self, server_id: str) -> None:
        """Release connection from server."""
        if server_id in self.metrics:
            self.metrics[server_id].active_connections = max(
                0, self.metrics[server_id].active_connections - 1
            )

    def record_request(
        self, server_id: str, response_time: float, success: bool = True
    ) -> None:
        """Record request metrics."""
        if server_id in self.metrics:
            self.metrics[server_id].record_request(response_time, success)


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round-robin load balancer."""

    def __init__(self) -> None:
        """Initialize weighted round-robin balancer."""
        self.weights: dict[str, int] = {}
        self.current_weights: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def select(self, servers: list[Any]) -> Any:
        """Select server based on weighted round-robin."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            # Initialize weights for new servers
            for server in servers:
                server_id = getattr(server, "id", str(server))
                if server_id not in self.weights:
                    self.weights[server_id] = 1
                    self.current_weights[server_id] = 0

            # Find server with highest current weight
            selected_server = None
            max_weight = -1
            total_weight = 0

            for server in servers:
                server_id = getattr(server, "id", str(server))
                weight = self.weights[server_id]
                current_weight = self.current_weights[server_id]

                # Increase current weight
                self.current_weights[server_id] = current_weight + weight
                total_weight += weight

                if self.current_weights[server_id] > max_weight:
                    max_weight = self.current_weights[server_id]
                    selected_server = server

            # Decrease selected server's current weight
            if selected_server:
                server_id = getattr(selected_server, "id", str(selected_server))
                self.current_weights[server_id] -= total_weight

            return selected_server

    def set_weight(self, server_id: str, weight: int) -> None:
        """Set server weight."""
        self.weights[server_id] = max(1, weight)
        if server_id not in self.current_weights:
            self.current_weights[server_id] = 0


class WeightedRandomBalancer(LoadBalancer):
    """Weighted random load balancer."""

    def __init__(self) -> None:
        """Initialize weighted random balancer."""
        self.weights: dict[str, int] = {}

    async def select(self, servers: list[Any]) -> Any:
        """Select server based on weighted random selection."""
        if not servers:
            raise ValueError("No servers available")

        # Build weighted list
        weighted_servers = []
        for server in servers:
            server_id = getattr(server, "id", str(server))
            weight = self.weights.get(server_id, 1)
            weighted_servers.extend([server] * weight)

        if not weighted_servers:
            return servers[0]

        return random.choice(weighted_servers)

    def set_weight(self, server_id: str, weight: int) -> None:
        """Set server weight."""
        self.weights[server_id] = max(1, weight)


class AdaptiveBalancer(LoadBalancer):
    """Adaptive load balancer based on response times and health."""

    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize adaptive balancer.

        Args:
            window_size: Size of response time window for averaging
        """
        self.window_size = window_size
        self.metrics: dict[str, ServerMetrics] = {}
        self._lock = asyncio.Lock()

    async def select(self, servers: list[Any]) -> Any:
        """Select server with best recent performance."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            # Initialize metrics for new servers
            for server in servers:
                server_id = getattr(server, "id", str(server))
                if server_id not in self.metrics:
                    self.metrics[server_id] = ServerMetrics(server_id=server_id)

            # Calculate scores for each server
            scored_servers = []
            for server in servers:
                server_id = getattr(server, "id", str(server))
                metrics = self.metrics[server_id]

                # Skip unhealthy servers
                if not metrics.is_healthy():
                    continue

                score = self._calculate_server_score(metrics)
                scored_servers.append((server, score))

            if not scored_servers:
                # All servers unhealthy, select randomly
                return random.choice(servers)

            # Select using weighted random based on scores
            return self._weighted_random_select(scored_servers)

    def _calculate_server_score(self, metrics: ServerMetrics) -> float:
        """Calculate server score for selection."""
        # Base score
        score = 1.0

        # Penalize high response times
        if metrics.response_times:
            avg_response_time = metrics.average_response_time
            # Lower response time = higher score
            score *= max(0.1, 1.0 - (avg_response_time / 10.0))

        # Penalize high error rates
        error_rate = metrics.error_rate
        score *= max(0.1, 1.0 - (error_rate / 100.0))

        # Penalize high active connections
        score *= max(0.1, 1.0 - (metrics.active_connections / 100.0))

        # Apply weight multiplier
        score *= metrics.weight

        # Slight randomization to prevent thundering herd
        score *= random.uniform(0.9, 1.1)

        return max(0.01, score)

    def _weighted_random_select(self, scored_servers: list[tuple]) -> Any:
        """Select server using weighted random based on scores."""
        total_score = sum(score for _, score in scored_servers)
        rand_value = random.uniform(0, total_score)

        cumulative = 0
        for server, score in scored_servers:
            cumulative += score
            if rand_value <= cumulative:
                server_id = getattr(server, "id", str(server))
                self.metrics[server_id].last_selected = time.time()
                return server

        # Fallback to last server
        return scored_servers[-1][0]

    def record_request(
        self, server_id: str, response_time: float, success: bool = True
    ) -> None:
        """Record request metrics for adaptive balancing."""
        if server_id not in self.metrics:
            self.metrics[server_id] = ServerMetrics(server_id=server_id)

        self.metrics[server_id].record_request(response_time, success)

    def set_server_weight(self, server_id: str, weight: int) -> None:
        """Set server weight."""
        if server_id not in self.metrics:
            self.metrics[server_id] = ServerMetrics(server_id=server_id)
        self.metrics[server_id].weight = max(1, weight)

    def get_server_metrics(self, server_id: str) -> dict[str, Any] | None:
        """Get metrics for specific server."""
        if server_id not in self.metrics:
            return None

        metrics = self.metrics[server_id]
        return {
            "server_id": server_id,
            "active_connections": metrics.active_connections,
            "total_requests": metrics.total_requests,
            "error_count": metrics.error_count,
            "error_rate": metrics.error_rate,
            "average_response_time": metrics.average_response_time,
            "weight": metrics.weight,
            "last_selected": metrics.last_selected,
            "is_healthy": metrics.is_healthy(),
        }

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all servers."""
        return {
            server_id: self.get_server_metrics(server_id) for server_id in self.metrics
        }


class ConsistentHashBalancer(LoadBalancer):
    """Consistent hash load balancer for sticky sessions."""

    def __init__(self, virtual_nodes: int = 150) -> None:
        """
        Initialize consistent hash balancer.

        Args:
            virtual_nodes: Number of virtual nodes per server
        """
        self.virtual_nodes = virtual_nodes
        self.ring: dict[int, str] = {}
        self.servers: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def select(self, servers: list[Any]) -> Any:
        """Select server using consistent hashing."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            # Update ring if servers changed
            current_server_ids = {getattr(s, "id", str(s)) for s in servers}
            if set(self.servers.keys()) != current_server_ids:
                await self._rebuild_ring(servers)

            # For now, select based on timestamp as hash key
            # In practice, you'd use session ID or other consistent key
            hash_key = int(time.time() * 1000) % (2**32)

            return self._find_server_for_hash(hash_key)

    async def select_by_key(self, servers: list[Any], key: str) -> Any:
        """Select server for specific key using consistent hashing."""
        if not servers:
            raise ValueError("No servers available")

        async with self._lock:
            # Update ring if servers changed
            current_server_ids = {getattr(s, "id", str(s)) for s in servers}
            if set(self.servers.keys()) != current_server_ids:
                await self._rebuild_ring(servers)

            # Hash the key
            hash_key = hash(key) % (2**32)

            return self._find_server_for_hash(hash_key)

    async def _rebuild_ring(self, servers: list[Any]) -> None:
        """Rebuild the consistent hash ring."""
        self.ring.clear()
        self.servers.clear()

        for server in servers:
            server_id = getattr(server, "id", str(server))
            self.servers[server_id] = server

            # Add virtual nodes for this server
            for i in range(self.virtual_nodes):
                virtual_key = hash(f"{server_id}:{i}") % (2**32)
                self.ring[virtual_key] = server_id

        logger.debug(f"Rebuilt hash ring with {len(self.ring)} virtual nodes")

    def _find_server_for_hash(self, hash_key: int) -> Any:
        """Find server responsible for hash key."""
        if not self.ring:
            raise ValueError("Hash ring is empty")

        # Find the first server with hash >= key
        sorted_hashes = sorted(self.ring.keys())

        for ring_hash in sorted_hashes:
            if ring_hash >= hash_key:
                server_id = self.ring[ring_hash]
                return self.servers[server_id]

        # Wrap around to first server
        server_id = self.ring[sorted_hashes[0]]
        return self.servers[server_id]


class LoadBalancerFactory:
    """Factory for creating load balancers."""

    @staticmethod
    def create(balancer_type: str, **kwargs) -> LoadBalancer:
        """
        Create load balancer instance.

        Args:
            balancer_type: Type of load balancer
            **kwargs: Additional configuration

        Returns:
            Load balancer instance

        Raises:
            ValueError: If unknown balancer type
        """
        balancer_classes = {
            "round_robin": RoundRobinBalancer,
            "random": RandomBalancer,
            "least_connections": LeastConnectionsBalancer,
            "weighted_round_robin": WeightedRoundRobinBalancer,
            "weighted_random": WeightedRandomBalancer,
            "adaptive": AdaptiveBalancer,
            "consistent_hash": ConsistentHashBalancer,
        }

        if balancer_type not in balancer_classes:
            raise ValueError(f"Unknown balancer type: {balancer_type}")

        balancer_class = balancer_classes[balancer_type]
        return balancer_class(**kwargs)

    @staticmethod
    def list_types() -> list[str]:
        """Get list of available balancer types."""
        return [
            "round_robin",
            "random",
            "least_connections",
            "weighted_round_robin",
            "weighted_random",
            "adaptive",
            "consistent_hash",
        ]
