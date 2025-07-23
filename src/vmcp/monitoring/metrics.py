"""
Metrics collection and aggregation for vMCP.

This module provides comprehensive metrics collection for monitoring
system performance, request patterns, and operational health.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    method: str
    server_id: str | None
    start_time: float
    end_time: float | None = None
    success: bool = False
    error_code: int | None = None
    cached: bool = False
    response_size: int = 0

    @property
    def duration(self) -> float:
        """Get request duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, window_size: int = 1000) -> None:
        """
        Initialize metrics collector.

        Args:
            window_size: Size of rolling window for request metrics
        """
        self.window_size = window_size

        # Request metrics
        self.request_metrics: deque[RequestMetrics] = deque(maxlen=window_size)
        self.method_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.server_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Counters (monotonically increasing)
        self.counters = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "requests_cached": 0,
            "connections_total": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }

        # Gauges (current values)
        self.gauges = {
            "active_requests": 0,
            "active_connections": 0,
            "servers_healthy": 0,
            "servers_total": 0,
            "cache_size": 0,
            "memory_used": 0,
            "transport_count": 0,
        }

        # Histograms (for response times)
        self.histograms = {
            "response_time": deque(maxlen=1000),
            "request_size": deque(maxlen=1000),
            "response_size": deque(maxlen=1000),
        }

        self._lock = asyncio.Lock()
        self._start_time = time.time()

    async def record_request(
        self,
        method: str,
        server_id: str | None = None,
        duration: float = 0.0,
        success: bool = True,
        error_code: int | None = None,
        cached: bool = False,
        response_size: int = 0,
    ) -> None:
        """
        Record request metrics.

        Args:
            method: Request method
            server_id: Server that handled request
            duration: Request duration in seconds
            success: Whether request was successful
            error_code: Error code if request failed
            cached: Whether response was cached
            response_size: Size of response in bytes
        """
        async with self._lock:
            # Create metrics entry
            now = time.time()
            metrics = RequestMetrics(
                method=method,
                server_id=server_id,
                start_time=now - duration,
                end_time=now,
                success=success,
                error_code=error_code,
                cached=cached,
                response_size=response_size,
            )

            # Add to collections
            self.request_metrics.append(metrics)
            self.method_metrics[method].append(metrics)
            if server_id:
                self.server_metrics[server_id].append(metrics)

            # Update counters
            self.counters["requests_total"] += 1
            if success:
                self.counters["requests_success"] += 1
            else:
                self.counters["requests_failed"] += 1
            if cached:
                self.counters["requests_cached"] += 1
            if response_size > 0:
                self.counters["bytes_sent"] += response_size

            # Update histograms
            if duration > 0:
                self.histograms["response_time"].append(duration)
            if response_size > 0:
                self.histograms["response_size"].append(response_size)

    async def update_gauge(self, name: str, value: float) -> None:
        """
        Update a gauge value.

        Args:
            name: Gauge name
            value: New value
        """
        async with self._lock:
            self.gauges[name] = value

    async def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Increment value
        """
        async with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    async def record_histogram(self, name: str, value: float) -> None:
        """
        Record value in histogram.

        Args:
            name: Histogram name
            value: Value to record
        """
        async with self._lock:
            if name not in self.histograms:
                self.histograms[name] = deque(maxlen=1000)
            self.histograms[name].append(value)

    async def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        async with self._lock:
            # Calculate request statistics
            recent_requests = list(self.request_metrics)
            request_stats = self._calculate_request_stats(recent_requests)

            # Method breakdown
            method_stats = {}
            for method, metrics_list in self.method_metrics.items():
                if metrics_list:
                    method_stats[method] = self._calculate_request_stats(
                        list(metrics_list)
                    )

            # Server breakdown
            server_stats = {}
            for server_id, metrics_list in self.server_metrics.items():
                if metrics_list:
                    server_stats[server_id] = self._calculate_request_stats(
                        list(metrics_list)
                    )

            # Histogram statistics
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    histogram_stats[name] = self._calculate_histogram_stats(
                        list(values)
                    )

            return {
                "timestamp": time.time(),
                "uptime": time.time() - self._start_time,
                "requests": request_stats,
                "methods": method_stats,
                "servers": server_stats,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": histogram_stats,
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "error_rate": self._calculate_error_rate(),
            }

    def _calculate_request_stats(
        self, requests: list[RequestMetrics]
    ) -> dict[str, Any]:
        """Calculate statistics for list of requests."""
        if not requests:
            return {
                "count": 0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "latency_mean": 0.0,
                "latency_median": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
                "throughput": 0.0,
            }

        durations = [r.duration for r in requests if r.duration > 0]
        success_count = sum(1 for r in requests if r.success)

        # Calculate throughput (requests per second)
        if requests:
            time_span = max(r.end_time for r in requests if r.end_time) - min(
                r.start_time for r in requests
            )
            throughput = len(requests) / max(time_span, 1.0)
        else:
            throughput = 0.0

        return {
            "count": len(requests),
            "success_rate": success_count / len(requests),
            "error_rate": (len(requests) - success_count) / len(requests),
            "latency_mean": statistics.mean(durations) if durations else 0.0,
            "latency_median": statistics.median(durations) if durations else 0.0,
            "latency_p95": self._percentile(durations, 0.95),
            "latency_p99": self._percentile(durations, 0.99),
            "throughput": throughput,
        }

    def _calculate_histogram_stats(self, values: list[float]) -> dict[str, Any]:
        """Calculate statistics for histogram values."""
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
        }

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.counters["requests_total"]
        cached_requests = self.counters["requests_cached"]

        if total_requests == 0:
            return 0.0
        return cached_requests / total_requests

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate."""
        total_requests = self.counters["requests_total"]
        failed_requests = self.counters["requests_failed"]

        if total_requests == 0:
            return 0.0
        return failed_requests / total_requests

    async def get_method_stats(self, method: str) -> dict[str, Any] | None:
        """Get statistics for specific method."""
        async with self._lock:
            if method not in self.method_metrics:
                return None
            metrics_list = list(self.method_metrics[method])
            return self._calculate_request_stats(metrics_list)

    async def get_server_stats(self, server_id: str) -> dict[str, Any] | None:
        """Get statistics for specific server."""
        async with self._lock:
            if server_id not in self.server_metrics:
                return None
            metrics_list = list(self.server_metrics[server_id])
            return self._calculate_request_stats(metrics_list)

    async def get_recent_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent request metrics."""
        async with self._lock:
            recent = list(self.request_metrics)[-limit:]
            return [
                {
                    "method": r.method,
                    "server_id": r.server_id,
                    "duration": r.duration,
                    "success": r.success,
                    "error_code": r.error_code,
                    "cached": r.cached,
                    "response_size": r.response_size,
                    "timestamp": r.end_time or r.start_time,
                }
                for r in recent
            ]

    async def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        async with self._lock:
            self.request_metrics.clear()
            self.method_metrics.clear()
            self.server_metrics.clear()

            # Reset counters
            for key in self.counters:
                self.counters[key] = 0

            # Reset gauges
            for key in self.gauges:
                self.gauges[key] = 0

            # Reset histograms
            for key in self.histograms:
                self.histograms[key].clear()

            self._start_time = time.time()

    def get_memory_usage(self) -> dict[str, int]:
        """Get approximate memory usage of metrics collections."""
        return {
            "request_metrics": len(self.request_metrics),
            "method_metrics": sum(
                len(metrics) for metrics in self.method_metrics.values()
            ),
            "server_metrics": sum(
                len(metrics) for metrics in self.server_metrics.values()
            ),
            "histogram_values": sum(len(values) for values in self.histograms.values()),
            "total_methods": len(self.method_metrics),
            "total_servers": len(self.server_metrics),
        }


class PrometheusExporter:
    """Export metrics in Prometheus format."""

    def __init__(self, metrics_collector: MetricsCollector) -> None:
        """
        Initialize Prometheus exporter.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.collector = metrics_collector

    async def export(self) -> str:
        """Export metrics in Prometheus text format."""
        metrics = await self.collector.get_metrics()
        lines = []

        # Add metadata
        lines.append("# vMCP Gateway Metrics")
        lines.append(f"# Generated at {time.time()}")
        lines.append("")

        # Counter metrics
        lines.append("# HELP vmcp_requests_total Total number of requests")
        lines.append("# TYPE vmcp_requests_total counter")
        lines.append(f"vmcp_requests_total {metrics['counters']['requests_total']}")
        lines.append("")

        lines.append("# HELP vmcp_requests_success_total Total successful requests")
        lines.append("# TYPE vmcp_requests_success_total counter")
        lines.append(
            f"vmcp_requests_success_total {metrics['counters']['requests_success']}"
        )
        lines.append("")

        lines.append("# HELP vmcp_requests_failed_total Total failed requests")
        lines.append("# TYPE vmcp_requests_failed_total counter")
        lines.append(
            f"vmcp_requests_failed_total {metrics['counters']['requests_failed']}"
        )
        lines.append("")

        # Gauge metrics
        lines.append("# HELP vmcp_active_requests Current number of active requests")
        lines.append("# TYPE vmcp_active_requests gauge")
        lines.append(f"vmcp_active_requests {metrics['gauges']['active_requests']}")
        lines.append("")

        lines.append("# HELP vmcp_servers_healthy Number of healthy servers")
        lines.append("# TYPE vmcp_servers_healthy gauge")
        lines.append(f"vmcp_servers_healthy {metrics['gauges']['servers_healthy']}")
        lines.append("")

        lines.append("# HELP vmcp_servers_total Total number of servers")
        lines.append("# TYPE vmcp_servers_total gauge")
        lines.append(f"vmcp_servers_total {metrics['gauges']['servers_total']}")
        lines.append("")

        # Histogram metrics
        lines.append("# HELP vmcp_request_duration_seconds Request duration")
        lines.append("# TYPE vmcp_request_duration_seconds histogram")

        # Method-specific metrics
        for method, stats in metrics["methods"].items():
            labels = f'method="{method}"'
            lines.append(
                f"vmcp_request_duration_seconds_count{{{labels}}} {stats['count']}"
            )
            lines.append(
                f"vmcp_request_duration_seconds_sum{{{labels}}} {stats['latency_mean'] * stats['count']}"
            )

            # Quantiles
            lines.append(
                f'vmcp_request_duration_seconds{{quantile="0.5",{labels}}} {stats["latency_median"]}'
            )
            lines.append(
                f'vmcp_request_duration_seconds{{quantile="0.95",{labels}}} {stats["latency_p95"]}'
            )
            lines.append(
                f'vmcp_request_duration_seconds{{quantile="0.99",{labels}}} {stats["latency_p99"]}'
            )

        lines.append("")

        # Rate metrics
        lines.append("# HELP vmcp_error_rate Current error rate")
        lines.append("# TYPE vmcp_error_rate gauge")
        lines.append(f"vmcp_error_rate {metrics['error_rate']}")
        lines.append("")

        lines.append("# HELP vmcp_cache_hit_rate Cache hit rate")
        lines.append("# TYPE vmcp_cache_hit_rate gauge")
        lines.append(f"vmcp_cache_hit_rate {metrics['cache_hit_rate']}")
        lines.append("")

        return "\n".join(lines)

    async def export_json(self) -> dict[str, Any]:
        """Export metrics in JSON format."""
        return await self.collector.get_metrics()
