"""
Circuit breaker pattern implementation for fault tolerance.

This module implements a production-grade circuit breaker that prevents
cascading failures by failing fast when a service is unreliable.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from ..errors import CircuitBreakerOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    half_open_max_calls: int = 3
    slow_call_threshold: float = 5.0  # Seconds
    slow_call_rate_threshold: float = 0.5  # 50%
    minimum_calls: int = 10  # Minimum calls before evaluating failure rate


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    failure_count: int = 0
    success_count: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    slow_calls: int = 0
    average_response_time: float = 0.0
    state_changes: list[tuple[CircuitState, float]] = field(default_factory=list)

    def get_failure_rate(self) -> float:
        """Get current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def get_slow_call_rate(self) -> float:
        """Get current slow call rate."""
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name for logging
            config: Configuration object
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._recent_calls: list[
            tuple[float, float, bool]
        ] = []  # (timestamp, duration, success)

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        if self._state != CircuitState.OPEN:
            return False

        # Check if we should transition to half-open
        if self._should_attempt_reset():
            asyncio.create_task(self._transition_to(CircuitState.HALF_OPEN))
            return False

        return True

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            if self.is_open:
                raise CircuitBreakerOpenError(
                    self.name, self._stats.failure_count, self._stats.last_failure_time
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        self.name,
                        self._stats.failure_count,
                        self._stats.last_failure_time,
                    )
                self._half_open_calls += 1

        # Execute the function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            duration = time.time() - start_time
            await self._on_success(duration)
            return result  # type: ignore[no-any-return]

        except Exception as e:
            if isinstance(e, self.config.expected_exception):
                duration = time.time() - start_time
                await self._on_failure(duration)
            raise
        finally:
            async with self._lock:
                self._stats.total_calls += 1

    async def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Alias for call method."""
        return await self.call(func, *args, **kwargs)

    async def _on_success(self, duration: float) -> None:
        """Handle successful call."""
        async with self._lock:
            now = time.time()

            # Update stats
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = now

            # Check if call was slow
            if duration > self.config.slow_call_threshold:
                self._stats.slow_calls += 1

            # Update average response time
            self._update_average_response_time(duration)

            # Record call
            self._recent_calls.append((now, duration, True))
            self._cleanup_old_calls()

            # State transitions
            if (
                self._state == CircuitState.HALF_OPEN
                and self._stats.consecutive_successes >= self.config.half_open_max_calls
            ):
                await self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, duration: float) -> None:
        """Handle failed call."""
        async with self._lock:
            now = time.time()

            # Update stats
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = now

            # Update average response time
            self._update_average_response_time(duration)

            # Record call
            self._recent_calls.append((now, duration, False))
            self._cleanup_old_calls()

            # State transitions
            if self._state == CircuitState.HALF_OPEN or (
                self._state == CircuitState.CLOSED and self._should_open_circuit()
            ):
                await self._transition_to(CircuitState.OPEN)

    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        # Not enough calls to make decision
        if self._stats.total_calls < self.config.minimum_calls:
            return False

        # Check consecutive failures
        if self._stats.consecutive_failures >= self.config.failure_threshold:
            return True

        # Check failure rate
        failure_rate = self._stats.get_failure_rate()
        if failure_rate > 0.5:  # 50% failure rate
            return True

        # Check slow call rate
        slow_call_rate = self._stats.get_slow_call_rate()
        return slow_call_rate > self.config.slow_call_rate_threshold

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self._stats.last_failure_time is not None
            and time.time() - self._stats.last_failure_time
            >= self.config.recovery_timeout
        )

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        now = time.time()

        logger.info(
            f"Circuit breaker '{self.name}' transitioning "
            f"from {old_state.value} to {new_state.value}"
        )

        # Record state change
        self._stats.state_changes.append((new_state, now))

        # Reset counters based on transition
        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.consecutive_failures = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._stats.consecutive_successes = 0
            self._stats.consecutive_failures = 0
        elif new_state == CircuitState.OPEN:
            self._half_open_calls = 0

    def _update_average_response_time(self, duration: float) -> None:
        """Update average response time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._stats.average_response_time == 0:
            self._stats.average_response_time = duration
        else:
            self._stats.average_response_time = (
                alpha * duration + (1 - alpha) * self._stats.average_response_time
            )

    def _cleanup_old_calls(self) -> None:
        """Remove old call records."""
        cutoff_time = time.time() - 300  # Keep last 5 minutes
        self._recent_calls = [
            call for call in self._recent_calls if call[0] > cutoff_time
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "total_calls": self._stats.total_calls,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
            "consecutive_successes": self._stats.consecutive_successes,
            "consecutive_failures": self._stats.consecutive_failures,
            "last_failure_time": self._stats.last_failure_time,
            "last_success_time": self._stats.last_success_time,
            "failure_rate": self._stats.get_failure_rate(),
            "slow_calls": self._stats.slow_calls,
            "slow_call_rate": self._stats.get_slow_call_rate(),
            "average_response_time": self._stats.average_response_time,
            "state_changes": len(self._stats.state_changes),
            "uptime_percentage": (
                self._stats.total_successes / max(1, self._stats.total_calls) * 100
            ),
            "half_open_calls": self._half_open_calls,
            "recent_calls": len(self._recent_calls),
        }

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitBreakerStats()
            self._half_open_calls = 0
            self._recent_calls.clear()
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    async def force_open(self) -> None:
        """Manually force circuit breaker open."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
            logger.warning(f"Circuit breaker '{self.name}' manually forced open")

    def get_recent_call_stats(self, window_seconds: int = 60) -> dict[str, Any]:
        """
        Get statistics for recent calls within time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Statistics for recent calls
        """
        cutoff_time = time.time() - window_seconds
        recent_calls = [call for call in self._recent_calls if call[0] > cutoff_time]

        if not recent_calls:
            return {
                "total_calls": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "slow_calls": 0,
                "slow_call_rate": 0.0,
            }

        success_count = sum(1 for call in recent_calls if call[2])
        failure_count = len(recent_calls) - success_count
        total_duration = sum(call[1] for call in recent_calls)
        slow_calls = sum(
            1 for call in recent_calls if call[1] > self.config.slow_call_threshold
        )

        return {
            "total_calls": len(recent_calls),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(recent_calls),
            "average_duration": total_duration / len(recent_calls),
            "slow_calls": slow_calls,
            "slow_call_rate": slow_calls / len(recent_calls),
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self) -> None:
        """Initialize circuit breaker registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration (uses default if None)

        Returns:
            Circuit breaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                logger.debug(f"Created circuit breaker: {name}")
            return self._breakers[name]

    async def remove_breaker(self, name: str) -> bool:
        """
        Remove circuit breaker.

        Args:
            name: Circuit breaker name

        Returns:
            True if breaker was removed, False if not found
        """
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.debug(f"Removed circuit breaker: {name}")
                return True
            return False

    def list_breakers(self) -> list[str]:
        """Get list of all circuit breaker names."""
        return list(self._breakers.keys())

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        async with self._lock:
            return {
                name: breaker.get_stats() for name, breaker in self._breakers.items()
            }

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()
            logger.info("Reset all circuit breakers")

    async def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary of all circuit breakers."""
        async with self._lock:
            if not self._breakers:
                return {
                    "total_breakers": 0,
                    "healthy_breakers": 0,
                    "degraded_breakers": 0,
                    "unhealthy_breakers": 0,
                    "overall_health": "healthy",
                }

            healthy = sum(1 for breaker in self._breakers.values() if breaker.is_closed)
            degraded = sum(
                1 for breaker in self._breakers.values() if breaker.is_half_open
            )
            unhealthy = sum(1 for breaker in self._breakers.values() if breaker.is_open)

            total = len(self._breakers)

            # Determine overall health
            if unhealthy == 0 and degraded == 0:
                overall_health = "healthy"
            elif unhealthy > total // 2:
                overall_health = "unhealthy"
            else:
                overall_health = "degraded"

            return {
                "total_breakers": total,
                "healthy_breakers": healthy,
                "degraded_breakers": degraded,
                "unhealthy_breakers": unhealthy,
                "overall_health": overall_health,
            }
