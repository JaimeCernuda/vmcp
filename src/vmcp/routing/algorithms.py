"""
Routing algorithms for vMCP request routing.

This module implements path-based and content-based routing algorithms
for efficiently routing requests to appropriate MCP servers.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from re import Pattern
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RouteRule:
    """Routing rule for path-based routing."""

    pattern: str
    server_id: str
    priority: int = 0
    method_filter: str | None = None
    enabled: bool = True
    description: str | None = None
    _regex: Pattern | None = None

    def __post_init__(self) -> None:
        """Compile pattern to regex after initialization."""
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Convert glob pattern to compiled regex."""
        try:
            # Escape special regex characters except * and ?
            escaped = re.escape(self.pattern)

            # Convert glob wildcards to regex
            regex_pattern = escaped.replace(r"\*", ".*").replace(r"\?", ".")

            # Ensure full match
            regex_pattern = f"^{regex_pattern}$"

            self._regex = re.compile(regex_pattern, re.IGNORECASE)
            logger.debug(
                f"Compiled route pattern '{self.pattern}' to regex: {regex_pattern}"
            )

        except re.error as e:
            logger.error(f"Invalid route pattern '{self.pattern}': {e}")
            self._regex = None

    def matches(self, method: str) -> bool:
        """
        Check if method matches this rule.

        Args:
            method: Method name to match

        Returns:
            True if method matches pattern
        """
        if not self.enabled or not self._regex:
            return False

        # Check method filter if specified
        if self.method_filter and not method.startswith(self.method_filter):
            return False

        return bool(self._regex.match(method))


@dataclass
class ContentRule:
    """Rule for content-based routing."""

    server_id: str
    priority: int = 0
    method: str | None = None
    tool_name: str | None = None
    resource_uri: str | None = None
    resource_pattern: str | None = None
    prompt_name: str | None = None
    condition: Callable[[dict[str, Any]], bool] | None = None
    enabled: bool = True
    description: str | None = None
    _resource_regex: Pattern | None = None

    def __post_init__(self) -> None:
        """Compile resource pattern if specified."""
        if self.resource_pattern:
            try:
                # Convert glob pattern to regex
                escaped = re.escape(self.resource_pattern)
                regex_pattern = escaped.replace(r"\*", ".*").replace(r"\?", ".")
                regex_pattern = f"^{regex_pattern}$"
                self._resource_regex = re.compile(regex_pattern, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Invalid resource pattern '{self.resource_pattern}': {e}")
                self._resource_regex = None


class RoutingAlgorithm(ABC):
    """Abstract base class for routing algorithms."""

    @abstractmethod
    def route(self, request: dict[str, Any]) -> str | None:
        """
        Route request to server ID.

        Args:
            request: JSON-RPC request

        Returns:
            Server ID or None if no match
        """
        pass

    @abstractmethod
    def add_rule(self, rule: Any) -> None:
        """Add routing rule."""
        pass

    @abstractmethod
    def remove_rule(self, rule_id: str) -> bool:
        """Remove routing rule."""
        pass

    @abstractmethod
    def get_rules(self) -> list[dict[str, Any]]:
        """Get all routing rules."""
        pass


class PathBasedRouter(RoutingAlgorithm):
    """Routes requests based on method path patterns."""

    def __init__(self) -> None:
        """Initialize path-based router."""
        self.rules: list[RouteRule] = []

    def route(self, request: dict[str, Any]) -> str | None:
        """
        Route request based on method path.

        Args:
            request: JSON-RPC request

        Returns:
            Server ID or None if no match
        """
        method = request.get("method", "")
        if not method:
            return None

        # Sort rules by priority (highest first) and iterate
        sorted_rules = sorted(
            [rule for rule in self.rules if rule.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        for rule in sorted_rules:
            if rule.matches(method):
                logger.debug(
                    f"Method '{method}' matched rule '{rule.pattern}' -> {rule.server_id}"
                )
                return rule.server_id

        logger.debug(f"Method '{method}' did not match any path-based rules")
        return None

    def add_rule(self, rule: Any) -> None:
        """Add routing rule. Use add_path_rule for PathBasedRouter."""
        if isinstance(rule, RouteRule):
            self.rules.append(rule)
        else:
            raise ValueError(
                f"PathBasedRouter only accepts RouteRule, got {type(rule)}"
            )

    def add_path_rule(
        self,
        pattern: str,
        server_id: str,
        priority: int = 0,
        method_filter: str | None = None,
        description: str | None = None,
    ) -> RouteRule:
        """
        Add a routing rule.

        Args:
            pattern: Glob pattern to match methods
            server_id: Target server ID
            priority: Rule priority (higher = checked first)
            method_filter: Optional method prefix filter
            description: Optional rule description

        Returns:
            Created rule
        """
        rule = RouteRule(
            pattern=pattern,
            server_id=server_id,
            priority=priority,
            method_filter=method_filter,
            description=description,
        )

        self.rules.append(rule)
        logger.info(
            f"Added path-based rule: '{pattern}' -> {server_id} (priority: {priority})"
        )
        return rule

    def remove_rule(self, pattern: str) -> bool:
        """
        Remove a routing rule by pattern.

        Args:
            pattern: Pattern to remove

        Returns:
            True if rule was removed
        """
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.pattern != pattern]
        removed = len(self.rules) < original_count

        if removed:
            logger.info(f"Removed path-based rule: '{pattern}'")
        else:
            logger.warning(f"Path-based rule not found: '{pattern}'")

        return removed

    def get_rules(self) -> list[dict[str, Any]]:
        """Get all routing rules."""
        return [
            {
                "pattern": rule.pattern,
                "server_id": rule.server_id,
                "priority": rule.priority,
                "method_filter": rule.method_filter,
                "enabled": rule.enabled,
                "description": rule.description,
            }
            for rule in self.rules
        ]

    def enable_rule(self, pattern: str) -> bool:
        """Enable a routing rule."""
        for rule in self.rules:
            if rule.pattern == pattern:
                rule.enabled = True
                logger.info(f"Enabled path-based rule: '{pattern}'")
                return True
        return False

    def disable_rule(self, pattern: str) -> bool:
        """Disable a routing rule."""
        for rule in self.rules:
            if rule.pattern == pattern:
                rule.enabled = False
                logger.info(f"Disabled path-based rule: '{pattern}'")
                return True
        return False


class ContentBasedRouter(RoutingAlgorithm):
    """Routes requests based on message content."""

    def __init__(self) -> None:
        """Initialize content-based router."""
        self.rules: list[ContentRule] = []

    def route(self, request: dict[str, Any]) -> str | None:
        """
        Route request based on content.

        Args:
            request: JSON-RPC request

        Returns:
            Server ID or None if no match
        """
        method = request.get("method", "")
        params = request.get("params", {})

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [rule for rule in self.rules if rule.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        for rule in sorted_rules:
            if self._matches_rule(method, params, rule):
                logger.debug(f"Request matched content rule -> {rule.server_id}")
                return rule.server_id

        logger.debug("Request did not match any content-based rules")
        return None

    def _matches_rule(
        self, method: str, params: dict[str, Any], rule: ContentRule
    ) -> bool:
        """
        Check if request matches rule.

        Args:
            method: Request method
            params: Request parameters
            rule: Rule to check

        Returns:
            True if request matches rule
        """
        # Check method match
        if rule.method and method != rule.method:
            return False

        # Check tool name for tools/call method
        if (
            rule.tool_name
            and method == "tools/call"
            and params.get("name") != rule.tool_name
        ):
            return False

        # Check resource URI for resources/read method
        if (
            rule.resource_uri
            and method == "resources/read"
            and params.get("uri") != rule.resource_uri
        ):
            return False

        # Check resource pattern for resources/read method
        if rule.resource_pattern and method == "resources/read":
            uri = params.get("uri", "")
            if rule._resource_regex and not rule._resource_regex.match(uri):
                return False

        # Check prompt name for prompts/get method
        if (
            rule.prompt_name
            and method == "prompts/get"
            and params.get("name") != rule.prompt_name
        ):
            return False

        # Check custom condition
        if rule.condition:
            try:
                return rule.condition({"method": method, "params": params})
            except Exception as e:
                logger.warning(f"Error evaluating rule condition: {e}")
                return False

        return True

    def add_rule(self, rule: ContentRule) -> None:
        """
        Add a content-based routing rule.

        Args:
            rule: Content rule to add
        """
        self.rules.append(rule)
        logger.info(
            f"Added content-based rule -> {rule.server_id} (priority: {rule.priority})"
        )

    def add_tool_rule(
        self,
        tool_name: str,
        server_id: str,
        priority: int = 0,
        description: str | None = None,
    ) -> ContentRule:
        """
        Add rule for specific tool.

        Args:
            tool_name: Tool name to match
            server_id: Target server ID
            priority: Rule priority
            description: Optional description

        Returns:
            Created rule
        """
        rule = ContentRule(
            server_id=server_id,
            priority=priority,
            method="tools/call",
            tool_name=tool_name,
            description=description or f"Route tool '{tool_name}' to {server_id}",
        )
        self.add_rule(rule)
        return rule

    def add_resource_rule(
        self,
        resource_pattern: str,
        server_id: str,
        priority: int = 0,
        description: str | None = None,
    ) -> ContentRule:
        """
        Add rule for resource pattern.

        Args:
            resource_pattern: Resource URI pattern to match
            server_id: Target server ID
            priority: Rule priority
            description: Optional description

        Returns:
            Created rule
        """
        rule = ContentRule(
            server_id=server_id,
            priority=priority,
            method="resources/read",
            resource_pattern=resource_pattern,
            description=description
            or f"Route resource pattern '{resource_pattern}' to {server_id}",
        )
        self.add_rule(rule)
        return rule

    def add_prompt_rule(
        self,
        prompt_name: str,
        server_id: str,
        priority: int = 0,
        description: str | None = None,
    ) -> ContentRule:
        """
        Add rule for specific prompt.

        Args:
            prompt_name: Prompt name to match
            server_id: Target server ID
            priority: Rule priority
            description: Optional description

        Returns:
            Created rule
        """
        rule = ContentRule(
            server_id=server_id,
            priority=priority,
            method="prompts/get",
            prompt_name=prompt_name,
            description=description or f"Route prompt '{prompt_name}' to {server_id}",
        )
        self.add_rule(rule)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove rule by server ID and description match.

        Args:
            rule_id: Rule identifier (server_id:description)

        Returns:
            True if rule was removed
        """
        original_count = len(self.rules)

        if ":" in rule_id:
            server_id, description = rule_id.split(":", 1)
            self.rules = [
                rule
                for rule in self.rules
                if not (rule.server_id == server_id and rule.description == description)
            ]
        else:
            # Remove all rules for server
            self.rules = [rule for rule in self.rules if rule.server_id != rule_id]

        removed = len(self.rules) < original_count

        if removed:
            logger.info(f"Removed content-based rule: {rule_id}")
        else:
            logger.warning(f"Content-based rule not found: {rule_id}")

        return removed

    def get_rules(self) -> list[dict[str, Any]]:
        """Get all routing rules."""
        return [
            {
                "server_id": rule.server_id,
                "priority": rule.priority,
                "method": rule.method,
                "tool_name": rule.tool_name,
                "resource_uri": rule.resource_uri,
                "resource_pattern": rule.resource_pattern,
                "prompt_name": rule.prompt_name,
                "enabled": rule.enabled,
                "description": rule.description,
                "has_condition": rule.condition is not None,
            }
            for rule in self.rules
        ]

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a routing rule."""
        count = 0
        for rule in self.rules:
            if (
                rule.server_id == rule_id
                or f"{rule.server_id}:{rule.description}" == rule_id
            ):
                rule.enabled = True
                count += 1

        if count > 0:
            logger.info(f"Enabled {count} content-based rule(s): {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a routing rule."""
        count = 0
        for rule in self.rules:
            if (
                rule.server_id == rule_id
                or f"{rule.server_id}:{rule.description}" == rule_id
            ):
                rule.enabled = False
                count += 1

        if count > 0:
            logger.info(f"Disabled {count} content-based rule(s): {rule_id}")
            return True
        return False


class HybridRouter(RoutingAlgorithm):
    """Combines path-based and content-based routing."""

    def __init__(self) -> None:
        """Initialize hybrid router."""
        self.path_router = PathBasedRouter()
        self.content_router = ContentBasedRouter()

    def route(self, request: dict[str, Any]) -> str | None:
        """
        Route using path-based first, then content-based.

        Args:
            request: JSON-RPC request

        Returns:
            Server ID or None if no match
        """
        # Try path-based routing first
        server_id = self.path_router.route(request)
        if server_id:
            return server_id

        # Fall back to content-based routing
        return self.content_router.route(request)

    def add_rule(self, rule: Any) -> None:
        """Add rule to appropriate router."""
        if isinstance(rule, RouteRule):
            self.path_router.rules.append(rule)
        elif isinstance(rule, ContentRule):
            self.content_router.rules.append(rule)
        else:
            raise ValueError(f"Unsupported rule type: {type(rule)}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule from both routers."""
        path_removed = self.path_router.remove_rule(rule_id)
        content_removed = self.content_router.remove_rule(rule_id)
        return path_removed or content_removed

    def get_rules(self) -> list[dict[str, Any]]:
        """Get all routing rules from both routers."""
        rules = []

        # Add path-based rules
        for rule in self.path_router.get_rules():
            rule["type"] = "path"
            rules.append(rule)

        # Add content-based rules
        for rule in self.content_router.get_rules():
            rule["type"] = "content"
            rules.append(rule)

        return rules


class CapabilityRouter:
    """Routes requests based on server capabilities."""

    def __init__(self, registry: Any) -> None:
        """
        Initialize capability router.

        Args:
            registry: Server registry
        """
        self.registry = registry

    def route_by_capability(
        self, request: dict[str, Any], required_capability: str
    ) -> str | None:
        """
        Route request to server with required capability.

        Args:
            request: JSON-RPC request
            required_capability: Required capability (e.g., "tools", "resources")

        Returns:
            Server ID or None if no capable server found
        """
        capable_servers = self.registry.get_servers_by_capability(required_capability)

        if not capable_servers:
            logger.debug(f"No servers found with capability '{required_capability}'")
            return None

        # For now, return first capable server
        # Could implement more sophisticated selection here
        selected = capable_servers[0]
        logger.debug(
            f"Routed request to {selected.config.id} "
            f"based on capability '{required_capability}'"
        )
        return str(selected.config.id)

    def get_method_capability(self, method: str) -> str | None:
        """
        Determine required capability for method.

        Args:
            method: JSON-RPC method

        Returns:
            Required capability or None
        """
        if method.startswith("tools/"):
            return "tools"
        elif method.startswith("resources/"):
            return "resources"
        elif method.startswith("prompts/"):
            return "prompts"
        elif method in ["completion/complete"]:
            return "completion"
        else:
            return None
