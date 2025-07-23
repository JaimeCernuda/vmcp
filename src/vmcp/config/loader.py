"""
Configuration loading and validation for vMCP.

This module provides comprehensive configuration loading from TOML files
with environment variable substitution, validation, and schema checking.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field, ValidationError

from ..errors import ConfigurationError

logger = logging.getLogger(__name__)


class TransportConfig(BaseModel):
    """Configuration for transport layer."""

    enabled: bool = False
    port: int | None = None
    host: str | None = None
    max_connections: int | None = None
    timeout: int | None = None

    class Config:
        extra = "allow"


class GatewayConfig(BaseModel):
    """Configuration for gateway settings."""

    registry_path: str = "~/.vmcp/registry"
    log_level: str = "INFO"
    log_format: str = "structured"
    log_file: str | None = None

    # Cache configuration
    cache_enabled: bool = True
    cache_ttl: int = 300
    cache_max_size: int = 1000

    # Performance configuration
    max_connections: int = 1000
    request_timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024

    # Health monitoring
    health_check_interval: int = 30
    health_check_timeout: int = 5

    # Security configuration
    enable_request_validation: bool = True
    max_concurrent_requests: int = 100
    rate_limit_requests: int = 1000

    class Config:
        extra = "allow"


class RoutingConfig(BaseModel):
    """Configuration for routing strategies."""

    default_strategy: str = "hybrid"
    load_balancer: str = "round_robin"
    cache_enabled: bool = True
    cache_ttl: int = 300

    # Path-based routing rules
    path_rules: list[dict[str, Any]] = Field(default_factory=list)

    # Content-based routing rules
    content_rules: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"


class ServerConfig(BaseModel):
    """Configuration for individual MCP server."""

    id: str
    name: str
    transport: str = "stdio"
    command: list[str]
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    working_directory: str | None = None
    enabled: bool = True

    # Capabilities
    capabilities: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Resource limits
    max_memory: int | None = None
    max_cpu: float | None = None
    timeout: int = 30

    # Health check configuration
    health_check_interval: int = 60
    health_check_timeout: int = 5
    restart_on_failure: bool = True
    max_restarts: int = 3

    class Config:
        extra = "allow"


class VMCPConfig(BaseModel):
    """Complete vMCP configuration schema."""

    version: str = "0.1.0"
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    transports: dict[str, TransportConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    servers: dict[str, ServerConfig] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class ConfigLoader:
    """Configuration loader with environment variable substitution."""

    def __init__(self) -> None:
        """Initialize configuration loader."""
        self.env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    def load_from_file(self, config_path: str | Path) -> dict[str, Any]:
        """
        Load configuration from TOML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary with environment variable substitution

        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid
        """
        try:
            config_path = Path(config_path).expanduser().resolve()

            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            logger.info(f"Loading configuration from {config_path}")

            # Load TOML file
            with open(config_path) as f:
                config_data = toml.load(f)

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            # Validate configuration
            errors = self.validate_config(config_data)
            if errors:
                raise ConfigurationError(
                    f"Configuration validation failed: {'; '.join(errors)}"
                )

            logger.info("Configuration loaded successfully")
            return config_data

        except toml.TomlDecodeError as e:
            raise ConfigurationError(f"Invalid TOML syntax: {e}") from e
        except OSError as e:
            raise ConfigurationError(f"Error reading configuration file: {e}") from e

    def load_defaults(self) -> dict[str, Any]:
        """
        Load default configuration.

        Returns:
            Default configuration dictionary
        """
        default_config = VMCPConfig()

        # Add default transport configurations
        default_config.transports = {
            "stdio": TransportConfig(enabled=True),
            "http": TransportConfig(enabled=False, port=3000, host="127.0.0.1"),
            "websocket": TransportConfig(enabled=False, port=3001, host="127.0.0.1"),
        }

        return default_config.dict()

    def load_from_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Processed configuration with environment variable substitution

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Substitute environment variables
        config_data = self._substitute_env_vars(config_dict)

        # Validate configuration
        errors = self.validate_config(config_data)
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

        return config_data

    def validate_config(self, config_data: dict[str, Any]) -> list[str]:
        """
        Validate configuration against schema.

        Args:
            config_data: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Validate using Pydantic model
            VMCPConfig(**config_data)
        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")

        # Additional custom validations
        errors.extend(self._validate_server_configs(config_data.get("servers", {})))
        errors.extend(
            self._validate_transport_configs(config_data.get("transports", {}))
        )
        errors.extend(self._validate_routing_config(config_data.get("routing", {})))

        return errors

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Args:
            obj: Configuration object (dict, list, or primitive)

        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_vars_in_string(obj)
        else:
            return obj

    def _substitute_env_vars_in_string(self, text: str) -> str:
        """
        Substitute environment variables in a string.

        Supports formats:
        - ${VAR} - Required variable (raises error if not found)
        - ${VAR:-default} - Variable with default value
        - ${VAR:default} - Variable with default value (alternative syntax)

        Args:
            text: String potentially containing environment variable references

        Returns:
            String with environment variables substituted

        Raises:
            ConfigurationError: If required environment variable is missing
        """

        def replace_var(match):
            var_expr = match.group(1)

            # Check for default value syntax
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
            elif ":" in var_expr and not var_expr.startswith(":"):
                var_name, default_value = var_expr.split(":", 1)
            else:
                var_name = var_expr
                default_value = None

            # Get environment variable value
            env_value = os.environ.get(var_name.strip())

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ConfigurationError(
                    f"Required environment variable not found: {var_name}"
                )

        return self.env_var_pattern.sub(replace_var, text)

    def _validate_server_configs(self, servers: dict[str, Any]) -> list[str]:
        """Validate server configurations."""
        errors = []

        for server_id, server_config in servers.items():
            if not isinstance(server_config, dict):
                errors.append(f"servers.{server_id}: must be a dictionary")
                continue

            # Validate required fields
            if "command" not in server_config:
                errors.append(f"servers.{server_id}: missing required field 'command'")

            # Validate command is a list
            command = server_config.get("command")
            if command is not None and not isinstance(command, list):
                errors.append(f"servers.{server_id}.command: must be a list")

            # Validate transport
            transport = server_config.get("transport", "stdio")
            valid_transports = ["stdio", "http", "websocket", "tcp"]
            if transport not in valid_transports:
                errors.append(
                    f"servers.{server_id}.transport: must be one of {valid_transports}"
                )

            # Validate capabilities
            capabilities = server_config.get("capabilities", {})
            if not isinstance(capabilities, dict):
                errors.append(f"servers.{server_id}.capabilities: must be a dictionary")

        return errors

    def _validate_transport_configs(self, transports: dict[str, Any]) -> list[str]:
        """Validate transport configurations."""
        errors = []

        for transport_name, transport_config in transports.items():
            if not isinstance(transport_config, dict):
                errors.append(f"transports.{transport_name}: must be a dictionary")
                continue

            # Validate port for network transports
            if transport_name in ["http", "websocket", "tcp"]:
                port = transport_config.get("port")
                if transport_config.get("enabled", False) and port is None:
                    errors.append(
                        f"transports.{transport_name}: port is required when enabled"
                    )
                elif port is not None and (
                    not isinstance(port, int) or port < 1 or port > 65535
                ):
                    errors.append(
                        f"transports.{transport_name}.port: must be an integer between 1 and 65535"
                    )

        return errors

    def _validate_routing_config(self, routing: dict[str, Any]) -> list[str]:
        """Validate routing configuration."""
        errors = []

        # Validate default strategy
        strategy = routing.get("default_strategy", "hybrid")
        valid_strategies = ["path", "content", "capability", "hybrid"]
        if strategy not in valid_strategies:
            errors.append(
                f"routing.default_strategy: must be one of {valid_strategies}"
            )

        # Validate load balancer
        load_balancer = routing.get("load_balancer", "round_robin")
        valid_balancers = [
            "round_robin",
            "random",
            "least_connections",
            "weighted_round_robin",
            "weighted_random",
            "adaptive",
            "consistent_hash",
        ]
        if load_balancer not in valid_balancers:
            errors.append(f"routing.load_balancer: must be one of {valid_balancers}")

        return errors

    def save_config(self, config_data: dict[str, Any], config_path: str | Path) -> None:
        """
        Save configuration to TOML file.

        Args:
            config_data: Configuration to save
            config_path: Path to save configuration file

        Raises:
            ConfigurationError: If configuration cannot be saved
        """
        try:
            config_path = Path(config_path).expanduser().resolve()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate before saving
            errors = self.validate_config(config_data)
            if errors:
                raise ConfigurationError(
                    f"Cannot save invalid configuration: {'; '.join(errors)}"
                )

            # Save to TOML file
            with open(config_path, "w") as f:
                toml.dump(config_data, f)

            logger.info(f"Configuration saved to {config_path}")

        except OSError as e:
            raise ConfigurationError(f"Error saving configuration file: {e}") from e

    def merge_configs(
        self, base_config: dict[str, Any], override_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get_default_config_path(self) -> Path:
        """
        Get default configuration file path.

        Returns:
            Default configuration path
        """
        return Path.home() / ".vmcp" / "config.toml"

    def create_example_config(self, config_path: str | Path) -> None:
        """
        Create example configuration file.

        Args:
            config_path: Path to create example configuration

        Raises:
            ConfigurationError: If example configuration cannot be created
        """
        example_config = {
            "version": "0.1.0",
            "gateway": {
                "registry_path": "~/.vmcp/registry",
                "log_level": "INFO",
                "log_format": "structured",
                "cache_enabled": True,
                "cache_ttl": 300,
                "max_connections": 1000,
                "request_timeout": 30,
                "health_check_interval": 30,
                "enable_request_validation": True,
                "max_concurrent_requests": 100,
                "rate_limit_requests": 1000,
            },
            "transports": {
                "stdio": {"enabled": True},
                "http": {
                    "enabled": False,
                    "port": 3000,
                    "host": "127.0.0.1",
                    "max_connections": 100,
                },
                "websocket": {"enabled": False, "port": 3001, "host": "127.0.0.1"},
            },
            "routing": {
                "default_strategy": "hybrid",
                "load_balancer": "round_robin",
                "cache_enabled": True,
                "cache_ttl": 300,
                "path_rules": [
                    {"pattern": "/tools/*", "server_id": "tools-server", "priority": 10}
                ],
                "content_rules": [
                    {
                        "tool_name": "file_operations",
                        "server_id": "file-server",
                        "priority": 10,
                    }
                ],
            },
            "servers": {
                "example-server": {
                    "id": "example-server",
                    "name": "Example MCP Server",
                    "transport": "stdio",
                    "command": ["python", "-m", "example_mcp_server"],
                    "args": [],
                    "env": {"MCP_SERVER_MODE": "production"},
                    "enabled": True,
                    "capabilities": {
                        "tools": {"list_changed": True},
                        "resources": {"subscribe": True, "list_changed": True},
                    },
                    "timeout": 30,
                    "health_check_interval": 60,
                    "restart_on_failure": True,
                    "max_restarts": 3,
                }
            },
        }

        self.save_config(example_config, config_path)
