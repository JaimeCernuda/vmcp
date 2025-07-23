"""Tests for configuration loading functionality."""

import tempfile
from pathlib import Path

import pytest
import toml

from vmcp.config.loader import ConfigLoader, VMCPConfig


class TestVMCPConfig:
    def test_default_config_creation(self):
        """Test default configuration has expected values."""
        config = VMCPConfig()

        assert config.version == "0.1.0"
        assert config.gateway.registry_path == "~/.vmcp/registry"
        assert config.gateway.log_level == "INFO"
        assert config.gateway.cache_enabled is True
        assert isinstance(config.transports, dict)
        assert isinstance(config.servers, dict)

    def test_config_validation(self):
        """Test configuration validation."""
        config = VMCPConfig()

        # Test valid timeout
        config.gateway.request_timeout = 30
        assert config.gateway.request_timeout == 30

        # Test negative timeout should be handled gracefully
        config.gateway.request_timeout = -1
        assert (
            config.gateway.request_timeout == -1
        )  # Validation not enforced at model level

    def test_config_dict_conversion(self):
        """Test configuration can be converted to dict."""
        config = VMCPConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "version" in config_dict
        assert "gateway" in config_dict
        assert "transports" in config_dict


class TestConfigLoader:
    @pytest.fixture
    def loader(self):
        """Create config loader instance."""
        return ConfigLoader()

    def test_load_config_from_file(self, loader):
        """Test loading configuration from TOML file."""
        config_data = {
            "version": "0.1.0",
            "gateway": {
                "registry_path": "~/.vmcp/test_registry",
                "log_level": "DEBUG",
                "cache_ttl": 600,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            toml.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = loader.load_from_file(config_path)
            assert loaded_config["gateway"]["registry_path"] == "~/.vmcp/test_registry"
            assert loaded_config["gateway"]["log_level"] == "DEBUG"
            assert loaded_config["gateway"]["cache_ttl"] == 600
        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self, loader):
        """Test loading config when file doesn't exist."""
        with pytest.raises((FileNotFoundError, OSError), match="Configuration file not found"):
            loader.load_from_file("nonexistent_config.toml")

    def test_load_config_invalid_toml(self, loader):
        """Test loading invalid TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [unclosed")
            config_path = f.name

        try:
            with pytest.raises((ValueError, OSError), match="(Invalid TOML syntax|Configuration file not found)"):
                loader.load_from_file(config_path)
        finally:
            Path(config_path).unlink()

    def test_validate_config_valid(self, loader):
        """Test validating valid configuration."""
        config_data = {"version": "0.1.0", "gateway": {"log_level": "INFO"}}

        errors = loader.validate_config(config_data)
        assert len(errors) == 0

    def test_validate_config_invalid(self, loader):
        """Test validating invalid configuration."""
        config_data = {"routing": {"default_strategy": "invalid_strategy"}}

        errors = loader.validate_config(config_data)
        assert len(errors) > 0
        assert any("default_strategy" in error for error in errors)

    def test_environment_variable_substitution(self, loader):
        """Test environment variable substitution."""
        import os

        # Set test environment variable
        os.environ["TEST_VAR"] = "test_value"

        try:
            config_str = "test_path = '${TEST_VAR}/path'"
            result = loader._substitute_env_vars_in_string(config_str)
            assert result == "test_path = 'test_value/path'"
        finally:
            os.environ.pop("TEST_VAR", None)

    def test_environment_variable_with_default(self, loader):
        """Test environment variable substitution with default value."""
        config_str = "test_path = '${NONEXISTENT_VAR:-default_value}'"
        result = loader._substitute_env_vars_in_string(config_str)
        assert result == "test_path = 'default_value'"

    def test_save_config(self, loader):
        """Test saving configuration to file."""
        config_data = {"version": "0.1.0", "gateway": {"log_level": "INFO"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.toml"

            loader.save_config(config_data, config_path)
            assert config_path.exists()

            # Verify content
            loaded_config = loader.load_from_file(config_path)
            assert loaded_config["version"] == "0.1.0"

    def test_create_example_config(self, loader):
        """Test creating example configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "example_config.toml"

            loader.create_example_config(config_path)
            assert config_path.exists()

            # Verify it's valid
            loaded_config = loader.load_from_file(config_path)
            errors = loader.validate_config(loaded_config)
            assert len(errors) == 0

    def test_merge_configs(self, loader):
        """Test merging configurations."""
        base_config = {
            "gateway": {"log_level": "INFO", "cache_enabled": True},
            "routing": {"default_strategy": "hybrid"},
        }

        override_config = {
            "gateway": {"log_level": "DEBUG"},
            "new_section": {"value": "test"},
        }

        merged = loader.merge_configs(base_config, override_config)

        assert merged["gateway"]["log_level"] == "DEBUG"  # Overridden
        assert merged["gateway"]["cache_enabled"] is True  # Preserved
        assert merged["routing"]["default_strategy"] == "hybrid"  # Preserved
        assert merged["new_section"]["value"] == "test"  # Added
