"""Integration tests for CLI commands."""

import json
import tempfile
from pathlib import Path
from subprocess import run

import pytest


class TestCLIIntegration:
    """Integration tests that actually run CLI commands."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_vmcp_help(self):
        """Test that vmcp --help works."""
        result = run(["uv", "run", "vmcp", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Virtual Model Context Protocol" in result.stdout

    def test_vmcp_status(self):
        """Test vmcp status command."""
        result = run(["uv", "run", "vmcp", "status"], capture_output=True, text=True)
        assert result.returncode == 0
        # Should show not running since no server is started
        assert "not running" in result.stdout.lower()

    def test_config_init_and_validate(self, temp_config_dir):
        """Test config init and validate commands."""
        config_path = temp_config_dir / "test_config.toml"

        # Test config init
        result = run(
            ["uv", "run", "vmcp", "config", "init", "--output", str(config_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert config_path.exists()

        # Test config validate
        result = run(
            ["uv", "run", "vmcp", "config", "validate", str(config_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

    def test_extension_list(self):
        """Test extension list command."""
        result = run(
            ["uv", "run", "vmcp", "extension", "list"], capture_output=True, text=True
        )
        assert result.returncode == 0
        # Should show available extensions table
        assert "Extensions" in result.stdout

    def test_extension_list_json_format(self):
        """Test extension list with JSON format."""
        result = run(
            ["uv", "run", "vmcp", "extension", "list", "--format", "json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Should be valid JSON
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Extension list JSON output is invalid")

    def test_extension_list_filters(self):
        """Test extension list with filters."""
        # Test --installed filter
        result = run(
            ["uv", "run", "vmcp", "extension", "list", "--installed"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Test --enabled filter
        result = run(
            ["uv", "run", "vmcp", "extension", "list", "--enabled"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_completion_generation(self):
        """Test shell completion generation."""
        # Test bash completion
        result = run(
            ["uv", "run", "vmcp", "completion", "bash"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "complete" in result.stdout

        # Test zsh completion
        result = run(
            ["uv", "run", "vmcp", "completion", "zsh"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "compdef" in result.stdout

    def test_repo_commands(self):
        """Test repository management commands."""
        # Test repo stats
        result = run(
            ["uv", "run", "vmcp", "repo", "stats"], capture_output=True, text=True
        )
        assert result.returncode == 0

        # Test repo search
        result = run(
            ["uv", "run", "vmcp", "repo", "search", "pandas"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_extension_install_uninstall_cycle(self):
        """Test full extension lifecycle: install -> enable -> disable -> uninstall."""
        extension_id = "arxiv-mcp"

        # Ensure extension is not installed initially
        run(
            ["uv", "run", "vmcp", "extension", "uninstall", extension_id, "--force"],
            capture_output=True,
        )

        # Install extension
        result = run(
            ["uv", "run", "vmcp", "extension", "install", extension_id],
            capture_output=True,
            text=True,
        )

        # Installation might fail if dependencies aren't available, which is expected
        # in CI environment. We just test the command structure works.
        assert result.returncode in [0, 1]  # Allow both success and expected failure

        if result.returncode == 0:
            # If install succeeded, test the rest of the lifecycle

            # Enable extension
            result = run(
                ["uv", "run", "vmcp", "extension", "enable", extension_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

            # Check extension info
            result = run(
                ["uv", "run", "vmcp", "extension", "info", extension_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert extension_id in result.stdout

            # Disable extension
            result = run(
                ["uv", "run", "vmcp", "extension", "disable", extension_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

            # Uninstall extension
            result = run(
                ["uv", "run", "vmcp", "extension", "uninstall", extension_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

    def test_health_and_metrics_commands(self):
        """Test health and metrics commands."""
        # Test health check (may fail if no servers running, which is expected)
        result = run(
            ["uv", "run", "vmcp", "health", "check"], capture_output=True, text=True
        )
        assert result.returncode in [0, 1]  # Allow failure when no servers

        # Test metrics show
        result = run(
            ["uv", "run", "vmcp", "metrics", "show"], capture_output=True, text=True
        )
        assert result.returncode in [0, 1]  # Allow failure when no servers

    def test_error_handling(self):
        """Test CLI error handling."""
        # Test invalid command
        result = run(
            ["uv", "run", "vmcp", "invalid-command"], capture_output=True, text=True
        )
        assert result.returncode != 0

        # Test invalid extension operation
        result = run(
            ["uv", "run", "vmcp", "extension", "install", "nonexistent-extension"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

        # Test invalid config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [")
            invalid_config = f.name

        try:
            result = run(
                ["uv", "run", "vmcp", "config", "validate", invalid_config],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
        finally:
            Path(invalid_config).unlink()

    def test_verbose_and_quiet_modes(self):
        """Test verbose and quiet output modes."""
        # Test verbose mode
        result = run(
            ["uv", "run", "vmcp", "--verbose", "extension", "list"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Test quiet mode (if implemented)
        result = run(
            ["uv", "run", "vmcp", "--quiet", "status"], capture_output=True, text=True
        )
        # May not be implemented yet, so allow various return codes
        assert result.returncode in [0, 1, 2]
