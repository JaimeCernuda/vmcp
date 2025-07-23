"""Tests for CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from vmcp.cli.main import cli, status_cmd, list_cmd
from vmcp.errors import VMCPError


class TestCLI:
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write('''
version = "0.1.0"

[gateway]
registry_path = "~/.vmcp/test_registry"
log_level = "INFO"

[transports.stdio]
enabled = true
            ''')
            return f.name

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Virtual Model Context Protocol' in result.output

    def test_status_command_not_running(self, runner):
        """Test status command when gateway is not running."""
        with patch('vmcp.cli.main.check_gateway_status') as mock_check:
            mock_check.return_value = False
            
            result = runner.invoke(status_cmd)
            assert result.exit_code == 0
            assert 'not running' in result.output.lower()

    def test_status_command_running(self, runner):
        """Test status command when gateway is running."""
        with patch('vmcp.cli.main.check_gateway_status') as mock_check:
            with patch('vmcp.cli.main.get_gateway_info') as mock_info:
                mock_check.return_value = True
                mock_info.return_value = {
                    "status": "running",
                    "uptime": "1h 30m",
                    "servers": 3,
                    "requests": 150
                }
                
                result = runner.invoke(status_cmd)
                assert result.exit_code == 0
                assert 'running' in result.output.lower()

    def test_list_command_empty(self, runner):
        """Test list command with no servers."""
        with patch('vmcp.registry.registry.ServerRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.list_servers.return_value = []
            mock_registry_class.return_value = mock_registry
            
            result = runner.invoke(list_cmd)
            assert result.exit_code == 0
            assert 'No servers registered' in result.output

    def test_list_command_with_servers(self, runner):
        """Test list command with servers."""
        from vmcp.registry.registry import ServerInfo, ServerState
        
        # Create mock servers
        server1 = ServerInfo(
            id="server1",
            name="Server 1",
            transport="stdio",
            command="python",
            args=["server.py"]
        )
        server1.state.status = "healthy"
        
        server2 = ServerInfo(
            id="server2", 
            name="Server 2",
            transport="http",
            command="node",
            args=["server.js"]
        )
        server2.state.status = "unhealthy"
        
        with patch('vmcp.registry.registry.ServerRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.list_servers.return_value = [server1, server2]
            mock_registry_class.return_value = mock_registry
            
            result = runner.invoke(list_cmd)
            assert result.exit_code == 0
            assert 'server1' in result.output
            assert 'server2' in result.output
            assert 'healthy' in result.output
            assert 'unhealthy' in result.output

    def test_config_init_command(self, runner):
        """Test config init command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.toml"
            
            result = runner.invoke(cli, ['config', 'init', '--output', str(config_path)])
            assert result.exit_code == 0
            assert config_path.exists()

    def test_config_validate_command_valid(self, runner, temp_config):
        """Test config validate command with valid config."""
        try:
            result = runner.invoke(cli, ['config', 'validate', temp_config])
            assert result.exit_code == 0
            assert 'valid' in result.output.lower()
        finally:
            Path(temp_config).unlink()

    def test_config_validate_command_invalid(self, runner):
        """Test config validate command with invalid config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("invalid toml content [unclosed")
            invalid_config = f.name
        
        try:
            result = runner.invoke(cli, ['config', 'validate', invalid_config])
            assert result.exit_code != 0
        finally:
            Path(invalid_config).unlink()

    def test_extension_list_command(self, runner):
        """Test extension list command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.list_available_extensions.return_value = [
                {
                    "id": "test-ext",
                    "name": "Test Extension",
                    "version": "1.0.0",
                    "enabled": True
                }
            ]
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'list'])
            assert result.exit_code == 0
            assert 'test-ext' in result.output

    def test_extension_install_command(self, runner):
        """Test extension install command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.install_extension.return_value = True
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'install', 'test-ext'])
            assert result.exit_code == 0
            mock_manager.install_extension.assert_called_once_with('test-ext', 'builtin')

    def test_extension_install_command_failure(self, runner):
        """Test extension install command failure."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.install_extension.side_effect = Exception("Installation failed")
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'install', 'test-ext'])
            assert result.exit_code != 0
            assert 'failed' in result.output.lower()

    def test_extension_enable_command(self, runner):
        """Test extension enable command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.enable_extension.return_value = True
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'enable', 'test-ext'])
            assert result.exit_code == 0
            mock_manager.enable_extension.assert_called_once_with('test-ext', None)

    def test_extension_disable_command(self, runner):
        """Test extension disable command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.disable_extension.return_value = True
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'disable', 'test-ext'])
            assert result.exit_code == 0
            mock_manager.disable_extension.assert_called_once_with('test-ext')

    def test_extension_uninstall_command(self, runner):
        """Test extension uninstall command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.uninstall_extension.return_value = True
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'uninstall', 'test-ext'])
            assert result.exit_code == 0
            mock_manager.uninstall_extension.assert_called_once_with('test-ext', False)

    def test_extension_info_command(self, runner):
        """Test extension info command."""
        with patch('vmcp.extensions.manager.ExtensionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_extension_info.return_value = {
                "id": "test-ext",
                "name": "Test Extension",
                "version": "1.0.0",
                "description": "A test extension",
                "enabled": True
            }
            mock_manager_class.return_value = mock_manager
            
            result = runner.invoke(cli, ['extension', 'info', 'test-ext'])
            assert result.exit_code == 0
            assert 'test-ext' in result.output
            assert 'Test Extension' in result.output

    def test_start_command(self, runner):
        """Test start command."""
        with patch('vmcp.cli.main.start_gateway') as mock_start:
            mock_start.return_value = True
            
            result = runner.invoke(cli, ['start'])
            assert result.exit_code == 0
            mock_start.assert_called_once()

    def test_stop_command(self, runner):
        """Test stop command."""
        with patch('vmcp.cli.main.stop_gateway') as mock_stop:
            mock_stop.return_value = True
            
            result = runner.invoke(cli, ['stop'])
            assert result.exit_code == 0
            mock_stop.assert_called_once()

    def test_health_check_command(self, runner):
        """Test health check command."""
        with patch('vmcp.monitoring.health.HealthChecker') as mock_health_class:
            mock_health = Mock()
            mock_health.check_all_servers.return_value = {
                "overall_status": "healthy", 
                "servers": {
                    "server1": {"status": "healthy", "latency": 50}
                }
            }
            mock_health_class.return_value = mock_health
            
            result = runner.invoke(cli, ['health', 'check'])
            assert result.exit_code == 0
            assert 'healthy' in result.output.lower()

    def test_metrics_show_command(self, runner):
        """Test metrics show command."""
        with patch('vmcp.monitoring.metrics.MetricsCollector') as mock_metrics_class:
            mock_metrics = Mock()
            mock_metrics.get_all_metrics.return_value = {
                "requests_total": 100,
                "requests_per_second": 2.5,
                "active_connections": 5
            }
            mock_metrics_class.return_value = mock_metrics
            
            result = runner.invoke(cli, ['metrics', 'show'])
            assert result.exit_code == 0
            assert '100' in result.output

    def test_completion_bash_command(self, runner):
        """Test bash completion generation."""
        result = runner.invoke(cli, ['completion', 'bash'])
        assert result.exit_code == 0
        assert 'complete' in result.output

    def test_completion_zsh_command(self, runner):
        """Test zsh completion generation.""" 
        result = runner.invoke(cli, ['completion', 'zsh'])
        assert result.exit_code == 0
        assert 'compdef' in result.output

    def test_repo_search_command(self, runner):
        """Test repository search command."""
        with patch('vmcp.repository.discovery.RepositoryManager') as mock_repo_class:
            mock_repo = Mock()
            mock_repo.search_extensions.return_value = [
                {"id": "pandas-mcp", "name": "Pandas MCP", "description": "Data analysis"}
            ]
            mock_repo_class.return_value = mock_repo
            
            result = runner.invoke(cli, ['repo', 'search', 'pandas'])
            assert result.exit_code == 0
            assert 'pandas-mcp' in result.output

    def test_repo_stats_command(self, runner):
        """Test repository stats command."""
        with patch('vmcp.repository.discovery.RepositoryManager') as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get_stats.return_value = {
                "total_extensions": 14,
                "installed_extensions": 3,
                "enabled_extensions": 2
            }
            mock_repo_class.return_value = mock_repo
            
            result = runner.invoke(cli, ['repo', 'stats'])
            assert result.exit_code == 0
            assert '14' in result.output