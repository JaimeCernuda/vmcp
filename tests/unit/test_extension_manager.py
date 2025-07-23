"""Tests for extension management functionality."""

import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vmcp.extensions.manager import ExtensionManager
from vmcp.errors import ExtensionError


class TestExtensionManager:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def extension_manager(self, temp_dir):
        """Create an extension manager with temporary directories."""
        config_path = temp_dir / "config.toml"
        registry_path = temp_dir / "registry"
        extensions_path = temp_dir / "extensions"
        repository_path = temp_dir / "repository"
        
        # Create necessary directories
        registry_path.mkdir()
        extensions_path.mkdir()
        repository_path.mkdir()
        
        with patch("vmcp.extensions.manager.load_config") as mock_load:
            mock_config = Mock()
            mock_config.gateway.registry_path = str(registry_path)
            mock_load.return_value = mock_config
            
            manager = ExtensionManager(
                config_path=str(config_path),
                extensions_path=str(extensions_path),
                repository_path=str(repository_path)
            )
            yield manager

    @pytest.fixture
    def sample_manifest(self):
        """Sample extension manifest."""
        return {
            "$schema": "https://vmcp.dev/schemas/manifest.json",
            "vmcp_version": "0.1.0",
            "name": "test-extension",
            "display_name": "Test Extension",
            "version": "1.0.0",
            "description": "A test extension",
            "author": {
                "name": "Test Author",
                "email": "test@example.com"
            },
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool"
                }
            ],
            "server": {
                "type": "python",
                "mcp": {
                    "command": "python",
                    "args": ["server.py"]
                }
            }
        }

    @pytest.fixture
    def sample_vext_file(self, temp_dir, sample_manifest):
        """Create a sample .vext file."""
        vext_path = temp_dir / "test-extension.vext"
        
        with zipfile.ZipFile(vext_path, 'w') as zf:
            # Add manifest
            zf.writestr("manifest.json", json.dumps(sample_manifest, indent=2))
            # Add a simple server file
            zf.writestr("src/server.py", "# Test server")
            # Add README
            zf.writestr("README.md", "# Test Extension")
        
        return vext_path

    def test_list_available_extensions(self, extension_manager, temp_dir):
        """Test listing available extensions from repository."""
        # Create repository.json
        repo_data = {
            "extensions": [
                {
                    "id": "test-ext",
                    "name": "Test Extension",
                    "version": "1.0.0",
                    "description": "Test extension"
                }
            ]
        }
        
        repo_file = temp_dir / "repository" / "repository.json"
        with open(repo_file, 'w') as f:
            json.dump(repo_data, f)
        
        extensions = extension_manager.list_available_extensions()
        assert len(extensions) == 1
        assert extensions[0]["id"] == "test-ext"

    def test_list_installed_extensions(self, extension_manager, temp_dir):
        """Test listing installed extensions."""
        # Create an installed extension directory with manifest
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        
        manifest = {
            "name": "test-ext",
            "version": "1.0.0"
        }
        
        with open(ext_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        installed = extension_manager.list_installed_extensions()
        assert len(installed) == 1
        assert installed[0]["name"] == "test-ext"

    def test_install_extension_success(self, extension_manager, sample_vext_file, sample_manifest):
        """Test successful extension installation."""
        # Mock repository lookup
        with patch.object(extension_manager, '_find_extension_package') as mock_find:
            mock_find.return_value = str(sample_vext_file)
            
            result = extension_manager.install_extension("test-extension")
            assert result is True

    def test_install_extension_not_found(self, extension_manager):
        """Test installing non-existent extension."""
        with patch.object(extension_manager, '_find_extension_package') as mock_find:
            mock_find.return_value = None
            
            with pytest.raises(ExtensionError, match="Extension .* not found"):
                extension_manager.install_extension("nonexistent")

    def test_install_extension_invalid_vext(self, extension_manager, temp_dir):
        """Test installing invalid .vext file."""
        # Create invalid .vext file
        invalid_vext = temp_dir / "invalid.vext"
        with open(invalid_vext, 'w') as f:
            f.write("not a zip file")
        
        with patch.object(extension_manager, '_find_extension_package') as mock_find:
            mock_find.return_value = str(invalid_vext)
            
            with pytest.raises(ExtensionError, match="Invalid extension package"):
                extension_manager.install_extension("invalid")

    def test_enable_extension(self, extension_manager, temp_dir):
        """Test enabling an installed extension."""
        # Create installed extension
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        
        manifest = {
            "name": "test-ext",
            "version": "1.0.0",
            "server": {
                "type": "python",
                "mcp": {
                    "command": "python",
                    "args": ["server.py"]
                }
            }
        }
        
        with open(ext_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Mock registry
        with patch.object(extension_manager, 'registry') as mock_registry:
            mock_registry.add_server.return_value = True
            
            result = extension_manager.enable_extension("test-ext")
            assert result is True
            mock_registry.add_server.assert_called_once()

    def test_enable_extension_not_installed(self, extension_manager):
        """Test enabling non-installed extension."""
        with pytest.raises(ExtensionError, match="Extension .* is not installed"):
            extension_manager.enable_extension("nonexistent")

    def test_disable_extension(self, extension_manager, temp_dir):
        """Test disabling an enabled extension."""
        # Mock registry with enabled extension
        with patch.object(extension_manager, 'registry') as mock_registry:
            mock_registry.get_server.return_value = {"id": "test-ext", "enabled": True}
            mock_registry.remove_server.return_value = True
            
            result = extension_manager.disable_extension("test-ext")
            assert result is True
            mock_registry.remove_server.assert_called_once_with("test-ext")

    def test_uninstall_extension_enabled(self, extension_manager, temp_dir):
        """Test uninstalling enabled extension without force."""
        # Create installed extension
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        
        # Mock registry showing extension is enabled
        with patch.object(extension_manager, 'registry') as mock_registry:
            mock_registry.get_server.return_value = {"id": "test-ext", "enabled": True}
            
            with pytest.raises(ExtensionError, match="Cannot uninstall enabled extension"):
                extension_manager.uninstall_extension("test-ext", force=False)

    def test_uninstall_extension_force(self, extension_manager, temp_dir):
        """Test force uninstalling enabled extension."""
        # Create installed extension
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        
        # Add a file to verify deletion
        (ext_dir / "test.txt").write_text("test")
        
        # Mock registry
        with patch.object(extension_manager, 'registry') as mock_registry:
            mock_registry.get_server.return_value = {"id": "test-ext", "enabled": True}
            mock_registry.remove_server.return_value = True
            
            result = extension_manager.uninstall_extension("test-ext", force=True)
            assert result is True
            assert not ext_dir.exists()

    def test_get_extension_info(self, extension_manager, temp_dir):
        """Test getting extension information."""
        # Create installed extension
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        
        manifest = {
            "name": "test-ext",
            "version": "1.0.0",
            "description": "Test extension"
        }
        
        with open(ext_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Mock registry
        with patch.object(extension_manager, 'registry') as mock_registry:
            mock_registry.get_server.return_value = {"id": "test-ext", "enabled": True}
            
            info = extension_manager.get_extension_info("test-ext")
            assert info["name"] == "test-ext"
            assert info["enabled"] is True

    def test_update_extension(self, extension_manager, sample_vext_file):
        """Test updating an extension."""
        # Mock installed extension
        with patch.object(extension_manager, 'is_installed') as mock_installed:
            mock_installed.return_value = True
            
            with patch.object(extension_manager, '_find_extension_package') as mock_find:
                mock_find.return_value = str(sample_vext_file)
                
                with patch.object(extension_manager, 'uninstall_extension') as mock_uninstall:
                    with patch.object(extension_manager, 'install_extension') as mock_install:
                        mock_uninstall.return_value = True
                        mock_install.return_value = True
                        
                        result = extension_manager.update_extension("test-extension")
                        assert result is True
                        mock_uninstall.assert_called_once()
                        mock_install.assert_called_once()

    def test_validate_manifest(self, extension_manager, sample_manifest):
        """Test manifest validation."""
        # Valid manifest should pass
        assert extension_manager._validate_manifest(sample_manifest) is True
        
        # Invalid manifest should fail
        invalid_manifest = {"name": "test"}  # Missing required fields
        assert extension_manager._validate_manifest(invalid_manifest) is False

    def test_is_installed(self, extension_manager, temp_dir):
        """Test checking if extension is installed."""
        # Non-existent extension
        assert extension_manager.is_installed("nonexistent") is False
        
        # Create installed extension
        ext_dir = temp_dir / "extensions" / "installed" / "test-ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "manifest.json").write_text("{}")
        
        assert extension_manager.is_installed("test-ext") is True