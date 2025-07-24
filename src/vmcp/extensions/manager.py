"""
Extension Manager for vMCP - DXT-inspired architecture.

Manages installation, enabling/disabling, and lifecycle of vMCP extensions.
"""

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..errors import ExtensionError, RepositoryError

logger = logging.getLogger(__name__)


class ExtensionManager:
    """Manages vMCP extensions following DXT patterns."""

    def __init__(self, vmcp_home: str = "~/.vmcp"):
        """
        Initialize extension manager.

        Args:
            vmcp_home: vMCP home directory path
        """
        self.vmcp_home = Path(vmcp_home).expanduser()
        self.installed_dir = self.vmcp_home / "installed"
        self.enabled_file = self.vmcp_home / "enabled.json"
        self.repositories_dir = Path("repositories")

        # Ensure directories exist
        self.installed_dir.mkdir(parents=True, exist_ok=True)
        self.vmcp_home.mkdir(parents=True, exist_ok=True)

        self._enabled_extensions: dict[str, Any] | None = None

    def list_repositories(self) -> list[dict[str, Any]]:
        """List all available repositories."""
        repositories: list[dict[str, Any]] = []

        if not self.repositories_dir.exists():
            return repositories

        for repo_dir in self.repositories_dir.iterdir():
            if repo_dir.is_dir():
                repo_manifest = repo_dir / "repository.json"
                if repo_manifest.exists():
                    try:
                        with open(repo_manifest) as f:
                            repo_data = json.load(f)
                        repositories.append(repo_data)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load repository {repo_dir.name}: {e}"
                        )

        return repositories

    def list_available_extensions(
        self, repository_id: str = "builtin"
    ) -> list[dict[str, Any]]:
        """
        List all available extensions in a repository.

        Args:
            repository_id: Repository identifier

        Returns:
            List of extension metadata
        """
        repo_dir = self.repositories_dir / repository_id
        repo_manifest = repo_dir / "repository.json"

        if not repo_manifest.exists():
            raise RepositoryError(f"Repository not found: {repository_id}")

        try:
            with open(repo_manifest) as f:
                repo_data = json.load(f)
            extensions = repo_data.get("extensions", [])
            return extensions  # type: ignore[no-any-return]
        except Exception as e:
            raise RepositoryError(
                f"Failed to load repository {repository_id}: {e}"
            ) from e

    def list_installed_extensions(self) -> list[dict[str, Any]]:
        """List all installed extensions."""
        installed: list[dict[str, Any]] = []

        if not self.installed_dir.exists():
            return installed

        for ext_dir in self.installed_dir.iterdir():
            if ext_dir.is_dir():
                manifest_path = ext_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)

                        # Add installation info
                        manifest["installed_path"] = str(ext_dir)
                        manifest["enabled"] = self.is_extension_enabled(
                            manifest.get("name", "")
                        )

                        installed.append(manifest)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load extension manifest {ext_dir.name}: {e}"
                        )

        return installed

    def install_extension(
        self, extension_id: str, repository_id: str = "builtin"
    ) -> bool:
        """
        Install an extension from a repository.

        Args:
            extension_id: Extension identifier
            repository_id: Repository identifier

        Returns:
            True if installation successful
        """
        logger.info(f"Installing extension: {extension_id} from {repository_id}")

        # Find extension in repository
        repo_dir = self.repositories_dir / repository_id
        extensions_dir = repo_dir / "extensions"
        vext_file = extensions_dir / f"{extension_id}.vext"

        if not vext_file.exists():
            raise ExtensionError(f"Extension package not found: {extension_id}")

        # Check if already installed
        install_path = self.installed_dir / extension_id
        if install_path.exists():
            logger.warning(f"Extension {extension_id} already installed, updating...")
            shutil.rmtree(install_path)

        try:
            # Extract .vext package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract package
                with zipfile.ZipFile(vext_file, "r") as zf:
                    zf.extractall(temp_path)

                # Validate manifest
                manifest_path = temp_path / "manifest.json"
                if not manifest_path.exists():
                    raise ExtensionError(
                        "Invalid extension package: missing manifest.json"
                    )

                # Load and validate manifest
                with open(manifest_path) as f:
                    manifest = json.load(f)

                if manifest.get("name") != extension_id:
                    raise ExtensionError(
                        f"Extension ID mismatch: expected {extension_id}, got {manifest.get('name')}"
                    )

                # Copy to installation directory
                shutil.copytree(temp_path, install_path)

                logger.info(f"Successfully installed {extension_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to install {extension_id}: {e}")
            # Cleanup on failure
            if install_path.exists():
                shutil.rmtree(install_path)
            raise ExtensionError(f"Installation failed: {e}") from e

    def uninstall_extension(self, extension_id: str) -> bool:
        """
        Uninstall an extension.

        Args:
            extension_id: Extension identifier

        Returns:
            True if uninstallation successful
        """
        logger.info(f"Uninstalling extension: {extension_id}")

        install_path = self.installed_dir / extension_id
        if not install_path.exists():
            raise ExtensionError(f"Extension not installed: {extension_id}")

        try:
            # Disable if enabled
            if self.is_extension_enabled(extension_id):
                self.disable_extension(extension_id)

            # Remove installation directory
            shutil.rmtree(install_path)

            logger.info(f"Successfully uninstalled {extension_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {extension_id}: {e}")
            raise ExtensionError(f"Uninstallation failed: {e}") from e

    def _load_enabled_extensions(self) -> dict[str, Any]:
        """Load enabled extensions configuration."""
        if self._enabled_extensions is not None:
            return self._enabled_extensions

        if not self.enabled_file.exists():
            # Create default enabled.json
            default_config = {
                "version": "0.1.0",
                "enabled": {},
                "settings": {
                    "auto_enable_on_install": False,
                    "max_concurrent_connections": 10,
                },
            }

            with open(self.enabled_file, "w") as f:
                json.dump(default_config, f, indent=2)

            self._enabled_extensions = default_config
            return default_config

        try:
            with open(self.enabled_file) as f:
                self._enabled_extensions = json.load(f)
            return self._enabled_extensions
        except Exception as e:
            logger.error(f"Failed to load enabled extensions: {e}")
            return {"version": "0.1.0", "enabled": {}, "settings": {}}

    def _save_enabled_extensions(self) -> None:
        """Save enabled extensions configuration."""
        if self._enabled_extensions is None:
            return

        try:
            with open(self.enabled_file, "w") as f:
                json.dump(self._enabled_extensions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save enabled extensions: {e}")
            raise ExtensionError(f"Failed to save configuration: {e}") from e

    def enable_extension(
        self, extension_id: str, config: dict[str, Any] | None = None
    ) -> bool:
        """
        Enable an extension.

        Args:
            extension_id: Extension identifier
            config: Optional extension configuration

        Returns:
            True if successfully enabled
        """
        logger.info(f"Enabling extension: {extension_id}")

        # Check if installed
        install_path = self.installed_dir / extension_id
        if not install_path.exists():
            raise ExtensionError(f"Extension not installed: {extension_id}")

        # Load manifest to validate
        manifest_path = install_path / "manifest.json"
        if not manifest_path.exists():
            raise ExtensionError("Invalid extension: missing manifest.json")

        try:
            with open(manifest_path) as f:
                json.load(f)
        except Exception as e:
            raise ExtensionError(f"Failed to load manifest: {e}") from e

        # Update enabled configuration
        enabled_config = self._load_enabled_extensions()
        enabled_config["enabled"][extension_id] = {
            "enabled": True,
            "config": config or {},
            "manifest_path": str(manifest_path),
            "install_path": str(install_path),
        }

        self._save_enabled_extensions()

        logger.info(f"Successfully enabled {extension_id}")
        return True

    def disable_extension(self, extension_id: str) -> bool:
        """
        Disable an extension.

        Args:
            extension_id: Extension identifier

        Returns:
            True if successfully disabled
        """
        logger.info(f"Disabling extension: {extension_id}")

        enabled_config = self._load_enabled_extensions()

        if extension_id not in enabled_config.get("enabled", {}):
            logger.warning(f"Extension {extension_id} is not enabled")
            return True

        # Remove from enabled configuration
        del enabled_config["enabled"][extension_id]
        self._save_enabled_extensions()

        logger.info(f"Successfully disabled {extension_id}")
        return True

    def is_extension_enabled(self, extension_id: str) -> bool:
        """
        Check if an extension is enabled.

        Args:
            extension_id: Extension identifier

        Returns:
            True if extension is enabled
        """
        enabled_config = self._load_enabled_extensions()
        return extension_id in enabled_config.get("enabled", {})

    def get_enabled_extensions(self) -> dict[str, dict[str, Any]]:
        """
        Get all enabled extensions.

        Returns:
            Dictionary of enabled extensions with their configurations
        """
        enabled_config = self._load_enabled_extensions()
        enabled = enabled_config.get("enabled", {})
        return enabled  # type: ignore[no-any-return]

    def get_extension_manifest(self, extension_id: str) -> dict[str, Any] | None:
        """
        Get extension manifest.

        Args:
            extension_id: Extension identifier

        Returns:
            Extension manifest or None if not found
        """
        install_path = self.installed_dir / extension_id
        manifest_path = install_path / "manifest.json"

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"Failed to load manifest for {extension_id}: {e}")
            return None

    def update_extension(
        self, extension_id: str, repository_id: str = "builtin"
    ) -> bool:
        """
        Update an extension to the latest version.

        Args:
            extension_id: Extension identifier
            repository_id: Repository identifier

        Returns:
            True if update successful
        """
        logger.info(f"Updating extension: {extension_id}")

        # Check if installed
        if not (self.installed_dir / extension_id).exists():
            raise ExtensionError(f"Extension not installed: {extension_id}")

        # Store enabled state and config
        was_enabled = self.is_extension_enabled(extension_id)
        enabled_config = self._load_enabled_extensions()
        ext_config = (
            enabled_config.get("enabled", {}).get(extension_id, {}).get("config", {})
        )

        try:
            # Reinstall
            self.install_extension(extension_id, repository_id)

            # Restore enabled state
            if was_enabled:
                self.enable_extension(extension_id, ext_config)

            logger.info(f"Successfully updated {extension_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update {extension_id}: {e}")
            raise ExtensionError(f"Update failed: {e}") from e
