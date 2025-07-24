"""
MCP Server Installation System.

This module provides installation, update, and management capabilities
for MCP servers from various sources.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import InstallationFailedError
from .discovery import MCPServerInfo

logger = logging.getLogger(__name__)


class MCPInstaller:
    """MCP Server Installation Manager."""

    def __init__(self, install_dir: str | None = None):
        """
        Initialize MCP installer.

        Args:
            install_dir: Directory for installing servers
        """
        self.install_dir = Path(install_dir or Path.home() / ".vmcp" / "servers")
        self.install_dir.mkdir(parents=True, exist_ok=True)

        # Installation metadata
        self.metadata_file = self.install_dir / "installations.json"
        self.installations: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize the installer."""
        logger.info(f"Initializing MCP installer (install_dir: {self.install_dir})")

        # Load existing installation metadata
        self._load_metadata()

        # Verify UV is available
        if not await self._verify_uv():
            raise InstallationFailedError(
                "uv", "UV package manager not found. Please install UV."
            )

    async def install_server(self, server_info: MCPServerInfo) -> str:
        """
        Install an MCP server.

        Args:
            server_info: Server information

        Returns:
            Installation path
        """
        server_id = server_info.id
        logger.info(f"Installing MCP server: {server_id}")

        # Check if already installed
        if await self.is_installed(server_id):
            logger.info(f"Server {server_id} already installed, updating instead")
            return await self.update_server(server_info)

        install_path = self.install_dir / server_id

        try:
            # Create installation directory
            install_path.mkdir(exist_ok=True)

            # Install based on source type
            if server_info.source_type == "local":
                await self._install_local(server_info, install_path)
            elif server_info.source_type == "git":
                await self._install_git(server_info, install_path)
            elif server_info.source_type == "pypi":
                await self._install_pypi(server_info, install_path)
            else:
                raise InstallationFailedError(
                    server_info.id,
                    f"Unsupported source type: {server_info.source_type}",
                )

            # Record installation
            self._record_installation(server_id, server_info, str(install_path))

            logger.info(f"Successfully installed {server_id} to {install_path}")
            return str(install_path)

        except Exception as e:
            # Cleanup on failure
            if install_path.exists():
                shutil.rmtree(install_path, ignore_errors=True)
            raise InstallationFailedError(
                server_id, f"Installation failed for {server_id}: {e}"
            ) from e

    async def update_server(self, server_info: MCPServerInfo) -> str:
        """
        Update an installed MCP server.

        Args:
            server_info: Server information

        Returns:
            Installation path
        """
        server_id = server_info.id
        logger.info(f"Updating MCP server: {server_id}")

        if not await self.is_installed(server_id):
            raise InstallationFailedError(
                server_id, f"Server not installed: {server_id}"
            )

        install_path = self.install_dir / server_id

        try:
            # Update based on source type
            if server_info.source_type == "local":
                await self._update_local(server_info, install_path)
            elif server_info.source_type == "git":
                await self._update_git(server_info, install_path)
            elif server_info.source_type == "pypi":
                await self._update_pypi(server_info, install_path)
            else:
                raise InstallationFailedError(
                    server_id, f"Unsupported source type: {server_info.source_type}"
                )

            # Update installation record
            self._record_installation(
                server_id, server_info, str(install_path), is_update=True
            )

            logger.info(f"Successfully updated {server_id}")
            return str(install_path)

        except Exception as e:
            raise InstallationFailedError(
                server_id, f"Update failed for {server_id}: {e}"
            ) from e

    async def uninstall_server(self, server_id: str) -> bool:
        """
        Uninstall an MCP server.

        Args:
            server_id: Server identifier

        Returns:
            True if successful
        """
        logger.info(f"Uninstalling MCP server: {server_id}")

        if not await self.is_installed(server_id):
            logger.warning(f"Server not installed: {server_id}")
            return True

        install_path = self.install_dir / server_id

        try:
            # Remove installation directory
            if install_path.exists():
                shutil.rmtree(install_path)

            # Remove from installation records
            if server_id in self.installations:
                del self.installations[server_id]
                self._save_metadata()

            logger.info(f"Successfully uninstalled {server_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {server_id}: {e}")
            raise InstallationFailedError(
                server_id, f"Uninstallation failed: {e}"
            ) from e

    async def is_installed(self, server_id: str) -> bool:
        """
        Check if a server is installed.

        Args:
            server_id: Server identifier

        Returns:
            True if installed
        """
        return (
            server_id in self.installations and (self.install_dir / server_id).exists()
        )

    async def list_installed(self) -> dict[str, dict[str, Any]]:
        """
        List all installed servers.

        Returns:
            Dictionary of installed servers and their metadata
        """
        # Verify installations still exist
        valid_installations = {}

        for server_id, metadata in self.installations.items():
            install_path = Path(metadata.get("path", self.install_dir / server_id))
            if install_path.exists():
                valid_installations[server_id] = metadata
            else:
                logger.warning(f"Installation missing for {server_id}: {install_path}")

        # Update installations if any were removed
        if len(valid_installations) != len(self.installations):
            self.installations = valid_installations
            self._save_metadata()

        return valid_installations

    async def verify_installation(self, server_id: str) -> dict[str, Any]:
        """
        Verify an installation and return status.

        Args:
            server_id: Server identifier

        Returns:
            Verification results
        """
        if not await self.is_installed(server_id):
            return {
                "status": "not_installed",
                "installed": False,
                "executable": False,
                "dependencies": False,
            }

        install_path = self.install_dir / server_id
        metadata = self.installations.get(server_id, {})

        results = {
            "status": "installed",
            "installed": True,
            "path": str(install_path),
            "metadata": metadata,
        }

        # Check if executable exists
        try:
            # Try to run the server with --help to verify it's working
            cmd = [
                "uv",
                "run",
                "--directory",
                str(install_path),
                "python",
                "-c",
                "import sys; print('OK')",
            ]
            result = await self._run_command(cmd, timeout=10)
            results["executable"] = result.returncode == 0

        except Exception as e:
            logger.debug(f"Executable check failed for {server_id}: {e}")
            results["executable"] = False

        # Check dependencies
        try:
            # Check if uv.lock exists and dependencies are satisfied
            lock_file = install_path / "uv.lock"
            results["dependencies"] = lock_file.exists()

        except Exception:
            results["dependencies"] = False

        # Determine overall status
        if results["executable"] and results["dependencies"]:
            results["status"] = "healthy"
        elif results["executable"]:
            results["status"] = "degraded"
        else:
            results["status"] = "broken"

        return results

    def get_install_stats(self) -> dict[str, Any]:
        """
        Get installation statistics.

        Returns:
            Installation statistics
        """
        total_installed = len(self.installations)

        # Count by source type
        source_types: dict[str, int] = {}
        install_dates = []

        for metadata in self.installations.values():
            source_type = metadata.get("source_type", "unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1

            if "install_date" in metadata:
                install_dates.append(metadata["install_date"])

        return {
            "total_installed": total_installed,
            "source_types": source_types,
            "install_directory": str(self.install_dir),
            "disk_usage": self._calculate_disk_usage(),
            "recent_installations": len(
                [d for d in install_dates if self._is_recent(d)]
            ),
        }

    async def _install_local(
        self, server_info: MCPServerInfo, install_path: Path
    ) -> None:
        """Install from local directory."""
        source_path = Path(server_info.source_location)

        if not source_path.exists():
            raise InstallationFailedError(
                server_info.id, f"Source directory not found: {source_path}"
            )

        # Copy source to installation directory
        if source_path.is_dir():
            # Copy entire directory
            shutil.copytree(source_path, install_path, dirs_exist_ok=True)
        else:
            raise InstallationFailedError(
                server_info.id, f"Source is not a directory: {source_path}"
            )

        # Install dependencies using UV
        await self._install_dependencies(install_path)

    async def _install_git(
        self, server_info: MCPServerInfo, install_path: Path
    ) -> None:
        """Install from Git repository."""
        import git

        repo_url = server_info.source_location

        # Clone repository
        try:
            git.Repo.clone_from(repo_url, install_path)
        except Exception as e:
            raise InstallationFailedError(
                server_info.id, f"Git clone failed: {e}"
            ) from e

        # Install dependencies
        await self._install_dependencies(install_path)

    async def _install_pypi(
        self, server_info: MCPServerInfo, install_path: Path
    ) -> None:
        """Install from PyPI."""
        # Create a basic project structure
        install_path.mkdir(exist_ok=True)

        # Create pyproject.toml with the package dependency
        pyproject_content = f"""[project]
name = "{server_info.id}"
version = "{server_info.version}"
description = "{server_info.description}"
dependencies = ["{server_info.id}"]

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"
"""

        (install_path / "pyproject.toml").write_text(pyproject_content)

        # Install dependencies
        await self._install_dependencies(install_path)

    async def _update_local(
        self, server_info: MCPServerInfo, install_path: Path
    ) -> None:
        """Update local installation."""
        # For local installations, copy updated source
        await self._install_local(server_info, install_path)

    async def _update_git(self, server_info: MCPServerInfo, install_path: Path) -> None:
        """Update Git installation."""
        import git

        try:
            repo = git.Repo(install_path)
            repo.remotes.origin.pull()
        except Exception as e:
            logger.warning(f"Git pull failed: {e}, reinstalling instead")
            shutil.rmtree(install_path)
            await self._install_git(server_info, install_path)
            return

        # Reinstall dependencies
        await self._install_dependencies(install_path)

    async def _update_pypi(
        self, server_info: MCPServerInfo, install_path: Path
    ) -> None:
        """Update PyPI installation."""
        # For PyPI packages, reinstall with latest version
        await self._install_pypi(server_info, install_path)

    async def _install_dependencies(self, install_path: Path) -> None:
        """Install dependencies using UV."""
        pyproject_file = install_path / "pyproject.toml"

        if not pyproject_file.exists():
            logger.warning(f"No pyproject.toml found in {install_path}")
            return

        # Use UV to sync dependencies
        cmd = ["uv", "sync", "--directory", str(install_path)]

        result = await self._run_command(cmd, timeout=300)  # 5 minute timeout

        if result.returncode != 0:
            raise InstallationFailedError(
                "dependencies", f"Dependency installation failed: {result.stderr}"
            )

    async def _verify_uv(self) -> bool:
        """Verify UV is available."""
        try:
            result = await self._run_command(["uv", "--version"], timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    async def _run_command(
        self, cmd: list[str], timeout: int = 30, cwd: str | None = None
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        logger.debug(f"Running command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode or 0,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
            )

        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise InstallationFailedError(
                "command", f"Command timed out: {' '.join(cmd)}"
            ) from e

    def _record_installation(
        self,
        server_id: str,
        server_info: MCPServerInfo,
        install_path: str,
        is_update: bool = False,
    ) -> None:
        """Record installation metadata."""
        now = datetime.now().isoformat()

        metadata = {
            "id": server_id,
            "name": server_info.name,
            "version": server_info.version,
            "description": server_info.description,
            "source_type": server_info.source_type,
            "source_location": server_info.source_location,
            "path": install_path,
            "install_date": self.installations.get(server_id, {}).get(
                "install_date", now
            ),
            "update_date": now if is_update else now,
            "entry_point": server_info.entry_point,
            "capabilities": server_info.capabilities,
        }

        self.installations[server_id] = metadata
        self._save_metadata()

    def _load_metadata(self) -> None:
        """Load installation metadata."""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file) as f:
                self.installations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load installation metadata: {e}")
            self.installations = {}

    def _save_metadata(self) -> None:
        """Save installation metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.installations, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save installation metadata: {e}")

    def _calculate_disk_usage(self) -> int:
        """Calculate total disk usage of installations."""
        total_size = 0

        for install_path in self.install_dir.iterdir():
            if install_path.is_dir():
                try:
                    for file_path in install_path.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                except Exception:
                    pass

        return total_size

    def _is_recent(self, date_str: str, days: int = 7) -> bool:
        """Check if date is within recent days."""
        try:
            install_date = datetime.fromisoformat(date_str)
            now = datetime.now()
            return (now - install_date).days <= days
        except Exception:
            return False
