"""
Repository Manager for vMCP.

This module provides centralized management of MCP server repositories,
including discovery, installation, and integration with the vMCP gateway.
"""

import logging
from pathlib import Path
from typing import Any

from ..errors import RepositoryError
from ..registry.registry import MCPServerConfig, Registry
from .discovery import MCPDiscovery, MCPServerInfo
from .installer import MCPInstaller

logger = logging.getLogger(__name__)


class RepositoryManager:
    """Central manager for MCP server repositories."""

    def __init__(
        self,
        registry: Registry,
        cache_dir: str | None = None,
        install_dir: str | None = None,
    ):
        """
        Initialize repository manager.

        Args:
            registry: vMCP server registry
            cache_dir: Directory for caching repository data
            install_dir: Directory for installing servers
        """
        self.registry = registry
        self.discovery = MCPDiscovery(cache_dir)
        self.installer = MCPInstaller(install_dir)

        # Repository configuration
        self.repositories: dict[str, str] = {
            "iowarp-mcps": "https://github.com/iowarp/iowarp-mcps.git",
            "official-mcps": "https://github.com/modelcontextprotocol/servers.git",
        }

    async def initialize(self) -> None:
        """Initialize the repository manager."""
        logger.info("Initializing repository manager")

        # Ensure installer is ready
        await self.installer.initialize()

        # Run initial discovery
        await self.discover_servers()

    async def discover_servers(
        self, sources: list[str] | None = None, refresh: bool = False
    ) -> dict[str, MCPServerInfo]:
        """
        Discover MCP servers from configured sources.

        Args:
            sources: Custom sources to discover from
            refresh: Whether to refresh cached results

        Returns:
            Dictionary of discovered servers
        """
        if sources is None:
            sources = self._get_discovery_sources()

        return await self.discovery.discover_all(sources, refresh)

    async def search_servers(
        self,
        query: str = "",
        tags: list[str] | None = None,
        capabilities: list[str] | None = None,
        installed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for MCP servers.

        Args:
            query: Search query
            tags: Required tags
            capabilities: Required capabilities
            installed_only: Only return installed servers

        Returns:
            List of server information with installation status
        """
        # Get search results from discovery
        servers = await self.discovery.search_servers(query, tags, capabilities)

        # Add installation status
        results = []
        for server in servers:
            server_dict = server.to_dict()

            # Check installation status
            is_installed = await self.installer.is_installed(server.id)
            server_dict["installed"] = is_installed

            # Check registry status
            is_registered = self.registry.get_server(server.id) is not None
            server_dict["registered"] = is_registered

            # Skip if only showing installed servers
            if installed_only and not is_installed:
                continue

            results.append(server_dict)

        return results

    async def install_server(
        self, server_id: str, register: bool = True, enable: bool = True
    ) -> bool:
        """
        Install and optionally register an MCP server.

        Args:
            server_id: Server identifier
            register: Whether to register with vMCP registry
            enable: Whether to enable the server

        Returns:
            True if successful
        """
        # Get server information
        server_info = await self.discovery.get_server_details(server_id)
        if not server_info:
            raise RepositoryError(f"Server not found: {server_id}")

        logger.info(f"Installing MCP server: {server_id}")

        try:
            # Install the server
            install_path = await self.installer.install_server(server_info)

            # Register with vMCP if requested
            if register:
                await self._register_installed_server(server_info, install_path, enable)

            logger.info(f"Successfully installed {server_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to install {server_id}: {e}")
            raise RepositoryError(f"Installation failed: {e}")

    async def uninstall_server(self, server_id: str, unregister: bool = True) -> bool:
        """
        Uninstall and optionally unregister an MCP server.

        Args:
            server_id: Server identifier
            unregister: Whether to unregister from vMCP registry

        Returns:
            True if successful
        """
        logger.info(f"Uninstalling MCP server: {server_id}")

        try:
            # Unregister from vMCP if requested
            if unregister:
                await self.registry.unregister_server(server_id)

            # Uninstall the server
            await self.installer.uninstall_server(server_id)

            logger.info(f"Successfully uninstalled {server_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {server_id}: {e}")
            raise RepositoryError(f"Uninstallation failed: {e}")

    async def update_server(self, server_id: str) -> bool:
        """
        Update an installed MCP server.

        Args:
            server_id: Server identifier

        Returns:
            True if successful
        """
        logger.info(f"Updating MCP server: {server_id}")

        try:
            # Check if server is installed
            if not await self.installer.is_installed(server_id):
                raise RepositoryError(f"Server not installed: {server_id}")

            # Get current server info
            server_info = await self.discovery.get_server_details(server_id)
            if not server_info:
                raise RepositoryError(f"Server not found in discovery: {server_id}")

            # Update the installation
            install_path = await self.installer.update_server(server_info)

            # Update registry configuration if registered
            server_config = self.registry.get_server(server_id)
            if server_config:
                # Update command path if needed
                new_config = MCPServerConfig(
                    id=server_config.config.id,
                    name=server_info.name,
                    transport=server_config.config.transport,
                    command=self._generate_command(server_info, install_path),
                    capabilities=server_info.capabilities,
                    enabled=server_config.config.enabled,
                )
                await self.registry.update_server(server_id, new_config)

            logger.info(f"Successfully updated {server_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update {server_id}: {e}")
            raise RepositoryError(f"Update failed: {e}")

    async def list_installed(self) -> list[dict[str, Any]]:
        """
        List all installed MCP servers.

        Returns:
            List of installed server information
        """
        installed_servers = await self.installer.list_installed()
        results = []

        for server_id, install_info in installed_servers.items():
            # Get discovery information if available
            server_info = await self.discovery.get_server_details(server_id)

            # Check registry status
            is_registered = self.registry.get_server(server_id) is not None

            result = {
                "id": server_id,
                "installed": True,
                "registered": is_registered,
                "install_path": install_info.get("path"),
                "install_date": install_info.get("date"),
                "version": install_info.get("version"),
            }

            # Add discovery info if available
            if server_info:
                result.update(
                    {
                        "name": server_info.name,
                        "description": server_info.description,
                        "capabilities": server_info.capabilities,
                        "source_type": server_info.source_type,
                    }
                )
            else:
                result.update(
                    {
                        "name": server_id,
                        "description": "Installed server (no discovery info)",
                        "capabilities": {},
                        "source_type": "unknown",
                    }
                )

            results.append(result)

        return results

    async def auto_register_iowarp_mcps(self, mcps_directory: str) -> int:
        """
        Automatically discover and register iowarp-mcps servers.

        Args:
            mcps_directory: Path to iowarp-mcps directory

        Returns:
            Number of servers registered
        """
        logger.info(f"Auto-registering iowarp-mcps from: {mcps_directory}")

        mcps_path = Path(mcps_directory)
        if not mcps_path.exists():
            raise RepositoryError(f"iowarp-mcps directory not found: {mcps_directory}")

        # Discover servers in the directory
        servers = await self.discovery.discover_local_directory(mcps_directory)

        registered_count = 0

        for server_id, server_info in servers.items():
            try:
                # Create server configuration
                server_config = MCPServerConfig(
                    id=server_id,
                    name=server_info.name,
                    transport="stdio",
                    command=self._generate_iowarp_command(server_info),
                    args=self._generate_iowarp_args(server_info),
                    capabilities=self._format_capabilities(server_info.capabilities),
                    enabled=True,
                )

                # Register with vMCP
                await self.registry.register_server(server_config)
                registered_count += 1

                logger.info(f"Registered iowarp MCP: {server_id}")

            except Exception as e:
                logger.error(f"Failed to register {server_id}: {e}")

        logger.info(f"Auto-registered {registered_count} iowarp-mcps servers")
        return registered_count

    async def sync_repositories(self) -> dict[str, int]:
        """
        Sync all configured repositories.

        Returns:
            Dictionary of repository sync results
        """
        logger.info("Syncing all repositories")

        results = {}

        for repo_name, repo_url in self.repositories.items():
            try:
                logger.info(f"Syncing repository: {repo_name}")

                # Discover servers from repository
                servers = await self.discovery.discover_git_repository(repo_url)

                results[repo_name] = len(servers)
                logger.info(f"Found {len(servers)} servers in {repo_name}")

            except Exception as e:
                logger.error(f"Failed to sync {repo_name}: {e}")
                results[repo_name] = 0

        # Refresh discovery cache
        await self.discovery.discover_all(refresh_cache=True)

        return results

    def get_repository_stats(self) -> dict[str, Any]:
        """
        Get repository management statistics.

        Returns:
            Repository statistics
        """
        # Get discovery stats
        discovery_stats = self.discovery.get_discovery_stats()

        # Get installer stats
        installer_stats = self.installer.get_install_stats()

        # Get registry stats
        registry_stats = self.registry.get_registry_stats()

        return {
            "discovery": discovery_stats,
            "installation": installer_stats,
            "registry": registry_stats,
            "repositories": list(self.repositories.keys()),
        }

    async def _register_installed_server(
        self, server_info: MCPServerInfo, install_path: str, enable: bool = True
    ) -> None:
        """Register an installed server with the vMCP registry."""
        # Generate args based on server info
        args = ["run", "--directory", install_path]
        if server_info.entry_point:
            args.append(server_info.entry_point)
        else:
            args.extend(["python", "-m", "server"])

        server_config = MCPServerConfig(
            id=server_info.id,
            name=server_info.name,
            transport="stdio",
            command=self._generate_command(server_info, install_path),
            args=args,
            capabilities=self._format_capabilities(server_info.capabilities),
            enabled=enable,
        )

        await self.registry.register_server(server_config)

    def _generate_command(self, server_info: MCPServerInfo, install_path: str) -> str:
        """Generate command to run an installed server."""
        if server_info.entry_point:
            return "uv"
        else:
            # Default command structure
            return "uv"

    def _generate_iowarp_command(self, server_info: MCPServerInfo) -> str:
        """Generate command to run an iowarp-mcps server."""
        # Use uv to run individual servers
        return "uv"

    def _generate_iowarp_args(self, server_info: MCPServerInfo) -> list[str]:
        """Generate args to run an iowarp-mcps server."""
        # Map server IDs to their actual directory names and paths
        server_path_map = {
            "parquet-mcp": "parquet",
            "node-hardware-mcp": "Node_Hardware",
            "arxiv-mcp": "Arxiv",
            "slurm-mcp": "Slurm",
            "plot-mcp": "Plot",
            "parallel-sort-mcp": "Parallel_Sort",
            "lmod-mcp": "lmod",
            "compression-mcp": "Compression",
            "pandas-mcp": "Pandas",
            "chronolog-mcp": "Chronolog",
            "darshan-mcp": "Darshan",
            "hdf5-mcp": "HDF5",
            "adios-mcp": "Adios",
            "jarvis-mcp": "Jarvis",
        }

        server_dir = server_path_map.get(server_info.id, server_info.id)
        server_path = f"iowarp-mcps/mcps/{server_dir}"

        # Use the entry point if available, otherwise use module execution
        return ["run", "--directory", server_path, server_info.id]

    def _format_capabilities(
        self, capabilities: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Format capabilities for MCPServerConfig."""
        formatted = {}
        for cap_type, cap_data in capabilities.items():
            if isinstance(cap_data, dict):
                # Convert dict to list format expected by registry
                if "list_changed" in cap_data:
                    formatted[cap_type] = []
                else:
                    formatted[cap_type] = [cap_data]
            elif isinstance(cap_data, list):
                formatted[cap_type] = cap_data
            else:
                formatted[cap_type] = []
        return formatted

    def _get_discovery_sources(self) -> list[str]:
        """Get configured discovery sources."""
        sources = []

        # Add local iowarp-mcps if available
        if Path("iowarp-mcps").exists():
            sources.append("iowarp-mcps")

        # Add repository URLs for git discovery
        sources.extend(self.repositories.values())

        return sources
