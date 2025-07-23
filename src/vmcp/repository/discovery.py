"""
MCP Server Discovery System.

This module provides comprehensive discovery of MCP servers from various sources
including local directories, GitHub repositories, and package registries.
"""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import git
import toml

from ..errors import RepositoryError

logger = logging.getLogger(__name__)


@dataclass
class MCPServerInfo:
    """Information about a discovered MCP server."""
    id: str
    name: str
    description: str
    version: str
    author: str
    capabilities: dict[str, Any]
    entry_point: str
    source_type: str  # "local", "git", "pypi", "npm"
    source_location: str
    dependencies: list[str] = field(default_factory=list)
    documentation_url: str | None = None
    homepage_url: str | None = None
    tags: list[str] = field(default_factory=list)
    install_command: str | None = None
    test_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "capabilities": self.capabilities,
            "entry_point": self.entry_point,
            "source_type": self.source_type,
            "source_location": self.source_location,
            "dependencies": self.dependencies,
            "documentation_url": self.documentation_url,
            "homepage_url": self.homepage_url,
            "tags": self.tags,
            "install_command": self.install_command,
            "test_command": self.test_command
        }


class MCPDiscovery:
    """MCP Server Discovery System."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize MCP discovery system.
        
        Args:
            cache_dir: Directory for caching discovery results
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".vmcp" / "discovery_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.discovered_servers: dict[str, MCPServerInfo] = {}

    async def discover_all(
        self,
        sources: list[str] | None = None,
        refresh_cache: bool = False
    ) -> dict[str, MCPServerInfo]:
        """
        Discover MCP servers from all configured sources.
        
        Args:
            sources: List of source paths/URLs to scan
            refresh_cache: Whether to refresh cached results
            
        Returns:
            Dictionary of discovered servers by ID
        """
        if not refresh_cache:
            cached = self._load_cache()
            if cached:
                self.discovered_servers = cached
                return cached

        logger.info("Starting comprehensive MCP server discovery")

        if sources is None:
            sources = self._get_default_sources()

        discovery_tasks = []

        for source in sources:
            if Path(source).exists():
                # Local directory discovery
                discovery_tasks.append(self.discover_local_directory(source))
            elif source.startswith(("http://", "https://", "git@")):
                # Git repository discovery
                discovery_tasks.append(self.discover_git_repository(source))
            else:
                logger.warning(f"Unknown source type: {source}")

        # Execute all discovery tasks concurrently
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

        # Merge all discovered servers
        for result in results:
            if isinstance(result, dict):
                self.discovered_servers.update(result)
            elif isinstance(result, Exception):
                logger.error(f"Discovery error: {result}")

        # Cache results
        self._save_cache(self.discovered_servers)

        logger.info(f"Discovery complete: found {len(self.discovered_servers)} MCP servers")
        return self.discovered_servers

    async def discover_local_directory(self, directory: str) -> dict[str, MCPServerInfo]:
        """
        Discover MCP servers in a local directory.
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            Dictionary of discovered servers
        """
        servers = {}
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return servers

        logger.info(f"Scanning local directory: {directory}")

        # Look for iowarp-mcps structure
        mcps_dir = directory_path / "mcps"
        if mcps_dir.exists():
            servers.update(await self._scan_iowarp_mcps(mcps_dir))

        # Look for individual pyproject.toml files
        for pyproject_file in directory_path.rglob("pyproject.toml"):
            server_info = await self._analyze_pyproject(pyproject_file)
            if server_info:
                servers[server_info.id] = server_info

        return servers

    async def discover_git_repository(self, repo_url: str) -> dict[str, MCPServerInfo]:
        """
        Discover MCP servers in a Git repository.
        
        Args:
            repo_url: Git repository URL
            
        Returns:
            Dictionary of discovered servers
        """
        servers = {}

        try:
            # Clone to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "repo"

                logger.info(f"Cloning repository: {repo_url}")
                git.Repo.clone_from(repo_url, temp_path, depth=1)

                # Discover servers in cloned repo
                servers = await self.discover_local_directory(str(temp_path))

                # Update source information
                for server in servers.values():
                    server.source_type = "git"
                    server.source_location = repo_url

        except Exception as e:
            logger.error(f"Failed to clone repository {repo_url}: {e}")
            raise RepositoryError(f"Git discovery failed: {e}")

        return servers

    async def _scan_iowarp_mcps(self, mcps_dir: Path) -> dict[str, MCPServerInfo]:
        """
        Scan iowarp-mcps directory structure.
        
        Args:
            mcps_dir: Path to mcps directory
            
        Returns:
            Dictionary of discovered servers
        """
        servers = {}

        for mcp_dir in mcps_dir.iterdir():
            if not mcp_dir.is_dir():
                continue

            pyproject_file = mcp_dir / "pyproject.toml"
            if not pyproject_file.exists():
                continue

            server_info = await self._analyze_pyproject(pyproject_file)
            if server_info:
                # Add iowarp-specific metadata
                server_info.tags.extend(["iowarp", "scientific-computing"])
                server_info.source_type = "local"
                server_info.source_location = str(mcp_dir)

                # Try to extract additional info from README
                readme_file = mcp_dir / "README.md"
                if readme_file.exists():
                    try:
                        readme_content = readme_file.read_text()
                        # Extract description from README if not in pyproject
                        if not server_info.description and "## " in readme_content:
                            lines = readme_content.split("\n")
                            for i, line in enumerate(lines):
                                if line.startswith("## ") and i + 1 < len(lines):
                                    server_info.description = lines[i + 1].strip()
                                    break
                    except Exception as e:
                        logger.debug(f"Could not read README for {mcp_dir.name}: {e}")

                servers[server_info.id] = server_info

        return servers

    async def _analyze_pyproject(self, pyproject_file: Path) -> MCPServerInfo | None:
        """
        Analyze a pyproject.toml file to extract MCP server information.
        
        Args:
            pyproject_file: Path to pyproject.toml file
            
        Returns:
            MCPServerInfo if valid MCP server found, None otherwise
        """
        try:
            with open(pyproject_file) as f:
                config = toml.load(f)

            project = config.get("project", {})

            # Check if this looks like an MCP server
            scripts = project.get("scripts", {})
            entry_points = project.get("entry-points", {})

            # Look for MCP-related entry points
            mcp_entries = []

            # Check scripts for MCP entries
            for name, command in scripts.items():
                if "mcp" in name.lower() or "server" in command.lower():
                    mcp_entries.append((name, command))

            # Check entry points
            console_scripts = entry_points.get("console_scripts", {})
            for name, command in console_scripts.items():
                if "mcp" in name.lower():
                    mcp_entries.append((name, command))

            if not mcp_entries:
                return None

            # Extract server information
            server_id = project.get("name", pyproject_file.parent.name)
            if not server_id:
                return None

            # Determine entry point (prefer the first MCP entry)
            entry_point = mcp_entries[0][1] if mcp_entries else ""

            # Try to determine capabilities by analyzing the source
            capabilities = await self._analyze_capabilities(pyproject_file.parent)

            # Extract dependencies
            dependencies = project.get("dependencies", [])

            server_info = MCPServerInfo(
                id=server_id,
                name=project.get("name", server_id),
                description=project.get("description", ""),
                version=project.get("version", "0.1.0"),
                author=self._extract_author(project.get("authors", [])),
                capabilities=capabilities,
                entry_point=entry_point,
                source_type="local",
                source_location=str(pyproject_file.parent),
                dependencies=dependencies,
                homepage_url=self._extract_url(project.get("urls", {}), "homepage"),
                documentation_url=self._extract_url(project.get("urls", {}), "documentation"),
                tags=self._extract_tags(project),
                install_command=f"uv pip install -e {pyproject_file.parent}",
                test_command="uv run pytest" if (pyproject_file.parent / "tests").exists() else None
            )

            return server_info

        except Exception as e:
            logger.debug(f"Could not analyze {pyproject_file}: {e}")
            return None

    async def _analyze_capabilities(self, server_dir: Path) -> dict[str, Any]:
        """
        Analyze server directory to determine capabilities.
        
        Args:
            server_dir: Path to server directory
            
        Returns:
            Dictionary of capabilities
        """
        capabilities = {}

        # Look for common capability patterns
        src_dirs = [
            server_dir / "src",
            server_dir,
        ]

        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            # Look for capabilities directory
            cap_dirs = list(src_dir.rglob("capabilities"))
            if cap_dirs:
                cap_dir = cap_dirs[0]
                for py_file in cap_dir.glob("*.py"):
                    if py_file.name == "__init__.py":
                        continue

                    # Try to determine capability type from filename
                    filename = py_file.stem
                    if any(keyword in filename.lower() for keyword in ["tool", "handler"]):
                        capabilities["tools"] = {"list_changed": True}
                    elif "resource" in filename.lower():
                        capabilities["resources"] = {"subscribe": True, "list_changed": True}
                    elif "prompt" in filename.lower():
                        capabilities["prompts"] = {"list_changed": True}

            # Look for server.py to analyze
            server_files = list(src_dir.rglob("server.py"))
            if server_files:
                try:
                    content = server_files[0].read_text()
                    if "tools" in content and "list_tools" in content:
                        capabilities["tools"] = {"list_changed": True}
                    if "resources" in content and "list_resources" in content:
                        capabilities["resources"] = {"subscribe": True, "list_changed": True}
                    if "prompts" in content and "list_prompts" in content:
                        capabilities["prompts"] = {"list_changed": True}
                except Exception:
                    pass

        # Default capabilities if none found
        if not capabilities:
            capabilities = {"tools": {"list_changed": True}}

        return capabilities

    def _extract_author(self, authors: list[dict[str, str]]) -> str:
        """Extract author name from authors list."""
        if not authors:
            return "Unknown"
        if isinstance(authors[0], dict):
            return authors[0].get("name", "Unknown")
        return str(authors[0])

    def _extract_url(self, urls: dict[str, str], key: str) -> str | None:
        """Extract URL from URLs dictionary."""
        return urls.get(key) or urls.get(key.title()) or urls.get(key.upper())

    def _extract_tags(self, project: dict[str, Any]) -> list[str]:
        """Extract tags from project metadata."""
        tags = []

        # From keywords
        keywords = project.get("keywords", [])
        if isinstance(keywords, list):
            tags.extend(keywords)
        elif isinstance(keywords, str):
            tags.extend(keywords.split(","))

        # From classifiers
        classifiers = project.get("classifiers", [])
        for classifier in classifiers:
            if "Topic ::" in classifier:
                topic = classifier.split("Topic ::")[1].strip()
                tags.append(topic.lower().replace(" ", "-"))

        return tags

    def _get_default_sources(self) -> list[str]:
        """Get default discovery sources."""
        sources = []

        # Check for iowarp-mcps in current directory
        if Path("iowarp-mcps").exists():
            sources.append("iowarp-mcps")

        # Check user's home directory
        home_iowarp = Path.home() / "iowarp-mcps"
        if home_iowarp.exists():
            sources.append(str(home_iowarp))

        # Add common MCP directories
        common_dirs = [
            Path.home() / ".vmcp" / "servers",
            Path("/opt/vmcp/servers"),
            Path("./servers"),
        ]

        for directory in common_dirs:
            if directory.exists():
                sources.append(str(directory))

        return sources

    def _load_cache(self) -> dict[str, MCPServerInfo] | None:
        """Load cached discovery results."""
        cache_file = self.cache_dir / "discovery_cache.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            servers = {}
            for server_id, server_data in data.items():
                servers[server_id] = MCPServerInfo(**server_data)

            logger.debug(f"Loaded {len(servers)} servers from cache")
            return servers

        except Exception as e:
            logger.debug(f"Could not load cache: {e}")
            return None

    def _save_cache(self, servers: dict[str, MCPServerInfo]) -> None:
        """Save discovery results to cache."""
        cache_file = self.cache_dir / "discovery_cache.json"

        try:
            data = {
                server_id: server.to_dict()
                for server_id, server in servers.items()
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(servers)} servers to cache")

        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    async def search_servers(
        self,
        query: str,
        tags: list[str] | None = None,
        capabilities: list[str] | None = None
    ) -> list[MCPServerInfo]:
        """
        Search discovered MCP servers.
        
        Args:
            query: Search query string
            tags: Required tags
            capabilities: Required capabilities
            
        Returns:
            List of matching servers
        """
        if not self.discovered_servers:
            await self.discover_all()

        matches = []
        query_lower = query.lower()

        for server in self.discovered_servers.values():
            # Text matching
            text_match = (
                query_lower in server.name.lower() or
                query_lower in server.description.lower() or
                query_lower in server.id.lower() or
                any(query_lower in tag.lower() for tag in server.tags)
            )

            if not text_match and query.strip():
                continue

            # Tag filtering
            if tags:
                if not all(tag.lower() in [t.lower() for t in server.tags] for tag in tags):
                    continue

            # Capability filtering
            if capabilities:
                server_caps = set(server.capabilities.keys())
                required_caps = set(cap.lower() for cap in capabilities)
                if not required_caps.issubset(cap.lower() for cap in server_caps):
                    continue

            matches.append(server)

        # Sort by relevance (simple scoring)
        def score_match(server):
            score = 0
            if query_lower in server.name.lower():
                score += 10
            if query_lower in server.id.lower():
                score += 8
            if query_lower in server.description.lower():
                score += 5
            return score

        matches.sort(key=score_match, reverse=True)
        return matches

    async def get_server_details(self, server_id: str) -> MCPServerInfo | None:
        """
        Get detailed information about a specific server.
        
        Args:
            server_id: Server identifier
            
        Returns:
            Server information or None if not found
        """
        if not self.discovered_servers:
            await self.discover_all()

        return self.discovered_servers.get(server_id)

    async def refresh_discovery(self) -> dict[str, MCPServerInfo]:
        """
        Refresh discovery cache.
        
        Returns:
            Updated discovered servers
        """
        return await self.discover_all(refresh_cache=True)

    def get_discovery_stats(self) -> dict[str, Any]:
        """
        Get discovery statistics.
        
        Returns:
            Discovery statistics
        """
        if not self.discovered_servers:
            return {
                "total_servers": 0,
                "source_types": {},
                "capabilities": {},
                "tags": {}
            }

        source_types = {}
        capabilities = {}
        tags = {}

        for server in self.discovered_servers.values():
            # Count source types
            source_types[server.source_type] = source_types.get(server.source_type, 0) + 1

            # Count capabilities
            for cap in server.capabilities.keys():
                capabilities[cap] = capabilities.get(cap, 0) + 1

            # Count tags
            for tag in server.tags:
                tags[tag] = tags.get(tag, 0) + 1

        return {
            "total_servers": len(self.discovered_servers),
            "source_types": source_types,
            "capabilities": capabilities,
            "tags": tags
        }
