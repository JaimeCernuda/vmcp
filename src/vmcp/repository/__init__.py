"""
Repository system for vMCP.

This package provides MCP server discovery, installation, and management
capabilities for the Virtual Model Context Protocol gateway.
"""

from .discovery import MCPDiscovery
from .installer import MCPInstaller
from .manager import RepositoryManager

__all__ = ["RepositoryManager", "MCPDiscovery", "MCPInstaller"]
