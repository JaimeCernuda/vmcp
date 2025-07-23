"""
Virtual Model Context Protocol (vMCP) - Unified MCP server gateway.

A production-ready system that provides unified access to multiple MCP servers
through a single interface, similar to how Virtual File Systems provide unified
access to different storage backends.
"""

__version__ = "1.0.0"
__author__ = "VMCP Team"
__email__ = "team@vmcp.dev"

from .errors import VMCPError, VMCPErrorCode

__all__ = ["VMCPError", "VMCPErrorCode"]
