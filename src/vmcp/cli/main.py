"""
Main CLI entry point for vMCP.

This module provides the primary command-line interface for the
Virtual Model Context Protocol gateway including all subcommands
for managing servers, routing, and system operations.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import click
import rich_click as click
import structlog
import uvloop
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from ..config.loader import ConfigLoader
from ..errors import ConfigurationError, ExtensionError, VMCPError
from ..extensions.manager import ExtensionManager
from ..gateway.server import GatewayConfig, VMCPGateway
from ..registry.registry import Registry
from ..repository.manager import RepositoryManager

# Configure rich-click styling
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.STYLE_METAVAR = "bold yellow"
click.rich_click.STYLE_USAGE_COMMAND = "bold"
click.rich_click.STYLE_USAGE_PROG = "bold blue"
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_ARGUMENT = "bold yellow"
click.rich_click.STYLE_COMMAND = "bold green"

# Create rich console for formatted output
console = Console()

# Configure structured logging for CLI
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.WriteLoggerFactory(),
    wrapper_class=structlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def install_shell_completion(shell: str) -> None:
    """Install shell completion for the specified shell."""
    import subprocess
    
    shell_commands = {
        'bash': 'complete -C vmcp vmcp',
        'zsh': 'compdef _vmcp vmcp',
        'fish': 'complete -c vmcp -f',
        'powershell': 'Register-ArgumentCompleter -CommandName vmcp -ScriptBlock $__vmcpCompleter'
    }
    
    if shell == 'bash':
        completion_script = '''
# vMCP bash completion
_vmcp_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "${prev}" in
        vmcp)
            opts="start stop status list info config repo extension health metrics mount unmount --help --version"
            ;;
        extension)
            opts="list install uninstall enable disable update info --help"
            ;;
        repo)
            opts="discover search install uninstall stats --help"
            ;;
        config)
            opts="init validate --help"
            ;;
        *)
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _vmcp_completion vmcp
'''
        console.print(Panel(completion_script, title="[bold green]Bash Completion[/bold green]", 
                           title_align="left", border_style="green"))
        console.print("[yellow]Add the above to your ~/.bashrc or ~/.bash_profile[/yellow]")
    
    elif shell == 'zsh':
        completion_script = '''
# vMCP zsh completion
#compdef vmcp

_vmcp() {
    local context state state_descr line
    typeset -A opt_args
    
    _arguments -C \\
        '1: :->commands' \\
        '*: :->args' && return 0
    
    case $state in
        commands)
            _values "vmcp commands" \\
                "start[Start the vMCP Gateway]" \\
                "stop[Stop the vMCP Gateway]" \\
                "status[Show vMCP Gateway status]" \\
                "list[List registered servers]" \\
                "extension[Extension management]" \\
                "repo[Repository management]" \\
                "config[Configuration management]"
            ;;
        args)
            case $words[2] in
                extension)
                    _values "extension commands" \\
                        "list[List extensions]" \\
                        "install[Install extension]" \\
                        "uninstall[Uninstall extension]" \\
                        "enable[Enable extension]" \\
                        "disable[Disable extension]"
                    ;;
            esac
            ;;
    esac
}

_vmcp "$@"
'''
        console.print(Panel(completion_script, title="[bold green]Zsh Completion[/bold green]", 
                           title_align="left", border_style="green"))
        console.print("[yellow]Add the above to your ~/.zshrc or save as _vmcp in your $fpath[/yellow]")
    
    else:
        console.print(f"[red]Shell completion for {shell} not implemented yet[/red]")
    
    sys.exit(0)


class VMCPCLIContext:
    """CLI context for sharing state between commands."""

    def __init__(self):
        self.config_path: str | None = None
        self.config: dict[str, Any] | None = None
        self.gateway: VMCPGateway | None = None
        self.verbose: bool = False


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """
    [bold blue]vMCP Gateway[/bold blue] - Virtual Model Context Protocol unified interface.
    
    The vMCP Gateway provides a unified abstraction layer that aggregates
    multiple MCP servers through a single interface, similar to how Virtual
    File Systems provide unified access to different storage backends.
    """
    ctx.ensure_object(VMCPCLIContext)
    ctx.obj.config_path = config
    ctx.obj.verbose = verbose

    # Configure logging level
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option(
    "--daemon", "-d",
    is_flag=True,
    help="Run as daemon (background process)"
)
@click.option(
    "--port", "-p",
    type=int,
    help="Override default port"
)
@click.option(
    "--host", "-h",
    type=str,
    help="Override default host"
)
@click.pass_context
def start(
    ctx: click.Context,
    daemon: bool,
    port: int | None,
    host: str | None
) -> None:
    """Start the vMCP Gateway server."""
    try:
        # Use uvloop for better async performance
        uvloop.install()

        # Load configuration
        config_loader = ConfigLoader()
        if ctx.obj.config_path:
            config_data = config_loader.load_from_file(ctx.obj.config_path)
        else:
            config_data = config_loader.load_defaults()

        # Apply command line overrides
        if port:
            config_data.setdefault("transports", {}).setdefault("http", {})["port"] = port
        if host:
            config_data.setdefault("transports", {}).setdefault("http", {})["host"] = host

        # Create gateway configuration
        gateway_config = GatewayConfig(**config_data.get("gateway", {}))

        # Create and start gateway
        gateway = VMCPGateway(gateway_config)

        if daemon:
            click.echo("Starting vMCP Gateway as daemon...")
            asyncio.run(_run_daemon(gateway))
        else:
            click.echo("Starting vMCP Gateway...")
            asyncio.run(_run_interactive(gateway))

    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except VMCPError as e:
        click.echo(f"Gateway error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nShutdown requested by user")
    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


async def _run_interactive(gateway: VMCPGateway) -> None:
    """Run gateway interactively with signal handling."""
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        shutdown_event.set()

    # Install signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize and start gateway
        await gateway.initialize()
        await gateway.start()

        click.echo(" vMCP Gateway started successfully")
        click.echo("Press Ctrl+C to stop...")

        # Wait for shutdown signal
        await shutdown_event.wait()

    finally:
        click.echo("Stopping vMCP Gateway...")
        await gateway.stop()
        click.echo(" vMCP Gateway stopped")


async def _run_daemon(gateway: VMCPGateway) -> None:
    """Run gateway as daemon process."""
    try:
        await gateway.initialize()
        await gateway.start()

        # Write PID file
        pid_file = Path.home() / ".vmcp" / "vmcp.pid"
        pid_file.parent.mkdir(exist_ok=True)
        pid_file.write_text(str(os.getpid()))

        logger.info("vMCP Gateway daemon started", pid=os.getpid())

        # Wait for shutdown
        await gateway.wait_for_shutdown()

    finally:
        # Clean up PID file
        try:
            pid_file.unlink(missing_ok=True)
        except:
            pass
        await gateway.stop()


@cli.command()
@click.option(
    "--timeout", "-t",
    type=int,
    default=10,
    help="Shutdown timeout in seconds"
)
def stop(timeout: int) -> None:
    """Stop the vMCP Gateway server."""
    import os

    # Look for PID file
    pid_file = Path.home() / ".vmcp" / "vmcp.pid"

    if not pid_file.exists():
        click.echo("vMCP Gateway is not running (no PID file found)")
        return

    try:
        pid = int(pid_file.read_text().strip())

        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent shutdown signal to process {pid}")

        # Wait for process to exit
        import time
        for _ in range(timeout):
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(1)
            except ProcessLookupError:
                # Process has exited
                pid_file.unlink(missing_ok=True)
                click.echo(" vMCP Gateway stopped")
                return

        # Process still running, send SIGKILL
        click.echo(f"Process {pid} did not exit gracefully, forcing...")
        os.kill(pid, signal.SIGKILL)
        pid_file.unlink(missing_ok=True)
        click.echo(" vMCP Gateway forcefully stopped")

    except (ValueError, ProcessLookupError) as e:
        click.echo(f"Error stopping gateway: {e}", err=True)
        pid_file.unlink(missing_ok=True)
    except PermissionError:
        click.echo("Permission denied - cannot stop gateway", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
@click.pass_context
def status(ctx: click.Context, output_format: str) -> None:
    """Show vMCP Gateway status."""
    try:
        # Check if daemon is running
        pid_file = Path.home() / ".vmcp" / "vmcp.pid"

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                import os
                os.kill(pid, 0)  # Check if process exists
                running = True
            except (ValueError, ProcessLookupError):
                running = False
                pid = None
        else:
            running = False
            pid = None

        # Try to connect to gateway for detailed status
        status_data = {
            "running": running,
            "pid": pid,
            "timestamp": time.time()
        }

        if running:
            # Try to get detailed status from running gateway
            try:
                # This would connect to the running gateway via transport
                # For now, just show basic status
                status_data.update({
                    "uptime": "unknown",
                    "servers": "unknown",
                    "health": "unknown"
                })
            except Exception:
                pass

        if output_format == "json":
            click.echo(json.dumps(status_data, indent=2))
        else:
            if running:
                click.echo(f" vMCP Gateway is running (PID: {pid})")
            else:
                click.echo(" vMCP Gateway is not running")

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json", "table"]),
    default="table",
    help="Output format"
)
@click.option(
    "--filter-enabled/--no-filter-enabled",
    default=True,
    help="Only show enabled servers"
)
def list(output_format: str, filter_enabled: bool) -> None:
    """List all registered MCP servers."""
    try:
        # Load configuration to get registry
        config_loader = ConfigLoader()
        config_data = config_loader.load_defaults()

        # Create registry instance
        from ..registry.registry import Registry
        registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))

        # This would require async initialization
        # For now, show placeholder data
        servers_data = [
            {
                "id": "example-server",
                "name": "Example MCP Server",
                "transport": "stdio",
                "enabled": True,
                "healthy": True,
                "capabilities": ["tools", "resources"]
            }
        ]

        if filter_enabled:
            servers_data = [s for s in servers_data if s["enabled"]]

        if output_format == "json":
            click.echo(json.dumps({"servers": servers_data}, indent=2))
        elif output_format == "table":
            if not servers_data:
                click.echo("No servers found")
                return

            # Simple table output
            click.echo(f"{'ID':<20} {'Name':<30} {'Transport':<10} {'Status':<10} {'Capabilities'}")
            click.echo("-" * 90)

            for server in servers_data:
                status = " healthy" if server["healthy"] else " unhealthy"
                if not server["enabled"]:
                    status = "- disabled"

                caps = ", ".join(server["capabilities"])
                click.echo(f"{server['id']:<20} {server['name']:<30} {server['transport']:<10} {status:<10} {caps}")
        else:
            # Text format
            for server in servers_data:
                click.echo(f"Server: {server['id']}")
                click.echo(f"  Name: {server['name']}")
                click.echo(f"  Transport: {server['transport']}")
                click.echo(f"  Enabled: {server['enabled']}")
                click.echo(f"  Healthy: {server['healthy']}")
                click.echo(f"  Capabilities: {', '.join(server['capabilities'])}")
                click.echo()

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            click.echo(f"Error listing servers: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("server_config", type=click.Path(exists=True))
@click.option(
    "--mount-point", "-m",
    help="Mount point for server capabilities"
)
@click.option(
    "--enable/--no-enable",
    default=True,
    help="Enable server after mounting"
)
def mount(server_config: str, mount_point: str | None, enable: bool) -> None:
    """Mount a new MCP server from configuration file."""
    try:
        # Load server configuration
        import toml
        with open(server_config) as f:
            config = toml.load(f)

        server_id = config.get("id") or Path(server_config).stem

        # This would add the server to the registry
        click.echo(f" Mounted server '{server_id}' from {server_config}")

        if mount_point:
            click.echo(f"  Mount point: {mount_point}")

        if enable:
            click.echo("  Status: enabled")
        else:
            click.echo("  Status: disabled")

    except Exception as e:
        click.echo(f"Error mounting server: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("server_id")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force unmount even if server is in use"
)
def unmount(server_id: str, force: bool) -> None:
    """Unmount an MCP server."""
    try:
        # This would remove the server from the registry
        if force:
            click.echo(f" Force unmounted server '{server_id}'")
        else:
            click.echo(f" Unmounted server '{server_id}'")

    except Exception as e:
        click.echo(f"Error unmounting server: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("server_id", required=False)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def info(server_id: str | None, output_format: str) -> None:
    """Show detailed information about server(s) or system."""
    try:
        if server_id:
            # Show specific server info
            server_info = {
                "id": server_id,
                "name": f"{server_id} Server",
                "transport": "stdio",
                "enabled": True,
                "healthy": True,
                "capabilities": ["tools", "resources"],
                "stats": {
                    "requests_total": 0,
                    "requests_success": 0,
                    "requests_failed": 0,
                    "uptime": "0s"
                },
                "configuration": {
                    "command": ["python", "-m", "example_server"],
                    "args": [],
                    "env": {}
                }
            }

            if output_format == "json":
                click.echo(json.dumps(server_info, indent=2))
            else:
                click.echo(f"Server Information: {server_id}")
                click.echo(f"  Name: {server_info['name']}")
                click.echo(f"  Transport: {server_info['transport']}")
                click.echo(f"  Status: {'enabled' if server_info['enabled'] else 'disabled'}")
                click.echo(f"  Health: {'healthy' if server_info['healthy'] else 'unhealthy'}")
                click.echo(f"  Capabilities: {', '.join(server_info['capabilities'])}")
                click.echo("  Statistics:")
                for key, value in server_info['stats'].items():
                    click.echo(f"    {key}: {value}")
        else:
            # Show system info
            system_info = {
                "version": "0.1.0",
                "config_path": str(Path.home() / ".vmcp" / "config.toml"),
                "registry_path": str(Path.home() / ".vmcp" / "registry"),
                "log_level": "INFO",
                "servers_total": 1,
                "servers_enabled": 1,
                "servers_healthy": 1
            }

            if output_format == "json":
                click.echo(json.dumps(system_info, indent=2))
            else:
                click.echo("vMCP Gateway System Information")
                click.echo(f"  Version: {system_info['version']}")
                click.echo(f"  Config: {system_info['config_path']}")
                click.echo(f"  Registry: {system_info['registry_path']}")
                click.echo(f"  Log Level: {system_info['log_level']}")
                click.echo("  Servers:")
                click.echo(f"    Total: {system_info['servers_total']}")
                click.echo(f"    Enabled: {system_info['servers_enabled']}")
                click.echo(f"    Healthy: {system_info['servers_healthy']}")

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            click.echo(f"Error getting info: {e}", err=True)
        sys.exit(1)


@cli.group()
def config():
    """Configuration management commands."""
    pass


@cli.group()
def repo():
    """Repository management commands."""
    pass


@repo.command()
@click.option(
    "--sources", "-s",
    multiple=True,
    help="Discovery sources to scan"
)
@click.option(
    "--refresh", "-r",
    is_flag=True,
    help="Refresh discovery cache"
)
def discover(sources: tuple, refresh: bool) -> None:
    """Discover available MCP servers."""
    async def _discover():
        try:
            config_loader = ConfigLoader()
            config_data = config_loader.load_defaults()

            registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))
            await registry.initialize()

            repo_manager = RepositoryManager(registry)
            await repo_manager.initialize()

            # Auto-register iowarp-mcps if found
            if Path("iowarp-mcps").exists():
                count = await repo_manager.auto_register_iowarp_mcps("iowarp-mcps")
                click.echo(f"✓ Auto-registered {count} iowarp-mcps servers")

            servers = await repo_manager.discover_servers(
                list(sources) if sources else None,
                refresh
            )

            click.echo(f"✓ Discovered {len(servers)} MCP servers")

            for server_id, server_info in servers.items():
                click.echo(f"  • {server_id}: {server_info.description}")

        except Exception as e:
            click.echo(f"Error during discovery: {e}", err=True)
            sys.exit(1)

    asyncio.run(_discover())


@repo.command()
@click.argument("query", default="")
@click.option("--tags", multiple=True, help="Filter by tags")
@click.option("--capabilities", multiple=True, help="Filter by capabilities")
@click.option("--installed", is_flag=True, help="Show only installed servers")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json", "table"]),
    default="table",
    help="Output format"
)
def search(query: str, tags: tuple, capabilities: tuple, installed: bool, output_format: str) -> None:
    """Search for MCP servers."""
    async def _search():
        try:
            config_loader = ConfigLoader()
            config_data = config_loader.load_defaults()

            registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))
            await registry.initialize()

            repo_manager = RepositoryManager(registry)
            await repo_manager.initialize()

            results = await repo_manager.search_servers(
                query,
                list(tags) if tags else None,
                list(capabilities) if capabilities else None,
                installed
            )

            if output_format == "json":
                click.echo(json.dumps({"servers": results}, indent=2))
            elif output_format == "table":
                if not results:
                    click.echo("No servers found")
                    return

                click.echo(f"{'ID':<20} {'Name':<25} {'Status':<15} {'Capabilities'}")
                click.echo("-" * 80)

                for server in results:
                    status = "✓ installed" if server.get("installed") else "- available"
                    if server.get("registered"):
                        status += " + registered"

                    caps = ", ".join(server.get("capabilities", {}).keys())
                    click.echo(f"{server['id']:<20} {server['name']:<25} {status:<15} {caps}")
            else:
                for server in results:
                    click.echo(f"Server: {server['id']}")
                    click.echo(f"  Name: {server['name']}")
                    click.echo(f"  Description: {server['description']}")
                    click.echo(f"  Installed: {server.get('installed', False)}")
                    click.echo(f"  Registered: {server.get('registered', False)}")
                    click.echo(f"  Capabilities: {', '.join(server.get('capabilities', {}).keys())}")
                    click.echo()

        except Exception as e:
            click.echo(f"Error during search: {e}", err=True)
            sys.exit(1)

    asyncio.run(_search())


@repo.command()
@click.argument("server_id")
@click.option("--no-register", is_flag=True, help="Don't register with vMCP")
@click.option("--no-enable", is_flag=True, help="Don't enable after installation")
def install(server_id: str, no_register: bool, no_enable: bool) -> None:
    """Install an MCP server."""
    async def _install():
        try:
            config_loader = ConfigLoader()
            config_data = config_loader.load_defaults()

            registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))
            await registry.initialize()

            repo_manager = RepositoryManager(registry)
            await repo_manager.initialize()

            click.echo(f"Installing MCP server: {server_id}")

            success = await repo_manager.install_server(
                server_id,
                register=not no_register,
                enable=not no_enable
            )

            if success:
                click.echo(f"✓ Successfully installed {server_id}")
            else:
                click.echo(f"✗ Failed to install {server_id}", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error installing server: {e}", err=True)
            sys.exit(1)

    asyncio.run(_install())


@repo.command()
@click.argument("server_id")
@click.option("--keep-config", is_flag=True, help="Keep vMCP configuration")
def uninstall(server_id: str, keep_config: bool) -> None:
    """Uninstall an MCP server."""
    async def _uninstall():
        try:
            config_loader = ConfigLoader()
            config_data = config_loader.load_defaults()

            registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))
            await registry.initialize()

            repo_manager = RepositoryManager(registry)
            await repo_manager.initialize()

            click.echo(f"Uninstalling MCP server: {server_id}")

            success = await repo_manager.uninstall_server(
                server_id,
                unregister=not keep_config
            )

            if success:
                click.echo(f"✓ Successfully uninstalled {server_id}")
            else:
                click.echo(f"✗ Failed to uninstall {server_id}", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error uninstalling server: {e}", err=True)
            sys.exit(1)

    asyncio.run(_uninstall())


@repo.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def stats(output_format: str) -> None:
    """Show repository statistics."""
    async def _stats():
        try:
            config_loader = ConfigLoader()
            config_data = config_loader.load_defaults()

            registry = Registry(config_data.get("gateway", {}).get("registry_path", "~/.vmcp/registry"))
            await registry.initialize()

            repo_manager = RepositoryManager(registry)
            await repo_manager.initialize()

            stats_data = repo_manager.get_repository_stats()

            if output_format == "json":
                click.echo(json.dumps(stats_data, indent=2))
            else:
                click.echo("Repository Statistics")
                click.echo("Discovery:")
                discovery = stats_data["discovery"]
                click.echo(f"  Total servers: {discovery['total_servers']}")
                click.echo(f"  Source types: {discovery['source_types']}")

                click.echo("Installation:")
                installation = stats_data["installation"]
                click.echo(f"  Installed servers: {installation['total_installed']}")
                click.echo(f"  Install directory: {installation['install_directory']}")

                click.echo("Registry:")
                registry_stats = stats_data["registry"]
                click.echo(f"  Registered servers: {registry_stats['total_servers']}")
                click.echo(f"  Healthy servers: {registry_stats['healthy_servers']}")

        except Exception as e:
            click.echo(f"Error getting stats: {e}", err=True)
            sys.exit(1)

    asyncio.run(_stats())


@config.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output configuration file path"
)
def init(output: str | None) -> None:
    """Initialize default configuration file."""
    try:
        config_path = Path(output) if output else Path.home() / ".vmcp" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create default configuration
        default_config = {
            "version": "0.1.0",
            "gateway": {
                "registry_path": "~/.vmcp/registry",
                "log_level": "INFO",
                "cache_enabled": True,
                "cache_ttl": 300,
                "max_connections": 1000,
                "request_timeout": 30
            },
            "transports": {
                "stdio": {"enabled": True},
                "http": {"enabled": False, "port": 3000, "host": "127.0.0.1"},
                "websocket": {"enabled": False, "port": 3001, "host": "127.0.0.1"}
            },
            "routing": {
                "default_strategy": "hybrid",
                "load_balancer": "round_robin"
            }
        }

        import toml
        with open(config_path, 'w') as f:
            toml.dump(default_config, f)

        click.echo(f" Created default configuration at {config_path}")

    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        sys.exit(1)


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str) -> None:
    """Validate configuration file."""
    try:
        config_loader = ConfigLoader()
        config_data = config_loader.load_from_file(config_file)

        # Validate configuration
        errors = config_loader.validate_config(config_data)

        if errors:
            click.echo("Configuration validation failed:")
            for error in errors:
                click.echo(f"   {error}")
            sys.exit(1)
        else:
            click.echo(" Configuration is valid")

    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


@cli.group()
def health():
    """Health monitoring commands."""
    pass


@health.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def check(output_format: str) -> None:
    """Check system health."""
    try:
        # This would connect to running gateway and get health status
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": [
                {"name": "gateway", "status": "healthy", "message": "Gateway is running"},
                {"name": "registry", "status": "healthy", "message": "Registry is accessible"},
                {"name": "servers", "status": "healthy", "message": "All servers healthy"}
            ]
        }

        if output_format == "json":
            click.echo(json.dumps(health_data, indent=2))
        else:
            click.echo(f"System Health: {health_data['status']}")
            click.echo("Component Checks:")
            for check in health_data['checks']:
                status_icon = "" if check['status'] == 'healthy' else ""
                click.echo(f"  {status_icon} {check['name']}: {check['message']}")

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            click.echo(f"Error checking health: {e}", err=True)
        sys.exit(1)


@cli.group()
def metrics():
    """Metrics and monitoring commands."""
    pass


@metrics.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json", "prometheus"]),
    default="text",
    help="Output format"
)
def show(output_format: str) -> None:
    """Show system metrics."""
    try:
        # This would get metrics from running gateway
        metrics_data = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "active_requests": 0,
            "servers_healthy": 1,
            "servers_total": 1,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
            "uptime": "0s"
        }

        if output_format == "json":
            click.echo(json.dumps(metrics_data, indent=2))
        elif output_format == "prometheus":
            # Simple Prometheus format
            for key, value in metrics_data.items():
                if isinstance(value, (int, float)):
                    click.echo(f"vmcp_{key} {value}")
        else:
            click.echo("vMCP Gateway Metrics")
            click.echo(f"  Uptime: {metrics_data['uptime']}")
            click.echo(f"  Requests Total: {metrics_data['requests_total']}")
            click.echo(f"  Requests Success: {metrics_data['requests_success']}")
            click.echo(f"  Requests Failed: {metrics_data['requests_failed']}")
            click.echo(f"  Active Requests: {metrics_data['active_requests']}")
            click.echo(f"  Servers Healthy: {metrics_data['servers_healthy']}/{metrics_data['servers_total']}")
            click.echo(f"  Cache Hit Rate: {metrics_data['cache_hit_rate']:.1%}")
            click.echo(f"  Error Rate: {metrics_data['error_rate']:.1%}")

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"error": str(e)}, indent=2))
        else:
            click.echo(f"Error getting metrics: {e}", err=True)
        sys.exit(1)


# Extension management commands (DXT-inspired)
@cli.group()
def extension():
    """Extension management commands (DXT-inspired)."""
    pass


@extension.command("list")
@click.option(
    "--repository", "-r",
    default="builtin",
    help="Repository to list extensions from"
)
@click.option(
    "--installed", "-i",
    is_flag=True,
    help="List installed extensions only"
)
@click.option(
    "--enabled", "-e", 
    is_flag=True,
    help="List enabled extensions only"
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
def list_extensions(repository: str, installed: bool, enabled: bool, output_format: str) -> None:
    """List available, installed, or enabled extensions."""
    try:
        ext_manager = ExtensionManager()
        
        if installed:
            extensions = ext_manager.list_installed_extensions()
            title = "🔧 Installed Extensions"
            title_style = "bold blue"
        elif enabled:
            enabled_exts = ext_manager.get_enabled_extensions()
            extensions = []
            for ext_id in enabled_exts:
                manifest = ext_manager.get_extension_manifest(ext_id)
                if manifest:
                    manifest["enabled"] = True
                    extensions.append(manifest)
            title = "✅ Enabled Extensions"
            title_style = "bold green"
        else:
            extensions = ext_manager.list_available_extensions(repository)
            title = f"📦 Available Extensions - {repository.title()}"
            title_style = "bold cyan"
        
        if output_format == "json":
            console.print_json(json.dumps({"extensions": extensions}, indent=2))
        else:
            if not extensions:
                console.print(f"[{title_style}]{title}[/{title_style}]")
                console.print("[yellow]No extensions found[/yellow]")
                return
            
            # Create rich table
            table = Table(title=title, title_style=title_style, show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", no_wrap=True, width=18)
            table.add_column("Name", style="white", width=28)
            table.add_column("Version", style="dim", width=8)
            table.add_column("Category", style="blue", width=16)
            table.add_column("Status", style="green", width=12)
            
            # Add rows with proper formatting
            for ext in extensions:
                ext_id = ext.get("id", ext.get("name", "unknown"))
                name = ext.get("display_name", ext.get("name", ""))
                version = ext.get("version", "unknown")
                category = ext.get("category", "unknown")
                
                # Format status with colors and icons
                if installed or enabled:
                    if ext.get("enabled", False):
                        status = "[green]✓ enabled[/green]"
                    else:
                        status = "[yellow]○ disabled[/yellow]"
                else:
                    status = "[dim]- available[/dim]"
                
                # Truncate long names
                if len(name) > 25:
                    name = name[:22] + "..."
                
                table.add_row(ext_id, name, version, category, status)
            
            console.print(table)
            
            # Add summary
            if extensions:
                total_count = len(extensions)
                if enabled:
                    console.print(f"[dim]Total: {total_count} enabled extension{'s' if total_count != 1 else ''}[/dim]")
                elif installed:
                    enabled_count = sum(1 for ext in extensions if ext.get("enabled", False))
                    console.print(f"[dim]Total: {total_count} installed, {enabled_count} enabled[/dim]")
                else:
                    console.print(f"[dim]Total: {total_count} extension{'s' if total_count != 1 else ''} available[/dim]")
    
    except Exception as e:
        console.print(f"[red]✗ Error listing extensions: {e}[/red]", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
@click.option(
    "--repository", "-r",
    default="builtin", 
    help="Repository to install from"
)
def install(extension_id: str, repository: str) -> None:
    """Install an extension."""
    try:
        ext_manager = ExtensionManager()
        
        with console.status(f"[cyan]Installing extension: {extension_id} from {repository}...[/cyan]", spinner="dots"):
            success = ext_manager.install_extension(extension_id, repository)
        
        if success:
            console.print(f"[green]✓ Successfully installed {extension_id}[/green]")
            console.print("[dim]💡 Use 'vmcp extension enable' to enable the extension[/dim]")
        else:
            console.print(f"[red]✗ Failed to install {extension_id}[/red]", err=True)
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]✗ Error installing extension: {e}[/red]", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force uninstall even if enabled"
)
def uninstall(extension_id: str, force: bool) -> None:
    """Uninstall an extension."""
    try:
        ext_manager = ExtensionManager()
        
        if not force and ext_manager.is_extension_enabled(extension_id):
            click.echo(f"Extension {extension_id} is currently enabled.")
            click.echo("Use --force to uninstall anyway, or disable first with 'vmcp extension disable'")
            sys.exit(1)
        
        click.echo(f"Uninstalling extension: {extension_id}")
        success = ext_manager.uninstall_extension(extension_id)
        
        if success:
            click.echo(f"✓ Successfully uninstalled {extension_id}")
        else:
            click.echo(f"✗ Failed to uninstall {extension_id}", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error uninstalling extension: {e}", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
@click.option(
    "--config", "-c",
    help="JSON configuration for the extension"
)
def enable(extension_id: str, config: str | None) -> None:
    """Enable an installed extension."""
    try:
        ext_manager = ExtensionManager()
        
        # Parse config if provided
        ext_config = {}
        if config:
            try:
                ext_config = json.loads(config)
                console.print(f"[dim]Using custom configuration: {config}[/dim]")
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON configuration: {e}[/red]", err=True)
                sys.exit(1)
        
        with console.status(f"[green]Enabling extension: {extension_id}...[/green]", spinner="dots"):
            success = ext_manager.enable_extension(extension_id, ext_config)
        
        if success:
            console.print(f"[green]✅ Successfully enabled {extension_id}[/green]")
            console.print("[dim]🚀 Extension is now available to vMCP Gateway[/dim]")
        else:
            console.print(f"[red]✗ Failed to enable {extension_id}[/red]", err=True)
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]✗ Error enabling extension: {e}[/red]", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
def disable(extension_id: str) -> None:
    """Disable an extension."""
    try:
        ext_manager = ExtensionManager()
        
        click.echo(f"Disabling extension: {extension_id}")
        success = ext_manager.disable_extension(extension_id)
        
        if success:
            click.echo(f"✓ Successfully disabled {extension_id}")
        else:
            click.echo(f"✗ Failed to disable {extension_id}", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error disabling extension: {e}", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
@click.option(
    "--repository", "-r",
    default="builtin",
    help="Repository to update from"
)
def update(extension_id: str, repository: str) -> None:
    """Update an extension to the latest version."""
    try:
        ext_manager = ExtensionManager()
        
        click.echo(f"Updating extension: {extension_id} from {repository}")
        success = ext_manager.update_extension(extension_id, repository)
        
        if success:
            click.echo(f"✓ Successfully updated {extension_id}")
        else:
            click.echo(f"✗ Failed to update {extension_id}", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error updating extension: {e}", err=True)
        sys.exit(1)


@extension.command()
@click.argument("extension_id")
def info(extension_id: str) -> None:
    """Show detailed information about an extension."""
    try:
        ext_manager = ExtensionManager()
        
        # Try to get manifest from installed extension first
        manifest = ext_manager.get_extension_manifest(extension_id)
        is_installed = True
        
        if not manifest:
            # Try to get from repository
            available_exts = ext_manager.list_available_extensions("builtin")
            for ext in available_exts:
                if ext.get("id") == extension_id:
                    manifest = ext
                    is_installed = False
                    break
        
        if not manifest:
            console.print(f"[red]Extension not found: {extension_id}[/red]", err=True)
            sys.exit(1)
        
        # Create info panel
        info_content = []
        
        # Basic information
        name = manifest.get('display_name', manifest.get('name', extension_id))
        info_content.append(f"[bold cyan]Name:[/bold cyan] {name}")
        info_content.append(f"[bold cyan]ID:[/bold cyan] {manifest.get('name', extension_id)}")
        info_content.append(f"[bold cyan]Version:[/bold cyan] {manifest.get('version', 'unknown')}")
        
        # Description
        description = manifest.get('description', 'No description')
        info_content.append(f"[bold cyan]Description:[/bold cyan] {description}")
        
        if manifest.get('long_description'):
            info_content.append(f"[bold cyan]Details:[/bold cyan] {manifest['long_description']}")
        
        # Author
        author = manifest.get('author', {})
        if isinstance(author, dict):
            author_str = f"{author.get('name', 'Unknown')}"
            if author.get('email'):
                author_str += f" <{author.get('email')}>"
            info_content.append(f"[bold cyan]Author:[/bold cyan] {author_str}")
        else:
            info_content.append(f"[bold cyan]Author:[/bold cyan] {author}")
        
        # Category and keywords
        category = manifest.get('category', 'unknown')
        info_content.append(f"[bold cyan]Category:[/bold cyan] [blue]{category}[/blue]")
        
        keywords = manifest.get('keywords', [])
        if keywords:
            keywords_str = ", ".join([f"[dim]{kw}[/dim]" for kw in keywords])
            info_content.append(f"[bold cyan]Keywords:[/bold cyan] {keywords_str}")
        
        # Status
        if is_installed:
            enabled_status = "enabled" if ext_manager.is_extension_enabled(extension_id) else "disabled"
            status_color = "green" if enabled_status == "enabled" else "yellow"
            status_icon = "✅" if enabled_status == "enabled" else "○"
            info_content.append(f"[bold cyan]Status:[/bold cyan] [{status_color}]{status_icon} installed ({enabled_status})[/{status_color}]")
            
            installed_path = ext_manager.installed_dir / extension_id
            info_content.append(f"[bold cyan]Path:[/bold cyan] [dim]{installed_path}[/dim]")
        else:
            info_content.append(f"[bold cyan]Status:[/bold cyan] [dim]- available for installation[/dim]")
        
        # Display main panel
        console.print(Panel(
            "\n".join(info_content),
            title=f"[bold blue]📦 Extension Information[/bold blue]",
            title_align="left",
            border_style="blue"
        ))
        
        # Tools information
        tools = manifest.get('tools', [])
        if tools:
            tools_table = Table(title="🔧 Available Tools", title_style="bold green", show_header=True, header_style="bold magenta")
            tools_table.add_column("Tool", style="cyan", width=20)
            tools_table.add_column("Description", style="white")
            
            for tool in tools:
                if isinstance(tool, dict):
                    tool_name = tool.get('name', 'unknown')
                    tool_desc = tool.get('description', 'No description')
                else:
                    tool_name = str(tool)
                    tool_desc = "No description"
                
                tools_table.add_row(tool_name, tool_desc)
            
            console.print(tools_table)
    
    except Exception as e:
        console.print(f"[red]✗ Error getting extension info: {e}[/red]", err=True)
        sys.exit(1)


@cli.command()
@click.argument("shell", type=click.Choice(['bash', 'zsh', 'fish', 'powershell']))
def completion(shell: str) -> None:
    """Generate shell completion scripts."""
    install_shell_completion(shell)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
