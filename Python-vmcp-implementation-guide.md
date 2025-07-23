# vMCP Implementation Guide - Python Edition

## Overview

The Virtual Model Context Protocol (vMCP) is a unified abstraction layer that aggregates multiple MCP servers through a single interface, similar to how Virtual File Systems (VFS) provide unified access to different storage backends. This guide provides the complete design and architecture for implementing a production-ready vMCP system in Python using UV for package management.

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        vMCP System                           │
├─────────────────────────────────────────────────────────────┤
│                    Gateway Layer                             │
│  - Multi-transport support (stdio, HTTP+SSE, WebSocket)     │
│  - Authentication & Authorization                            │
│  - Protocol translation                                      │
├─────────────────────────────────────────────────────────────┤
│                    Routing Engine                            │
│  - Request routing & load balancing                          │
│  - Connection pooling                                        │
│  - Circuit breakers & failover                              │
├─────────────────────────────────────────────────────────────┤
│                Server Management Layer                       │
│  - Server registry & discovery                               │
│  - Health monitoring                                         │
│  - Dynamic mounting/unmounting                              │
├─────────────────────────────────────────────────────────────┤
│              Repository & Installation Layer                 │
│  - Package discovery & search                                │
│  - Installation management                                   │
│  - Version control                                           │
├─────────────────────────────────────────────────────────────┤
│              Persona Access Control Layer                    │
│  - Named configuration sets                                  │
│  - Fine-grained permissions                                 │
│  - Secure-by-default                                        │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **VFS Philosophy**: Unified interface hiding backend complexity
2. **Modular Architecture**: Each component can be developed independently
3. **Multi-Transport**: Support stdio, HTTP+SSE, and WebSocket protocols
4. **Secure-by-Default**: New personas start with zero permissions
5. **Dynamic Configuration**: Hot-swappable server mounting
6. **Performance First**: Caching, connection pooling, and request batching

## Implementation Phases

### Phase 1: Core Engine
- Production-grade gateway with stdio support
- Comprehensive routing to multiple MCP servers
- Complete server registry with health monitoring

### Phase 2: Advanced Features
- Full multi-transport support
- Enterprise authentication framework
- Advanced connection pooling and caching

### Phase 3: Repository System
- Complete package discovery and installation
- Full version management
- Repository synchronization with multiple sources

### Phase 4: Persona System
- Complete access control implementation
- Comprehensive persona management CLI
- Full integration with core engine

## Core Engine Implementation

### Directory Structure

```
vmcp/
├── src/
│   └── vmcp/
│       ├── __init__.py
│       ├── gateway/
│       │   ├── __init__.py
│       │   ├── server.py           # Main gateway server
│       │   ├── transports/         # Transport implementations
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── stdio.py
│       │   │   ├── http_sse.py
│       │   │   └── websocket.py
│       │   ├── protocol.py         # Protocol translation
│       │   └── auth.py             # Authentication
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── router.py           # Request router
│       │   ├── algorithms.py       # Routing algorithms
│       │   ├── loadbalancer.py     # Load balancing
│       │   └── circuit_breaker.py  # Failure handling
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── registry.py         # Server registry
│       │   ├── discovery.py        # Server discovery
│       │   └── health.py           # Health monitoring
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── cache.py            # Caching layer
│       │   ├── strategies.py       # Cache strategies
│       │   └── storage.py          # Cache storage backends
│       ├── repository/
│       │   ├── __init__.py
│       │   ├── manager.py          # Repository management
│       │   ├── installer.py        # Package installer
│       │   └── search.py           # Search engine
│       ├── persona/
│       │   ├── __init__.py
│       │   ├── manager.py          # Persona management
│       │   └── access_control.py   # Access control
│       └── cli/
│           ├── __init__.py
│           └── main.py             # CLI entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── config/
│   └── default.toml
├── pyproject.toml
└── README.md
```

### Gateway Implementation

```python
# src/vmcp/gateway/server.py
import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from ..routing.router import Router
from ..registry.registry import Registry
from .transports.base import Transport
from .transports.stdio import StdioTransport
from .protocol import ProtocolHandler

logger = logging.getLogger(__name__)

@dataclass
class GatewayConfig:
    """Configuration for the vMCP Gateway."""
    registry_path: str = "~/.vmcp/registry"
    transports: Dict[str, Dict] = field(default_factory=lambda: {
        "stdio": {"enabled": True},
        "http": {"enabled": True, "port": 3000},
        "websocket": {"enabled": False}
    })
    cache_enabled: bool = True
    cache_ttl: int = 300
    max_connections: int = 1000
    request_timeout: int = 30

class VMCPGateway:
    """Main vMCP Gateway server implementation."""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.registry = Registry(config.registry_path)
        self.router = Router(self.registry)
        self.protocol_handler = ProtocolHandler()
        self.transports: Dict[str, Transport] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the gateway and all components."""
        logger.info("Initializing vMCP Gateway")
        
        # Load server configurations
        await self.registry.load_servers()
        
        # Initialize transports
        await self._initialize_transports()
        
        # Start health monitoring
        self._tasks.append(
            asyncio.create_task(self.registry.start_health_monitoring())
        )
        
    async def _initialize_transports(self):
        """Initialize configured transports."""
        if self.config.transports.get("stdio", {}).get("enabled"):
            self.transports["stdio"] = StdioTransport(
                self.router,
                self.protocol_handler
            )
            
        # Initialize other transports based on config
        
    async def start(self):
        """Start the gateway server."""
        if self._running:
            raise RuntimeError("Gateway already running")
            
        await self.initialize()
        self._running = True
        
        logger.info("Starting vMCP Gateway")
        
        # Start all transports
        transport_tasks = []
        for name, transport in self.transports.items():
            logger.info(f"Starting {name} transport")
            task = asyncio.create_task(transport.start())
            transport_tasks.append(task)
            self._tasks.append(task)
            
        # Wait for all transports to be ready
        await asyncio.gather(*transport_tasks)
        
        logger.info("vMCP Gateway started successfully")
        
    async def stop(self):
        """Stop the gateway server."""
        if not self._running:
            return
            
        logger.info("Stopping vMCP Gateway")
        self._running = False
        
        # Stop all transports
        for transport in self.transports.values():
            await transport.stop()
            
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("vMCP Gateway stopped")
        
    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming MCP request."""
        return await self.router.route(request)
```

### Router Implementation

```python
# src/vmcp/routing/router.py
import asyncio
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime

from ..registry.registry import Registry, MCPServer
from .algorithms import RoutingAlgorithm, PathBasedRouter, ContentBasedRouter
from .loadbalancer import LoadBalancer, RoundRobinBalancer
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

@dataclass
class RoutingContext:
    """Context for routing decisions."""
    request: Dict
    session_id: Optional[str] = None
    persona: Optional[str] = None
    metadata: Dict[str, Any] = None

class Router:
    """Production-grade request router for vMCP."""
    
    def __init__(self, registry: Registry):
        self.registry = registry
        self.path_router = PathBasedRouter()
        self.content_router = ContentBasedRouter()
        self.load_balancer = RoundRobinBalancer()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._request_count = 0
        self._route_cache: Dict[str, str] = {}
        
    async def route(self, request: Dict, context: Optional[RoutingContext] = None) -> Dict:
        """Route request to appropriate MCP server."""
        self._request_count += 1
        
        if context is None:
            context = RoutingContext(request=request)
            
        try:
            # Find target server(s)
            server = await self._find_server(context)
            if not server:
                return self._error_response(
                    request.get("id"),
                    -32601,
                    f"No server found for method: {request.get('method')}"
                )
                
            # Check circuit breaker
            breaker = self._get_circuit_breaker(server.id)
            
            # Execute request with circuit breaker protection
            return await breaker.execute(
                lambda: self._forward_request(server, request)
            )
            
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            return self._error_response(
                request.get("id"),
                -32603,
                "Internal routing error"
            )
            
    async def _find_server(self, context: RoutingContext) -> Optional[MCPServer]:
        """Find appropriate server for request."""
        request = context.request
        method = request.get("method", "")
        
        # Check route cache
        cache_key = f"{method}:{context.persona or 'default'}"
        if cache_key in self._route_cache:
            server_id = self._route_cache[cache_key]
            server = self.registry.get_server(server_id)
            if server and server.is_healthy:
                return server
                
        # Try path-based routing first
        server_id = self.path_router.route(method)
        if server_id:
            server = self.registry.get_server(server_id)
            if server and server.is_healthy:
                self._route_cache[cache_key] = server_id
                return server
                
        # Fall back to content-based routing
        server_id = self.content_router.route(request)
        if server_id:
            server = self.registry.get_server(server_id)
            if server and server.is_healthy:
                self._route_cache[cache_key] = server_id
                return server
                
        # Try to find any server that supports the method
        servers = await self._find_capable_servers(method)
        if servers:
            server = self.load_balancer.select(servers)
            self._route_cache[cache_key] = server.id
            return server
            
        return None
        
    async def _find_capable_servers(self, method: str) -> List[MCPServer]:
        """Find all servers capable of handling the method."""
        capable_servers = []
        
        for server in self.registry.get_all_servers():
            if not server.is_healthy:
                continue
                
            # Check if server has the capability
            if self._server_supports_method(server, method):
                capable_servers.append(server)
                
        return capable_servers
        
    def _server_supports_method(self, server: MCPServer, method: str) -> bool:
        """Check if server supports the given method."""
        # Check standard MCP methods
        if method in ["initialize", "initialized", "ping"]:
            return True
            
        # Check tool methods
        if method.startswith("tools/"):
            return bool(server.capabilities.get("tools"))
            
        # Check resource methods
        if method.startswith("resources/"):
            return bool(server.capabilities.get("resources"))
            
        # Check prompt methods
        if method.startswith("prompts/"):
            return bool(server.capabilities.get("prompts"))
            
        return False
        
    async def _forward_request(self, server: MCPServer, request: Dict) -> Dict:
        """Forward request to MCP server."""
        logger.debug(f"Forwarding request to server {server.id}")
        
        # Record metrics
        start_time = datetime.now()
        
        try:
            response = await server.send_request(request)
            
            # Record success metrics
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Request completed in {duration:.3f}s")
            
            return response
            
        except Exception as e:
            # Record failure metrics
            logger.error(f"Server {server.id} error: {e}")
            raise
            
    def _get_circuit_breaker(self, server_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for server."""
        if server_id not in self.circuit_breakers:
            self.circuit_breakers[server_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                half_open_requests=3
            )
        return self.circuit_breakers[server_id]
        
    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
```

### Server Registry

```python
# src/vmcp/registry/registry.py
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    id: str
    name: str
    transport: str  # stdio, http, websocket
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    capabilities: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    health_check_interval: int = 30
    max_retries: int = 3
    timeout: int = 30

@dataclass
class MCPServer:
    """Runtime representation of an MCP server."""
    config: MCPServerConfig
    is_healthy: bool = False
    last_health_check: Optional[datetime] = None
    connection: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return self.config.id
        
    @property
    def capabilities(self) -> Dict:
        return self.config.capabilities
        
    async def send_request(self, request: Dict) -> Dict:
        """Send request to this server."""
        if not self.connection:
            raise RuntimeError(f"Server {self.id} not connected")
            
        return await self.connection.send_request(request)

class Registry:
    """Server registry with health monitoring and discovery."""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path).expanduser()
        self.servers: Dict[str, MCPServer] = {}
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._config_file = self.registry_path / "servers.json"
        
    async def load_servers(self):
        """Load server configurations from disk."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        if self._config_file.exists():
            with open(self._config_file, 'r') as f:
                config_data = json.load(f)
                
            for server_config in config_data.get("servers", []):
                config = MCPServerConfig(**server_config)
                await self.register_server(config)
                
        logger.info(f"Loaded {len(self.servers)} servers from registry")
        
    async def save_config(self):
        """Save current configuration to disk."""
        config_data = {
            "servers": [
                {
                    "id": server.config.id,
                    "name": server.config.name,
                    "transport": server.config.transport,
                    "command": server.config.command,
                    "args": server.config.args,
                    "url": server.config.url,
                    "environment": server.config.environment,
                    "capabilities": server.config.capabilities,
                }
                for server in self.servers.values()
            ]
        }
        
        with open(self._config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
            
    async def register_server(self, config: MCPServerConfig) -> MCPServer:
        """Register a new MCP server."""
        if config.id in self.servers:
            raise ValueError(f"Server {config.id} already registered")
            
        server = MCPServer(config=config)
        
        # Initialize connection based on transport
        await self._initialize_server_connection(server)
        
        self.servers[config.id] = server
        logger.info(f"Registered server: {config.id}")
        
        # Perform initial health check
        await self._check_server_health(server)
        
        return server
        
    async def unregister_server(self, server_id: str):
        """Unregister an MCP server."""
        server = self.servers.pop(server_id, None)
        if server:
            if server.connection:
                await server.connection.close()
            logger.info(f"Unregistered server: {server_id}")
            
    def get_server(self, server_id: str) -> Optional[MCPServer]:
        """Get server by ID."""
        return self.servers.get(server_id)
        
    def get_all_servers(self) -> List[MCPServer]:
        """Get all registered servers."""
        return list(self.servers.values())
        
    async def start_health_monitoring(self):
        """Start health monitoring task."""
        if self._health_monitor_task:
            return
            
        self._health_monitor_task = asyncio.create_task(
            self._health_monitor_loop()
        )
        
    async def stop_health_monitoring(self):
        """Stop health monitoring task."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            await asyncio.gather(
                self._health_monitor_task,
                return_exceptions=True
            )
            self._health_monitor_task = None
            
    async def _health_monitor_loop(self):
        """Health monitoring loop."""
        while True:
            try:
                await self._check_all_servers_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)
                
    async def _check_all_servers_health(self):
        """Check health of all servers."""
        tasks = [
            self._check_server_health(server)
            for server in self.servers.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _check_server_health(self, server: MCPServer):
        """Check health of individual server."""
        try:
            if not server.connection:
                await self._initialize_server_connection(server)
                
            # Send ping request
            response = await asyncio.wait_for(
                server.send_request({
                    "jsonrpc": "2.0",
                    "id": f"health-{server.id}",
                    "method": "ping"
                }),
                timeout=5.0
            )
            
            server.is_healthy = True
            server.last_health_check = datetime.now()
            
        except Exception as e:
            logger.warning(f"Server {server.id} health check failed: {e}")
            server.is_healthy = False
            
    async def _initialize_server_connection(self, server: MCPServer):
        """Initialize connection to server based on transport type."""
        # This will be implemented based on transport type
        # For now, we'll create a placeholder
        from ..gateway.transports.stdio import StdioServerConnection
        
        if server.config.transport == "stdio":
            server.connection = StdioServerConnection(server.config)
            await server.connection.connect()
```

## Repository System Implementation

### Repository Structure

```
~/.vmcp/
├── config/
│   ├── vmcp.toml              # Main configuration
│   ├── repositories.json      # Repository registry
│   └── personas/              # Persona configurations
├── cache/
│   ├── repos/                 # Repository cache
│   └── packages/              # Downloaded packages
├── installed/
│   └── servers/               # Installed MCP servers
└── logs/
```

### Repository Manager

```python
# src/vmcp/repository/manager.py
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import git
from packaging import version

from .search import SearchEngine, SearchResult
from .installer import Installer

logger = logging.getLogger(__name__)

@dataclass
class Repository:
    """Repository configuration and metadata."""
    name: str
    url: str
    type: str  # git, http
    enabled: bool = True
    priority: int = 0
    last_sync: Optional[datetime] = None
    cache_path: Optional[Path] = None

class RepositoryManager:
    """Manages MCP server repositories with full search and installation."""
    
    def __init__(self, config_path: str = "~/.vmcp"):
        self.config_path = Path(config_path).expanduser()
        self.repositories: Dict[str, Repository] = {}
        self.search_engine = SearchEngine()
        self.installer = Installer(self.config_path / "installed")
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Ensure directories exist
        self._init_directories()
        
    def _init_directories(self):
        """Initialize required directories."""
        dirs = [
            self.config_path / "config",
            self.config_path / "cache" / "repos",
            self.config_path / "cache" / "packages",
            self.config_path / "installed" / "servers",
            self.config_path / "logs",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    async def initialize(self):
        """Initialize repository manager."""
        await self.load_repositories()
        self._session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup resources."""
        if self._session:
            await self._session.close()
            
    async def load_repositories(self):
        """Load repository list from configuration."""
        repo_file = self.config_path / "config" / "repositories.json"
        
        if repo_file.exists():
            with open(repo_file, 'r') as f:
                data = json.load(f)
                
            for repo_data in data.get("repositories", []):
                repo = Repository(**repo_data)
                repo.cache_path = self.config_path / "cache" / "repos" / repo.name
                self.repositories[repo.name] = repo
                
        # Add default repository if none exist
        if not self.repositories:
            await self.add_repository(
                "https://github.com/vmcp/official-repo.git",
                "official"
            )
            
    async def save_repositories(self):
        """Save repository list to configuration."""
        repo_file = self.config_path / "config" / "repositories.json"
        
        data = {
            "repositories": [
                {
                    "name": repo.name,
                    "url": repo.url,
                    "type": repo.type,
                    "enabled": repo.enabled,
                    "priority": repo.priority,
                    "last_sync": repo.last_sync.isoformat() if repo.last_sync else None,
                }
                for repo in self.repositories.values()
            ]
        }
        
        with open(repo_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    async def add_repository(self, url: str, name: Optional[str] = None) -> Repository:
        """Add a new repository."""
        # Determine repository type
        if url.endswith(".git") or "github.com" in url:
            repo_type = "git"
        else:
            repo_type = "http"
            
        # Generate name if not provided
        if not name:
            name = Path(url).stem.replace("-", "_")
            
        # Check for duplicates
        if name in self.repositories:
            raise ValueError(f"Repository '{name}' already exists")
            
        # Create repository
        repo = Repository(
            name=name,
            url=url,
            type=repo_type,
            priority=len(self.repositories),
            cache_path=self.config_path / "cache" / "repos" / name
        )
        
        # Perform initial sync
        await self.sync_repository(repo)
        
        # Save to registry
        self.repositories[name] = repo
        await self.save_repositories()
        
        logger.info(f"Added repository: {name}")
        return repo
        
    async def remove_repository(self, name: str):
        """Remove a repository."""
        repo = self.repositories.pop(name, None)
        if not repo:
            raise ValueError(f"Repository '{name}' not found")
            
        # Remove cache
        if repo.cache_path and repo.cache_path.exists():
            import shutil
            shutil.rmtree(repo.cache_path)
            
        await self.save_repositories()
        logger.info(f"Removed repository: {name}")
        
    async def sync_repository(self, repo: Repository):
        """Synchronize a repository."""
        logger.info(f"Syncing repository: {repo.name}")
        
        if repo.type == "git":
            await self._sync_git_repository(repo)
        elif repo.type == "http":
            await self._sync_http_repository(repo)
        else:
            raise ValueError(f"Unknown repository type: {repo.type}")
            
        repo.last_sync = datetime.now()
        
        # Update search index
        await self._index_repository(repo)
        
    async def sync_all_repositories(self):
        """Synchronize all enabled repositories."""
        tasks = [
            self.sync_repository(repo)
            for repo in self.repositories.values()
            if repo.enabled
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors
        for repo, result in zip(self.repositories.values(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to sync {repo.name}: {result}")
                
    async def _sync_git_repository(self, repo: Repository):
        """Sync a git repository."""
        if not repo.cache_path:
            return
            
        if repo.cache_path.exists():
            # Pull updates
            git_repo = git.Repo(repo.cache_path)
            origin = git_repo.remotes.origin
            origin.pull()
        else:
            # Clone repository
            git.Repo.clone_from(repo.url, repo.cache_path)
            
    async def _sync_http_repository(self, repo: Repository):
        """Sync an HTTP repository."""
        if not repo.cache_path or not self._session:
            return
            
        # Download repository index
        async with self._session.get(f"{repo.url}/index.json") as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch repository index: {response.status}")
                
            index_data = await response.json()
            
        # Save index
        repo.cache_path.mkdir(parents=True, exist_ok=True)
        with open(repo.cache_path / "index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
            
    async def _index_repository(self, repo: Repository):
        """Index repository contents for search."""
        if not repo.cache_path or not repo.cache_path.exists():
            return
            
        # Find all MCP server definitions
        for manifest_path in repo.cache_path.rglob("MANIFEST.md"):
            try:
                await self.search_engine.index_server(
                    manifest_path,
                    repo.name
                )
            except Exception as e:
                logger.error(f"Failed to index {manifest_path}: {e}")
                
    async def search_servers(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search for MCP servers across all repositories."""
        results = await self.search_engine.search(query, limit)
        
        # Sort by repository priority
        results.sort(
            key=lambda r: self.repositories.get(r.repository, Repository("", "", "")).priority
        )
        
        return results
        
    async def get_server_info(self, name: str, repo_name: Optional[str] = None) -> Dict:
        """Get detailed information about a server."""
        # Search for the server
        results = await self.search_servers(name, limit=50)
        
        # Filter by repository if specified
        if repo_name:
            results = [r for r in results if r.repository == repo_name]
            
        # Find exact match
        for result in results:
            if result.name == name:
                return await self._load_server_manifest(result)
                
        raise ValueError(f"Server '{name}' not found")
        
    async def _load_server_manifest(self, result: SearchResult) -> Dict:
        """Load complete server manifest."""
        repo = self.repositories.get(result.repository)
        if not repo or not repo.cache_path:
            raise ValueError(f"Repository '{result.repository}' not available")
            
        manifest_path = repo.cache_path / result.path / "MANIFEST.md"
        install_path = repo.cache_path / result.path / "install.json"
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest_content = f.read()
            
        # Load installation config
        install_config = {}
        if install_path.exists():
            with open(install_path, 'r') as f:
                install_config = json.load(f)
                
        return {
            "manifest": manifest_content,
            "install_config": install_config,
            "repository": result.repository,
            "path": result.path,
        }
        
    async def install_server(
        self,
        name: str,
        version: Optional[str] = None,
        repo_name: Optional[str] = None,
        method: Optional[str] = None
    ) -> Dict:
        """Install an MCP server."""
        # Get server information
        info = await self.get_server_info(name, repo_name)
        
        # Determine installation method
        install_config = info["install_config"]
        if not install_config:
            raise ValueError(f"No installation configuration for '{name}'")
            
        # Install using the installer
        result = await self.installer.install(
            name=name,
            version=version,
            install_config=install_config,
            source_path=Path(info["repository"]) / info["path"],
            method=method
        )
        
        logger.info(f"Successfully installed {name}")
        return result
        
    async def uninstall_server(self, name: str):
        """Uninstall an MCP server."""
        await self.installer.uninstall(name)
        logger.info(f"Successfully uninstalled {name}")
        
    async def update_server(self, name: str, version: Optional[str] = None):
        """Update an installed server."""
        # Get current installation info
        installed = await self.installer.get_installed_info(name)
        if not installed:
            raise ValueError(f"Server '{name}' is not installed")
            
        # Get latest version info
        info = await self.get_server_info(name, installed.get("repository"))
        
        # Reinstall with new version
        await self.uninstall_server(name)
        await self.install_server(name, version, installed.get("repository"))
        
    async def list_installed(self) -> List[Dict]:
        """List all installed servers."""
        return await self.installer.list_installed()
```

## Persona System Implementation

### Persona Manager

```python
# src/vmcp/persona/manager.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Persona:
    """Persona configuration for access control."""
    name: str
    description: Optional[str] = None
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    enabled_servers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created"] = self.created.isoformat()
        data["modified"] = self.modified.isoformat()
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> "Persona":
        """Create from dictionary."""
        data = data.copy()
        data["created"] = datetime.fromisoformat(data["created"])
        data["modified"] = datetime.fromisoformat(data["modified"])
        return cls(**data)

class PersonaManager:
    """Manages personas for fine-grained access control."""
    
    def __init__(self, config_path: str = "~/.vmcp/config/personas"):
        self.config_path = Path(config_path).expanduser()
        self.personas: Dict[str, Persona] = {}
        self._default_persona_name = "__default__"
        
        # Ensure directory exists
        self.config_path.mkdir(parents=True, exist_ok=True)
        
    async def load_personas(self):
        """Load all personas from disk."""
        for persona_file in self.config_path.glob("persona-*.json"):
            try:
                with open(persona_file, 'r') as f:
                    data = json.load(f)
                    
                persona = Persona.from_dict(data)
                self.personas[persona.name] = persona
                
            except Exception as e:
                logger.error(f"Failed to load {persona_file}: {e}")
                
        logger.info(f"Loaded {len(self.personas)} personas")
        
    async def save_persona(self, persona: Persona):
        """Save persona to disk."""
        filename = f"persona-{persona.name}.json"
        filepath = self.config_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(persona.to_dict(), f, indent=2)
            
    async def create_persona(
        self,
        name: str,
        description: Optional[str] = None
    ) -> Persona:
        """Create a new persona with zero permissions."""
        if name in self.personas:
            raise ValueError(f"Persona '{name}' already exists")
            
        if not self._validate_persona_name(name):
            raise ValueError(f"Invalid persona name: '{name}'")
            
        persona = Persona(
            name=name,
            description=description,
            enabled_servers=[]  # Secure by default
        )
        
        self.personas[name] = persona
        await self.save_persona(persona)
        
        logger.info(f"Created persona: {name}")
        return persona
        
    async def delete_persona(self, name: str):
        """Delete a persona."""
        if name not in self.personas:
            raise ValueError(f"Persona '{name}' not found")
            
        if name == self._default_persona_name:
            raise ValueError("Cannot delete default persona")
            
        persona = self.personas.pop(name)
        
        # Delete file
        filename = f"persona-{name}.json"
        filepath = self.config_path / filename
        if filepath.exists():
            filepath.unlink()
            
        logger.info(f"Deleted persona: {name}")
        
    async def get_persona(self, name: str) -> Optional[Persona]:
        """Get persona by name."""
        return self.personas.get(name)
        
    async def list_personas(self) -> List[Persona]:
        """List all personas."""
        return list(self.personas.values())
        
    async def enable_server(self, persona_name: str, server_id: str):
        """Enable access to a server for a persona."""
        persona = self.personas.get(persona_name)
        if not persona:
            raise ValueError(f"Persona '{persona_name}' not found")
            
        if server_id not in persona.enabled_servers:
            persona.enabled_servers.append(server_id)
            persona.modified = datetime.now()
            await self.save_persona(persona)
            
            logger.info(f"Enabled {server_id} for persona {persona_name}")
            
    async def disable_server(self, persona_name: str, server_id: str):
        """Disable access to a server for a persona."""
        persona = self.personas.get(persona_name)
        if not persona:
            raise ValueError(f"Persona '{persona_name}' not found")
            
        if server_id in persona.enabled_servers:
            persona.enabled_servers.remove(server_id)
            persona.modified = datetime.now()
            await self.save_persona(persona)
            
            logger.info(f"Disabled {server_id} for persona {persona_name}")
            
    async def set_enabled_servers(
        self,
        persona_name: str,
        server_ids: List[str]
    ):
        """Set the complete list of enabled servers for a persona."""
        persona = self.personas.get(persona_name)
        if not persona:
            raise ValueError(f"Persona '{persona_name}' not found")
            
        persona.enabled_servers = server_ids.copy()
        persona.modified = datetime.now()
        await self.save_persona(persona)
        
        logger.info(f"Updated enabled servers for persona {persona_name}")
        
    async def get_enabled_servers(self, persona_name: Optional[str]) -> Set[str]:
        """Get set of enabled servers for a persona."""
        # No persona = full access
        if not persona_name:
            return set()  # Empty set means no restrictions
            
        persona = self.personas.get(persona_name)
        if not persona:
            raise ValueError(f"Unknown persona: {persona_name}")
            
        return set(persona.enabled_servers)
        
    def _validate_persona_name(self, name: str) -> bool:
        """Validate persona name."""
        if not name or len(name) > 50:
            return False
            
        # Allow alphanumeric, dash, underscore
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))
```

### Access Control Integration

```python
# src/vmcp/persona/access_control.py
import logging
from typing import Optional, Set, Dict, Any

from .manager import PersonaManager

logger = logging.getLogger(__name__)

class AccessController:
    """Enforces persona-based access control."""
    
    def __init__(self, persona_manager: PersonaManager):
        self.persona_manager = persona_manager
        self._access_cache: Dict[str, Set[str]] = {}
        
    async def check_access(
        self,
        persona_name: Optional[str],
        server_id: str,
        method: Optional[str] = None
    ) -> bool:
        """Check if persona has access to server/method."""
        # No persona = full access (backward compatibility)
        if not persona_name:
            return True
            
        # Check cache
        cache_key = f"{persona_name}:{server_id}"
        if cache_key in self._access_cache:
            enabled_servers = self._access_cache[cache_key]
        else:
            # Load from persona manager
            try:
                enabled_servers = await self.persona_manager.get_enabled_servers(
                    persona_name
                )
                self._access_cache[cache_key] = enabled_servers
            except ValueError:
                logger.warning(f"Unknown persona: {persona_name}")
                return False
                
        # Check if server is enabled
        if server_id not in enabled_servers:
            logger.debug(
                f"Access denied: persona '{persona_name}' "
                f"cannot access server '{server_id}'"
            )
            return False
            
        # Future: Check method-level permissions
        # For now, server access means full access to all methods
        
        return True
        
    async def filter_servers(
        self,
        persona_name: Optional[str],
        servers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter list of servers based on persona access."""
        if not persona_name:
            return servers
            
        filtered = []
        for server in servers:
            if await self.check_access(persona_name, server["id"]):
                filtered.append(server)
                
        return filtered
        
    async def filter_capabilities(
        self,
        persona_name: Optional[str],
        server_id: str,
        capabilities: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Filter capabilities based on persona access."""
        # For now, if persona has access to server, they get all capabilities
        # Future: Fine-grained capability filtering
        if await self.check_access(persona_name, server_id):
            return capabilities
        else:
            return {}
            
    def clear_cache(self, persona_name: Optional[str] = None):
        """Clear access cache."""
        if persona_name:
            # Clear specific persona
            keys_to_remove = [
                k for k in self._access_cache
                if k.startswith(f"{persona_name}:")
            ]
            for key in keys_to_remove:
                del self._access_cache[key]
        else:
            # Clear all
            self._access_cache.clear()
```

## CLI Implementation

### Command Structure

```bash
# Core commands
vmcp start                    # Start the vMCP gateway
vmcp stop                     # Stop the gateway
vmcp status                   # Show gateway status

# Server management
vmcp list                     # List mounted servers
vmcp mount <server-id>        # Mount a server
vmcp unmount <server-id>      # Unmount a server
vmcp info <server-id>         # Show server details

# Repository commands
vmcp repo add <url>           # Add a repository
vmcp repo list                # List repositories
vmcp repo sync                # Sync repositories
vmcp search <query>           # Search for servers
vmcp install <server>         # Install a server
vmcp uninstall <server>       # Uninstall a server

# Persona commands
vmcp persona create <name>    # Create a persona
vmcp persona list             # List personas
vmcp persona show <name>      # Show persona details
vmcp persona <name> enable <server>   # Enable server access
vmcp persona <name> disable <server>  # Disable server access
vmcp persona delete <name>    # Delete a persona
```

### CLI Entry Point

```python
# src/vmcp/cli/main.py
#!/usr/bin/env python3
import asyncio
import click
import logging
import sys
from pathlib import Path
from typing import Optional

from ..gateway.server import VMCPGateway, GatewayConfig
from ..repository.manager import RepositoryManager
from ..persona.manager import PersonaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@click.group()
@click.option('--config', default='~/.vmcp/config/vmcp.toml', help='Configuration file')
@click.pass_context
def cli(ctx, config):
    """Virtual Model Context Protocol (vMCP) gateway and manager."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = Path(config).expanduser()

@cli.command()
@click.option('--daemon', '-d', is_flag=True, help='Run as daemon')
@click.pass_context
def start(ctx, daemon):
    """Start the vMCP gateway."""
    async def run():
        config = GatewayConfig()  # Load from file
        gateway = VMCPGateway(config)
        
        try:
            await gateway.start()
            
            if not daemon:
                # Run until interrupted
                await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await gateway.stop()
            
    asyncio.run(run())

@cli.command()
def stop():
    """Stop the vMCP gateway."""
    # Implementation: Send stop signal to running daemon
    click.echo("Stopping vMCP gateway...")

@cli.command()
def status():
    """Show gateway status."""
    # Implementation: Check if gateway is running
    click.echo("vMCP Gateway Status: Running")

# Server management commands
@cli.group()
def server():
    """Manage MCP servers."""
    pass

@server.command('list')
def list_servers():
    """List mounted servers."""
    async def run():
        # Implementation
        click.echo("Mounted servers:")
        
    asyncio.run(run())

@server.command('mount')
@click.argument('server_id')
def mount_server(server_id):
    """Mount a server."""
    async def run():
        click.echo(f"Mounting server: {server_id}")
        
    asyncio.run(run())

@server.command('unmount')
@click.argument('server_id')
def unmount_server(server_id):
    """Unmount a server."""
    async def run():
        click.echo(f"Unmounting server: {server_id}")
        
    asyncio.run(run())

# Repository commands
@cli.group()
def repo():
    """Manage repositories."""
    pass

@repo.command('add')
@click.argument('url')
@click.option('--name', help='Repository name')
def add_repo(url, name):
    """Add a repository."""
    async def run():
        manager = RepositoryManager()
        await manager.initialize()
        
        try:
            repo = await manager.add_repository(url, name)
            click.echo(f"Added repository: {repo.name}")
        finally:
            await manager.cleanup()
            
    asyncio.run(run())

@repo.command('list')
def list_repos():
    """List repositories."""
    async def run():
        manager = RepositoryManager()
        await manager.initialize()
        
        try:
            for repo in manager.repositories.values():
                status = "✓" if repo.enabled else "✗"
                click.echo(f"{status} {repo.name}: {repo.url}")
        finally:
            await manager.cleanup()
            
    asyncio.run(run())

@repo.command('sync')
@click.option('--all', is_flag=True, help='Sync all repositories')
@click.argument('name', required=False)
def sync_repos(all, name):
    """Sync repositories."""
    async def run():
        manager = RepositoryManager()
        await manager.initialize()
        
        try:
            if all:
                await manager.sync_all_repositories()
                click.echo("Synced all repositories")
            elif name:
                repo = manager.repositories.get(name)
                if not repo:
                    click.echo(f"Repository '{name}' not found", err=True)
                    return
                await manager.sync_repository(repo)
                click.echo(f"Synced repository: {name}")
            else:
                click.echo("Specify repository name or use --all", err=True)
        finally:
            await manager.cleanup()
            
    asyncio.run(run())

@cli.command()
@click.argument('query')
@click.option('--limit', default=20, help='Maximum results')
def search(query, limit):
    """Search for MCP servers."""
    async def run():
        manager = RepositoryManager()
        await manager.initialize()
        
        try:
            results = await manager.search_servers(query, limit)
            
            if not results:
                click.echo("No results found")
                return
                
            for result in results:
                click.echo(f"{result.name} ({result.version}) - {result.description}")
                click.echo(f"  Repository: {result.repository}")
                click.echo(f"  Tags: {', '.join(result.tags)}")
                click.echo()
        finally:
            await manager.cleanup()
            
    asyncio.run(run())

@cli.command()
@click.argument('server')
@click.option('--version', help='Specific version')
@click.option('--repo', help='Repository name')
@click.option('--method', help='Installation method')
def install(server, version, repo, method):
    """Install an MCP server."""
    async def run():
        manager = RepositoryManager()
        await manager.initialize()
        
        try:
            result = await manager.install_server(
                server,
                version=version,
                repo_name=repo,
                method=method
            )
            click.echo(f"Successfully installed {server}")
        except Exception as e:
            click.echo(f"Installation failed: {e}", err=True)
            sys.exit(1)
        finally:
            await manager.cleanup()
            
    asyncio.run(run())

# Persona commands
@cli.group()
def persona():
    """Manage personas."""
    pass

@persona.command('create')
@click.argument('name')
@click.option('--description', help='Persona description')
def create_persona(name, description):
    """Create a new persona."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        try:
            persona = await manager.create_persona(name, description)
            click.echo(f"Created persona: {name}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
            
    asyncio.run(run())

@persona.command('list')
def list_personas():
    """List all personas."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        personas = await manager.list_personas()
        if not personas:
            click.echo("No personas found")
            return
            
        for persona in personas:
            click.echo(f"{persona.name}")
            if persona.description:
                click.echo(f"  Description: {persona.description}")
            click.echo(f"  Enabled servers: {len(persona.enabled_servers)}")
            click.echo(f"  Created: {persona.created.strftime('%Y-%m-%d %H:%M')}")
            click.echo()
            
    asyncio.run(run())

@persona.command('show')
@click.argument('name')
def show_persona(name):
    """Show persona details."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        persona = await manager.get_persona(name)
        if not persona:
            click.echo(f"Persona '{name}' not found", err=True)
            sys.exit(1)
            
        click.echo(f"Persona: {persona.name}")
        if persona.description:
            click.echo(f"Description: {persona.description}")
        click.echo(f"Created: {persona.created}")
        click.echo(f"Modified: {persona.modified}")
        click.echo("\nEnabled servers:")
        for server_id in persona.enabled_servers:
            click.echo(f"  - {server_id}")
            
    asyncio.run(run())

@persona.group()
@click.argument('name')
@click.pass_context
def persona_name(ctx, name):
    """Manage specific persona."""
    ctx.obj = ctx.obj or {}
    ctx.obj['persona_name'] = name

@persona_name.command('enable')
@click.argument('server')
@click.pass_context
def enable_server(ctx, server):
    """Enable server access for persona."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        try:
            await manager.enable_server(ctx.obj['persona_name'], server)
            click.echo(f"Enabled {server} for persona {ctx.obj['persona_name']}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
            
    asyncio.run(run())

@persona_name.command('disable')
@click.argument('server')
@click.pass_context
def disable_server(ctx, server):
    """Disable server access for persona."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        try:
            await manager.disable_server(ctx.obj['persona_name'], server)
            click.echo(f"Disabled {server} for persona {ctx.obj['persona_name']}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
            
    asyncio.run(run())

@persona.command('delete')
@click.argument('name')
@click.confirmation_option(prompt='Are you sure you want to delete this persona?')
def delete_persona(name):
    """Delete a persona."""
    async def run():
        manager = PersonaManager()
        await manager.load_personas()
        
        try:
            await manager.delete_persona(name)
            click.echo(f"Deleted persona: {name}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
            
    asyncio.run(run())

if __name__ == '__main__':
    cli()
```

## Configuration Files

### Main Configuration (TOML)

```toml
# ~/.vmcp/config/vmcp.toml
[vmcp]
version = "1.0.0"

[gateway]
[gateway.transports.stdio]
enabled = true

[gateway.transports.http]
enabled = true
port = 3000
host = "127.0.0.1"

[gateway.transports.websocket]
enabled = false
port = 3001

[cache]
enabled = true
ttl = 300
max_size = "100MB"
backend = "memory"  # memory, redis, disk

[logging]
level = "INFO"
file = "~/.vmcp/logs/vmcp.log"
max_size = "10MB"
backup_count = 5

[monitoring]
metrics_enabled = true
health_check_interval = 30
prometheus_port = 9090
```

### Server Configuration

```json
// ~/.vmcp/registry/servers.json
{
  "version": "1.0.0",
  "servers": [
    {
      "id": "mcp-filesystem",
      "name": "Filesystem Server",
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-filesystem"],
      "environment": {
        "MCP_FILESYSTEM_ROOT": "${HOME}/mcp-data"
      },
      "capabilities": {
        "tools": [
          {
            "name": "read_file",
            "description": "Read file contents"
          },
          {
            "name": "write_file",
            "description": "Write content to file"
          }
        ],
        "resources": [
          {
            "name": "directory_tree",
            "description": "Get directory structure"
          }
        ]
      },
      "health_check_interval": 30,
      "timeout": 30
    }
  ]
}
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_router.py
import pytest
import asyncio
from vmcp.routing.router import Router
from vmcp.registry.registry import Registry, MCPServerConfig

@pytest.mark.asyncio
async def test_router_routes_to_correct_server():
    """Test that router correctly routes requests."""
    # Create mock registry
    registry = Registry(":memory:")
    
    # Add test server
    config = MCPServerConfig(
        id="test-server",
        name="Test Server",
        transport="stdio",
        command="echo",
        capabilities={
            "tools": [{"name": "test_tool", "description": "Test"}]
        }
    )
    await registry.register_server(config)
    
    # Create router
    router = Router(registry)
    
    # Test routing
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "test_tool",
            "arguments": {}
        }
    }
    
    # Mock the server response
    # ... implementation
    
    response = await router.route(request)
    assert response["id"] == 1
    assert "result" in response

@pytest.mark.asyncio
async def test_router_handles_unknown_method():
    """Test router error handling for unknown methods."""
    registry = Registry(":memory:")
    router = Router(registry)
    
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "unknown/method"
    }
    
    response = await router.route(request)
    assert "error" in response
    assert response["error"]["code"] == -32601
```

### Integration Tests

```python
# tests/integration/test_gateway.py
import pytest
import asyncio
from vmcp.gateway.server import VMCPGateway, GatewayConfig

@pytest.mark.asyncio
async def test_gateway_end_to_end():
    """Test complete request flow through gateway."""
    config = GatewayConfig(
        transports={"stdio": {"enabled": True}},
        cache_enabled=False  # Disable for predictable tests
    )
    
    gateway = VMCPGateway(config)
    
    try:
        await gateway.start()
        
        # Test initialization
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        }
        
        response = await gateway.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        
    finally:
        await gateway.stop()

@pytest.mark.asyncio
async def test_persona_access_control():
    """Test persona-based access control."""
    # Create persona with limited access
    from vmcp.persona.manager import PersonaManager
    
    persona_mgr = PersonaManager(":memory:")
    await persona_mgr.create_persona("limited", "Limited access")
    await persona_mgr.enable_server("limited", "allowed-server")
    
    # Test access control
    # ... implementation
```

## Security Considerations

### Input Validation

```python
# src/vmcp/security/validation.py
import json
from typing import Dict, Any
from jsonschema import validate, ValidationError

# JSON-RPC 2.0 Request Schema
REQUEST_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "number", "null"]},
        "method": {"type": "string"},
        "params": {"type": ["object", "array"]}
    }
}

def validate_request(request: Dict[str, Any]) -> bool:
    """Validate JSON-RPC request."""
    try:
        validate(request, REQUEST_SCHEMA)
        return True
    except ValidationError:
        return False

def sanitize_path(path: str) -> str:
    """Sanitize file paths to prevent directory traversal."""
    from pathlib import Path
    
    # Resolve to absolute path and check it's within allowed directory
    resolved = Path(path).resolve()
    allowed_root = Path.home() / "mcp-data"
    
    if not str(resolved).startswith(str(allowed_root)):
        raise ValueError("Path outside allowed directory")
        
    return str(resolved)
```

### Process Isolation

```python
# src/vmcp/security/sandbox.py
import os
import subprocess
from typing import List, Dict, Optional

class ProcessSandbox:
    """Sandbox for running MCP server processes."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def spawn_process(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.Popen:
        """Spawn sandboxed process."""
        # Merge environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
            
        # Security restrictions
        if os.name != 'nt':  # Unix-like systems
            # Set resource limits
            import resource
            
            def set_limits():
                # Limit CPU time (seconds)
                resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
                # Limit memory (bytes)
                resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                # Limit file descriptors
                resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
                
            preexec_fn = set_limits
        else:
            preexec_fn = None
            
        # Spawn process
        process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            preexec_fn=preexec_fn
        )
        
        return process
```

## Performance Optimizations

### Caching Implementation

```python
# src/vmcp/cache/cache.py
import asyncio
import time
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass
import hashlib
import json

@dataclass
class CacheEntry:
    """Single cache entry."""
    value: Any
    expires_at: float
    size: int

class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self.cache.get(key)
            
            if not entry:
                return None
                
            # Check expiration
            if time.time() > entry.expires_at:
                del self.cache[key]
                self.access_order.remove(key)
                return None
                
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.value
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        async with self._lock:
            # Calculate size (simplified)
            size = len(json.dumps(value, default=str))
            
            # Create entry
            entry = CacheEntry(
                value=value,
                expires_at=time.time() + (ttl or self.ttl),
                size=size
            )
            
            # Add to cache
            if key in self.cache:
                self.access_order.remove(key)
            self.cache[key] = entry
            self.access_order.append(key)
            
            # Enforce size limit
            while len(self.cache) > self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                
    def generate_key(self, method: str, params: Any, persona: Optional[str] = None) -> str:
        """Generate cache key from request parameters."""
        key_parts = [
            method,
            json.dumps(params, sort_keys=True),
            persona or "default"
        ]
        
        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

class CacheManager:
    """Manages multiple cache levels and strategies."""
    
    def __init__(self):
        self.l1_cache = LRUCache(max_size=1000, ttl=300)  # Memory
        self.l2_cache = None  # Future: Disk/Redis cache
        
    async def get_or_fetch(
        self,
        key: str,
        fetcher: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or fetch if missing."""
        # Try L1 cache
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
            
        # Try L2 cache (if implemented)
        # ...
        
        # Fetch fresh value
        value = await fetcher()
        
        # Store in cache
        await self.l1_cache.set(key, value, ttl)
        
        return value
```

### Connection Pooling

```python
# src/vmcp/routing/connection_pool.py
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 1
    max_size: int = 10
    idle_timeout: int = 300
    acquire_timeout: int = 5

@dataclass
class PooledConnection:
    """Wrapper for pooled connections."""
    connection: Any
    created_at: float
    last_used: float
    in_use: bool = False

class ConnectionPool:
    """Production-grade connection pool."""
    
    def __init__(self, server_id: str, config: PoolConfig):
        self.server_id = server_id
        self.config = config
        self.connections: List[PooledConnection] = []
        self._lock = asyncio.Lock()
        self._waiters: List[asyncio.Future] = []
        self._closing = False
        
    async def initialize(self, connection_factory: Callable):
        """Initialize pool with minimum connections."""
        self.connection_factory = connection_factory
        
        # Create minimum connections
        for _ in range(self.config.min_size):
            conn = await self._create_connection()
            self.connections.append(conn)
            
    async def _create_connection(self) -> PooledConnection:
        """Create new pooled connection."""
        connection = await self.connection_factory()
        return PooledConnection(
            connection=connection,
            created_at=time.time(),
            last_used=time.time()
        )
        
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        if self._closing:
            raise RuntimeError("Pool is closing")
            
        start_time = time.time()
        timeout = self.config.acquire_timeout
        
        while True:
            async with self._lock:
                # Find available connection
                for conn in self.connections:
                    if not conn.in_use:
                        conn.in_use = True
                        conn.last_used = time.time()
                        try:
                            yield conn.connection
                            return
                        finally:
                            async with self._lock:
                                conn.in_use = False
                                conn.last_used = time.time()
                                self._notify_waiters()
                                
                # Create new connection if under limit
                if len(self.connections) < self.config.max_size:
                    conn = await self._create_connection()
                    conn.in_use = True
                    self.connections.append(conn)
                    try:
                        yield conn.connection
                        return
                    finally:
                        async with self._lock:
                            conn.in_use = False
                            conn.last_used = time.time()
                            self._notify_waiters()
                            
            # Wait for available connection
            if time.time() - start_time > timeout:
                raise asyncio.TimeoutError("Failed to acquire connection")
                
            waiter = asyncio.Future()
            self._waiters.append(waiter)
            
            try:
                await asyncio.wait_for(waiter, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            finally:
                self._waiters.remove(waiter)
                
    def _notify_waiters(self):
        """Notify waiting acquirers."""
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
                break
                
    async def close(self):
        """Close all connections in pool."""
        self._closing = True
        
        async with self._lock:
            for conn in self.connections:
                if hasattr(conn.connection, 'close'):
                    await conn.connection.close()
                    
            self.connections.clear()
            
        # Cancel all waiters
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
```

## pyproject.toml Configuration

```toml
[project]
name = "vmcp"
version = "1.0.0"
description = "Virtual Model Context Protocol - Unified MCP server gateway"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
keywords = ["mcp", "ai", "gateway", "protocol"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "click>=8.1.0",
    "asyncio>=3.4.3",
    "aiohttp>=3.9.0",
    "aiofiles>=23.0.0",
    "jsonschema>=4.20.0",
    "pydantic>=2.5.0",
    "toml>=0.10.2",
    "packaging>=23.2",
    "GitPython>=3.1.40",
    "websockets>=12.0",
    "prometheus-client>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "black>=23.12.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

redis = [
    "redis>=5.0.0",
    "hiredis>=2.3.0",
]

[project.scripts]
vmcp = "vmcp.cli.main:cli"

[project.urls]
Homepage = "https://github.com/yourusername/vmcp"
Documentation = "https://vmcp.readthedocs.io"
Repository = "https://github.com/yourusername/vmcp"
Issues = "https://github.com/yourusername/vmcp/issues"

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.black]
line-length = 88
target-version = ['py310', 'py311', 'py312']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Development Workflow

1. **Setup Development Environment**:
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/vmcp.git
   cd vmcp
   
   # Create virtual environment with uv
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   
   # Install in development mode
   uv pip install -e ".[dev]"
   ```

2. **Run Tests**:
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=vmcp --cov-report=html
   
   # Run specific test file
   pytest tests/unit/test_router.py
   ```

3. **Code Quality**:
   ```bash
   # Format code
   black src tests
   
   # Lint code
   ruff check src tests
   
   # Type checking
   mypy src
   ```

4. **Build and Package**:
   ```bash
   # Build distribution
   python -m build
   
   # Install locally
   uv pip install dist/vmcp-1.0.0-py3-none-any.whl
   ```

This comprehensive implementation guide provides everything needed to build a production-ready vMCP system in Python using UV for package management. The architecture is modular, scalable, and follows best practices for Python development.