# vMCP Technical Specification - Python Edition

## Protocol Details

### JSON-RPC 2.0 Message Format

All MCP communication uses JSON-RPC 2.0 messages. vMCP must preserve message integrity while routing.

#### Request Format
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "/tmp/example.txt"
    }
  }
}
```

#### Response Format
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "File contents here"
      }
    ]
  }
}
```

#### Error Format
```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {
      "method": "unknown/method"
    }
  }
}
```

### MCP Methods to Support

#### Core Methods
- `initialize`: Establishes connection and exchanges capabilities
- `initialized`: Confirms initialization complete
- `tools/list`: Lists available tools
- `tools/call`: Invokes a specific tool
- `resources/list`: Lists available resources
- `resources/read`: Reads a resource
- `resources/subscribe`: Subscribes to resource changes
- `resources/unsubscribe`: Unsubscribes from resources
- `prompts/list`: Lists available prompts
- `prompts/get`: Gets a specific prompt
- `completion/complete`: Requests completion

#### vMCP Extensions
- `vmcp/servers/list`: Lists mounted servers
- `vmcp/servers/info`: Gets server information
- `vmcp/servers/health`: Checks server health
- `vmcp/cache/clear`: Clears cache
- `vmcp/metrics`: Gets system metrics

### Transport Protocol Specifications

#### Standard I/O (stdio)
```python
# src/vmcp/gateway/transports/stdio.py
import asyncio
import json
import sys
from typing import Optional, Callable, Dict, Any
import struct

class StdioTransport:
    """Standard I/O transport implementation."""
    
    def __init__(self, message_handler: Callable[[Dict], Any]):
        self.message_handler = message_handler
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._read_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the stdio transport."""
        if self._running:
            return
            
        self._running = True
        
        # Create async streams for stdin/stdout
        loop = asyncio.get_event_loop()
        self._reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        
        # Get writer for stdout
        transport, _ = await loop.connect_write_pipe(
            lambda: asyncio.Protocol(),
            sys.stdout
        )
        self._writer = asyncio.StreamWriter(transport, None, self._reader, loop)
        
        # Start reading messages
        self._read_task = asyncio.create_task(self._read_loop())
        
    async def stop(self):
        """Stop the stdio transport."""
        self._running = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
                
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            
    async def _read_loop(self):
        """Read messages from stdin."""
        while self._running:
            try:
                message = await self._read_message()
                if message:
                    response = await self.message_handler(message)
                    if response:
                        await self._write_message(response)
            except Exception as e:
                # Log error but continue
                pass
                
    async def _read_message(self) -> Optional[Dict]:
        """Read a single message using length-prefixed format."""
        if not self._reader:
            return None
            
        # Read content length header
        header = await self._reader.readline()
        if not header:
            return None
            
        # Parse content length
        try:
            if header.startswith(b"Content-Length: "):
                length = int(header[16:].strip())
            else:
                # Try simple length prefix format
                length = int(header.strip())
        except ValueError:
            return None
            
        # Read empty line after header (if using Content-Length format)
        if header.startswith(b"Content-Length: "):
            await self._reader.readline()
            
        # Read message content
        content = await self._reader.readexactly(length)
        
        # Parse JSON
        try:
            return json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            return None
            
    async def _write_message(self, message: Dict):
        """Write a message using length-prefixed format."""
        if not self._writer:
            return
            
        # Serialize to JSON
        content = json.dumps(message, separators=(',', ':')).encode('utf-8')
        length = len(content)
        
        # Write length prefix and content
        self._writer.write(f"{length}\n".encode('utf-8'))
        self._writer.write(content)
        self._writer.write(b"\n")
        
        await self._writer.drain()
```

#### HTTP with Server-Sent Events (SSE)
```python
# src/vmcp/gateway/transports/http_sse.py
import asyncio
import json
from aiohttp import web
from typing import Dict, Any, Callable, Set
import uuid

class HttpSseTransport:
    """HTTP with Server-Sent Events transport."""
    
    def __init__(self, message_handler: Callable, port: int = 3000):
        self.message_handler = message_handler
        self.port = port
        self.app = web.Application()
        self.connections: Dict[str, web.StreamResponse] = {}
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post('/message', self._handle_message)
        self.app.router.add_get('/sse', self._handle_sse)
        self.app.router.add_get('/health', self._handle_health)
        
    async def start(self):
        """Start HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
    async def _handle_message(self, request: web.Request) -> web.Response:
        """Handle incoming JSON-RPC message."""
        try:
            # Parse request
            data = await request.json()
            
            # Get connection ID from headers
            connection_id = request.headers.get('X-Connection-ID')
            if not connection_id:
                return web.json_response(
                    {"error": "Missing X-Connection-ID header"},
                    status=400
                )
                
            # Process message
            response = await self.message_handler(data)
            
            # Send response via SSE if available
            if connection_id in self.connections:
                sse_response = self.connections[connection_id]
                await self._send_sse_message(sse_response, response)
                
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
            
    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE connection."""
        # Create unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Setup SSE response
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Connection-ID'] = connection_id
        
        await response.prepare(request)
        
        # Store connection
        self.connections[connection_id] = response
        
        # Send initial connection event
        await self._send_sse_message(response, {
            "type": "connection",
            "connectionId": connection_id
        })
        
        try:
            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                await response.write(b':keepalive\n\n')
        except Exception:
            pass
        finally:
            # Remove connection
            self.connections.pop(connection_id, None)
            
        return response
        
    async def _send_sse_message(self, response: web.StreamResponse, data: Dict):
        """Send message via SSE."""
        message = f"data: {json.dumps(data)}\n\n"
        await response.write(message.encode('utf-8'))
        
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy"})
```

#### WebSocket
```python
# src/vmcp/gateway/transports/websocket.py
import asyncio
import json
import websockets
from typing import Dict, Any, Callable, Set

class WebSocketTransport:
    """WebSocket transport implementation."""
    
    def __init__(self, message_handler: Callable, port: int = 3001):
        self.message_handler = message_handler
        self.port = port
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        
    async def start(self):
        """Start WebSocket server."""
        self.server = await websockets.serve(
            self._handle_connection,
            'localhost',
            self.port
        )
        
    async def stop(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Close all connections
        for ws in list(self.connections):
            await ws.close()
            
    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        self.connections.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON-RPC message
                    data = json.loads(message)
                    
                    # Process message
                    response = await self.message_handler(data)
                    
                    # Send response
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    error = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    await websocket.send(json.dumps(error))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)
            
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        if self.connections:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[ws.send(message_str) for ws in self.connections],
                return_exceptions=True
            )
```

## Routing Algorithms

### Path-Based Routing

```python
# src/vmcp/routing/algorithms.py
import re
from typing import Dict, List, Optional, Pattern
from dataclasses import dataclass

@dataclass
class RouteRule:
    """Routing rule for path-based routing."""
    pattern: str
    server_id: str
    priority: int = 0
    regex: Optional[Pattern] = None
    
    def __post_init__(self):
        """Compile pattern to regex."""
        # Convert wildcards to regex
        regex_pattern = self.pattern
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = regex_pattern.replace('?', '.')
        self.regex = re.compile(f'^{regex_pattern}$')

class PathBasedRouter:
    """Routes requests based on method path patterns."""
    
    def __init__(self):
        self.rules: List[RouteRule] = []
        
    def add_rule(self, pattern: str, server_id: str, priority: int = 0):
        """Add a routing rule."""
        rule = RouteRule(pattern, server_id, priority)
        self.rules.append(rule)
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
    def route(self, method: str) -> Optional[str]:
        """Find server for given method."""
        for rule in self.rules:
            if rule.regex and rule.regex.match(method):
                return rule.server_id
        return None
        
    def remove_rule(self, pattern: str):
        """Remove a routing rule."""
        self.rules = [r for r in self.rules if r.pattern != pattern]
        
    def get_rules(self) -> List[Dict]:
        """Get all routing rules."""
        return [
            {
                "pattern": rule.pattern,
                "server_id": rule.server_id,
                "priority": rule.priority
            }
            for rule in self.rules
        ]
```

### Content-Based Routing

```python
# src/vmcp/routing/algorithms.py (continued)
from typing import Any, Callable

@dataclass
class ContentRule:
    """Rule for content-based routing."""
    server_id: str
    method: Optional[str] = None
    tool_name: Optional[str] = None
    resource_uri: Optional[str] = None
    condition: Optional[Callable[[Dict], bool]] = None

class ContentBasedRouter:
    """Routes requests based on message content."""
    
    def __init__(self):
        self.rules: List[ContentRule] = []
        
    def add_rule(self, rule: ContentRule):
        """Add a content-based routing rule."""
        self.rules.append(rule)
        
    def route(self, request: Dict[str, Any]) -> Optional[str]:
        """Find server based on request content."""
        method = request.get("method", "")
        params = request.get("params", {})
        
        for rule in self.rules:
            if self._matches_rule(method, params, rule):
                return rule.server_id
                
        return None
        
    def _matches_rule(
        self,
        method: str,
        params: Dict,
        rule: ContentRule
    ) -> bool:
        """Check if request matches rule."""
        # Check method
        if rule.method and method != rule.method:
            return False
            
        # Check tool name
        if rule.tool_name and method == "tools/call":
            if params.get("name") != rule.tool_name:
                return False
                
        # Check resource URI
        if rule.resource_uri and method == "resources/read":
            if params.get("uri") != rule.resource_uri:
                return False
                
        # Check custom condition
        if rule.condition:
            try:
                return rule.condition({"method": method, "params": params})
            except Exception:
                return False
                
        return True
```

### Load Balancing Strategies

```python
# src/vmcp/routing/loadbalancer.py
import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ServerMetrics:
    """Metrics for load balancing decisions."""
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    last_error_time: Optional[float] = None
    weight: int = 1

class LoadBalancer(ABC):
    """Abstract base class for load balancers."""
    
    @abstractmethod
    def select(self, servers: List[Any]) -> Any:
        """Select a server from the list."""
        pass

class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""
    
    def __init__(self):
        self._index = 0
        self._lock = asyncio.Lock()
        
    async def select(self, servers: List[Any]) -> Any:
        """Select next server in round-robin fashion."""
        if not servers:
            raise ValueError("No servers available")
            
        async with self._lock:
            server = servers[self._index % len(servers)]
            self._index += 1
            return server

class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer."""
    
    def __init__(self):
        self.metrics: Dict[str, ServerMetrics] = {}
        
    def select(self, servers: List[Any]) -> Any:
        """Select server with least active connections."""
        if not servers:
            raise ValueError("No servers available")
            
        # Get metrics for each server
        server_metrics = []
        for server in servers:
            if server.id not in self.metrics:
                self.metrics[server.id] = ServerMetrics()
            server_metrics.append((server, self.metrics[server.id]))
            
        # Sort by active connections
        server_metrics.sort(key=lambda x: x[1].active_connections)
        
        return server_metrics[0][0]
        
    def update_metrics(self, server_id: str, **kwargs):
        """Update server metrics."""
        if server_id not in self.metrics:
            self.metrics[server_id] = ServerMetrics()
            
        metrics = self.metrics[server_id]
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

class WeightedBalancer(LoadBalancer):
    """Weighted random load balancer."""
    
    def __init__(self):
        self.weights: Dict[str, int] = {}
        
    def set_weight(self, server_id: str, weight: int):
        """Set server weight."""
        self.weights[server_id] = max(1, weight)
        
    def select(self, servers: List[Any]) -> Any:
        """Select server based on weights."""
        if not servers:
            raise ValueError("No servers available")
            
        # Build weighted list
        weighted_servers = []
        for server in servers:
            weight = self.weights.get(server.id, 1)
            weighted_servers.extend([server] * weight)
            
        # Random selection from weighted list
        return random.choice(weighted_servers)

class AdaptiveBalancer(LoadBalancer):
    """Adaptive load balancer based on response times."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times: Dict[str, List[float]] = {}
        
    def select(self, servers: List[Any]) -> Any:
        """Select server with best recent performance."""
        if not servers:
            raise ValueError("No servers available")
            
        # Calculate scores for each server
        scores = []
        for server in servers:
            if server.id not in self.response_times:
                # New server gets neutral score
                scores.append((server, 1.0))
            else:
                times = self.response_times[server.id]
                if times:
                    # Lower average time = higher score
                    avg_time = sum(times) / len(times)
                    score = 1.0 / (avg_time + 0.001)
                else:
                    score = 1.0
                scores.append((server, score))
                
        # Weighted random selection based on scores
        total_score = sum(score for _, score in scores)
        rand_value = random.uniform(0, total_score)
        
        cumulative = 0
        for server, score in scores:
            cumulative += score
            if rand_value <= cumulative:
                return server
                
        return scores[0][0]  # Fallback
        
    def record_response_time(self, server_id: str, response_time: float):
        """Record response time for adaptive balancing."""
        if server_id not in self.response_times:
            self.response_times[server_id] = []
            
        times = self.response_times[server_id]
        times.append(response_time)
        
        # Keep only recent times
        if len(times) > self.window_size:
            self.response_times[server_id] = times[-self.window_size:]
```

## Connection Management

### Connection Pool Implementation

```python
# src/vmcp/routing/connection_pool.py
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 1
    max_size: int = 10
    idle_timeout: int = 300
    acquire_timeout: int = 5
    validation_interval: int = 60
    retry_attempts: int = 3
    retry_delay: float = 0.1

@dataclass
class PooledConnection:
    """Wrapper for pooled connections."""
    connection: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False
    healthy: bool = True

class ConnectionPool:
    """Production-grade connection pool implementation."""
    
    def __init__(
        self,
        server_id: str,
        connection_factory: Callable,
        config: Optional[PoolConfig] = None
    ):
        self.server_id = server_id
        self.connection_factory = connection_factory
        self.config = config or PoolConfig()
        
        self.connections: List[PooledConnection] = []
        self._waiters: List[asyncio.Future] = []
        self._lock = asyncio.Lock()
        self._closing = False
        self._stats = {
            "acquired": 0,
            "released": 0,
            "created": 0,
            "destroyed": 0,
            "timeouts": 0,
            "errors": 0
        }
        
    async def initialize(self):
        """Initialize pool with minimum connections."""
        logger.info(f"Initializing pool for {self.server_id}")
        
        create_tasks = []
        for _ in range(self.config.min_size):
            create_tasks.append(self._create_connection())
            
        connections = await asyncio.gather(*create_tasks, return_exceptions=True)
        
        async with self._lock:
            for conn in connections:
                if isinstance(conn, PooledConnection):
                    self.connections.append(conn)
                else:
                    logger.error(f"Failed to create connection: {conn}")
                    
    async def _create_connection(self) -> PooledConnection:
        """Create new pooled connection."""
        for attempt in range(self.config.retry_attempts):
            try:
                connection = await self.connection_factory()
                pooled = PooledConnection(connection=connection)
                self._stats["created"] += 1
                logger.debug(f"Created connection for {self.server_id}")
                return pooled
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
                    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        if self._closing:
            raise RuntimeError("Pool is closing")
            
        start_time = time.time()
        timeout = self.config.acquire_timeout
        connection = None
        
        try:
            while time.time() - start_time < timeout:
                async with self._lock:
                    # Find available healthy connection
                    for conn in self.connections:
                        if not conn.in_use and conn.healthy:
                            # Validate if needed
                            if await self._should_validate(conn):
                                if not await self._validate_connection(conn):
                                    continue
                                    
                            conn.in_use = True
                            conn.last_used = time.time()
                            conn.use_count += 1
                            connection = conn
                            self._stats["acquired"] += 1
                            break
                            
                    # Create new connection if under limit
                    if not connection and len(self.connections) < self.config.max_size:
                        try:
                            conn = await self._create_connection()
                            conn.in_use = True
                            self.connections.append(conn)
                            connection = conn
                            self._stats["acquired"] += 1
                        except Exception as e:
                            logger.error(f"Failed to create connection: {e}")
                            self._stats["errors"] += 1
                            
                if connection:
                    break
                    
                # Wait for available connection
                waiter = asyncio.Future()
                self._waiters.append(waiter)
                
                try:
                    await asyncio.wait_for(
                        waiter,
                        timeout=min(1.0, timeout - (time.time() - start_time))
                    )
                except asyncio.TimeoutError:
                    pass
                finally:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                        
            if not connection:
                self._stats["timeouts"] += 1
                raise asyncio.TimeoutError(
                    f"Failed to acquire connection within {timeout}s"
                )
                
            yield connection.connection
            
        except Exception:
            # Mark connection as unhealthy on error
            if connection:
                connection.healthy = False
            raise
            
        finally:
            if connection:
                await self._release_connection(connection)
                
    async def _release_connection(self, conn: PooledConnection):
        """Release connection back to pool."""
        async with self._lock:
            conn.in_use = False
            conn.last_used = time.time()
            self._stats["released"] += 1
            
            # Remove unhealthy connections
            if not conn.healthy or self._should_retire(conn):
                self.connections.remove(conn)
                self._stats["destroyed"] += 1
                asyncio.create_task(self._destroy_connection(conn))
                
            # Notify waiters
            self._notify_waiters()
            
    def _notify_waiters(self):
        """Notify waiting acquirers."""
        for waiter in self._waiters[:]:
            if not waiter.done():
                waiter.set_result(None)
                break
                
    async def _should_validate(self, conn: PooledConnection) -> bool:
        """Check if connection needs validation."""
        return (
            time.time() - conn.last_validated > self.config.validation_interval
        )
        
    async def _validate_connection(self, conn: PooledConnection) -> bool:
        """Validate connection health."""
        try:
            # Attempt to use connection for validation
            if hasattr(conn.connection, 'ping'):
                await conn.connection.ping()
            conn.last_validated = time.time()
            conn.healthy = True
            return True
        except Exception:
            conn.healthy = False
            return False
            
    def _should_retire(self, conn: PooledConnection) -> bool:
        """Check if connection should be retired."""
        # Retire old connections
        age = time.time() - conn.created_at
        if age > self.config.idle_timeout * 10:  # 10x idle timeout
            return True
            
        # Retire heavily used connections
        if conn.use_count > 1000:
            return True
            
        return False
        
    async def _destroy_connection(self, conn: PooledConnection):
        """Destroy a connection."""
        try:
            if hasattr(conn.connection, 'close'):
                await conn.connection.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
            
    async def close(self):
        """Close all connections in pool."""
        logger.info(f"Closing pool for {self.server_id}")
        self._closing = True
        
        # Cancel all waiters
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
                
        # Close all connections
        async with self._lock:
            close_tasks = []
            for conn in self.connections:
                close_tasks.append(self._destroy_connection(conn))
                
            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.connections.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "size": len(self.connections),
            "available": sum(1 for c in self.connections if not c.in_use),
            "healthy": sum(1 for c in self.connections if c.healthy),
            "waiters": len(self._waiters)
        }
```

### Session Management

```python
# src/vmcp/gateway/session.py
import asyncio
import time
import uuid
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class Session:
    """User session information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: Optional[str] = None
    persona: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    error_count: int = 0
    
    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
        
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired."""
        expiry = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiry

class SessionManager:
    """Manages user sessions with cleanup and persistence."""
    
    def __init__(
        self,
        timeout_minutes: int = 30,
        cleanup_interval: int = 300
    ):
        self.timeout_minutes = timeout_minutes
        self.cleanup_interval = cleanup_interval
        
        self.sessions: Dict[str, Session] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    async def create_session(
        self,
        connection_id: Optional[str] = None,
        persona: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Session:
        """Create new session."""
        async with self._lock:
            session = Session(
                connection_id=connection_id,
                persona=persona,
                metadata=metadata or {}
            )
            
            self.sessions[session.id] = session
            
            if connection_id:
                self.connection_sessions[connection_id] = session.id
                
            logger.info(f"Created session {session.id} for persona {persona}")
            return session
            
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired(self.timeout_minutes):
                session.touch()
                return session
            return None
            
    async def get_session_by_connection(
        self,
        connection_id: str
    ) -> Optional[Session]:
        """Get session by connection ID."""
        async with self._lock:
            session_id = self.connection_sessions.get(connection_id)
            if session_id:
                return await self.get_session(session_id)
            return None
            
    async def update_session(
        self,
        session_id: str,
        **kwargs
    ) -> Optional[Session]:
        """Update session attributes."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.touch()
                return session
            return None
            
    async def delete_session(self, session_id: str):
        """Delete a session."""
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session and session.connection_id:
                self.connection_sessions.pop(session.connection_id, None)
                
            logger.info(f"Deleted session {session_id}")
            
    async def record_request(
        self,
        session_id: str,
        success: bool = True
    ):
        """Record request for session."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.request_count += 1
                if not success:
                    session.error_count += 1
                session.touch()
                
    async def get_active_sessions(self) -> List[Session]:
        """Get all active sessions."""
        async with self._lock:
            return [
                session for session in self.sessions.values()
                if not session.is_expired(self.timeout_minutes)
            ]
            
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        async with self._lock:
            active_sessions = [
                s for s in self.sessions.values()
                if not s.is_expired(self.timeout_minutes)
            ]
            
            persona_counts = {}
            for session in active_sessions:
                persona = session.persona or "default"
                persona_counts[persona] = persona_counts.get(persona, 0) + 1
                
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": len(active_sessions),
                "expired_sessions": len(self.sessions) - len(active_sessions),
                "personas": persona_counts,
                "total_requests": sum(s.request_count for s in self.sessions.values()),
                "total_errors": sum(s.error_count for s in self.sessions.values())
            }
            
    async def _cleanup_loop(self):
        """Periodic cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        async with self._lock:
            expired = []
            for session_id, session in self.sessions.items():
                if session.is_expired(self.timeout_minutes):
                    expired.append(session_id)
                    
            for session_id in expired:
                session = self.sessions.pop(session_id)
                if session.connection_id:
                    self.connection_sessions.pop(session.connection_id, None)
                    
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
```

## Cache Implementation

### Multi-Level Cache

```python
# src/vmcp/cache/cache.py
import asyncio
import time
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import aiofiles
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    size: int
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)

class CacheBackend:
    """Abstract cache backend."""
    
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
        
    async def set(self, key: str, value: Any, ttl: int):
        raise NotImplementedError
        
    async def delete(self, key: str):
        raise NotImplementedError
        
    async def clear(self):
        raise NotImplementedError
        
    async def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError

class MemoryCache(CacheBackend):
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory: int = 100 * 1024 * 1024):
        self.max_size = max_size
        self.max_memory = max_memory
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_used": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self.cache.get(key)
            
            if not entry:
                self._stats["misses"] += 1
                return None
                
            # Check expiration
            if time.time() > entry.expires_at:
                await self._remove_entry(key)
                self._stats["misses"] += 1
                return None
                
            # Update access order and stats
            self.access_order.remove(key)
            self.access_order.append(key)
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1
            
            return entry.value
            
    async def set(self, key: str, value: Any, ttl: int):
        """Set value in cache."""
        async with self._lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                await self._remove_entry(key)
                
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                expires_at=time.time() + ttl
            )
            
            # Add to cache
            self.cache[key] = entry
            self.access_order.append(key)
            self._stats["memory_used"] += size
            
            # Enforce limits
            await self._enforce_limits()
            
    async def delete(self, key: str):
        """Delete entry from cache."""
        async with self._lock:
            await self._remove_entry(key)
            
    async def clear(self):
        """Clear entire cache."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self._stats["memory_used"] = 0
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            return {
                **self._stats,
                "entries": len(self.cache),
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            }
            
    async def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.access_order.remove(key)
            self._stats["memory_used"] -= entry.size
            
    async def _enforce_limits(self):
        """Enforce size and memory limits."""
        # Enforce memory limit
        while self._stats["memory_used"] > self.max_memory and self.access_order:
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
            
        # Enforce size limit
        while len(self.cache) > self.max_size and self.access_order:
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback to JSON size estimation
            return len(json.dumps(value, default=str))

class DiskCache(CacheBackend):
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str = "~/.vmcp/cache/disk"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Load cache index."""
        if self.index_file.exists():
            async with aiofiles.open(self.index_file, 'r') as f:
                content = await f.read()
                self.index = json.loads(content)
                
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        async with self._lock:
            meta = self.index.get(key)
            if not meta:
                return None
                
            # Check expiration
            if time.time() > meta["expires_at"]:
                await self._remove_entry(key)
                return None
                
            # Read from disk
            file_path = self.cache_dir / meta["filename"]
            if not file_path.exists():
                await self._remove_entry(key)
                return None
                
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                
            return pickle.loads(data)
            
    async def set(self, key: str, value: Any, ttl: int):
        """Set value in disk cache."""
        async with self._lock:
            # Generate filename
            filename = f"{hashlib.sha256(key.encode()).hexdigest()}.cache"
            file_path = self.cache_dir / filename
            
            # Serialize and write
            data = pickle.dumps(value)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
                
            # Update index
            self.index[key] = {
                "filename": filename,
                "size": len(data),
                "created_at": time.time(),
                "expires_at": time.time() + ttl
            }
            
            await self._save_index()
            
    async def delete(self, key: str):
        """Delete entry from disk cache."""
        async with self._lock:
            await self._remove_entry(key)
            
    async def clear(self):
        """Clear entire disk cache."""
        async with self._lock:
            # Remove all cache files
            for meta in self.index.values():
                file_path = self.cache_dir / meta["filename"]
                if file_path.exists():
                    file_path.unlink()
                    
            self.index.clear()
            await self._save_index()
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_size = sum(meta["size"] for meta in self.index.values())
            return {
                "entries": len(self.index),
                "disk_usage": total_size
            }
            
    async def _remove_entry(self, key: str):
        """Remove entry from disk cache."""
        meta = self.index.pop(key, None)
        if meta:
            file_path = self.cache_dir / meta["filename"]
            if file_path.exists():
                file_path.unlink()
            await self._save_index()
            
    async def _save_index(self):
        """Save index to disk."""
        async with aiofiles.open(self.index_file, 'w') as f:
            await f.write(json.dumps(self.index, indent=2))

class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (disk) layers."""
    
    def __init__(
        self,
        l1_size: int = 1000,
        l1_memory: int = 100 * 1024 * 1024,
        l2_enabled: bool = True
    ):
        self.l1 = MemoryCache(max_size=l1_size, max_memory=l1_memory)
        self.l2 = DiskCache() if l2_enabled else None
        self.key_generator = CacheKeyGenerator()
        
    async def initialize(self):
        """Initialize cache layers."""
        if self.l2:
            await self.l2.initialize()
            
    async def get(
        self,
        method: str,
        params: Any,
        persona: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache."""
        key = self.key_generator.generate(method, params, persona)
        
        # Try L1
        value = await self.l1.get(key)
        if value is not None:
            return value
            
        # Try L2
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                # Promote to L1
                await self.l1.set(key, value, 300)  # 5 min in L1
                return value
                
        return None
        
    async def set(
        self,
        method: str,
        params: Any,
        value: Any,
        ttl: int,
        persona: Optional[str] = None
    ):
        """Set value in cache."""
        key = self.key_generator.generate(method, params, persona)
        
        # Always set in L1
        await self.l1.set(key, value, ttl)
        
        # Set in L2 for larger values or longer TTL
        if self.l2 and (ttl > 600 or self._is_large_value(value)):
            await self.l2.set(key, value, ttl)
            
    def _is_large_value(self, value: Any) -> bool:
        """Check if value should go to L2."""
        try:
            size = len(pickle.dumps(value))
            return size > 10 * 1024  # 10KB
        except:
            return False
            
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        # This would need pattern matching implementation
        pass

class CacheKeyGenerator:
    """Generates cache keys for MCP requests."""
    
    def generate(
        self,
        method: str,
        params: Any,
        persona: Optional[str] = None
    ) -> str:
        """Generate cache key."""
        # Create deterministic key components
        components = [
            "mcp",
            method,
            self._hash_params(params),
            persona or "default"
        ]
        
        return ":".join(components)
        
    def _hash_params(self, params: Any) -> str:
        """Create hash of parameters."""
        # Ensure consistent ordering
        if isinstance(params, dict):
            sorted_params = json.dumps(params, sort_keys=True)
        else:
            sorted_params = json.dumps(params)
            
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:12]
```

## Error Handling

### Error Types and Codes

```python
# src/vmcp/errors.py
from enum import IntEnum
from typing import Optional, Dict, Any

class VMCPErrorCode(IntEnum):
    """vMCP error codes following JSON-RPC conventions."""
    
    # JSON-RPC standard errors (-32768 to -32000)
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Transport errors (1xxx)
    TRANSPORT_ERROR = 1000
    CONNECTION_FAILED = 1001
    CONNECTION_TIMEOUT = 1002
    TRANSPORT_CLOSED = 1003
    
    # Routing errors (2xxx)
    NO_SERVER_FOUND = 2000
    SERVER_UNAVAILABLE = 2001
    ROUTING_FAILED = 2002
    ALL_SERVERS_DOWN = 2003
    
    # Permission errors (3xxx)
    UNAUTHORIZED = 3000
    PERSONA_NOT_FOUND = 3001
    SERVER_NOT_ALLOWED = 3002
    INSUFFICIENT_PERMISSIONS = 3003
    
    # Protocol errors (4xxx)
    PROTOCOL_ERROR = 4000
    UNSUPPORTED_VERSION = 4001
    INVALID_MESSAGE = 4002
    
    # Server errors (5xxx)
    GATEWAY_ERROR = 5000
    BACKEND_ERROR = 5001
    CACHE_ERROR = 5002
    CONFIGURATION_ERROR = 5003

class VMCPError(Exception):
    """Base exception for vMCP errors."""
    
    def __init__(
        self,
        code: VMCPErrorCode,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}
        
    def to_json_rpc_error(self, request_id: Any = None) -> Dict:
        """Convert to JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": int(self.code),
                "message": self.message,
                "data": self.data
            }
        }
        
    def __repr__(self) -> str:
        return f"VMCPError({self.code}, {self.message!r})"

# Convenience error classes
class TransportError(VMCPError):
    """Transport-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(VMCPErrorCode.TRANSPORT_ERROR, message, kwargs)

class RoutingError(VMCPError):
    """Routing-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(VMCPErrorCode.ROUTING_FAILED, message, kwargs)

class PermissionError(VMCPError):
    """Permission-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(VMCPErrorCode.UNAUTHORIZED, message, kwargs)

class ServerNotFoundError(VMCPError):
    """Server not found error."""
    def __init__(self, server_id: str):
        super().__init__(
            VMCPErrorCode.NO_SERVER_FOUND,
            f"Server '{server_id}' not found",
            {"server_id": server_id}
        )

class PersonaNotFoundError(VMCPError):
    """Persona not found error."""
    def __init__(self, persona_name: str):
        super().__init__(
            VMCPErrorCode.PERSONA_NOT_FOUND,
            f"Persona '{persona_name}' not found",
            {"persona": persona_name}
        )
```

### Circuit Breaker Pattern

```python
# src/vmcp/routing/circuit_breaker.py
import asyncio
import time
from enum import Enum
from typing import Callable, TypeVar, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    half_open_max_calls: int = 3
    
@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    failure_count: int = 0
    success_count: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: List[Tuple[CircuitState, float]] = field(default_factory=list)

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state
        
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
        
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        if self._state != CircuitState.OPEN:
            return False
            
        # Check if we should transition to half-open
        if self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
            return False
            
        return True
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.is_open:
                raise VMCPError(
                    VMCPErrorCode.SERVER_UNAVAILABLE,
                    f"Circuit breaker '{self.name}' is open",
                    {
                        "circuit_breaker": self.name,
                        "failure_count": self._stats.failure_count,
                        "last_failure": self._stats.last_failure_time
                    }
                )
                
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise VMCPError(
                        VMCPErrorCode.SERVER_UNAVAILABLE,
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
                
        # Execute the function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
        finally:
            self._stats.total_calls += 1
            
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
                    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif (self._state == CircuitState.CLOSED and 
                  self._stats.consecutive_failures >= self.config.failure_threshold):
                self._transition_to(CircuitState.OPEN)
                
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self._stats.last_failure_time is not None and
            time.time() - self._stats.last_failure_time >= self.config.recovery_timeout
        )
        
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        if self._state != new_state:
            logger.info(
                f"Circuit breaker '{self.name}' transitioning "
                f"from {self._state.value} to {new_state.value}"
            )
            
            old_state = self._state
            self._state = new_state
            self._stats.state_changes.append((new_state, time.time()))
            
            # Reset counters based on transition
            if new_state == CircuitState.CLOSED:
                self._stats.failure_count = 0
                self._half_open_calls = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._stats.consecutive_successes = 0
                self._stats.consecutive_failures = 0
                
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "total_calls": self._stats.total_calls,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
            "consecutive_successes": self._stats.consecutive_successes,
            "consecutive_failures": self._stats.consecutive_failures,
            "last_failure_time": self._stats.last_failure_time,
            "state_changes": len(self._stats.state_changes),
            "uptime_percentage": (
                self._stats.total_successes / max(1, self._stats.total_calls) * 100
            )
        }
        
    async def reset(self):
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitBreakerStats()
            logger.info(f"Circuit breaker '{self.name}' manually reset")
```

## Monitoring and Metrics

### Metrics Collection

```python
# src/vmcp/monitoring/metrics.py
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    method: str
    server_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_code: Optional[int] = None
    cached: bool = False
    
    @property
    def duration(self) -> float:
        """Get request duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Request metrics
        self.request_metrics: deque[RequestMetrics] = deque(maxlen=window_size)
        self.method_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.server_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Counters
        self.counters = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "requests_cached": 0,
            "connections_active": 0,
            "connections_total": 0,
        }
        
        # Gauges
        self.gauges = {
            "servers_healthy": 0,
            "servers_total": 0,
            "cache_size": 0,
            "memory_used": 0,
        }
        
        self._lock = asyncio.Lock()
        
    async def record_request(
        self,
        method: str,
        server_id: Optional[str] = None,
        duration: float = 0.0,
        success: bool = True,
        error_code: Optional[int] = None,
        cached: bool = False
    ):
        """Record request metrics."""
        async with self._lock:
            # Create metrics entry
            metrics = RequestMetrics(
                method=method,
                server_id=server_id,
                start_time=time.time() - duration,
                end_time=time.time(),
                success=success,
                error_code=error_code,
                cached=cached
            )
            
            # Add to collections
            self.request_metrics.append(metrics)
            self.method_metrics[method].append(metrics)
            if server_id:
                self.server_metrics[server_id].append(metrics)
                
            # Update counters
            self.counters["requests_total"] += 1
            if success:
                self.counters["requests_success"] += 1
            else:
                self.counters["requests_failed"] += 1
            if cached:
                self.counters["requests_cached"] += 1
                
    async def update_gauge(self, name: str, value: float):
        """Update a gauge value."""
        async with self._lock:
            self.gauges[name] = value
            
    async def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        async with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value
            
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        async with self._lock:
            # Calculate request statistics
            recent_requests = list(self.request_metrics)
            if recent_requests:
                durations = [r.duration for r in recent_requests]
                success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests)
                
                request_stats = {
                    "count": len(recent_requests),
                    "success_rate": success_rate,
                    "latency_mean": statistics.mean(durations),
                    "latency_median": statistics.median(durations),
                    "latency_p95": self._percentile(durations, 0.95),
                    "latency_p99": self._percentile(durations, 0.99),
                }
            else:
                request_stats = {
                    "count": 0,
                    "success_rate": 0.0,
                    "latency_mean": 0.0,
                    "latency_median": 0.0,
                    "latency_p95": 0.0,
                    "latency_p99": 0.0,
                }
                
            # Method breakdown
            method_stats = {}
            for method, metrics in self.method_metrics.items():
                if metrics:
                    method_durations = [m.duration for m in metrics]
                    method_stats[method] = {
                        "count": len(metrics),
                        "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                        "latency_mean": statistics.mean(method_durations),
                    }
                    
            # Server breakdown
            server_stats = {}
            for server_id, metrics in self.server_metrics.items():
                if metrics:
                    server_durations = [m.duration for m in metrics]
                    server_stats[server_id] = {
                        "count": len(metrics),
                        "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                        "latency_mean": statistics.mean(server_durations),
                    }
                    
            return {
                "timestamp": time.time(),
                "requests": request_stats,
                "methods": method_stats,
                "servers": server_stats,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "cache_hit_rate": (
                    self.counters["requests_cached"] / 
                    max(1, self.counters["requests_total"])
                ),
            }
            
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

class PrometheusExporter:
    """Export metrics in Prometheus format."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        
    async def export(self) -> str:
        """Export metrics in Prometheus text format."""
        metrics = await self.collector.get_metrics()
        lines = []
        
        # Request metrics
        lines.append("# HELP vmcp_requests_total Total number of requests")
        lines.append("# TYPE vmcp_requests_total counter")
        lines.append(f"vmcp_requests_total {metrics['counters']['requests_total']}")
        
        lines.append("# HELP vmcp_requests_success_total Total successful requests")
        lines.append("# TYPE vmcp_requests_success_total counter")
        lines.append(f"vmcp_requests_success_total {metrics['counters']['requests_success']}")
        
        lines.append("# HELP vmcp_request_duration_seconds Request duration")
        lines.append("# TYPE vmcp_request_duration_seconds histogram")
        for method, stats in metrics['methods'].items():
            lines.append(
                f'vmcp_request_duration_seconds{{method="{method}",quantile="0.5"}} '
                f'{stats["latency_mean"]}'
            )
            
        # Gauge metrics
        lines.append("# HELP vmcp_servers_healthy Number of healthy servers")
        lines.append("# TYPE vmcp_servers_healthy gauge")
        lines.append(f"vmcp_servers_healthy {metrics['gauges']['servers_healthy']}")
        
        return "\n".join(lines)
```

### Health Check Implementation

```python
# src/vmcp/monitoring/health.py
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.checks: List[HealthCheck] = []
        self._last_check_time: Optional[float] = None
        
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        start_time = time.time()
        checks = []
        
        # Check gateway
        checks.append(await self._check_gateway())
        
        # Check servers
        if "registry" in self.components:
            server_checks = await self._check_servers()
            checks.extend(server_checks)
            
        # Check cache
        if "cache" in self.components:
            checks.append(await self._check_cache())
            
        # Check database connections
        if "database" in self.components:
            checks.append(await self._check_database())
            
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        self._last_check_time = time.time()
        self.checks = checks
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                    "duration": check.duration,
                }
                for check in checks
            ],
        }
        
    async def _check_gateway(self) -> HealthCheck:
        """Check gateway health."""
        start = time.time()
        
        try:
            gateway = self.components.get("gateway")
            if gateway and hasattr(gateway, "_running") and gateway._running:
                return HealthCheck(
                    name="gateway",
                    status=HealthStatus.HEALTHY,
                    message="Gateway is running",
                    duration=time.time() - start
                )
            else:
                return HealthCheck(
                    name="gateway",
                    status=HealthStatus.UNHEALTHY,
                    message="Gateway is not running",
                    duration=time.time() - start
                )
        except Exception as e:
            return HealthCheck(
                name="gateway",
                status=HealthStatus.UNHEALTHY,
                message=f"Gateway check failed: {e}",
                duration=time.time() - start
            )
            
    async def _check_servers(self) -> List[HealthCheck]:
        """Check all registered servers."""
        checks = []
        registry = self.components["registry"]
        
        for server in registry.get_all_servers():
            start = time.time()
            
            try:
                # Ping server
                response = await asyncio.wait_for(
                    server.send_request({
                        "jsonrpc": "2.0",
                        "id": "health",
                        "method": "ping"
                    }),
                    timeout=5.0
                )
                
                if "result" in response:
                    status = HealthStatus.HEALTHY
                    message = "Server responding"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "Invalid response"
                    
            except asyncio.TimeoutError:
                status = HealthStatus.UNHEALTHY
                message = "Server timeout"
            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"Server error: {e}"
                
            checks.append(HealthCheck(
                name=f"server:{server.id}",
                status=status,
                message=message,
                duration=time.time() - start,
                details={"server_id": server.id}
            ))
            
        return checks
        
    async def _check_cache(self) -> HealthCheck:
        """Check cache health."""
        start = time.time()
        
        try:
            cache = self.components["cache"]
            
            # Test cache operations
            test_key = "__health_check__"
            test_value = {"timestamp": time.time()}
            
            await cache.set("test", {}, test_value, 60)
            retrieved = await cache.get("test", {})
            
            if retrieved == test_value:
                stats = await cache.l1.get_stats()
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache operational",
                    duration=time.time() - start,
                    details=stats
                )
            else:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.UNHEALTHY,
                    message="Cache test failed",
                    duration=time.time() - start
                )
                
        except Exception as e:
            return HealthCheck(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache error: {e}",
                duration=time.time() - start
            )
            
    async def _check_database(self) -> HealthCheck:
        """Check database connectivity."""
        # Implementation depends on database backend
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connected"
        )
        
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health."""
        if not checks:
            return HealthStatus.UNHEALTHY
            
        unhealthy_count = sum(1 for c in checks if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > len(checks) // 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
            
    async def get_liveness(self) -> Dict[str, Any]:
        """Simple liveness check."""
        return {
            "status": "alive",
            "timestamp": time.time()
        }
        
    async def get_readiness(self) -> Dict[str, Any]:
        """Readiness check for load balancers."""
        # Quick check of critical components
        gateway_ready = self.components.get("gateway", {})._running
        
        if gateway_ready:
            return {
                "status": "ready",
                "timestamp": time.time()
            }
        else:
            return {
                "status": "not_ready",
                "timestamp": time.time(),
                "reason": "Gateway not running"
            }
```

## Testing Utilities

### Mock MCP Server

```python
# tests/mocks/mock_server.py
import asyncio
from typing import Dict, Any, Callable, Optional, List
import json
import logging

logger = logging.getLogger(__name__)

class MockMCPServer:
    """Mock MCP server for testing."""
    
    def __init__(self, server_id: str = "mock-server"):
        self.server_id = server_id
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, str] = {}
        self.request_log: List[Dict] = []
        self.response_delay: float = 0.0
        self.should_fail: bool = False
        self.failure_rate: float = 0.0
        
    def add_tool(
        self,
        name: str,
        handler: Optional[Callable] = None,
        description: str = ""
    ):
        """Add a tool to the mock server."""
        if handler is None:
            handler = lambda args: {"result": f"Mock result for {name}"}
        self.tools[name] = {
            "handler": handler,
            "description": description
        }
        
    def add_resource(self, uri: str, content: Any):
        """Add a resource to the mock server."""
        self.resources[uri] = content
        
    def add_prompt(self, name: str, template: str):
        """Add a prompt to the mock server."""
        self.prompts[name] = template
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        self.request_log.append(request)
        
        # Simulate delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
            
        # Simulate failures
        if self.should_fail or (self.failure_rate > 0 and random.random() < self.failure_rate):
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": "Mock server error"
                }
            }
            
        method = request.get("method")
        params = request.get("params", {})
        
        # Route to appropriate handler
        if method == "initialize":
            return await self._handle_initialize(request)
        elif method == "tools/list":
            return await self._handle_tools_list(request)
        elif method == "tools/call":
            return await self._handle_tools_call(request)
        elif method == "resources/list":
            return await self._handle_resources_list(request)
        elif method == "resources/read":
            return await self._handle_resources_read(request)
        elif method == "prompts/list":
            return await self._handle_prompts_list(request)
        elif method == "prompts/get":
            return await self._handle_prompts_get(request)
        elif method == "ping":
            return await self._handle_ping(request)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    async def _handle_initialize(self, request: Dict) -> Dict:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    },
                    "resources": {
                        "subscribe": True,
                        "listChanged": True
                    },
                    "prompts": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": self.server_id,
                    "version": "1.0.0"
                }
            }
        }
        
    async def _handle_tools_list(self, request: Dict) -> Dict:
        """Handle tools/list request."""
        tools = [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            for name, info in self.tools.items()
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": tools
            }
        }
        
    async def _handle_tools_call(self, request: Dict) -> Dict:
        """Handle tools/call request."""
        tool_name = request["params"]["name"]
        arguments = request["params"].get("arguments", {})
        
        if tool_name not in self.tools:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32602,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
            
        try:
            handler = self.tools[tool_name]["handler"]
            result = await handler(arguments) if asyncio.iscoroutinefunction(handler) else handler(arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result) if not isinstance(result, str) else result
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Tool execution error: {e}"
                }
            }
            
    async def _handle_resources_list(self, request: Dict) -> Dict:
        """Handle resources/list request."""
        resources = [
            {
                "uri": uri,
                "name": uri.split("/")[-1],
                "mimeType": "application/json"
            }
            for uri in self.resources
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "resources": resources
            }
        }
        
    async def _handle_resources_read(self, request: Dict) -> Dict:
        """Handle resources/read request."""
        uri = request["params"]["uri"]
        
        if uri not in self.resources:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32602,
                    "message": f"Resource not found: {uri}"
                }
            }
            
        content = self.resources[uri]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(content) if not isinstance(content, str) else content
                    }
                ]
            }
        }
        
    async def _handle_prompts_list(self, request: Dict) -> Dict:
        """Handle prompts/list request."""
        prompts = [
            {
                "name": name,
                "description": f"Mock prompt: {name}"
            }
            for name in self.prompts
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "prompts": prompts
            }
        }
        
    async def _handle_prompts_get(self, request: Dict) -> Dict:
        """Handle prompts/get request."""
        name = request["params"]["name"]
        
        if name not in self.prompts:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32602,
                    "message": f"Prompt not found: {name}"
                }
            }
            
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "prompt": {
                    "name": name,
                    "template": self.prompts[name]
                }
            }
        }
        
    async def _handle_ping(self, request: Dict) -> Dict:
        """Handle ping request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "pong": True,
                "timestamp": time.time()
            }
        }
```

### Integration Test Framework

```python
# tests/integration/test_framework.py
import asyncio
import pytest
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import json

from vmcp.gateway.server import VMCPGateway, GatewayConfig
from vmcp.repository.manager import RepositoryManager
from vmcp.persona.manager import PersonaManager
from tests.mocks.mock_server import MockMCPServer

class VMCPTestFramework:
    """Integration test framework for vMCP."""
    
    def __init__(self):
        self.temp_dir = None
        self.gateway: Optional[VMCPGateway] = None
        self.repo_manager: Optional[RepositoryManager] = None
        self.persona_manager: Optional[PersonaManager] = None
        self.mock_servers: Dict[str, MockMCPServer] = {}
        
    async def setup(self):
        """Setup test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        config_path = Path(self.temp_dir) / ".vmcp"
        
        # Initialize components
        self.repo_manager = RepositoryManager(str(config_path))
        await self.repo_manager.initialize()
        
        self.persona_manager = PersonaManager(str(config_path / "personas"))
        await self.persona_manager.load_personas()
        
        # Create gateway config
        gateway_config = GatewayConfig(
            registry_path=str(config_path / "registry"),
            transports={"stdio": {"enabled": True}},
            cache_enabled=False  # Disable for predictable tests
        )
        
        self.gateway = VMCPGateway(gateway_config)
        
    async def teardown(self):
        """Cleanup test environment."""
        if self.gateway:
            await self.gateway.stop()
            
        if self.repo_manager:
            await self.repo_manager.cleanup()
            
        # Cleanup temp directory
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            
    async def add_mock_server(
        self,
        server_id: str,
        setup_func: Optional[Callable[[MockMCPServer], None]] = None
    ) -> MockMCPServer:
        """Add a mock MCP server."""
        mock_server = MockMCPServer(server_id)
        
        if setup_func:
            setup_func(mock_server)
            
        self.mock_servers[server_id] = mock_server
        
        # Register with gateway
        config = {
            "id": server_id,
            "name": f"Mock {server_id}",
            "transport": "mock",
            "capabilities": {
                "tools": [
                    {"name": name, "description": info["description"]}
                    for name, info in mock_server.tools.items()
                ]
            }
        }
        
        await self.gateway.registry.register_server(config)
        
        return mock_server
        
    async def create_persona(
        self,
        name: str,
        enabled_servers: List[str]
    ) -> None:
        """Create a test persona."""
        persona = await self.persona_manager.create_persona(name)
        for server_id in enabled_servers:
            await self.persona_manager.enable_server(name, server_id)
            
    async def send_request(
        self,
        request: Dict[str, Any],
        persona: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send request through gateway."""
        # Create context with persona
        from vmcp.routing.router import RoutingContext
        
        context = RoutingContext(
            request=request,
            persona=persona
        )
        
        return await self.gateway.router.route(request, context)
        
    def assert_success_response(self, response: Dict[str, Any]):
        """Assert response is successful."""
        assert "error" not in response
        assert "result" in response
        assert response.get("jsonrpc") == "2.0"
        
    def assert_error_response(
        self,
        response: Dict[str, Any],
        error_code: Optional[int] = None
    ):
        """Assert response is an error."""
        assert "error" in response
        assert "result" not in response
        
        if error_code is not None:
            assert response["error"]["code"] == error_code
            
# Example test using the framework
@pytest.mark.asyncio
async def test_persona_access_control():
    """Test that personas correctly restrict access."""
    framework = VMCPTestFramework()
    
    try:
        await framework.setup()
        
        # Add two mock servers
        server1 = await framework.add_mock_server("server1")
        server1.add_tool("tool1", lambda args: {"result": "from server1"})
        
        server2 = await framework.add_mock_server("server2")
        server2.add_tool("tool2", lambda args: {"result": "from server2"})
        
        # Create persona with access to only server1
        await framework.create_persona("limited", ["server1"])
        
        # Test access to allowed server
        response = await framework.send_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "tool1",
                    "arguments": {}
                }
            },
            persona="limited"
        )
        framework.assert_success_response(response)
        
        # Test access to restricted server
        response = await framework.send_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "tool2",
                    "arguments": {}
                }
            },
            persona="limited"
        )
        framework.assert_error_response(response, 3002)  # SERVER_NOT_ALLOWED
        
    finally:
        await framework.teardown()
```

This comprehensive technical specification provides all the detailed protocol information, algorithms, and implementation patterns needed to build vMCP in Python. Combined with the main implementation guide, it gives Claude Code everything needed to create a production-ready system using Python and UV.