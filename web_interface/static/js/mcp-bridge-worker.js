// Keep this explicit and in sync with mcp-server/bridge.js. `localhost` can
// resolve to IPv4 while Node has selected an IPv6-only listener on macOS.
const RELAY_URL = 'ws://127.0.0.1:3700';
let ws = null;
const ports = [];

function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  try {
    ws = new WebSocket(RELAY_URL);

    ws.onopen = () => {
      broadcast({ type: 'MCP_STATUS', status: 'connected' });
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);

        if (msg.type === 'CLIENT_INFO') {
          broadcast({ type: 'MCP_CLIENT_INFO', name: msg.name, version: msg.version });
          return;
        }

        if (ports.length > 0) {
          ports[0].postMessage({ type: 'BRIDGE_REQUEST', id: msg.id, method: msg.method, params: msg.params });
        } else {
          ws.send(JSON.stringify({ id: msg.id, error: 'No Vectoria tab connected' }));
        }
      } catch (e) {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      broadcast({ type: 'MCP_STATUS', status: 'disconnected' });
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      broadcast({ type: 'MCP_STATUS', status: 'disconnected' });
    };
  } catch (e) {
    setTimeout(connect, 2000);
  }
}

function broadcast(msg) {
  for (const port of ports) {
    try { port.postMessage(msg); } catch (e) { /* ignore dead ports */ }
  }
}

self.onconnect = (e) => {
  const port = e.ports[0];
  ports.push(port);

  port.onmessage = (evt) => {
    if (evt.data.type === 'BRIDGE_RESPONSE') {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ id: evt.data.id, result: evt.data.result }));
      }
    } else if (evt.data.type === 'MCP_PING') {
      port.postMessage({
        type: 'MCP_PONG',
        connected: ws !== null && ws.readyState === WebSocket.OPEN
      });
    } else if (evt.data.type === 'MCP_DISCONNECT') {
      const idx = ports.indexOf(port);
      if (idx > -1) ports.splice(idx, 1);
    }
  };

  // Send current status to new port
  port.postMessage({
    type: 'MCP_STATUS',
    status: ws && ws.readyState === WebSocket.OPEN ? 'connected' : 'disconnected'
  });

  // Start connecting if not already
  if (!ws || ws.readyState > 1) connect();
};
