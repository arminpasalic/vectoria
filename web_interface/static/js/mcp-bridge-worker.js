// Keep this explicit and in sync with mcp-server/bridge.js. `localhost` can
// resolve to IPv4 while Node has selected an IPv6-only listener on macOS.
const RELAY_URL = 'ws://127.0.0.1:3700';
const RELAY_PROTOCOL_VERSION = 2;
let ws = null;
const ports = [];
let relayInfo = { protocolVersion: null, compatible: null };
let clientInfo = null;

function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  try {
    ws = new WebSocket(RELAY_URL);

    ws.onopen = () => {
      relayInfo = { protocolVersion: null, compatible: null };
      broadcast({ type: 'MCP_STATUS', status: 'connected' });
      const openedSocket = ws;
      setTimeout(() => {
        if (ws === openedSocket && ws.readyState === WebSocket.OPEN && relayInfo.protocolVersion === null) {
          relayInfo = { protocolVersion: 1, compatible: false };
          broadcast({ type: 'MCP_RELAY_INFO', ...relayInfo, expectedProtocolVersion: RELAY_PROTOCOL_VERSION });
        }
      }, 1500);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);

        if (msg.type === 'RELAY_HELLO') {
          relayInfo = {
            protocolVersion: msg.protocolVersion,
            compatible: msg.protocolVersion === RELAY_PROTOCOL_VERSION
          };
          broadcast({ type: 'MCP_RELAY_INFO', ...relayInfo, expectedProtocolVersion: RELAY_PROTOCOL_VERSION });
          return;
        }

        if (msg.type === 'CLIENT_INFO') {
          clientInfo = msg;
          broadcast({
            type: 'MCP_CLIENT_INFO',
            name: msg.name,
            version: msg.version,
            protocolVersion: msg.protocolVersion
          });
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
      clientInfo = null;
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
        connected: ws !== null && ws.readyState === WebSocket.OPEN,
        relayInfo,
        clientInfo
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
  if (relayInfo.protocolVersion !== null) port.postMessage({ type: 'MCP_RELAY_INFO', ...relayInfo, expectedProtocolVersion: RELAY_PROTOCOL_VERSION });
  if (clientInfo) port.postMessage({ type: 'MCP_CLIENT_INFO', ...clientInfo });

  // Start connecting if not already
  if (!ws || ws.readyState > 1) connect();
};
