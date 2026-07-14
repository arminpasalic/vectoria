import { WebSocketServer } from 'ws';

export class BrowserBridge {
  constructor(port = 3700, host = '127.0.0.1') {
    this._port = parseInt(process.env.VECTORIA_BRIDGE_PORT || port, 10);
    this._host = process.env.VECTORIA_BRIDGE_HOST || host;
    this._wss = null;
    this._client = null;
    this._pending = new Map(); // id → {resolve, reject, timer}
    this._clientInfo = null; // last known MCP client identity
  }

  async start() {
    return new Promise((resolve, reject) => {
      // Bind the same IPv4 loopback address used by the browser worker. On
      // macOS, binding without a host can create an IPv6-only listener while
      // `localhost` resolves to 127.0.0.1, leaving the UI waiting forever.
      this._wss = new WebSocketServer({ port: this._port, host: this._host });
      this._wss.on('connection', (ws) => {
        console.error(`[MCP Bridge] Browser tab connected`);
        this._client = ws;

        // Push client info to newly connected browser tab
        if (this._clientInfo) {
          ws.send(JSON.stringify({ type: 'CLIENT_INFO', ...this._clientInfo }));
        }

        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            const entry = this._pending.get(msg.id);
            if (!entry) return;
            clearTimeout(entry.timer);
            this._pending.delete(msg.id);
            if (msg.error) {
              entry.reject(new Error(msg.error));
            } else {
              entry.resolve(msg.result);
            }
          } catch (e) {
            console.error('[MCP Bridge] Bad message from browser:', e.message);
          }
        });

        ws.on('close', () => {
          console.error('[MCP Bridge] Browser tab disconnected');
          this._client = null;
          for (const [id, entry] of this._pending) {
            clearTimeout(entry.timer);
            entry.reject(new Error('Browser tab disconnected'));
            this._pending.delete(id);
          }
        });

        ws.on('error', (err) => {
          console.error('[MCP Bridge] WebSocket error:', err.message);
        });
      });

      this._wss.on('listening', () => {
        console.error(`[MCP Bridge] Relay listening on ws://${this._host}:${this._port}`);
        resolve();
      });

      this._wss.on('error', (err) => {
        console.error('[MCP Bridge] Server error:', err.message);
        reject(err);
      });
    });
  }

  get isConnected() {
    return this._client !== null && this._client.readyState === 1;
  }

  sendClientInfo(info) {
    this._clientInfo = { name: info.name, version: info.version || '' };
    if (this.isConnected) {
      this._client.send(JSON.stringify({ type: 'CLIENT_INFO', ...this._clientInfo }));
    }
  }

  call(method, params = {}, timeoutMs = 30000) {
    if (!this.isConnected) {
      return Promise.reject(new Error(
        'Vectoria browser tab not connected. Open Vectoria in your browser, then enable MCP Bridge in Advanced Settings.'
      ));
    }

    const id = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pending.delete(id);
        reject(new Error(`Request timed out after ${timeoutMs}ms (method: ${method})`));
      }, timeoutMs);

      this._pending.set(id, { resolve, reject, timer });
      this._client.send(JSON.stringify({ id, method, params }));
    });
  }
}
