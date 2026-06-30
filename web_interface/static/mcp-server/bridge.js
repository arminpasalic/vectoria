import { WebSocketServer } from 'ws';

export class BrowserBridge {
  constructor(port = 3700) {
    this._port = parseInt(process.env.VECTORIA_BRIDGE_PORT || port, 10);
    this._wss = null;
    this._client = null;
    this._pending = new Map(); // id → {resolve, reject, timer}
    this._clientInfo = null; // last known MCP client identity
  }

  async start() {
    return new Promise((resolve) => {
      this._wss = new WebSocketServer({ port: this._port });
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
        console.error(`[MCP Bridge] Relay listening on ws://localhost:${this._port}`);
        resolve();
      });

      this._wss.on('error', (err) => {
        console.error('[MCP Bridge] Server error:', err.message);
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
        'Vectoria browser tab not connected. Open http://localhost:5050 first, then enable MCP Bridge in Advanced Settings.'
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
