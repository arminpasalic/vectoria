import { WebSocketServer } from 'ws';
import { createServer } from 'node:http';

export const BRIDGE_PROTOCOL_VERSION = 2;

export class BrowserBridge {
  constructor(port = 3700, host = '127.0.0.1', allowedOrigins = []) {
    this._port = parseInt(process.env.VECTORIA_BRIDGE_PORT || port, 10);
    this._host = process.env.VECTORIA_BRIDGE_HOST || host;
    this._allowedOrigins = new Set([
      'https://vectoria.app',
      'https://www.vectoria.app',
      'http://localhost:5050',
      'http://127.0.0.1:5050',
      ...String(process.env.VECTORIA_ALLOWED_ORIGINS || '').split(','),
      ...allowedOrigins
    ].map((value) => this._normalizeOrigin(value)).filter(Boolean));
    this._httpServer = null;
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
      this._httpServer = createServer((req, res) => this._handleHttpRequest(req, res));
      this._wss = new WebSocketServer({ server: this._httpServer });
      this._wss.on('connection', (ws, req) => {
        const origin = this._normalizeOrigin(req.headers.origin);
        if (!origin || !this._allowedOrigins.has(origin)) {
          console.error(`[MCP Bridge] Rejected browser origin: ${origin || '(missing)'}`);
          ws.close(1008, 'Origin not allowed');
          return;
        }

        console.error(`[MCP Bridge] Browser tab connected`);
        if (this._client && this._client !== ws) {
          this._client.close(1000, 'Replaced by another Vectoria tab');
        }
        this._client = ws;

        ws.send(JSON.stringify({
          type: 'RELAY_HELLO',
          protocolVersion: BRIDGE_PROTOCOL_VERSION,
          service: 'vectoria-mcp-relay'
        }));

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
          if (this._client === ws) this._client = null;
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

      this._httpServer.on('listening', () => {
        console.error(`[MCP Bridge] Relay listening on ws://${this._host}:${this._port}`);
        resolve();
      });

      this._httpServer.on('error', (err) => {
        console.error('[MCP Bridge] Server error:', err.message);
        reject(err);
      });

      this._httpServer.listen(this._port, this._host);
    });
  }

  _normalizeOrigin(value) {
    if (!value) return '';
    try {
      return new URL(String(value).trim()).origin;
    } catch (_) {
      return '';
    }
  }

  _handleHttpRequest(req, res) {
    const origin = this._normalizeOrigin(req.headers.origin);
    if (!origin || !this._allowedOrigins.has(origin)) {
      res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: false, error: 'Origin not allowed' }));
      return;
    }

    const corsHeaders = {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Cache-Control': 'no-store',
      'Vary': 'Origin'
    };

    if (req.method === 'OPTIONS') {
      res.writeHead(204, corsHeaders);
      res.end();
      return;
    }

    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200, { ...corsHeaders, 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: true, service: 'vectoria-mcp-relay' }));
      return;
    }

    res.writeHead(404, { ...corsHeaders, 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: false, error: 'Not found' }));
  }

  get isConnected() {
    return this._client !== null && this._client.readyState === 1;
  }

  sendClientInfo(info) {
    this._clientInfo = {
      name: info.name,
      version: info.version || '',
      protocolVersion: BRIDGE_PROTOCOL_VERSION
    };
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
