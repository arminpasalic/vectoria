import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { BrowserBridge } from './bridge.js';
import { registerSearchTools } from './tools/search.js';
import { registerRagTools } from './tools/rag.js';
import { registerDataTools } from './tools/data.js';
import { registerMetadataTools } from './tools/metadata.js';
import { registerConfigTools } from './tools/config.js';
import { registerDatasetTools } from './tools/dataset.js';
import { registerAnalysisTools } from './tools/analysis.js';
import { registerAnnotationTools } from './tools/annotations.js';
import { registerClusterTools } from './tools/clusters.js';
import { registerMetricTools } from './tools/metrics.js';
import { registerSessionTools } from './tools/sessions.js';

const bridge = new BrowserBridge();
await bridge.start();

const server = new McpServer({
  name: 'vectoria',
  version: '1.0.0'
});

registerSearchTools(server, bridge);
registerRagTools(server, bridge);
registerDataTools(server, bridge);
registerMetadataTools(server, bridge);
registerConfigTools(server, bridge);
registerDatasetTools(server, bridge);
registerAnalysisTools(server, bridge);
registerAnnotationTools(server, bridge);
registerClusterTools(server, bridge);
registerMetricTools(server, bridge);
registerSessionTools(server, bridge);

const transport = new StdioServerTransport();
await server.connect(transport);

// Poll briefly for clientInfo after handshake completes (non-blocking)
(async () => {
  for (let i = 0; i < 40; i++) {
    await new Promise(r => setTimeout(r, 100));
    try {
      const info = server.server?.getClientVersion?.();
      if (info?.name) {
        console.error(`[Vectoria MCP] Client: ${info.name} ${info.version || ''}`);
        bridge.sendClientInfo(info);
        return;
      }
    } catch (_) {}
  }
})();

console.error('[Vectoria MCP] Server ready. Open Vectoria in your browser and enable MCP Bridge in Advanced Settings.');
