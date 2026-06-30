import { z } from 'zod';

export function registerClusterTools(server, bridge) {
  server.tool(
    'set_cluster_label',
    'Set a cluster\'s display label. Used by the external-summarizer path: after summarize_cluster returns exemplars + a prompt template, synthesize a 3-5 word label and call this to persist it. The new label propagates immediately to the WebGL viz, the legend, and the cluster panel.',
    {
      cluster_id: z.number().int(),
      label:      z.string().describe('3-5 word topic label'),
      source:     z.enum(['mcp', 'local', 'manual', 'session']).default('mcp')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/clusters/label', params, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
