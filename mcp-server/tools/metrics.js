import { z } from 'zod';

export function registerMetricTools(server, bridge) {
  server.tool(
    'register_metric',
    'Define a reusable derived metric as a small whitelisted arithmetic formula over metadata fields, e.g. "likes + 2*shares". After registering, the metric name can be passed wherever a metric is expected (e.g. aggregate metric="engagement"). Supports +, -, *, /, parentheses, and unary minus; identifiers resolve to doc.metadata[field].',
    {
      name:        z.string().describe('Metric name (used to reference it later)'),
      formula:     z.string().describe('Arithmetic expression over metadata field names, e.g. "likes + 2*shares"'),
      description: z.string().optional()
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/metrics/register', params, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'list_metrics',
    'List all currently registered metrics with a sample evaluation on the first document.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/metrics/list', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
