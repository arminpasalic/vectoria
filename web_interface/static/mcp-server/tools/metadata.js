import { z } from 'zod';

export function registerMetadataTools(server, bridge) {
  server.tool(
    'get_metadata_schema',
    'Get metadata field definitions: names, types, unique value counts, and sample values.',
    {},
    async () => {
      const result = await bridge.call('GET /metadata_schema', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'set_metadata_filters',
    'Apply metadata filters to scope all subsequent searches. Filters persist until cleared. Example: {"category": "news"} or {"year": {"min": 2020, "max": 2024}}.',
    {
      filters: z.record(z.any()).describe('Key-value metadata filters')
    },
    async (params) => {
      const result = await bridge.call('POST /api/set-metadata-filters', { filters: params.filters }, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'clear_metadata_filters',
    'Remove all active metadata filters so searches return results from the full dataset.',
    {},
    async () => {
      const result = await bridge.call('POST /api/set-metadata-filters', { filters: {} }, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
