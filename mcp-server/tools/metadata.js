import { z } from 'zod';

export function registerMetadataTools(server, bridge) {
  server.tool(
    'get_metadata_schema',
    'Get metadata field definitions: names, types, unique value counts, and sample values.',
    {},
    async () => {
      const result = await bridge.call('GET /metadata_schema', {}, 10000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'set_metadata_filters',
    'Apply persistent metadata filters to all subsequent filter-aware MCP tools. Inline filters override the same fields for one call. Filters reset when the dataset changes or when cleared. Example: {"category": "news"} or {"year": {"min": 2020, "max": 2024}}.',
    {
      filters: z.record(z.any()).describe('Key-value metadata filters')
    },
    async (params) => {
      const result = await bridge.call('POST /api/set-metadata-filters', { filters: params.filters }, 10000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'clear_metadata_filters',
    'Remove all persistent MCP metadata filters so filter-aware tools use the full dataset unless they receive inline filters.',
    {},
    async () => {
      const result = await bridge.call('POST /api/set-metadata-filters', { filters: {} }, 10000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );
}
