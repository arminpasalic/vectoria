import { z } from 'zod';

export function registerSearchTools(server, bridge) {
  server.tool(
    'search',
    'Search the Vectoria dataset using keyword (BM25) or semantic vector search.',
    {
      query:            z.string().describe('Search query'),
      search_type:      z.enum(['fast', 'semantic']).default('fast').describe('fast = BM25 keyword, semantic = vector similarity'),
      k:                z.number().int().default(10).describe('Number of results'),
      include_metadata: z.boolean().default(true),
      metadata_filters: z.record(z.any()).optional().describe('Per-call filters, e.g. {"category": "news"}; override matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /search', {
        query: params.query,
        search_type: params.search_type ?? 'fast',
        k: params.k ?? 10,
        include_metadata: params.include_metadata ?? true,
        metadata_filters: params.metadata_filters ?? {},
        use_persistent_filters: true
      }, 15000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'hybrid_search',
    'Combined vector + BM25 hybrid search with configurable weight balance.',
    {
      query:            z.string().describe('Search query'),
      k:                z.number().int().default(10),
      vector_weight:    z.number().min(0).max(1).default(0.6).describe('Weight for vector search (0-1). Remainder goes to BM25.'),
      include_metadata: z.boolean().default(true),
      metadata_filters: z.record(z.any()).optional().describe('Per-call filters that override matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /search', {
        query: params.query,
        search_type: 'hybrid',
        k: params.k ?? 10,
        vector_weight: params.vector_weight ?? 0.6,
        include_metadata: params.include_metadata ?? true,
        metadata_filters: params.metadata_filters ?? {},
        use_persistent_filters: true
      }, 15000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'get_document',
    'Fetch a single document by index, including full text and metadata.',
    {
      index: z.number().int().describe('Document index (0-based)')
    },
    async (params) => {
      const result = await bridge.call('GET /bridge/document', { index: params.index }, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'get_documents_by_cluster',
    'Get all documents belonging to a specific cluster. Use -1 for outliers/noise.',
    {
      cluster_id: z.number().int().describe('Cluster ID (-1 = outliers)')
    },
    async (params) => {
      const result = await bridge.call('GET /bridge/documents_by_cluster', { cluster_id: params.cluster_id }, 15000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
