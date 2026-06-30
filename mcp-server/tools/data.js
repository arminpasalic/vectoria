import { z } from 'zod';

export function registerDataTools(server, bridge) {
  server.tool(
    'get_visualization_data',
    'Get the 2D UMAP coordinates, cluster assignments, and colors for all documents.',
    {
      include_text: z.boolean().default(false).describe('Include document text in each point (makes response much larger)')
    },
    async (params) => {
      const result = await bridge.call('GET /api/visualization_data', {}, 15000);
      if (!params.include_text && result.points) {
        result.points = result.points.map(({ text: _t, ...rest }) => rest);
      }
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'get_cluster_summary',
    'Get a summary of all clusters: IDs, labels, sizes, and top keywords.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/cluster_summary', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'get_dataset_stats',
    'Get statistics about the loaded dataset: document count, cluster count, embedding cache stats, RAG chunk count.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/status', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'get_point_neighbors',
    'Find the K nearest neighbors (most similar documents) to a given document by index.',
    {
      index: z.number().int().describe('Document index (0-based)'),
      k:     z.number().int().default(5).describe('Number of neighbors to return')
    },
    async (params) => {
      const result = await bridge.call('GET /bridge/point_neighbors', {
        index: params.index,
        k: params.k ?? 5
      }, 15000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
