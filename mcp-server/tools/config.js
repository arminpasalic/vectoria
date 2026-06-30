import { z } from 'zod';

export function registerConfigTools(server, bridge) {
  server.tool(
    'get_config',
    'Read the full Vectoria pipeline configuration: LLM settings, search weights, chunking, UMAP/clustering parameters, visualization preferences.',
    {},
    async () => {
      const result = await bridge.call('GET /config', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'set_config',
    'Update Vectoria configuration fields. Pass a partial config object — only provided fields are updated. Sections: llm (temperature, max_tokens, top_p), search (num_results, retrieval_k, vector_weight), chunking (enabled, chunk_size), visualization (point_size, opacity).',
    {
      config: z.record(z.any()).describe('Partial config object with fields to update')
    },
    async (params) => {
      const result = await bridge.call('POST /config', params.config, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
