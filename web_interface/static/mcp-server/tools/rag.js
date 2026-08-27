import { z } from 'zod';

export function registerRagTools(server, bridge) {
  server.tool(
    'query_rag_local',
    'Ask Vectoria\'s active dataset a document-grounded question using the browser-local WebLLM model. Returns an AI-generated answer with cited sources. The MCP host owns conversational routing and memory. Can take 30-120 seconds.',
    {
      question:    z.string().describe('The question to answer'),
      search_type: z.enum(['semantic', 'hybrid', 'fast']).default('semantic'),
      num_results: z.number().int().default(5).describe('Number of chunks to retrieve'),
      metadata_filters: z.record(z.any()).optional().describe('Per-call filters that override matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/query_local', {
        question: params.question,
        search_type: params.search_type ?? 'semantic',
        num_results: params.num_results ?? 5,
        generation_provider: 'local',
        scope: 'all',
        metadata_filters: params.metadata_filters || {},
        use_persistent_filters: true
      }, 120000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'query_rag_external',
    `Retrieve relevant context when the user asks about Vectoria's active dataset. Do not call this for ordinary client conversation. Returns a formatted context_prompt and source list; the MCP host owns conversational routing and memory. IMPORTANT: After calling this tool, use the returned context_prompt to answer the user's question yourself — do not call another tool.`,
    {
      question:    z.string().describe('The question to retrieve context for'),
      k:           z.number().int().default(5).describe('Number of context chunks to retrieve'),
      search_type: z.enum(['semantic', 'hybrid', 'fast']).default('semantic'),
      scope:       z.enum(['all', 'current']).default('all').describe('Search the full dataset or the active UI selection/filters'),
      metadata_filters: z.record(z.any()).optional().describe('Per-call filters that override matching persistent filter fields'),
      include_metadata: z.boolean().default(true),
      metadata_fields: z.array(z.string()).optional()
    },
    async (params) => {
      const data = await bridge.call('POST /bridge/retrieve_context', {
        question: params.question,
        k: params.k ?? 5,
        search_type: params.search_type ?? 'semantic',
        scope: params.scope ?? 'all',
        metadata_filters: params.metadata_filters || {},
        include_metadata: params.include_metadata !== false,
        metadata_fields: params.metadata_fields,
        use_persistent_filters: true
      }, 30000);

      if (data.error) {
        return { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], isError: true };
      }

      if (!data.sources || data.sources.length === 0) {
        return { content: [{ type: 'text', text: JSON.stringify({
          context_prompt: data.context_prompt || `No relevant documents found for: "${params.question}".`,
          context: data.context || '',
          sources: [],
          question: params.question,
          scope: data.scope,
          retrieval_metadata: data.retrieval_metadata,
          generation_provider: data.generation_provider
        }, null, 2) }] };
      }

      return { content: [{ type: 'text', text: JSON.stringify({
        context_prompt: data.context_prompt,
        context: data.context,
        sources: data.sources,
        question: params.question,
        scope: data.scope,
        retrieval_metadata: data.retrieval_metadata,
        generation_provider: data.generation_provider,
        instruction: 'Use context_prompt above to answer the question.'
      }, null, 2) }] };
    }
  );
}
