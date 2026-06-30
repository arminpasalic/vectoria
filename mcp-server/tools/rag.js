import { z } from 'zod';

export function registerRagTools(server, bridge) {
  server.tool(
    'query_rag_local',
    'Ask a question using RAG with the locally running ONNX model in the browser. Returns an AI-generated answer with cited sources. Can take 30-120 seconds.',
    {
      question:    z.string().describe('The question to answer'),
      search_type: z.enum(['semantic', 'hybrid', 'fast']).default('semantic'),
      num_results: z.number().int().default(5).describe('Number of chunks to retrieve')
    },
    async (params) => {
      const result = await bridge.call('POST /query', {
        question: params.question,
        search_type: params.search_type ?? 'semantic',
        num_results: params.num_results ?? 5
      }, 120000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'query_rag_external',
    `Retrieve relevant context from the Vectoria dataset for a question. Returns a formatted context_prompt and source list. IMPORTANT: After calling this tool, use the returned context_prompt to answer the user's question yourself — do not call another tool.`,
    {
      question:    z.string().describe('The question to retrieve context for'),
      k:           z.number().int().default(5).describe('Number of context chunks to retrieve'),
      search_type: z.enum(['semantic', 'hybrid', 'fast']).default('semantic')
    },
    async (params) => {
      const data = await bridge.call('POST /search', {
        query: params.question,
        k: params.k ?? 5,
        search_type: params.search_type ?? 'semantic',
        include_metadata: true
      }, 15000);

      if (!data.results || data.results.length === 0) {
        return { content: [{ type: 'text', text: JSON.stringify({
          context_prompt: `No relevant documents found for: "${params.question}".`,
          sources: [],
          question: params.question
        }, null, 2) }] };
      }

      const contextLines = data.results.map((r, i) =>
        `[Source ${i + 1} | score: ${(r.score || 0).toFixed(3)}]\n${r.text}`
      ).join('\n\n---\n\n');

      return { content: [{ type: 'text', text: JSON.stringify({
        context_prompt: `Answer the following question using ONLY the sources below. Cite sources by number.\n\nQuestion: ${params.question}\n\n${contextLines}`,
        sources: data.results,
        question: params.question,
        instruction: 'Use context_prompt above to answer the question.'
      }, null, 2) }] };
    }
  );
}
