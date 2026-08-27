import { z } from 'zod';

export function registerAnalysisTools(server, bridge) {
  server.tool(
    'summarize_cluster',
    'Abstractively label and summarize an HDBSCAN cluster using quantile-based exemplar sampling (low/mid/high HDBSCAN confidence). Two modes: summarizer="local" runs the in-browser Gemma ONNX model and returns {label, summary, exemplars, keywords, coverage}; summarizer="external" returns exemplars + a ready-made prompt_template for you to synthesize, then call set_cluster_label to persist. Default n_exemplars is adaptive to cluster size (3/6/9/12-15).',
    {
      cluster_id:    z.number().int().describe('Cluster id (-1 for the noise/outlier bucket)'),
      summarizer:    z.enum(['local', 'external']).default('external'),
      n_exemplars:   z.number().int().optional().describe('Override adaptive exemplar count'),
      persist_label: z.boolean().default(true).describe('When local mode produces a label, store it as the cluster\'s display label')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/summarize_cluster', params, 180000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'get_outliers',
    'Return documents that did not cluster well: HDBSCAN noise (cluster_id = -1) plus any whose membership probability falls below the threshold. Use this for "what do my exotic / fringe documents look like" investigations.',
    {
      threshold:    z.number().min(0).max(1).default(0.5).describe('Probability cutoff for low-confidence members'),
      k:            z.number().int().default(50),
      include_text: z.boolean().default(false)
    },
    async (params) => {
      const result = await bridge.call('GET /bridge/outliers', params, 30000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'cross_tabulate',
    'Pivot the dataset on two fields and return the contingency table plus chi-square / Cramér\'s V. Use col_field="__cluster__" to test whether a metadata field is independent of cluster membership.',
    {
      row_field:  z.string().describe('Metadata field for the rows'),
      col_field:  z.string().describe('Metadata field for the columns, or "__cluster__"'),
      normalize:  z.enum(['none', 'row', 'col', 'total']).default('none'),
      filter:     z.record(z.any()).optional().describe('Optional per-call metadata filter; overrides matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/cross_tabulate', params, 30000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'aggregate',
    'Group documents by a field (or "__cluster__") and aggregate a metric. Metric can be "__count__", a metadata field, or a registered metric name (see register_metric).',
    {
      group_by: z.string().describe('Field to group by, or "__cluster__"'),
      metric:   z.string().default('__count__'),
      agg:      z.enum(['count', 'sum', 'mean', 'median', 'min', 'max']).default('count'),
      filter:   z.record(z.any()).optional().describe('Optional per-call metadata filter; overrides matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/aggregate', params, 30000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'compare_clusters',
    'Compare two or more clusters across metadata fields. Returns per-field distributions plus a divergence score (Jensen-Shannon for categorical, Welch\'s t for numeric).',
    {
      cluster_ids: z.array(z.number().int()).describe('Cluster ids to compare (>=2)'),
      fields:      z.array(z.string()).optional().describe('Restrict comparison to these metadata fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/compare_clusters', params, 30000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'multi_vector_search',
    'Run several queries (text strings, or pre-computed vectors) and fuse the result lists with Reciprocal Rank Fusion. Each result includes contributing_queries so you can see which queries surfaced it.',
    {
      queries:          z.array(z.union([z.string(), z.object({ text: z.string().optional(), vector: z.array(z.number()).optional(), label: z.string().optional() })])),
      k:                z.number().int().default(10),
      rrf_k:            z.number().int().default(60).describe('RRF damping constant (higher = flatter contribution)'),
      fuse:             z.enum(['rrf', 'mean']).default('rrf'),
      metadata_filters: z.record(z.any()).optional().describe('Optional per-call metadata filters; override matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/multi_vector_search', params, 60000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'query_with_citations',
    'Run a RAG query where the generated answer is split into per-sentence claims, each linked back to the supporting source documents with a confidence score. Use for audit-style answers where every statement needs provenance.',
    {
      question:             z.string(),
      k:                    z.number().int().default(5),
      search_type:          z.enum(['semantic', 'hybrid']).default('semantic'),
      confidence_threshold: z.number().min(0).max(1).default(0.0).describe('Drop claims with confidence below this'),
      metadata_filters:     z.record(z.any()).optional().describe('Per-call filters that override matching persistent filter fields')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/query_with_citations', params, 180000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );

  server.tool(
    'filter_to_subset',
    'Create a named in-memory subset from a metadata filter. Returns a subset_id and the matching doc_indices, useful as a checkpoint inside a multi-step analysis (e.g. "now examine only the news-category docs").',
    {
      filters: z.record(z.any()).describe('Metadata filter, same shape as set_metadata_filters; overrides matching persistent filter fields'),
      name:    z.string().optional()
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/filter_to_subset', params, 30000);
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
        ...(result?.error ? { isError: true } : {})
      };
    }
  );
}
