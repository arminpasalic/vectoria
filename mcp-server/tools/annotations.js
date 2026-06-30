import { z } from 'zod';

export function registerAnnotationTools(server, bridge) {
  server.tool(
    'annotate_documents',
    'Attach a labelled annotation (tag + optional note/color) to one or more documents. Annotations stay in browser memory for the rest of the session and ride along inside vectoria.json save files so a multi-step analysis can accumulate findings.',
    {
      doc_indices: z.array(z.number().int()).describe('Document indices to annotate'),
      tag:         z.string().describe('Short label, e.g. "interesting", "review", "fraud-signal"'),
      note:        z.string().optional().describe('Free-form note attached to each annotation'),
      color:       z.string().optional().describe('Optional CSS-color string used by the UI')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/annotations/add', params, 15000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'list_annotations',
    'List previously created annotations, optionally filtered by tag or by a specific document index.',
    {
      tag:       z.string().optional(),
      doc_index: z.number().int().optional()
    },
    async (params) => {
      const result = await bridge.call('GET /bridge/annotations/list', params || {}, 15000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
