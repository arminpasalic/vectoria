import { z } from 'zod';

export function registerSessionTools(server, bridge) {
  server.tool(
    'save_analysis_session',
    'Snapshot the current analysis state (active filters, registered metrics, annotations, custom cluster labels, subsets, plus caller-supplied findings) into a tamper-evidenced session record. Optionally downloads it as a standalone vectoria-session-*.json with a SHA-256 signature.',
    {
      name:     z.string().describe('Human-readable session name'),
      findings: z.string().optional().describe('Narrative findings to include in the session'),
      download: z.boolean().default(true).describe('Also download a signed JSON artifact in the browser')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/sessions/save', params, 30000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'list_sessions',
    'List in-memory analysis sessions saved this browser session.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/sessions/list', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'load_analysis_session',
    'Restore an analysis session — either by session_id (from list_sessions) or by passing the full json_payload from a previously downloaded vectoria-session-*.json. Hash verification will fail if the artifact has been tampered with.',
    {
      session_id:   z.string().optional(),
      json_payload: z.record(z.any()).optional().describe('Full session JSON, including the {version, payload, sha256} wrapper')
    },
    async (params) => {
      const result = await bridge.call('POST /bridge/sessions/load', params, 30000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
