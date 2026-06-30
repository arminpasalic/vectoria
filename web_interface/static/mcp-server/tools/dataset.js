import { z } from 'zod';

export function registerDatasetTools(server, bridge) {
  server.tool(
    'list_datasets',
    'Check if a dataset is loaded in Vectoria: readiness, document count, dataset ID, filename, cluster count.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/status', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    'get_dataset_info',
    'Get detailed information about the loaded dataset: filename, file type, text column, document counts, empty/duplicate row stats, and cluster keywords.',
    {},
    async () => {
      const result = await bridge.call('GET /bridge/dataset_info', {}, 10000);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    }
  );
}
