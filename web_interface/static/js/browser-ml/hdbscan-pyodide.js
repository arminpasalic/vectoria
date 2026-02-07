let worker;
let workerReadyPromise;
const pendingRequests = new Map();
let nextRequestId = 0;

function ensureWorker() {
  if (!worker) {
    worker = new Worker(new URL('./pyodide-hdbscan-worker.js', import.meta.url), {
      type: 'module'
    });

    worker.onmessage = (event) => {
      const { type, id, result, error } = event.data || {};
      if (type === 'ready') {
        if (workerReadyPromise) {
          workerReadyPromise.resolve();
        }
        return;
      }

      if (type === 'result' || type === 'error') {
        const deferred = pendingRequests.get(id);
        if (!deferred) {
          console.warn('Received response for unknown request id', id);
          return;
        }
        pendingRequests.delete(id);
        if (type === 'result') {
          deferred.resolve(result);
        } else {
          deferred.reject(new Error(error));
        }
      }
    };

    worker.onerror = (event) => {
      const message = event?.message || 'Unknown worker error';
      if (workerReadyPromise) {
        workerReadyPromise.reject(new Error(message));
      }
      pendingRequests.forEach(({ reject }) => reject(new Error(message)));
      pendingRequests.clear();
    };

    workerReadyPromise = createDeferred();
    worker.postMessage({ type: 'init' });
  }

  return workerReadyPromise.promise;
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Runs HDBSCAN inside the Pyodide worker.
 * @param {number[][]} data - 2D array of points (rows = samples).
 * @param {{ minClusterSize: number, minSamples: number, metric: string }} options
 * @returns {Promise<{ labels: number[], probabilities: number[] }>}
 */
export async function runHDBSCANPyodide(
  data,
  {
    minClusterSize,
    minSamples,
    metric,
    documents = null,
    keywordOptions = null
  }
) {
  await ensureWorker();

  const requestId = nextRequestId++;
  const deferred = createDeferred();
  pendingRequests.set(requestId, deferred);

  worker.postMessage({
    type: 'run',
    id: requestId,
    payload: {
      data,
      minClusterSize,
      minSamples,
      metric,
      documents,
      keywordOptions
    }
  });

  return deferred.promise;
}
