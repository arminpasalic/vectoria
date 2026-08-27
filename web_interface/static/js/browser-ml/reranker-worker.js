/** Browser-local q8 multilingual cross-encoder worker. */

import {
    AutoModelForSequenceClassification,
    AutoTokenizer,
    env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm';
import { createPairTokenizationInput, createRerankerBatches } from './reranker.js';

env.allowLocalModels = false;
env.useBrowserCache = true;

try {
    const cores = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
    if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = cores;
        env.backends.onnx.wasm.simd = true;
        env.backends.onnx.wasm.proxy = false;
    }
} catch (_) { /* best-effort WASM tuning */ }

let tokenizer = null;
let model = null;
let loadedRevision = null;

function sigmoid(value) {
    if (value >= 0) return 1 / (1 + Math.exp(-value));
    const exp = Math.exp(value);
    return exp / (1 + exp);
}

async function initialize(modelConfig, requestId) {
    if (tokenizer && model && loadedRevision === modelConfig.revision) return;
    await model?.dispose?.();
    tokenizer = null;
    model = null;
    loadedRevision = null;

    const progress_callback = progress => self.postMessage({
        type: 'progress',
        requestId,
        progress: {
            status: progress?.status || null,
            file: progress?.file || null,
            loaded: Number(progress?.loaded) || 0,
            total: Number(progress?.total) || 0
        }
    });
    const common = { revision: modelConfig.revision, progress_callback };
    tokenizer = await AutoTokenizer.from_pretrained(modelConfig.id, common);
    model = await AutoModelForSequenceClassification.from_pretrained(modelConfig.id, {
        ...common,
        dtype: modelConfig.dtype,
        device: modelConfig.device
    });
    loadedRevision = modelConfig.revision;
}

async function scoreBatch(query, passages, modelConfig) {
    const pair = createPairTokenizationInput(query, passages, modelConfig.maxLength);
    const inputs = tokenizer(pair.queries, pair.options);
    const output = await model(inputs);
    const data = Array.from(output.logits?.data || []);
    if (data.length < passages.length) throw new Error('reranker_missing_logits');
    return data.slice(0, passages.length).map(sigmoid);
}

self.addEventListener('message', async event => {
    const message = event.data || {};
    if (message.type === 'dispose') {
        await model?.dispose?.();
        tokenizer = null;
        model = null;
        loadedRevision = null;
        return;
    }
    if (message.type !== 'rerank') return;

    try {
        const modelConfig = message.model || {};
        await initialize(modelConfig, message.requestId);
        const passages = Array.isArray(message.passages) ? message.passages : [];
        const scores = [];
        for (const batch of createRerankerBatches(passages, modelConfig.batchSize)) {
            scores.push(...await scoreBatch(message.query, batch, modelConfig));
        }
        self.postMessage({ type: 'result', requestId: message.requestId, scores });
    } catch (error) {
        self.postMessage({
            type: 'error',
            requestId: message.requestId,
            code: error?.code || 'reranker_failed',
            error: error?.message || String(error)
        });
    }
});
