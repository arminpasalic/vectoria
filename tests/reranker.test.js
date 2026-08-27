import test from 'node:test';
import assert from 'node:assert/strict';

import {
    BrowserReranker,
    RERANKER_MODEL,
    createPairTokenizationInput,
    createRerankerBatches,
    fuseRerankerRanks
} from '../web_interface/static/js/browser-ml/reranker.js';

class FakeWorker {
    constructor(responder = null) {
        this.responder = responder;
        this.listeners = new Map();
        this.messages = [];
        this.terminated = false;
    }

    addEventListener(type, listener) {
        this.listeners.set(type, listener);
    }

    postMessage(message) {
        this.messages.push(message);
        this.responder?.(message, payload => queueMicrotask(() => {
            this.listeners.get('message')?.({ data: payload });
        }));
    }

    terminate() {
        this.terminated = true;
    }
}

function makeResults(count) {
    return Array.from({ length: count }, (_, index) => ({ id: `doc-${index + 1}`, text: `Passage ${index + 1}` }));
}

test('reranker constants pin one multilingual browser model and its runtime limits', () => {
    assert.equal(RERANKER_MODEL.canonicalId, 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1');
    assert.equal(RERANKER_MODEL.revision, '6772eee1ea62dc82e1fcaaf3ed90c269fdcb1fcf');
    assert.equal(RERANKER_MODEL.batchSize, 8);
    assert.equal(RERANKER_MODEL.maxLength, 512);
    assert.equal(RERANKER_MODEL.maxCandidates, 30);
    assert.equal(RERANKER_MODEL.stageOneWeight, 0.3);
    assert.equal(RERANKER_MODEL.neuralWeight, 0.7);
    assert.equal(RERANKER_MODEL.license, 'Apache-2.0');
});

test('30 candidates are batched as 8/8/8/6 and every pair is truncated to 512 tokens', () => {
    assert.deepEqual(createRerankerBatches(makeResults(30), 8).map(batch => batch.length), [8, 8, 8, 6]);
    const input = createPairTokenizationInput('query', ['first', 'second']);
    assert.deepEqual(input.queries, ['query', 'query']);
    assert.deepEqual(input.options, {
        text_pair: ['first', 'second'],
        padding: true,
        truncation: true,
        max_length: 512
    });
});

test('rank fusion uses 30/70 weighted RRF, preserves stable ties, and annotates results', () => {
    const source = makeResults(4);
    const reranked = fuseRerankerRanks(source, [0.1, 0.9, 0.9]);

    assert.deepEqual(reranked.map(result => result.id), ['doc-2', 'doc-3', 'doc-1', 'doc-4']);
    assert.equal(reranked[0].pre_rerank_rank, 2);
    assert.equal(reranked[0].reranker_rank, 1);
    assert.equal(reranked[0].reranker_score, 0.9);
    assert.equal(reranked[0].reranker_applied, true);
    assert.equal(reranked[3], source[3], 'candidates beyond the scored head remain untouched');
    assert.deepEqual(source.map(result => Object.keys(result)), [['id', 'text'], ['id', 'text'], ['id', 'text'], ['id', 'text']]);
});

test('worker is lazy, receives at most 30 candidates, and can be disposed before generation', async () => {
    let factoryCalls = 0;
    const worker = new FakeWorker((message, reply) => {
        reply({
            type: 'result',
            requestId: message.requestId,
            scores: message.passages.map((_, index) => index)
        });
    });
    const reranker = new BrowserReranker({
        workerFactory: () => {
            factoryCalls += 1;
            return worker;
        }
    });

    assert.equal(factoryCalls, 0);
    const outcome = await reranker.rerank('multilingual query', makeResults(35));
    assert.equal(factoryCalls, 1);
    assert.equal(worker.messages[0].passages.length, 30);
    assert.equal(worker.messages[0].model.batchSize, 8);
    assert.equal(worker.messages[0].model.maxLength, 512);
    assert.equal(outcome.diagnostics.reranker_applied, true);
    assert.equal(outcome.diagnostics.reranker_candidates, 30);
    assert.equal(outcome.results.length, 35);
    assert.equal(outcome.results.at(-1).id, 'doc-35');

    assert.equal(reranker.releaseForGeneration(), true);
    assert.equal(worker.terminated, true);
    assert.equal(reranker.worker, null);
});

test('a warm worker is released after its configured idle interval', async () => {
    const worker = new FakeWorker((message, reply) => {
        reply({ type: 'result', requestId: message.requestId, scores: message.passages.map(() => 0.5) });
    });
    const reranker = new BrowserReranker({ workerFactory: () => worker, idleTimeoutMs: 5 });
    await reranker.rerank('query', makeResults(1));
    await new Promise(resolve => setTimeout(resolve, 15));

    assert.equal(worker.terminated, true);
    assert.equal(reranker.worker, null);
});

test('initialization and inference errors fail open to the original ranking', async () => {
    const worker = new FakeWorker((message, reply) => {
        reply({ type: 'error', requestId: message.requestId, code: 'model_download_failed', error: 'offline cache miss' });
    });
    const reranker = new BrowserReranker({ workerFactory: () => worker });
    const source = makeResults(3);
    const outcome = await reranker.rerank('query', source);

    assert.deepEqual(outcome.results, source);
    assert.equal(outcome.diagnostics.reranker_applied, false);
    assert.equal(outcome.diagnostics.reranker_fallback_reason, 'model_download_failed');
    reranker.suspend();
});

test('a failed load can recover on the next request without changing the standard fallback', async () => {
    let attempts = 0;
    const worker = new FakeWorker((message, reply) => {
        attempts += 1;
        if (attempts === 1) {
            reply({ type: 'error', requestId: message.requestId, code: 'cache_miss', error: 'not cached' });
        } else {
            reply({ type: 'result', requestId: message.requestId, scores: message.passages.map((_, index) => index) });
        }
    });
    const reranker = new BrowserReranker({ workerFactory: () => worker });
    const first = await reranker.rerank('query', makeResults(2));
    const second = await reranker.rerank('query', makeResults(2));

    assert.equal(first.diagnostics.reranker_applied, false);
    assert.deepEqual(first.results.map(result => result.id), ['doc-1', 'doc-2']);
    assert.equal(second.diagnostics.reranker_applied, true);
    assert.equal(attempts, 2);
    reranker.suspend();
});

test('cancellation terminates the worker and returns the standard ranking diagnostic', async () => {
    const worker = new FakeWorker();
    const reranker = new BrowserReranker({ workerFactory: () => worker, requestTimeoutMs: 1000 });
    const controller = new AbortController();
    const pending = reranker.rerank('query', makeResults(2), { signal: controller.signal });
    controller.abort();
    const outcome = await pending;

    assert.deepEqual(outcome.results.map(result => result.id), ['doc-1', 'doc-2']);
    assert.equal(outcome.diagnostics.reranker_applied, false);
    assert.equal(outcome.diagnostics.reranker_fallback_reason, 'aborted');
    assert.equal(worker.terminated, true);
});

test('empty queries and passages do not initialize the worker', async () => {
    let factoryCalls = 0;
    const reranker = new BrowserReranker({ workerFactory: () => {
        factoryCalls += 1;
        return new FakeWorker();
    } });

    assert.equal((await reranker.rerank('', makeResults(2))).diagnostics.reranker_fallback_reason, 'empty_query');
    assert.equal((await reranker.rerank('query', [{ id: 'empty', text: '' }])).diagnostics.reranker_fallback_reason, 'no_passages');
    assert.equal(factoryCalls, 0);
});
