import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { normalizeLocalAIError } from '../web_interface/static/js/browser-ml/chat-context.js';
import { rerankAndDiversify } from '../web_interface/static/js/browser-ml/retrieval-ranking.js';
import { DEFAULT_CONFIG } from '../web_interface/static/js/config-manager.js';

const rawSource = await readFile(new URL('../web_interface/static/js/browser-ml/llm-rag.js', import.meta.url), 'utf8');
const executableSource = rawSource
    .replace(/^import .*?;\n/gm, '')
    .replace('export async function isWebLLMModelCached', 'async function isWebLLMModelCached')
    .replace(/import\.meta\.url/g, '"https://vectoria.invalid/llm-rag.js"')
    .replace('export class BrowserRAG', 'class BrowserRAG')
    .concat('\nglobalThis.BrowserRAG = BrowserRAG;');

function loadBrowserRAG() {
    const context = {
        console,
        setTimeout,
        clearTimeout,
        performance: { now: () => 0 },
        localStorage: { getItem: () => null, setItem() {} },
        getModelConstraints: () => ({
            contextWindow: 2048,
            temp: [0, 2],
            maxTokens: [1, 2048],
            hasThinkMode: false
        }),
        rerankAndDiversify,
        // llm-rag.js imports its fallbacks from config-manager.js, so the VM
        // harness supplies the real DEFAULT_CONFIG rather than a stub.
        DEFAULT_CONFIG,
        prebuiltAppConfig: {},
        hasModelInCache: async () => false,
        CreateWebWorkerMLCEngine: async () => { throw new Error('not used in lifecycle tests'); },
        CustomEvent: class CustomEvent { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        document: { dispatchEvent() {} }
    };
    context.window = context;
    vm.createContext(context);
    vm.runInContext(executableSource, context);
    return context.BrowserRAG;
}

function createRAG(vectorSearch = null, chunkVectorSearch = null, bm25Search = null, reranker = null) {
    const BrowserRAG = loadBrowserRAG();
    const rag = new BrowserRAG(vectorSearch, chunkVectorSearch, bm25Search, reranker);
    rag._scheduleIdleSuspend = () => {};
    return rag;
}

test('a transient shard download failure is retried instead of failing the load', async () => {
    const rag = createRAG();
    // Counters must be real numbers on a fresh instance: `undefined < 3` is
    // false, which would silently disable the retry on a first load.
    assert.equal(rag._shardRetries, 0);
    assert.equal(rag._idbRetried, false);

    const shardError = 'Failed to store https://huggingface.co/mlc-ai/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC/resolve/main/params_shard_15.bin with error: Error: Network response was not ok';
    const matcher = /failed to store .*(network response was not ok|failed to fetch|networkerror)/i;

    // WebLLM capitalises "Network", so a case-sensitive /network/ test misses it.
    assert.ok(matcher.test(shardError));
    assert.ok(matcher.test(shardError.toLowerCase()));
    assert.ok(matcher.test('Failed to store https://x/p.bin with error: TypeError: Failed to fetch'));

    // Genuine, non-transient failures must still surface immediately.
    assert.ok(!matcher.test('WebGPU is not supported in this browser'));
    assert.ok(!matcher.test('Out of memory while allocating KV cache'));
    assert.ok(!matcher.test('ArtifactIndexedDBCache failed to open'));
});

test('shard retries are bounded so a persistent outage cannot loop forever', async () => {
    const MAX_SHARD_RETRIES = 3;
    const matcher = /failed to store .*(network response was not ok|failed to fetch|networkerror)/i;
    const shardError = 'Failed to store https://x/params_shard_15.bin with error: Error: Network response was not ok';

    const run = async failures => {
        let remaining = failures;
        let attempts = 0;
        let retries = 0;
        const attempt = async () => {
            attempts += 1;
            try {
                if (remaining > 0) { remaining -= 1; throw new Error(shardError); }
                return 'loaded';
            } catch (error) {
                if (matcher.test(error.message) && retries < MAX_SHARD_RETRIES) {
                    retries += 1;
                    return attempt();
                }
                throw error;
            }
        };
        try { return { result: await attempt(), attempts }; }
        catch { return { result: 'failed', attempts }; }
    };

    assert.deepEqual(await run(0), { result: 'loaded', attempts: 1 });
    assert.deepEqual(await run(3), { result: 'loaded', attempts: 4 });
    assert.deepEqual(await run(9), { result: 'failed', attempts: MAX_SHARD_RETRIES + 1 });
});

test('operation-scoped cancellation is idle-safe and cannot leak into the next turn', () => {
    const rag = createRAG();
    assert.equal(rag.abort('chat'), false);

    const first = rag.beginOperation('chat', { datasetId: 'one' });
    assert.equal(rag.abort({ id: first.id + 1 }), false);
    assert.equal(rag.shouldAbort, false);
    assert.equal(rag.abort(first), true);
    assert.equal(rag.shouldAbort, true);
    assert.throws(() => rag.throwIfOperationCancelled(first), /stopped by user/);
    assert.equal(rag.endOperation(first), true);

    const second = rag.beginOperation('chat', { datasetId: 'one' });
    assert.equal(rag.shouldAbort, false);
    assert.doesNotThrow(() => rag.throwIfOperationCancelled(second));
    rag.endOperation(second);
});

test('stopping model loading leaves the worker cold and reloadable', () => {
    const rag = createRAG();
    let terminated = false;
    rag.worker = { terminate() { terminated = true; } };
    const operation = rag.beginOperation('chat');
    rag.setOperationPhase(operation, 'loading-model');
    assert.equal(rag.abort(operation), true);
    assert.equal(terminated, true);
    assert.equal(rag.worker, null);
    assert.equal(rag.engine, null);
    assert.equal(rag.isInitialized, false);
    assert.equal(rag.isSuspended, true);
    assert.equal(rag.workerUnloaded, false);
    rag.endOperation(operation);
});

test('every local generation readiness check releases the WASM reranker first', async () => {
    let releases = 0;
    const rag = createRAG(null, null, null, { releaseForGeneration() { releases += 1; } });
    rag.engine = {};
    rag.isInitialized = true;

    await rag.ensureEngineReady();
    assert.equal(releases, 1);
});

test('chat generation resets WebLLM once before streaming and rejects empty completions', async () => {
    const rag = createRAG();
    const calls = [];
    let requestMessages = [];
    rag.ensureEngineReady = async () => {};
    rag.engine = {
        async resetChat() { calls.push('reset'); },
        chat: { completions: { async create(request) {
            calls.push('create');
            requestMessages = request.messages;
            return (async function* () {
                yield { choices: [{ delta: { content: 'Grounded answer' }, finish_reason: null }] };
                yield { choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 } };
            })();
        } } }
    };
    rag.isInitialized = true;
    const result = await rag.generateFromMessages([
        { role: 'user', content: 'Question' },
        { role: 'system', content: 'Primary policy' },
        { role: 'system', content: 'Continuity digest' }
    ]);
    assert.deepEqual(calls, ['reset', 'create']);
    assert.equal(result.answer, 'Grounded answer');
    assert.deepEqual(Array.from(requestMessages, message => message.role), ['system', 'user']);
    assert.match(requestMessages[0].content, /Primary policy[\s\S]*Continuity digest/);
    assert.deepEqual(Array.from(result.metadata.promptCompatibility.roles), ['system', 'user']);
    assert.equal(result.metadata.promptCompatibility.messageCount, 2);
    assert.equal('content' in result.metadata.promptCompatibility, false);

    rag.engine.chat.completions.create = async () => (async function* () {
        yield { choices: [{ delta: {}, finish_reason: 'stop' }] };
    })();
    await assert.rejects(
        rag.generateFromMessages([{ role: 'user', content: 'Question' }]),
        /empty completion/i
    );
});

test('prompt compatibility diagnostics compare Gemma roles without recording content', () => {
    const rag = createRAG();
    const messages = [{ role: 'system', content: 'secret policy' }, { role: 'user', content: 'private question' }];

    rag.modelId = 'gemma-2-9b-it-q4f16_1-MLC';
    const gemma = rag._promptCompatibilityDiagnostic(messages);
    assert.equal(gemma.modelFamily, 'gemma-legacy-chat-template');
    assert.deepEqual(Array.from(gemma.roles), ['system', 'user']);
    assert.deepEqual(Array.from(gemma.foldedFirstUserRoles), ['user']);
    assert.doesNotMatch(JSON.stringify(gemma), /secret|private/);

    for (const modelId of [
        'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC',
        'Qwen3-4B-q4f16_1-MLC',
        'Llama-3.1-8B-Instruct-q4f16_1-MLC'
    ]) {
        rag.modelId = modelId;
        const diagnostic = rag._promptCompatibilityDiagnostic(messages);
        assert.equal(diagnostic.modelFamily, 'native-or-mlc-system-template');
        assert.equal(diagnostic.foldedFirstUserRoles, null);
    }
});

test('Qwen3 applies the selected direct or reasoning switch to the final user turn', () => {
    const rag = createRAG();
    rag.modelConstraints = {
        responseMode: 'switchable',
        noThinkSwitch: '/no_think',
        thinkSwitch: '/think'
    };

    rag.reasoningMode = 'direct';
    let messages = rag._chatMessages([{ role: 'user', content: 'Answer this' }]);
    assert.match(messages[0].content, /\/no_think$/);

    rag.reasoningMode = 'reasoning';
    messages = rag._chatMessages([{ role: 'user', content: 'Answer this' }]);
    assert.match(messages[0].content, /\/think$/);
});

test('chat metadata changes evidence text without changing retrieval ranking', async () => {
    const rag = createRAG();
    const calls = [];
    rag.vectorSearch = {
        search(_embedding, k, options) {
            calls.push({ k, includeMetadata: options.includeMetadata, hasFilter: Boolean(options.filter) });
            return [
                { index: 0, score: 0.9, doc_id: 'amazon', text: 'Fresh bread', metadata: { text: 'Fresh bread', platform: 'Amazon' } },
                { index: 1, score: 0.8, doc_id: 'walmart', text: 'Stale bread', metadata: { text: 'Stale bread', platform: 'Walmart' } }
            ];
        }
    };
    const without = await rag.retrieveContext('bread', [1], { numResults: 2, includeMetadata: false });
    const selected = await rag.retrieveContext('bread', [1], { numResults: 2, includeMetadata: true, metadataFields: ['platform'] });

    assert.deepEqual(Array.from(without.sources, source => source.doc_id), ['amazon', 'walmart']);
    assert.deepEqual(Array.from(selected.sources, source => source.doc_id), ['amazon', 'walmart']);
    assert.deepEqual(calls, [
        { k: 8, includeMetadata: true, hasFilter: false },
        { k: 8, includeMetadata: true, hasFilter: false }
    ]);
    assert.doesNotMatch(without.context, /platform:/i);
    assert.match(selected.context, /platform: Amazon/i);
    assert.match(selected.context, /platform: Walmart/i);
});

test('reranker-disabled retrieval is equivalent and never initializes neural ranking', async () => {
    let rerankerCalls = 0;
    const vectorSearch = {
        search() {
            return [
                { index: 0, score: 0.9, doc_id: 'one', text: 'Alpha policy answer', metadata: { text: 'Alpha policy answer' } },
                { index: 1, score: 0.8, doc_id: 'two', text: 'Beta policy answer', metadata: { text: 'Beta policy answer' } }
            ];
        }
    };
    const rag = createRAG(vectorSearch, null, null, { async rerank() { rerankerCalls += 1; } });
    const implicitOff = await rag.retrieveContext('policy answer', [1], { numResults: 2 });
    const explicitOff = await rag.retrieveContext('policy answer', [1], { numResults: 2, rerankerEnabled: false });

    assert.deepEqual(Array.from(implicitOff.sources, source => source.doc_id), Array.from(explicitOff.sources, source => source.doc_id));
    assert.equal(rerankerCalls, 0);
    assert.equal(explicitOff.metadata.retrieval.reranker_applied, false);
    assert.equal(explicitOff.metadata.retrieval.reranker_fallback_reason, null);
});

test('document scope is enforced before reranking and cannot be reintroduced', async () => {
    let rerankerInput = [];
    const documents = [
        { index: 0, score: 0.95, doc_id: 'outside', text: 'Outside secret', metadata: { doc_id: 'outside', text: 'Outside secret' } },
        { index: 1, score: 0.85, doc_id: 'inside', text: 'Inside evidence', metadata: { doc_id: 'inside', text: 'Inside evidence' } }
    ];
    const vectorSearch = {
        search(_embedding, _k, options) {
            return documents.filter(document => !options.filter || options.filter(document.metadata));
        }
    };
    const reranker = {
        async rerank(_query, results) {
            rerankerInput = results.slice();
            return {
                results: results.slice().reverse(),
                diagnostics: { reranker_applied: true, reranker_model: 'test-model', reranker_candidates: results.length }
            };
        }
    };
    const rag = createRAG(vectorSearch, null, null, reranker);
    const retrieved = await rag.retrieveContext('evidence', [1], {
        numResults: 5,
        allowedDocIds: ['inside'],
        rerankerEnabled: true
    });

    assert.deepEqual(rerankerInput.map(result => result.doc_id), ['inside']);
    assert.deepEqual(Array.from(retrieved.sources, source => source.doc_id), ['inside']);
});

test('HyDE semantic embedding never replaces the resolved human query used by BM25 and reranking', async () => {
    const calls = { vectorEmbedding: null, bm25Query: null, rerankerQuery: null };
    const chunk = {
        chunk_id: 'chunk-one', parent_id: 'doc-one', doc_id: 'doc-one', index: 0,
        score: 0.9, text: 'The actual evidence', metadata: { parent_id: 'doc-one', doc_id: 'doc-one', text: 'The actual evidence' }
    };
    const vectorSearch = { search() { return []; } };
    const chunkVectorSearch = {
        isBuilt: true,
        search(embedding) {
            calls.vectorEmbedding = embedding;
            return [chunk];
        }
    };
    const bm25Search = {
        isBuilt: true,
        search(query) {
            calls.bm25Query = query;
            return [chunk];
        }
    };
    const reranker = {
        async rerank(query, results) {
            calls.rerankerQuery = query;
            return {
                results,
                diagnostics: { reranker_applied: true, reranker_model: 'test-model', reranker_candidates: results.length }
            };
        }
    };
    const rag = createRAG(vectorSearch, chunkVectorSearch, bm25Search, reranker);
    const hypothesisEmbedding = [9, 9, 9];
    await rag.retrieveContext('hypothetical semantic passage', hypothesisEmbedding, {
        keywordQuery: 'When did Project Fjord launch?',
        numResults: 1,
        similarityThreshold: 0,
        rerankerEnabled: true
    });

    assert.deepEqual(calls.vectorEmbedding, hypothesisEmbedding);
    assert.equal(calls.bm25Query, 'When did Project Fjord launch?');
    assert.equal(calls.rerankerQuery, 'When did Project Fjord launch?');
});

test('HyDE resets WebLLM before completion and reset failures remain non-fatal', async () => {
    const rag = createRAG();
    const calls = [];
    rag.ensureEngineReady = async () => {};
    rag.engine = {
        async resetChat() {
            calls.push('reset');
            throw new Error('reset unavailable');
        },
        chat: { completions: { async create() {
            calls.push('create');
            return (async function* () {
                yield { choices: [{ delta: { content: 'Hypothetical retrieval passage' } }] };
            })();
        } } }
    };
    rag.isInitialized = true;
    const result = await rag.generateHyDE('What happened?');
    assert.deepEqual(calls, ['reset', 'create']);
    assert.equal(result, 'Hypothetical retrieval passage');
});

test('local AI failures normalize thrown values into stable recovery classes', () => {
    assert.equal(normalizeLocalAIError(null).code, 'local_ai_generation_failed');
    assert.equal(normalizeLocalAIError('ModelNotLoadedError: model not loaded').code, 'model_not_loaded');
    assert.equal(normalizeLocalAIError({ reason: 'Worker message channel terminated' }).code, 'worker_unavailable');
    assert.equal(normalizeLocalAIError(new Error('WebLLM returned an empty completion.')).code, 'empty_completion');
    assert.equal(normalizeLocalAIError(new Error('GPUDevice was lost')).code, 'gpu_device_lost');
});

test('deprecated runtimeStatsText liveness probing is absent', () => {
    assert.doesNotMatch(rawSource, /runtimeStatsText\s*\(/);
});
