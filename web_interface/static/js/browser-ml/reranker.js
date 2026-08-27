/**
 * Lazy browser-local multilingual cross-encoder reranking.
 *
 * The model is intentionally fixed for this release so the product exposes a
 * single understandable quality switch instead of another model marketplace.
 * Documents never leave the browser: the worker downloads model assets once,
 * then scores query/passage pairs locally with ONNX Runtime WASM.
 */

export const RERANKER_MODEL = Object.freeze({
    id: 'SugoLabs/mmarco-mMiniLMv2-L12-H384-v1',
    canonicalId: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    revision: '6772eee1ea62dc82e1fcaaf3ed90c269fdcb1fcf',
    dtype: 'q8',
    device: 'wasm',
    batchSize: 8,
    maxLength: 512,
    maxCandidates: 30,
    stageOneWeight: 0.3,
    neuralWeight: 0.7,
    rrfK: 60,
    estimatedDownloadBytes: 140 * 1024 * 1024,
    license: 'Apache-2.0'
});

function passageText(result) {
    if (Array.isArray(result?.chunks) && result.chunks.length) {
        return result.chunks
            .map(chunk => chunk?.text || chunk?.metadata?.text || '')
            .filter(Boolean)
            .join('\n');
    }
    return String(result?.text || result?.content || result?.metadata?.text || '');
}

function stableNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

export function createRerankerBatches(passages, batchSize = RERANKER_MODEL.batchSize) {
    const source = Array.isArray(passages) ? passages : [];
    const size = Math.max(1, Number(batchSize) || RERANKER_MODEL.batchSize);
    const batches = [];
    for (let offset = 0; offset < source.length; offset += size) {
        batches.push(source.slice(offset, offset + size));
    }
    return batches;
}

export function createPairTokenizationInput(query, passages, maxLength = RERANKER_MODEL.maxLength) {
    const pairs = Array.isArray(passages) ? passages : [];
    return {
        queries: new Array(pairs.length).fill(String(query || '')),
        options: {
            text_pair: pairs,
            padding: true,
            truncation: true,
            max_length: Math.max(1, Number(maxLength) || RERANKER_MODEL.maxLength)
        }
    };
}

/**
 * Fuse the original stage-one ordering with cross-encoder ranks. Rank fusion
 * is deliberately used instead of raw-score blending because vector, BM25,
 * and classifier scores have unrelated scales.
 */
export function fuseRerankerRanks(results, scores, {
    stageOneWeight = RERANKER_MODEL.stageOneWeight,
    neuralWeight = RERANKER_MODEL.neuralWeight,
    rrfK = RERANKER_MODEL.rrfK,
    model = RERANKER_MODEL.canonicalId,
    latencyMs = 0
} = {}) {
    const source = Array.isArray(results) ? results.filter(Boolean) : [];
    const headCount = Math.min(source.length, Array.isArray(scores) ? scores.length : 0);
    if (!headCount) return source.slice();

    const neuralOrder = Array.from({ length: headCount }, (_, index) => index)
        .sort((left, right) => stableNumber(scores[right], -Infinity) - stableNumber(scores[left], -Infinity) || left - right);
    const neuralRank = new Map(neuralOrder.map((sourceIndex, rank) => [sourceIndex, rank]));
    const rerankedHead = source.slice(0, headCount).map((result, sourceIndex) => {
        const rerankerRank = neuralRank.get(sourceIndex) ?? sourceIndex;
        const fusedScore = (stageOneWeight / (rrfK + sourceIndex + 1))
            + (neuralWeight / (rrfK + rerankerRank + 1));
        return {
            result: {
                ...result,
                pre_rerank_rank: sourceIndex + 1,
                reranker_rank: rerankerRank + 1,
                reranker_score: stableNumber(scores[sourceIndex]),
                reranker_applied: true,
                reranker_model: model,
                reranker_latency_ms: latencyMs
            },
            sourceIndex,
            fusedScore
        };
    }).sort((left, right) => right.fusedScore - left.fusedScore || left.sourceIndex - right.sourceIndex);

    return [
        ...rerankedHead.map(entry => entry.result),
        ...source.slice(headCount)
    ];
}

function fallbackResult(results, reason, startedAt) {
    return {
        results: Array.isArray(results) ? results.slice() : [],
        diagnostics: {
            reranker_applied: false,
            reranker_model: RERANKER_MODEL.canonicalId,
            reranker_candidates: 0,
            reranker_latency_ms: Math.max(0, Date.now() - startedAt),
            reranker_fallback_reason: reason || 'unavailable'
        }
    };
}

export class BrowserReranker {
    constructor({
        workerFactory = null,
        idleTimeoutMs = 5 * 60 * 1000,
        requestTimeoutMs = 180000
    } = {}) {
        this.workerFactory = workerFactory;
        this.idleTimeoutMs = idleTimeoutMs;
        this.requestTimeoutMs = requestTimeoutMs;
        this.worker = null;
        this.pending = new Map();
        this.sequence = 0;
        this.idleTimer = null;
        this.lastDiagnostics = null;
    }

    _createWorker() {
        if (this.workerFactory) return this.workerFactory();
        if (typeof Worker === 'undefined') return null;
        return new Worker(new URL('./reranker-worker.js', import.meta.url), { type: 'module' });
    }

    _ensureWorker() {
        if (this.worker) return this.worker;
        const worker = this._createWorker();
        if (!worker) throw new Error('reranker_worker_unavailable');
        worker.addEventListener('message', event => this._handleMessage(event.data || {}));
        worker.addEventListener('error', event => {
            const message = event?.message || 'reranker_worker_error';
            this._rejectPending(new Error(message));
            this.suspend('worker-error');
        });
        this.worker = worker;
        return worker;
    }

    _handleMessage(message) {
        if (message.type === 'progress') {
            const pending = this.pending.get(message.requestId);
            pending?.onProgress?.(message.progress || {});
            return;
        }
        const pending = this.pending.get(message.requestId);
        if (!pending) return;
        clearTimeout(pending.timeoutId);
        this.pending.delete(message.requestId);
        if (message.type === 'error') {
            const error = new Error(message.error || 'reranker_failed');
            error.code = message.code || 'reranker_failed';
            pending.reject(error);
        } else {
            pending.resolve(Array.isArray(message.scores) ? message.scores : []);
        }
    }

    _rejectPending(error) {
        for (const pending of this.pending.values()) {
            clearTimeout(pending.timeoutId);
            pending.reject(error);
        }
        this.pending.clear();
    }

    _scheduleIdleSuspend() {
        clearTimeout(this.idleTimer);
        this.idleTimer = setTimeout(() => this.suspend('idle'), this.idleTimeoutMs);
    }

    _request(query, passages, { signal = null, onProgress = null } = {}) {
        const worker = this._ensureWorker();
        const requestId = ++this.sequence;
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                this.pending.delete(requestId);
                const error = new Error('reranker_timeout');
                error.code = 'reranker_timeout';
                reject(error);
                this.suspend('timeout');
            }, this.requestTimeoutMs);
            const abort = () => {
                clearTimeout(timeoutId);
                this.pending.delete(requestId);
                const error = new Error('reranker_aborted');
                error.name = 'AbortError';
                reject(error);
                this.suspend('abort');
            };
            if (signal?.aborted) return abort();
            signal?.addEventListener('abort', abort, { once: true });
            this.pending.set(requestId, {
                resolve: value => {
                    signal?.removeEventListener('abort', abort);
                    resolve(value);
                },
                reject: error => {
                    signal?.removeEventListener('abort', abort);
                    reject(error);
                },
                timeoutId,
                onProgress
            });
            worker.postMessage({
                type: 'rerank',
                requestId,
                query,
                passages,
                model: RERANKER_MODEL
            });
        });
    }

    async rerank(query, results, options = {}) {
        const startedAt = Date.now();
        const source = Array.isArray(results) ? results.filter(Boolean) : [];
        const maxCandidates = Math.max(1, Number(options.maxCandidates) || RERANKER_MODEL.maxCandidates);
        const candidates = source.slice(0, maxCandidates);
        const passages = candidates.map(passageText);
        if (!String(query || '').trim()) return fallbackResult(source, 'empty_query', startedAt);
        if (!passages.length || passages.every(text => !text.trim())) return fallbackResult(source, 'no_passages', startedAt);

        try {
            const scores = await this._request(String(query), passages, options);
            if (scores.length !== passages.length) throw new Error('reranker_score_count_mismatch');
            const latencyMs = Math.max(0, Date.now() - startedAt);
            const reranked = fuseRerankerRanks(source, scores, { latencyMs });
            const diagnostics = {
                reranker_applied: true,
                reranker_model: RERANKER_MODEL.canonicalId,
                reranker_revision: RERANKER_MODEL.revision,
                reranker_candidates: candidates.length,
                reranker_latency_ms: latencyMs,
                reranker_fallback_reason: null
            };
            this.lastDiagnostics = diagnostics;
            this._scheduleIdleSuspend();
            return { results: reranked, diagnostics };
        } catch (error) {
            const reason = error?.name === 'AbortError' ? 'aborted' : (error?.code || error?.message || 'failed');
            console.warn('Multilingual reranker unavailable; using standard ranking.', reason);
            const fallback = fallbackResult(source, reason, startedAt);
            this.lastDiagnostics = fallback.diagnostics;
            return fallback;
        }
    }

    suspend(_reason = 'manual') {
        clearTimeout(this.idleTimer);
        this.idleTimer = null;
        if (!this.worker) return false;
        this._rejectPending(new Error('reranker_suspended'));
        this.worker.terminate();
        this.worker = null;
        return true;
    }

    releaseForGeneration() {
        return this.suspend('local-generation');
    }
}
