/**
 * Browser-based RAG (Retrieval-Augmented Generation) using WebLLM
 * Runs the configured language model locally in the browser via WebGPU.
 */

import { CreateWebWorkerMLCEngine, hasModelInCache, prebuiltAppConfig } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.84/+esm";
import { getModelConstraints } from "../model-constraints.js";
// Defaults are owned by config-manager.js. Never re-declare them here: this
// file previously kept its own copies, which silently drifted out of sync.
import { DEFAULT_CONFIG } from "../config-manager.js";
import { sanitizeCitationBounds } from "./chat-context.js";
import { rerankAndDiversify } from "./retrieval-ranking.js";

/**
 * WebLLM reports a weight shard that could not be fetched as
 * "Failed to store <url> with error: Error: Network response was not ok".
 * Note the capital N — a case-sensitive /network/ test misses it.
 */
const SHARD_DOWNLOAD_FAILURE = /failed to store .*(network response was not ok|failed to fetch|networkerror)/i;
const MAX_SHARD_RETRIES = 3;
const WEBLLM_APP_CONFIG = { ...prebuiltAppConfig, cacheBackend: 'indexeddb' };

export async function isWebLLMModelCached(modelId) {
    if (!modelId) return false;
    try {
        return await hasModelInCache(modelId, WEBLLM_APP_CONFIG);
    } catch (error) {
        console.warn('Unable to inspect the WebLLM model cache:', error);
        return false;
    }
}

function normalizeWebLLMMessages(messages) {
    const systemParts = [];
    const conversation = [];
    for (const message of Array.isArray(messages) ? messages : []) {
        const role = String(message?.role || '').toLowerCase();
        const content = String(message?.content || '').trim();
        if (!content) continue;
        if (role === 'system') systemParts.push(content);
        else if (role === 'user' || role === 'assistant') conversation.push({ role, content });
    }
    return systemParts.length
        ? [{ role: 'system', content: systemParts.join('\n\n') }, ...conversation]
        : conversation;
}

// Stall-based timeout: rejects only if no progress for `ms` milliseconds.
// More forgiving than a wall-clock timeout for multi-GB downloads on slow links.
function makeStallTimeout(ms, label) {
    let timeoutId;
    let rejectFn;
    const promise = new Promise((_, reject) => {
        rejectFn = reject;
        timeoutId = setTimeout(() => {
            const err = new Error(`${label} stalled — no progress for ${Math.round(ms / 1000)}s`);
            err.code = 'download_timeout';
            reject(err);
        }, ms);
    });
    return {
        promise,
        reset() {
            if (timeoutId) clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                const err = new Error(`${label} stalled — no progress for ${Math.round(ms / 1000)}s`);
                err.code = 'download_timeout';
                rejectFn(err);
            }, ms);
        },
        clear() {
            if (timeoutId) clearTimeout(timeoutId);
            timeoutId = null;
        }
    };
}

export class BrowserRAG {
    constructor(vectorSearch, chunkVectorSearch = null, bm25Search = null, reranker = null) {
        this.vectorSearch = vectorSearch;        // Parent document index (not used for retrieval)
        this.chunkVectorSearch = chunkVectorSearch; // Chunk index for RAG retrieval
        this.bm25Search = bm25Search;            // BM25 keyword search for hybrid retrieval
        this.reranker = reranker;                // Optional lazy multilingual cross-encoder
        this.engine = null;
        this.worker = null;

        // Abort control for stopping generation
        this.currentGenerationReject = null;
        this.needsReinit = false; // Flag to track if engine needs reinitialization after abort
        this.workerUnloaded = false; // Set true by unloadWorker() to permanently block re-init

        // Soft suspension state (distinct from unloadWorker which is permanent).
        // suspended = worker terminated to free RAM, but allowed to lazily re-init on next use.
        this.isSuspended = false;
        this._idleTimer = null;
        this._idleTimeoutMs = 5 * 60 * 1000; // 5 min idle → auto-suspend

        // Every local generation surface (Chat, HyDE, cluster labels, one-shot
        // RAG and MCP local generation) shares one WebLLM engine. Keep a single
        // explicit owner so concurrent features cannot issue overlapping engine
        // requests or accidentally stop/unload one another.
        this._activeOperation = null;
        this._operationSequence = 0;

        // Load saved configuration
        const savedConfig = this.loadSavedConfig();

        // Every fallback below comes from DEFAULT_CONFIG so this file can never
        // drift from the shipped defaults. Change values in config-manager.js.
        const llmDefaults = DEFAULT_CONFIG.llm;
        const searchDefaults = DEFAULT_CONFIG.search;
        const promptDefaults = DEFAULT_CONFIG.rag_prompts;
        const hydeDefaults = DEFAULT_CONFIG.hyde;

        // Load model ID from saved config or use default
        this.modelId = savedConfig.model_id || llmDefaults.model_id;

        // Get model constraints
        this.modelConstraints = getModelConstraints(this.modelId);

        this.isInitialized = false;

        // Recovery counters for initialize(). Declared here so the first load
        // compares against real numbers rather than undefined.
        this._idbRetried = false;
        this._shardRetries = 0;

        // Load LLM generation parameters from saved config
        this.temperature = savedConfig.temperature ?? llmDefaults.temperature;
        this.maxTokens = savedConfig.max_tokens ?? llmDefaults.max_tokens;
        this.topP = savedConfig.top_p ?? llmDefaults.top_p;
        this.repeatPenalty = savedConfig.repeat_penalty ?? llmDefaults.repeat_penalty;
        this.reasoningMode = savedConfig.reasoning_mode ?? llmDefaults.reasoning_mode;
        // Never fall back to the model's own ceiling: long-context models report
        // 128K-256K there, and allocating a KV cache that size would exhaust VRAM.
        this.maxContextLength = Math.min(
            savedConfig.context_window_size || llmDefaults.context_window_size,
            this.modelConstraints.contextWindow || llmDefaults.context_window_size
        );

        // Load RAG parameters
        this.numResults = savedConfig.num_results ?? searchDefaults.num_results;
        this.similarityThreshold = savedConfig.similarity_threshold ?? searchDefaults.similarity_threshold;
        this.retrievalK = savedConfig.retrieval_k ?? searchDefaults.retrieval_k;
        this.vectorWeight = savedConfig.vector_weight ?? searchDefaults.vector_weight;
        this.maxChunksPerParent = savedConfig.max_chunks_per_parent ?? searchDefaults.max_chunks_per_parent;

        // Conversation history for export
        this.conversationHistory = [];

        // Load RAG prompts
        this.systemPrompt = savedConfig.system_prompt || promptDefaults.system_prompt;
        this.userTemplate = savedConfig.user_template || promptDefaults.user_template;

        // Load HyDE prompts and settings
        this.hydePrompt = savedConfig.hyde_prompt || hydeDefaults.prompt;
        this.hydeTemperature = savedConfig.hyde_temperature ?? hydeDefaults.temperature;
        this.hydeMaxTokens = savedConfig.hyde_max_tokens ?? hydeDefaults.max_tokens;
    }

    /**
     * Get model constraints for UI validation
     * @returns {Object} Model constraints
     */
    getModelConstraints() {
        return this.modelConstraints;
    }

    /** Append Qwen3's documented mode switch to the final user turn. */
    _applyReasoningSwitch(messages) {
        if (this.modelConstraints?.responseMode !== 'switchable') return messages;
        const configuredMode = window.ConfigManager?.getConfig()?.llm?.reasoning_mode
            || this.reasoningMode
            || 'direct';
        const control = configuredMode === 'reasoning'
            ? this.modelConstraints?.thinkSwitch
            : this.modelConstraints?.noThinkSwitch;
        if (!control) return messages;
        for (let index = messages.length - 1; index >= 0; index--) {
            const message = messages[index];
            if (message.role !== 'user') continue;
            if (String(message.content).includes(control)) return messages;
            const next = messages.slice();
            next[index] = { ...message, content: `${message.content}\n\n${control}` };
            return next;
        }
        return messages;
    }

    /** Single choke point for everything handed to WebLLM's chat completions. */
    _chatMessages(messages) {
        return this._applyReasoningSwitch(normalizeWebLLMMessages(messages));
    }

    /** Role-only prompt diagnostics. Never include user or document content. */
    _promptCompatibilityDiagnostic(messages) {
        const roles = Array.from(messages || [], message => message.role);
        const gemmaWithoutSystemRole = /^(?:gemma-2-|gemma3-)/i.test(this.modelId || '');
        const foldedFirstUserRoles = gemmaWithoutSystemRole && roles[0] === 'system'
            ? (roles.length > 1 ? roles.slice(1) : ['user'])
            : null;
        return {
            modelFamily: gemmaWithoutSystemRole ? 'gemma-legacy-chat-template' : 'native-or-mlc-system-template',
            roles,
            messageCount: roles.length,
            foldedFirstUserRoles
        };
    }

    get activeOperation() {
        if (!this._activeOperation) return null;
        const { id, owner, datasetId, startedAt, phase } = this._activeOperation;
        return { id, owner, datasetId, startedAt, phase };
    }

    get isGenerating() {
        return this._activeOperation !== null;
    }

    get shouldAbort() {
        return this._activeOperation?.cancelled === true;
    }

    _operationLabel(owner) {
        return ({
            chat: 'answering an Ask question',
            hyde: 'creating a HyDE retrieval hypothesis',
            'cluster-label': 'labelling clusters',
            'suggested-questions': 'drafting suggested questions',
            rag: 'answering a RAG question',
            mcp: 'answering a connected-client request'
        })[owner] || 'running another local AI task';
    }

    _emitOperationState(operation, phase, detail = {}) {
        if (operation && this._activeOperation?.id === operation.id) {
            this._activeOperation.phase = phase;
        }
        if (typeof document !== 'undefined') {
            document.dispatchEvent(new CustomEvent('vectoria:local-ai-operation', {
                detail: {
                    active: phase !== 'idle',
                    operation: operation ? this.activeOperation : null,
                    phase,
                    ...detail
                }
            }));
        }
    }

    beginOperation(owner = 'local-ai', { datasetId = null } = {}) {
        if (this._activeOperation) {
            const error = new Error(`Local AI is currently ${this._operationLabel(this._activeOperation.owner)}. Stop or finish that task before starting another.`);
            error.code = 'local_ai_busy';
            error.activeOwner = this._activeOperation.owner;
            throw error;
        }
        const operation = {
            id: ++this._operationSequence,
            owner,
            datasetId: datasetId === undefined || datasetId === null ? null : String(datasetId),
            startedAt: Date.now(),
            phase: 'starting',
            cancelled: false,
            cancelReason: null,
            abortController: typeof AbortController === 'function' ? new AbortController() : null
        };
        this._activeOperation = operation;
        this._clearIdleTimer();
        this._emitOperationState(operation, 'starting');
        return operation;
    }

    assertOperation(operation) {
        if (!operation || this._activeOperation?.id !== operation.id) {
            const error = new Error('The local AI task is no longer active.');
            error.code = 'local_ai_operation_stale';
            throw error;
        }
        return this._activeOperation;
    }

    throwIfOperationCancelled(operation) {
        const active = this.assertOperation(operation);
        if (!active.cancelled) return;
        const error = new Error(active.cancelReason || 'Generation aborted');
        error.name = 'AbortError';
        error.code = 'local_ai_aborted';
        throw error;
    }

    setOperationPhase(operation, phase, detail = {}) {
        this.assertOperation(operation);
        this._emitOperationState(operation, phase, detail);
        if (phase === 'awaiting-input') this._scheduleIdleSuspend();
    }

    endOperation(operation, { state = 'idle' } = {}) {
        if (!operation || this._activeOperation?.id !== operation.id) return false;
        const finished = this._activeOperation;
        this._activeOperation = null;
        this.currentGenerationReject = null;
        this._scheduleIdleSuspend();
        this._emitOperationState(finished, state);
        return true;
    }

    async withOperation(owner, options, task) {
        const supplied = options?.operationToken || null;
        const operation = supplied || this.beginOperation(owner, { datasetId: options?.datasetId });
        const ownsOperation = !supplied;
        this.assertOperation(operation);
        try {
            // A multi-stage owner may be stopped while it is embedding or
            // retrieving. Do not erase that stop or warm WebLLM afterward.
            if (supplied) this.throwIfOperationCancelled(operation);
            return await task(operation);
        } finally {
            if (ownsOperation) this.endOperation(operation);
        }
    }

    /**
     * Abort the current RAG generation
     * Sets a flag that the streaming loop checks to stop gracefully
     */
    abort(ownerOrOperation = null) {
        if (!this._activeOperation) return false;
        if (ownerOrOperation) {
            const requestedId = typeof ownerOrOperation === 'object' ? ownerOrOperation.id : null;
            const requestedOwner = typeof ownerOrOperation === 'string' ? ownerOrOperation : null;
            if ((requestedId && requestedId !== this._activeOperation.id)
                || (requestedOwner && requestedOwner !== this._activeOperation.owner)) {
                return false;
            }
        }
        this._activeOperation.cancelled = true;
        this._activeOperation.cancelReason = 'Generation stopped by user';
        this._activeOperation.abortController?.abort?.();
        this._clearIdleTimer();

        // Model restoration/download has no engine yet to interrupt. Terminate
        // its worker so Stop does not wait for a multi-gigabyte cold load to
        // finish; cached chunks remain reusable on the next attempt.
        if (this._activeOperation?.phase === 'loading-model' && this.worker && !this.engine) {
            try { this.worker.terminate(); } catch (_) {}
            this.worker = null;
            this.engine = null;
            this.isInitialized = false;
            this.needsReinit = false;
            this.isSuspended = true;
        }

        // Try to interrupt the WebLLM engine directly
        if (this.engine && typeof this.engine.interruptGenerate === 'function') {
            this.engine.interruptGenerate();
        }

        if (this.currentGenerationReject) {
            this.currentGenerationReject(new Error('Generation stopped by user'));
        }
        if (this._activeOperation) this._emitOperationState(this._activeOperation, 'stopping');
        return true;
    }

    async resetConversationState() {
        if (this._activeOperation || !this.engine || !this.isInitialized) return false;
        try {
            await this.engine.resetChat?.();
            return true;
        } catch (error) {
            console.warn('Unable to reset WebLLM conversation cache:', error);
            return false;
        }
    }

    async _resetEngineChatBestEffort(purpose) {
        if (!this.engine || typeof this.engine.resetChat !== 'function') return false;
        try {
            await this.engine.resetChat();
            return true;
        } catch (error) {
            console.warn(`Unable to reset WebLLM state before ${purpose}:`, {
                name: error?.name || null,
                message: error?.message || String(error || 'Unknown reset error'),
                constructor: error?.constructor?.name || typeof error
            });
            return false;
        }
    }

    unloadWorker() {
        if (this._activeOperation) {
            console.warn(`LLM unload skipped while ${this._activeOperation.owner} owns local generation`);
            return false;
        }
        this._clearIdleTimer();
        if (this.worker) {
            try { this.worker.terminate(); } catch (_) {}
            this.worker = null;
        }
        this.engine = null;
        this.isInitialized = false;
        this.needsReinit = false;
        this.isSuspended = false;
        this.workerUnloaded = true; // Block any future re-initialization
        console.log('🤖 LLM worker unloaded — re-init blocked until page reload');
        return true;
    }

    /**
     * Soft-suspend the LLM worker to free GPU/RAM during heavy stages.
     * Unlike unloadWorker(), this allows lazy re-initialization on next query.
     * Safe to call when worker is not loaded (no-op).
     */
    suspendWorker(reason = 'idle') {
        this._clearIdleTimer();
        if (this._activeOperation && this._activeOperation.phase !== 'awaiting-input') {
            console.warn(`LLM suspension skipped while ${this._activeOperation.owner} owns local generation`);
            return false;
        }
        if (!this.worker && !this.isInitialized) {
            return true;
        }
        if (this.workerUnloaded) {
            return true; // Permanent unload already in effect
        }
        if (this.worker) {
            try { this.worker.terminate(); } catch (_) {}
            this.worker = null;
        }
        this.engine = null;
        this.isInitialized = false;
        this.needsReinit = false;
        this.isSuspended = true;
        console.log(`🤖 LLM worker suspended (${reason}) — will lazy-reload on next query`);
        return true;
    }

    /**
     * Start (or restart) the idle auto-suspend timer.
     * Called after each successful generation to suspend if user goes idle.
     */
    _scheduleIdleSuspend() {
        this._clearIdleTimer();
        if (this.workerUnloaded) return;
        this._idleTimer = setTimeout(() => {
            this._idleTimer = null;
            if (this.isInitialized && !this.workerUnloaded) {
                this.suspendWorker('idle-timeout');
            }
        }, this._idleTimeoutMs);
    }

    _clearIdleTimer() {
        if (this._idleTimer) {
            clearTimeout(this._idleTimer);
            this._idleTimer = null;
        }
    }

    get workerLoaded() {
        return this.worker !== null && this.isInitialized;
    }

    async _initializeWithAbort(onProgress = null, operation = null) {
        let rejectAbort;
        const abortPromise = new Promise((_, reject) => { rejectAbort = reject; });
        this.currentGenerationReject = rejectAbort;
        try {
            if (operation) this.throwIfOperationCancelled(operation);
            return await Promise.race([this.initialize(onProgress), abortPromise]);
        } finally {
            if (this.currentGenerationReject === rejectAbort) this.currentGenerationReject = null;
        }
    }

    /**
     * Reinitialize the engine (needed after abort corrupts the engine state)
     */
    async reinitializeEngine(onProgress = null, operation = null) {
        if (this.workerUnloaded) {
            console.log('🤖 LLM reinitializeEngine() skipped — worker intentionally unloaded');
            return;
        }
        // Terminate the old worker
        if (this.worker) {
            try {
                this.worker.terminate();
            } catch (_) { /* noop */ }
            this.worker = null;
        }
        this.engine = null;
        this.isInitialized = false;
        this.needsReinit = false;

        // Reinitialize
        await this._initializeWithAbort(onProgress, operation);
    }

    /**
     * Check if engine needs reinitialization and do it if needed
     */
    async ensureEngineReady(onStatus = null, operation = null) {
        this._clearIdleTimer();
        // WASM reranking is intentionally short-lived around local generation.
        // Terminating its worker releases linear memory before WebLLM allocates
        // or expands the local model runtime.
        this.reranker?.releaseForGeneration?.();
        if (this.workerUnloaded) {
            throw new Error('Local LLM is unloaded. Use query_rag_external (Claude as RAG) instead, or reload the page to re-enable the local LLM.');
        }
        // Cached installations and soft-suspended sessions both start cold.
        // Transparently create the runtime only once a processed dataset query
        // actually needs generation; model files remain cached in IndexedDB.
        if (!this.isInitialized) {
            onStatus?.('loading-model', { cached: true });
            if (operation) this._emitOperationState(operation, 'loading-model');
            console.log(this.isSuspended
                ? '🤖 LLM was suspended — resuming for query...'
                : '🤖 LLM is cached — loading for first query...');
            this.isSuspended = false;
            await this._initializeWithAbort(progress => {
                onStatus?.('model-progress', progress);
                if (operation) this._emitOperationState(operation, 'loading-model', { progress });
            }, operation);
            onStatus?.('model-ready', { cached: true });
        }
        if (this.needsReinit) {
            onStatus?.('loading-model', { recovering: true });
            await this.reinitializeEngine(progress => onStatus?.('model-progress', progress), operation);
        }
        if (!this.isInitialized) throw new Error('LLM could not be initialized.');
        if (operation) this.throwIfOperationCancelled(operation);
    }

    async recoverEngine(operation, onStatus = null) {
        this.throwIfOperationCancelled(operation);
        onStatus?.('loading-model', { recovering: true });
        this.needsReinit = true;
        this._idbRetried = false;
        this._shardRetries = 0;
        await this.reinitializeEngine(progress => onStatus?.('model-progress', progress), operation);
        this.throwIfOperationCancelled(operation);
        return this.isInitialized;
    }

    /**
     * Load saved configuration from localStorage via ConfigManager
     */
    loadSavedConfig() {
        try {
            // Use ConfigManager if available (centralized config system)
            const config = window.ConfigManager ? window.ConfigManager.getConfig() : null;

            if (config) {
                return {
                    // LLM settings
                    model_id: config.llm?.model_id,
                    temperature: config.llm?.temperature,
                    max_tokens: config.llm?.max_tokens,
                    top_p: config.llm?.top_p,
                    repeat_penalty: config.llm?.repeat_penalty,
                    reasoning_mode: config.llm?.reasoning_mode,
                    context_window_size: config.llm?.context_window_size,
                    // RAG settings
                    num_results: config.search?.num_results,
                    similarity_threshold: config.search?.similarity_threshold,
                    retrieval_k: config.search?.retrieval_k,
                    vector_weight: config.search?.vector_weight,
                    reranker_enabled: config.search?.reranker_enabled,
                    max_chunks_per_parent: config.search?.max_chunks_per_parent,
                    // RAG Prompts
                    system_prompt: config.rag_prompts?.system_prompt,
                    user_template: config.rag_prompts?.user_template,
                    // HyDE settings
                    hyde_prompt: config.hyde?.prompt,
                    hyde_temperature: config.hyde?.temperature,
                    hyde_max_tokens: config.hyde?.max_tokens
                };
            }
        } catch (error) {
            console.warn('Failed to load saved config:', error);
        }
        return {};
    }

    /**
     * Initialize the LLM engine
     * @param {Function} onProgress - Progress callback
     */
    async initialize(onProgress = null) {
        if (this.workerUnloaded || window.__vectoriaLLMUnloaded) {
            this.workerUnloaded = true;
            console.log('🤖 LLM initialize() skipped — worker intentionally unloaded');
            return;
        }
        if (this.isInitialized) {
            return;
        }

        const originalConsoleLog = console.log;
        const originalConsoleInfo = console.info;
        const originalConsoleWarn = console.warn;

        try {
            // Suppress verbose WebLLM logging
            console.log = () => {}; // Suppress all logs during init
            console.info = () => {};
            console.warn = () => {};

            if (this.worker) {
                try {
                    this.worker.terminate();
                } catch (_) { /* noop */ }
                this.worker = null;
            }

            this.worker = new Worker(new URL('./llm-worker.js', import.meta.url), {
                type: 'module'
            });

            const stall = makeStallTimeout(120000, 'LLM model download');
            const enginePromise = CreateWebWorkerMLCEngine(
                this.worker,
                this.modelId,
                {
                    initProgressCallback: (progress) => {
                        stall.reset();
                        // Only send to UI callback for modal display
                        if (onProgress) {
                            const text = String(progress.text || '');
                            const phase = /^Fetching param cache/i.test(text)
                                ? 'download'
                                : /^Loading model from cache/i.test(text)
                                    ? 'loading-cache'
                                    : 'preparing';
                            onProgress({
                                status: 'loading',
                                text,
                                progress: progress.progress ?? 0,
                                phase
                            });
                        }
                    },
                    appConfig: WEBLLM_APP_CONFIG
                },
                // chatOpts (third arg) — overrides for mlc-chat-config.json.
                // Disable sliding window so models like Gemma 3 (which default to SWA=512)
                // don't fail WebLLM's "only one of context_window_size / sliding_window_size
                // can be positive" check.
                {
                    context_window_size: this.maxContextLength,
                    sliding_window_size: -1,
                    attention_sink_size: 0
                }
            );
            try {
                this.engine = await Promise.race([enginePromise, stall.promise]);
            } finally {
                stall.clear();
            }

            // If unloadWorker() was called while we were awaiting, honor it
            if (this.workerUnloaded) {
                if (this.worker) { try { this.worker.terminate(); } catch (_) {} this.worker = null; }
                this.engine = null;
                console.log = originalConsoleLog;
                console.info = originalConsoleInfo;
                console.warn = originalConsoleWarn;
                console.log('🤖 LLM init completed but discarded — worker was unloaded during init');
                return;
            }

            this.isInitialized = true;
            this.isSuspended = false;
            console.log = originalConsoleLog;
            console.info = originalConsoleInfo;
            console.warn = originalConsoleWarn;

            if (onProgress) {
                onProgress({ status: 'ready', progress: 1, text: 'Model ready!' });
            }
        } catch (error) {
            console.log = originalConsoleLog;
            console.info = originalConsoleInfo;
            console.warn = originalConsoleWarn;
            if (this.worker) {
                try { this.worker.terminate(); } catch (_) {}
                this.worker = null;
            }

            // ArtifactIndexedDBCache error = stale/corrupt IndexedDB cache for this model
            // Retry once with cache flushed for this model
            const rawMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : JSON.stringify(error));
            if (rawMsg.includes('ArtifactIndexedDBCache') && !this._idbRetried) {
                this._idbRetried = true;
                originalConsoleWarn('⚠️ Stale IndexedDB cache detected, clearing and retrying...');
                try {
                    // Clear all caches named after the model (WebLLM uses cache names matching model id)
                    const cacheKeys = await caches.keys();
                    for (const key of cacheKeys) {
                        if (key.includes(this.modelId) || key.includes('webllm') || key.includes('mlc')) {
                            await caches.delete(key);
                        }
                    }
                    // Delete IndexedDB stores WebLLM uses — wrap in Promise so we can await
                    const dbNames = ['webllm-cache', `webllm/${this.modelId}`, this.modelId];
                    await Promise.all(dbNames.map(dbName => new Promise(resolve => {
                        try {
                            const req = indexedDB.deleteDatabase(dbName);
                            req.onsuccess = resolve;
                            req.onerror = resolve;
                            req.onblocked = resolve;
                        } catch (_) { resolve(); }
                    })));
                } catch (_) {}
                // Tail-call the retry — finally block will restore console after this returns
                return this.initialize(onProgress);
            }

            // A single weight shard failing mid-download is usually transient
            // (CDN hiccup, flaky connection). Shards already stored stay
            // cached, so retrying resumes rather than restarting the download.
            if (SHARD_DOWNLOAD_FAILURE.test(rawMsg) && this._shardRetries < MAX_SHARD_RETRIES) {
                this._shardRetries += 1;
                originalConsoleWarn(
                    `⚠️ Model shard download failed (attempt ${this._shardRetries}/${MAX_SHARD_RETRIES}), retrying…`
                );
                onProgress?.({
                    status: 'loading',
                    text: `Download interrupted — retrying (${this._shardRetries}/${MAX_SHARD_RETRIES})…`,
                    progress: 0
                });
                await new Promise(resolve => setTimeout(resolve, 1000 * this._shardRetries));
                return this.initialize(onProgress);
            }

            console.error('❌ Failed to initialize LLM:', error);
            let msg = rawMsg;
            if (rawMsg.includes('ArtifactIndexedDBCache')) {
                msg = `Model cache is corrupted. Please go to Advanced Settings → Language Model → "Clear Model Cache" and try again.`;
            } else if (SHARD_DOWNLOAD_FAILURE.test(rawMsg) || /network|fetch|failed to store/i.test(rawMsg)) {
                msg = `Model download failed (network error). This usually means:\n` +
                    `• The model files couldn't be fetched from HuggingFace\n` +
                    `• A firewall or VPN is blocking the download\n` +
                    `• HuggingFace is temporarily unavailable\n\n` +
                    `Your progress is cached, so retrying continues where it stopped. ` +
                    `Try again, or switch to a smaller model in Settings → Models.`;
            }
            throw new Error(`LLM initialization failed: ${msg}`);
        } finally {
            console.log = originalConsoleLog;
            console.info = originalConsoleInfo;
            console.warn = originalConsoleWarn;
        }
    }

    /**
     * Generate HyDE (Hypothetical Document Embeddings) text
     * Creates a hypothetical answer that can be used for better semantic search
     *
     * @param {string} question - User question
     * @returns {Promise<string>} Generated hypothetical answer
     */
    async generateHyDE(question, options = {}) {
        return this.withOperation(options.owner || 'hyde', options, async operation => {
            await this.ensureEngineReady(options.onStatus, operation);
            await this._resetEngineChatBestEffort('HyDE generation');

            const freshConfig = this.loadSavedConfig();
            const hydePrompt = options.prompt || freshConfig.hyde_prompt || this.hydePrompt;
            const hydeTemperature = options.temperature ?? freshConfig.hyde_temperature ?? this.hydeTemperature;
            let maxTokens = options.maxTokens ?? freshConfig.hyde_max_tokens ?? this.hydeMaxTokens;

            // Think-mode models need extra token budget for reasoning + answer
            if (this.modelConstraints?.hasThinkMode) {
                const boosted = maxTokens * 3;
                maxTokens = Math.min(boosted, this.modelConstraints.maxTokens[1]);
            }

            const userPrompt = `${hydePrompt}

${question}`;

            try {
                this._emitOperationState(operation, 'generating');
                const completion = await this.engine.chat.completions.create({
                    messages: this._chatMessages([{ role: 'user', content: userPrompt }]),
                    temperature: hydeTemperature,
                    max_tokens: maxTokens,
                    top_p: 0.9,
                    stream: true
                });

                let hydeText = '';
                let wasStopped = false;
                const thinkFilter = this._createThinkFilter();

                for await (const chunk of completion) {
                    if (this.shouldAbort) {
                        wasStopped = true;
                        break;
                    }
                    const content = chunk.choices[0]?.delta?.content || '';
                    hydeText += thinkFilter.push(content);
                }

                hydeText += thinkFilter.flush();
                if (wasStopped) {
                    this.needsReinit = true;
                    throw new Error('HyDE generation stopped by user');
                }

                return this._stripThinkingTokens(hydeText);
            } catch (error) {
                console.error('❌ HyDE generation failed:', error);
                throw new Error(`Failed to generate HyDE: ${error.message}`);
            }
        });
    }

    /**
     * Retrieve and format RAG context without initializing the local LLM.
     * This is the common retrieval path for connected-AI generation and MCP.
     */
    async retrieveContext(question, questionEmbedding, options = {}) {
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.num_results) this.numResults = freshConfig.num_results;
        if (freshConfig.similarity_threshold !== undefined) this.similarityThreshold = freshConfig.similarity_threshold;
        if (freshConfig.retrieval_k !== undefined) this.retrievalK = freshConfig.retrieval_k;
        if (freshConfig.vector_weight !== undefined) this.vectorWeight = freshConfig.vector_weight;
        if (freshConfig.max_chunks_per_parent !== undefined) this.maxChunksPerParent = freshConfig.max_chunks_per_parent;
        if (freshConfig.system_prompt) this.systemPrompt = freshConfig.system_prompt;
        if (freshConfig.user_template) this.userTemplate = freshConfig.user_template;

        const {
            numResults = this.numResults,
            includeMetadata = true,
            metadataFields = undefined,
            similarityThreshold = this.similarityThreshold,
            allowedDocIds = null,
            retrievalK = this.retrievalK ?? this.numResults * 3
        } = options;
        const vectorWeight = options.vectorWeight !== undefined ? options.vectorWeight : this.vectorWeight;
        const rerankerEnabled = options.rerankerEnabled !== undefined
            ? options.rerankerEnabled === true
            : freshConfig.reranker_enabled === true;
        const rankingQuery = String(options.keywordQuery || question);
        const allowedDocIdSet = this._normalizeDocScope(allowedDocIds);
        const allowDoc = (candidate) => {
            if (!allowedDocIdSet) return true;
            if (candidate === undefined || candidate === null) return false;
            return allowedDocIdSet.has(String(candidate));
        };
        const chunkFilter = allowedDocIdSet
            ? metadata => allowDoc(metadata?.parent_id ?? metadata?.doc_id ?? metadata?.id)
            : null;
        const docFilter = allowedDocIdSet
            ? metadata => allowDoc(metadata?.doc_id ?? metadata?.id)
            : null;

        let results;
        let isChunkBased = false;
        let retrievalMetrics;

        if (this.chunkVectorSearch && this.chunkVectorSearch.isBuilt) {
            const chunkRetrievalK = Math.max(numResults, retrievalK || numResults * 3);
            const useBM25 = vectorWeight < 1 && this.bm25Search?.isBuilt;
            const useVector = vectorWeight > 0;
            let vectorChunkResults = useVector
                ? this.chunkVectorSearch.search(questionEmbedding, chunkRetrievalK, {
                    minScore: similarityThreshold,
                    includeMetadata: true,
                    filter: chunkFilter
                })
                : [];
            const keywordQuery = rankingQuery;
            const bm25ChunkResults = useBM25
                ? this.bm25Search.search(keywordQuery, chunkRetrievalK, { filter: chunkFilter })
                : [];

            let fusedChunks;
            if (vectorWeight >= 1) fusedChunks = vectorChunkResults;
            else if (vectorWeight <= 0 && bm25ChunkResults.length) fusedChunks = bm25ChunkResults;
            else if (bm25ChunkResults.length) {
                fusedChunks = this._fuseResults(vectorChunkResults, bm25ChunkResults, {
                    k: 60,
                    vectorWeight,
                    topK: chunkRetrievalK
                });
            } else fusedChunks = vectorChunkResults;

            const maxTotalChunks = numResults * this.maxChunksPerParent * 2;
            fusedChunks = fusedChunks.slice(0, maxTotalChunks);
            const parentCandidateCount = Math.max(numResults, Math.min(chunkRetrievalK, numResults * 4));
            const parentGroups = this._groupChunksByParent(fusedChunks, parentCandidateCount, this.maxChunksPerParent);

            if (allowedDocIdSet && parentGroups.length === 0) {
                results = this.vectorSearch.search(questionEmbedding, parentCandidateCount, {
                    minScore: similarityThreshold,
                    includeMetadata: true,
                    filter: docFilter
                }).map(result => ({ ...result, text: result.text || result.metadata?.text || '' }));
            } else {
                results = parentGroups;
                isChunkBased = true;
            }

            retrievalMetrics = {
                vector_count: vectorChunkResults.length,
                bm25_count: bm25ChunkResults.length,
                fused_count: fusedChunks.length,
                parent_count: results.length,
                fusion_method: vectorWeight >= 1 ? 'vector-only' :
                    vectorWeight <= 0 ? 'bm25-only' :
                    bm25ChunkResults.length ? 'RRF' : 'vector-only',
                scope_size: allowedDocIdSet?.size ?? null,
                requested_k: chunkRetrievalK
            };
        } else {
            const parentCandidateCount = Math.max(numResults, Math.min(retrievalK || numResults * 4, numResults * 4));
            results = this.vectorSearch.search(questionEmbedding, parentCandidateCount, {
                minScore: similarityThreshold,
                includeMetadata: true,
                filter: docFilter
            }).map(result => ({ ...result, text: result.text || result.metadata?.text || '' }));
            retrievalMetrics = {
                vector_count: results.length,
                bm25_count: 0,
                fused_count: results.length,
                parent_count: results.length,
                fusion_method: 'vector-only',
                scope_size: allowedDocIdSet?.size ?? null,
                requested_k: retrievalK
            };
        }

        if (rerankerEnabled && this.reranker && results.length) {
            options.onStatus?.('reranking');
            const neural = await this.reranker.rerank(rankingQuery, results, {
                signal: options.signal,
                onProgress: progress => {
                    options.onRerankerProgress?.(progress);
                    options.onStatus?.('reranking', progress);
                }
            });
            results = neural.results;
            Object.assign(retrievalMetrics, neural.diagnostics);
        } else {
            Object.assign(retrievalMetrics, {
                reranker_applied: false,
                reranker_model: null,
                reranker_candidates: 0,
                reranker_latency_ms: 0,
                reranker_fallback_reason: rerankerEnabled ? 'unavailable' : null
            });
        }

        const beforeQualitySelection = results.length;
        results = rerankAndDiversify(results, rankingQuery, { maxResults: numResults });
        retrievalMetrics.candidate_parent_count = beforeQualitySelection;
        retrievalMetrics.selected_count = results.length;
        retrievalMetrics.dropped_count = Math.max(0, beforeQualitySelection - results.length);
        retrievalMetrics.post_fusion = retrievalMetrics.reranker_applied
            ? 'neural_rrf_coverage_mmr_v1'
            : 'coverage_mmr_v1';

        const contextResult = isChunkBased
            ? this._buildChunkedContext(results, includeMetadata, metadataFields)
            : this._buildContext(results, includeMetadata, metadataFields);
        return {
            question,
            sources: results,
            context: contextResult.context,
            contextPrompt: this._buildRAGPrompt(question, contextResult.context),
            metadata: {
                numSources: results.length,
                searchType: 'hybrid',
                retrieval: retrievalMetrics,
                scope: allowedDocIdSet ? { type: 'doc_filter', size: allowedDocIdSet.size } : { type: 'all' },
                contextLimited: contextResult.contextLimited,
                tokensUsed: contextResult.tokensUsed,
                maxContextTokens: contextResult.maxTokens,
                generationProvider: 'none'
            }
        };
    }

    async _generateFromRetrievedContext(question, questionEmbedding, options = {}) {
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.temperature !== undefined) this.temperature = freshConfig.temperature;
        if (freshConfig.max_tokens) this.maxTokens = freshConfig.max_tokens;
        if (freshConfig.top_p !== undefined) this.topP = freshConfig.top_p;
        if (freshConfig.repeat_penalty !== undefined) this.repeatPenalty = freshConfig.repeat_penalty;

        const retrievalStarted = Date.now();
        const retrieval = await this.retrieveContext(question, questionEmbedding, {
            ...options,
            signal: options.signal || options.operationToken?.abortController?.signal || null
        });
        const retrievalTimeMs = Date.now() - retrievalStarted;
        this.reranker?.releaseForGeneration?.();
        if (!retrieval.sources.length) {
            return {
                answer: 'No relevant documents were found for this question.',
                sources: [],
                metadata: {
                    ...retrieval.metadata,
                    generationProvider: 'local',
                    model: this.modelId,
                    retrieval_time_ms: retrievalTimeMs,
                    generationTime: 0,
                    wasStopped: false
                }
            };
        }

        const rawTemperature = options.temperature ?? this.temperature;
        const temperature = Math.max(this.modelConstraints.temp[0], Math.min(this.modelConstraints.temp[1], rawTemperature));
        let maxTokens = Math.max(
            this.modelConstraints.maxTokens[0],
            Math.min(this.modelConstraints.maxTokens[1], options.maxTokens ?? this.maxTokens)
        );
        if (this.modelConstraints?.hasThinkMode) {
            maxTokens = Math.min(maxTokens * 3, this.modelConstraints.maxTokens[1]);
        }

        const generationStarted = Date.now();
        const completion = await this.engine.chat.completions.create({
            messages: this._chatMessages([
                { role: 'system', content: this.systemPrompt },
                { role: 'user', content: retrieval.contextPrompt }
            ]),
            temperature,
            max_tokens: maxTokens,
            top_p: this.topP,
            repetition_penalty: this.repeatPenalty,
            stream: true
        });

        let answer = '';
        let wasStopped = false;
        const thinkFilter = this._createThinkFilter();
        for await (const chunk of completion) {
            if (this.shouldAbort) {
                wasStopped = true;
                this.needsReinit = true;
                break;
            }
            const visible = thinkFilter.push(chunk.choices[0]?.delta?.content || '');
            answer += visible;
            if (visible && typeof options.onChunk === 'function') options.onChunk(visible, answer);
        }
        answer += thinkFilter.flush();
        answer = this._stripThinkingTokens(answer);

        const metadata = {
            ...retrieval.metadata,
            generationProvider: 'local',
            model: this.modelId,
            temperature,
            retrieval_time_ms: retrievalTimeMs,
            generationTime: (Date.now() - generationStarted) / 1000,
            wasStopped
        };
        const result = { answer, sources: retrieval.sources, metadata };
        this.conversationHistory.push({
            timestamp: new Date().toISOString(),
            query: question,
            answer,
            sources: retrieval.sources,
            metadata
        });
        this._scheduleIdleSuspend();
        return result;
    }

    /**
     * Perform RAG query
     * @param {string} question - User question
     * @param {number[]} questionEmbedding - Embedding of the question
     * @param {Object} options - Query options
     * @returns {Promise<Object>} RAG response with answer and sources
     */
    async query(question, questionEmbedding, options = {}) {
        return this.withOperation(options.owner || 'rag', options, async operation => {
            await this.ensureEngineReady(options.onStatus, operation);
            this._emitOperationState(operation, 'generating');
            return this._generateFromRetrievedContext(question, questionEmbedding, { ...options, operationToken: operation });
        });
    }
    /**
     * Stream RAG response (for real-time display)
     * @param {string} question
     * @param {number[]} questionEmbedding
     * @param {Function} onChunk - Callback for each generated chunk
     * @param {Object} options
     */
    async queryStream(question, questionEmbedding, onChunk, options = {}) {
        return this.withOperation(options.owner || 'rag', options, async operation => {
            await this.ensureEngineReady(options.onStatus, operation);
            this._emitOperationState(operation, 'generating');
            return this._generateFromRetrievedContext(question, questionEmbedding, { ...options, onChunk, operationToken: operation });
        });
    }
    /**
     * Reciprocal Rank Fusion (RRF) for hybrid search
     * Combines results from multiple retrievers with rank-based scoring
     *
     * @param {Array} vectorResults - Results from vector search [{doc_id, score, ...}]
     * @param {Array} bm25Results - Results from BM25 search [{doc_id, score, ...}]
     * @param {Object} options - Fusion options
     * @returns {Array} Fused and re-ranked results
     */
    /**
     * Create a streaming think-token filter for think-mode models.
     * Buffers content while inside <think>...</think>, only forwards post-think text.
     * For non-think models, passes everything through immediately.
     */
    _createThinkFilter() {
        const isThinkModel = this.modelConstraints?.hasThinkMode;
        let buffer = '';
        let insideThink = false;
        let thinkDone = false;

        return {
            /**
             * Process a chunk of streamed text.
             * Returns the text that should be shown to the user (may be empty while thinking).
             */
            push(chunk) {
                if (!isThinkModel || thinkDone) return chunk;

                buffer += chunk;

                // Check if we've entered a think block
                if (!insideThink) {
                    const thinkStart = buffer.indexOf('<think>');
                    if (thinkStart !== -1) {
                        insideThink = true;
                        // Any text before <think> is real output
                        const before = buffer.substring(0, thinkStart);
                        buffer = buffer.substring(thinkStart + 7); // skip '<think>'
                        if (before.trim()) {
                            thinkDone = true;
                            return before;
                        }
                    } else if (buffer.length > 20) {
                        // No <think> tag found after enough chars — not a think-mode output
                        thinkDone = true;
                        const out = buffer;
                        buffer = '';
                        return out;
                    }
                    // Still buffering, waiting for potential <think> tag
                    return '';
                }

                // Inside think block — look for </think>
                const thinkEnd = buffer.indexOf('</think>');
                if (thinkEnd !== -1) {
                    const thinkContent = buffer.substring(0, thinkEnd);
                    const afterThink = buffer.substring(thinkEnd + 8); // skip '</think>'
                    buffer = '';
                    thinkDone = true;
                    insideThink = false;
                    return afterThink;
                }

                // Still inside think block, suppress output
                return '';
            },

            /**
             * Finalize — if the model ended mid-think (token limit), return nothing.
             * The reasoning was internal, there's no answer to show.
             */
            flush() {
                if (insideThink && !thinkDone) {
                    // Model used all tokens on thinking — return the reasoning as the answer
                    const out = buffer;
                    buffer = '';
                    return out;
                }
                // Return any remaining buffer
                const out = buffer;
                buffer = '';
                return out;
            }
        };
    }

    _stripThinkingTokens(text) {
        // Remove complete <think>...</think> blocks (including multiline)
        let cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, '');
        // Remove unclosed <think> block at end (model stopped mid-thought)
        cleaned = cleaned.replace(/<think>[\s\S]*$/g, '');
        cleaned = cleaned.trim();

        // Fallback: if stripping removed ALL content, just strip the tags themselves
        if (!cleaned && text.trim()) {
            cleaned = text.replace(/<\/?think>/g, '').trim();
        }

        return cleaned;
    }

    _fuseResults(vectorResults, bm25Results, options = {}) {
        const {
            k = 60,              // RRF constant (higher = more weight on top ranks)
            vectorWeight = 0.6,  // Weight for vector search (0.6 = 60% vector, 40% BM25)
            topK = 10            // Number of results to return
        } = options;

        // Build score map: doc_id -> { vector_rank, bm25_rank, vector_score, bm25_score, metadata }
        const scoreMap = new Map();

        // Process vector results
        vectorResults.forEach((result, rank) => {
            const docId = result.doc_id || result.metadata?.doc_id || result.id;
            if (!docId) return;

            scoreMap.set(docId, {
                doc_id: docId,
                vector_rank: rank,
                vector_score: result.score,
                bm25_rank: null,
                bm25_score: null,
                metadata: result.metadata || {},
                text: result.text || result.metadata?.text || '',
                parent_id: result.parent_id || result.metadata?.parent_id,
                chunks: result.chunks || []
            });
        });

        // Process BM25 results
        bm25Results.forEach((result, rank) => {
            const docId = result.doc_id || result.metadata?.doc_id || result.id;
            if (!docId) return;

            if (scoreMap.has(docId)) {
                // Document found in both - update BM25 info
                const entry = scoreMap.get(docId);
                entry.bm25_rank = rank;
                entry.bm25_score = result.score;
            } else {
                // Document only in BM25
                scoreMap.set(docId, {
                    doc_id: docId,
                    vector_rank: null,
                    vector_score: null,
                    bm25_rank: rank,
                    bm25_score: result.score,
                    metadata: result.metadata || {},
                    text: result.text || result.metadata?.text || '',
                    parent_id: result.parent_id || result.metadata?.parent_id,
                    chunks: result.chunks || []
                });
            }
        });

        // Compute RRF scores
        const fusedResults = [];
        scoreMap.forEach((entry) => {
            // RRF formula: score = sum(1 / (k + rank))
            let rrfScore = 0;

            if (entry.vector_rank !== null) {
                rrfScore += vectorWeight * (1 / (k + entry.vector_rank + 1));
            }

            if (entry.bm25_rank !== null) {
                rrfScore += (1 - vectorWeight) * (1 / (k + entry.bm25_rank + 1));
            }

            fusedResults.push({
                doc_id: entry.doc_id,
                score: rrfScore,
                vector_score: entry.vector_score,
                bm25_score: entry.bm25_score,
                vector_rank: entry.vector_rank,
                bm25_rank: entry.bm25_rank,
                metadata: entry.metadata,
                text: entry.text,
                parent_id: entry.parent_id,
                chunks: entry.chunks,
                fusion_method: 'RRF'
            });
        });

        // Sort by RRF score and return top K
        fusedResults.sort((a, b) => b.score - a.score);
        return fusedResults.slice(0, topK);
    }

    /**
     * Build context string from retrieved documents
     * @param {Array} results - Retrieved documents
     * @param {boolean} includeMetadata - Whether to include metadata
     * @param {Array} metadataFields - Optional array of specific metadata fields to include
     */
    _buildContext(results, includeMetadata, metadataFields = undefined) {
        const maxContextTokens = this._calculateContextBudget();
        let contextParts = [];
        let estimatedTokens = 0;
        let contextLimited = false;

        for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const text = result.text || result.metadata?.text || '';

            let contextItem = `[Doc ${i + 1}] ${text}`;

            if (includeMetadata && result.metadata) {
                // Filter metadata fields if specific fields are requested
                let metadataEntries = Object.entries(result.metadata)
                    .filter(([key]) => key !== 'text');

                if (metadataFields && Array.isArray(metadataFields) && metadataFields.length > 0) {
                    // Only include requested metadata fields
                    metadataEntries = metadataEntries.filter(([key]) => metadataFields.includes(key));
                }

                const metadataStr = metadataEntries
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ');

                if (metadataStr) {
                    contextItem += `\n   (${metadataStr})`;
                }
            }

            const itemTokens = this._estimateTokens(contextItem);
            if (estimatedTokens + itemTokens < maxContextTokens) {
                contextParts.push(contextItem);
                estimatedTokens += itemTokens;
            } else {
                console.warn(`⚠️ Context budget reached at ${estimatedTokens} tokens (max: ${maxContextTokens})`);
                contextLimited = true;
                break;
            }
        }

        return {
            context: contextParts.join('\n\n'),
            contextLimited,
            tokensUsed: estimatedTokens,
            maxTokens: maxContextTokens
        };
    }

    /**
     * Build RAG prompt using saved template
     */
    _buildRAGPrompt(question, context) {
        // Use saved user template, replacing placeholders
        return this.userTemplate
            .replace('{context}', context)
            .replace('{question}', question);
    }

    /**
     * Group chunk search results by parent document
     * @param {Array} chunkResults - Raw chunk search results
     * @param {number} topK - Number of parent documents to return
     * @param {number} maxChunksPerParent - Maximum chunks to keep per parent document
     * @returns {Array} Grouped results with chunks sorted by position and parent document data
     */
    _groupChunksByParent(chunkResults, topK, maxChunksPerParent = 5) {
        const parentGroups = new Map();

        chunkResults.forEach(chunk => {
            const parentId = chunk.metadata?.parent_id || chunk.parent_id;

            if (!parentGroups.has(parentId)) {
                parentGroups.set(parentId, {
                    doc_id: parentId,
                    parent_id: parentId,
                    chunks: [],
                    maxScore: 0,
                    avgScore: 0,
                    score: 0
                });
            }

            const group = parentGroups.get(parentId);
            group.chunks.push(chunk);
            group.maxScore = Math.max(group.maxScore, chunk.score);
        });

        // Calculate average scores and limit/sort chunks within each group
        parentGroups.forEach((group, parentId) => {
            // Sort by score (descending) to keep best chunks
            group.chunks.sort((a, b) => b.score - a.score);

            // LIMIT: Keep only top N chunks per parent to prevent context overflow
            if (group.chunks.length > maxChunksPerParent) {
                group.chunks = group.chunks.slice(0, maxChunksPerParent);
            }

            const totalScore = group.chunks.reduce((sum, c) => sum + c.score, 0);
            group.avgScore = totalScore / group.chunks.length;
            group.score = group.maxScore;

            // Re-sort chunks by position (reading order) after limiting
            group.chunks.sort((a, b) => {
                const posA = a.metadata?.chunk_index ?? a.position ?? 0;
                const posB = b.metadata?.chunk_index ?? b.position ?? 0;
                return posA - posB;
            });

            // Get parent document metadata and text from parent index
            if (group.chunks.length > 0) {
                const firstChunk = group.chunks[0];
                group.metadata = {};

                if (firstChunk.metadata) {
                    Object.keys(firstChunk.metadata).forEach(key => {
                        if (!key.startsWith('chunk_') && key !== 'parent_id') {
                            group.metadata[key] = firstChunk.metadata[key];
                        }
                    });
                }

                // Try to get full parent document text from parent vector index
                if (this.vectorSearch && this.vectorSearch.getDocument) {
                    const parentDoc = this.vectorSearch.getDocument(parentId);
                    if (parentDoc && parentDoc.metadata) {
                        // Use parent document's full text if available
                        group.text = parentDoc.metadata.text || group.metadata.text || '';
                        // Merge parent metadata (parent has full info)
                        if (parentDoc.metadata) {
                            group.metadata = { ...parentDoc.metadata, ...group.metadata };
                        }
                    }
                }

                // Fallback: reconstruct text from chunks if parent doc not found
                if (!group.text) {
                    group.text = group.chunks.map(c => c.text || c.metadata?.text || '').join(' ');
                }
            }
        });

        // Sort parent groups by max score and return top K
        return Array.from(parentGroups.values())
            .sort((a, b) => b.maxScore - a.maxScore)
            .slice(0, topK);
    }

    /**
     * Estimate token count for text
     * Uses 3.5 chars/token for English with 10% safety buffer
     * @param {string} text - Text to estimate
     * @returns {number} Estimated token count
     */
    _estimateTokens(text) {
        return Math.ceil(text.length / 3.5 * 1.1);
    }

    /**
     * Calculate available context token budget based on model's context window
     * Reserves space for system prompt, question, and answer generation
     * @returns {number} Available tokens for context
     */
    _calculateContextBudget() {
        const contextWindow = this.maxContextLength || 2048;
        const systemPromptTokens = this._estimateTokens(this.systemPrompt);
        const questionBuffer = 150; // Reserve for question + template overhead
        const answerReserve = this.maxTokens || 768;

        const availableContext = contextWindow - systemPromptTokens - questionBuffer - answerReserve;

        // Ensure minimum context budget
        const minContext = 500;
        const budget = Math.max(minContext, availableContext);

        return budget;
    }

    /**
     * Build context from chunked results with token budget management
     * Handles dynamic chunking at retrieval time if index-time chunking was disabled
     * @param {Array} parentGroups - Grouped chunk results
     * @param {boolean} includeMetadata - Whether to include metadata
     * @param {Array} metadataFields - Optional specific metadata fields
     * @returns {Object} { context: string, contextLimited: boolean, tokensUsed: number, maxTokens: number }
     */
    _buildChunkedContext(parentGroups, includeMetadata, metadataFields) {
        const maxContextTokens = this._calculateContextBudget();
        const contextParts = [];
        let estimatedTokens = 0;
        let contextLimited = false;

        // Dynamic chunking threshold: if a single chunk exceeds 30% of context budget
        const dynamicChunkThreshold = maxContextTokens * 0.3;
        const dynamicChunkTargetTokens = Math.floor(maxContextTokens * 0.15);

        for (let i = 0; i < parentGroups.length; i++) {
            const group = parentGroups[i];
            let parentContext = `\n[Doc ${i + 1}]`;

            // Add document-level metadata once (from first chunk)
            if (includeMetadata && group.chunks[0].metadata) {
                let metadataEntries = Object.entries(group.chunks[0].metadata)
                    .filter(([key]) => !key.startsWith('chunk_') && key !== 'parent_id' && key !== 'text');

                if (metadataFields && Array.isArray(metadataFields) && metadataFields.length > 0) {
                    metadataEntries = metadataEntries.filter(([key]) => metadataFields.includes(key));
                }

                const metadataStr = metadataEntries
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ');

                if (metadataStr) {
                    parentContext += `\n   Metadata: ${metadataStr}`;
                }
            }

            parentContext += '\n   Relevant passages:';

            // Add relevant chunks, with dynamic chunking for oversized single-chunk documents
            for (const chunk of group.chunks) {
                const chunkTextRaw = chunk.text || chunk.metadata?.text || '';
                const chunkTokensRaw = this._estimateTokens(chunkTextRaw);

                // Check if this is an oversized chunk (likely from disabled chunking)
                if (chunkTokensRaw > dynamicChunkThreshold && group.chunks.length === 1) {
                    const dynamicChunks = this._dynamicChunk(chunkTextRaw, dynamicChunkTargetTokens);

                    // Add first few dynamic chunks that fit
                    for (let j = 0; j < Math.min(dynamicChunks.length, 3); j++) {
                        const dynChunkText = `\n   » ${dynamicChunks[j]}`;
                        const dynChunkTokens = this._estimateTokens(dynChunkText);

                        if (estimatedTokens + dynChunkTokens < maxContextTokens) {
                            parentContext += dynChunkText;
                            estimatedTokens += dynChunkTokens;
                        } else {
                            console.warn(`⚠️ Context budget reached during dynamic chunking at ${estimatedTokens} tokens`);
                            contextLimited = true;
                            break;
                        }
                    }
                } else {
                    // Normal chunk processing
                    const chunkText = `\n   » ${chunkTextRaw}`;
                    const chunkTokens = this._estimateTokens(chunkText);

                    if (estimatedTokens + chunkTokens < maxContextTokens) {
                        parentContext += chunkText;
                        estimatedTokens += chunkTokens;
                    } else {
                        console.warn(`⚠️ Context budget reached at ${estimatedTokens} tokens (max: ${maxContextTokens})`);
                        contextLimited = true;
                        break;
                    }
                }
            }

            contextParts.push(parentContext);

            if (estimatedTokens >= maxContextTokens) {
                contextLimited = true;
                break;
            }
        }

        return {
            context: contextParts.join('\n\n'),
            contextLimited,
            tokensUsed: estimatedTokens,
            maxTokens: maxContextTokens
        };
    }

    _normalizeDocScope(allowedDocIds) {
        if (allowedDocIds === null || allowedDocIds === undefined) {
            return null;
        }

        if (allowedDocIds instanceof Set) {
            return new Set(Array.from(allowedDocIds).map(id => String(id)));
        }

        if (Array.isArray(allowedDocIds)) {
            return new Set(allowedDocIds.map(id => String(id)));
        }

        return new Set([String(allowedDocIds)]);
    }

    /**
     * Reset the conversation context
     */
    async resetContext() {
        if (this.isInitialized) {
            await this.engine.resetChat();
        }
    }

    /**
     * Get model statistics
     */
    getStats() {
        return {
            modelId: this.modelId,
            isInitialized: this.isInitialized,
            maxContextLength: this.maxContextLength
        };
    }

    /**
     * Update chunk vector search index (called after processing)
     */
    setChunkVectorSearch(chunkVectorSearch) {
        this.chunkVectorSearch = chunkVectorSearch;
    }

    /**
     * Update BM25 search index for hybrid retrieval (called after processing)
     */
    setBM25Search(bm25Search) {
        this.bm25Search = bm25Search;
    }

    setReranker(reranker) {
        this.reranker = reranker;
    }

    /**
     * Export conversation history
     * @param {string} format - 'json' or 'csv'
     * @returns {void} Triggers download
     */
    exportConversation(format = 'json') {
        if (typeof window.exportRAGConversation === 'function') {
            window.exportRAGConversation(this.conversationHistory, format);
        } else {
            console.error('Export function not available. Make sure export-import.js is loaded.');
        }
    }

    /**
     * Clear conversation history
     */
    clearConversationHistory() {
        this.conversationHistory = [];
    }

    /**
     * Get conversation history length
     */
    getConversationLength() {
        return this.conversationHistory.length;
    }

    /**
     * Dynamically chunk text at RAG retrieval time
     * Used when chunking is disabled at index time but documents are too long
     * @param {string} text - Text to chunk
     * @param {number} targetTokens - Target tokens per chunk
     * @returns {Array<string>} Array of text chunks
     */
    _dynamicChunk(text, targetTokens) {
        const targetChars = targetTokens * 3.5; // ~3.5 chars per token
        const overlap = Math.floor(targetChars * 0.15); // 15% overlap
        const chunks = [];

        for (let i = 0; i < text.length; i += (targetChars - overlap)) {
            const chunk = text.slice(i, i + targetChars).trim();
            if (chunk.length > 0) {
                chunks.push(chunk);
            }
        }

        return chunks;
    }

    /**
     * Generate text from a raw prompt without retrieval. Used for cluster
     * summarization and other prompt-only tasks.
     */
    async generateRaw(prompt, options = {}) {
        return this.withOperation(options.owner || 'cluster-label', options, async operation => {
            await this.ensureEngineReady(options.onStatus, operation);
            const {
                temperature = Math.min(0.7, this.modelConstraints.temp[1]),
                maxTokens: rawMax = Math.min(256, this.modelConstraints.maxTokens[1]),
                topP = 0.9,
                systemPrompt = null
            } = options;

            const maxTokens = Math.max(
                this.modelConstraints.maxTokens[0],
                Math.min(this.modelConstraints.maxTokens[1], rawMax)
            );

            const messages = [];
            if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
            messages.push({ role: 'user', content: prompt });
            this._emitOperationState(operation, 'generating');

            const completion = await this.engine.chat.completions.create({
                messages: this._chatMessages(messages),
                temperature,
                max_tokens: maxTokens,
                top_p: topP,
                stream: true
            });

            let text = '';
            const thinkFilter = this._createThinkFilter();
            for await (const chunk of completion) {
                if (this.shouldAbort) {
                    this.needsReinit = true;
                    throw new Error('Generation aborted');
                }
                const content = chunk.choices[0]?.delta?.content || '';
                text += thinkFilter.push(content);
            }
            text += thinkFilter.flush();
            return this._stripThinkingTokens(text).trim();
        });
    }

    /**
     * Stream a pre-budgeted chat prompt. Retrieval and memory allocation live
     * above this layer so one-shot RAG and MCP behavior remain unchanged.
     */
    async generateFromMessages(messages, options = {}) {
        return this.withOperation(options.owner || 'chat', options, async operation => {
            await this.ensureEngineReady(options.onStatus, operation);
            await this._resetEngineChatBestEffort('chat completion');
            // Older persisted plans and independent callers may still contain
            // a later system message. Gemma rejects that shape, so enforce one
            // leading system message at the WebLLM boundary as defense-in-depth.
            const orderedMessages = this._chatMessages(messages);
            if (!orderedMessages.length) throw new Error('No valid chat messages were provided.');
            const promptCompatibility = this._promptCompatibilityDiagnostic(orderedMessages);
            if (globalThis.__VECTORIA_DEBUG_AI__ === true) {
                console.debug('WebLLM prompt role diagnostic:', promptCompatibility);
            }
            const freshConfig = this.loadSavedConfig();
            const rawTemperature = options.temperature ?? freshConfig.temperature ?? this.temperature;
            const temperature = Math.max(this.modelConstraints.temp[0], Math.min(this.modelConstraints.temp[1], rawTemperature));
            const requestedMax = options.maxTokens ?? freshConfig.max_tokens ?? this.maxTokens;
            const maxTokens = Math.max(
                this.modelConstraints.maxTokens[0],
                Math.min(this.modelConstraints.maxTokens[1], requestedMax)
            );
            const startedAt = Date.now();
            const topP = options.topP ?? freshConfig.top_p ?? this.topP;
            const repeatPenalty = options.repeatPenalty ?? freshConfig.repeat_penalty ?? this.repeatPenalty;
            this._emitOperationState(operation, 'generating');
            const completion = await this.engine.chat.completions.create({
                messages: orderedMessages,
                temperature,
                max_tokens: maxTokens,
                top_p: topP,
                repetition_penalty: repeatPenalty,
                stream: true,
                stream_options: { include_usage: true }
            });

            let answer = '';
            let wasStopped = false;
            let finishReason = null;
            let usage = null;
            const thinkFilter = this._createThinkFilter();
            for await (const chunk of completion) {
                if (chunk?.usage) usage = chunk.usage;
                const chunkFinishReason = chunk?.choices?.[0]?.finish_reason;
                if (chunkFinishReason) finishReason = chunkFinishReason;
                if (this.shouldAbort) {
                    wasStopped = true;
                    finishReason = 'abort';
                    this.needsReinit = true;
                    break;
                }
                const visible = thinkFilter.push(chunk.choices[0]?.delta?.content || '');
                answer += visible;
                if (visible) options.onChunk?.(visible, answer);
            }
            answer += thinkFilter.flush();
            answer = this._stripThinkingTokens(answer).trim();
            if (!wasStopped && !answer) {
                const error = new Error('WebLLM returned an empty completion.');
                error.code = 'empty_completion';
                throw error;
            }
            if (finishReason === 'abort') wasStopped = true;
            const usageExtra = usage?.extra || {};
            return {
                answer,
                wasStopped,
                finishReason: finishReason || (wasStopped ? 'abort' : 'stop'),
                metadata: {
                    model: this.modelId,
                    temperature,
                    topP,
                    repeatPenalty,
                    maxTokens,
                    generationTime: (Date.now() - startedAt) / 1000,
                    wasStopped,
                    finishReason: finishReason || (wasStopped ? 'abort' : 'stop'),
                    promptCompatibility,
                    actualUsage: usage ? {
                        promptTokens: Number(usage.prompt_tokens) || 0,
                        completionTokens: Number(usage.completion_tokens) || 0,
                        totalTokens: Number(usage.total_tokens) || 0,
                        e2eLatencySeconds: Number(usageExtra.e2e_latency_s) || null,
                        prefillTokensPerSecond: Number(usageExtra.prefill_tokens_per_s) || null,
                        decodeTokensPerSecond: Number(usageExtra.decode_tokens_per_s) || null,
                        timeToFirstTokenSeconds: Number(usageExtra.time_to_first_token_s) || null
                    } : null
                }
            };
        });
    }

    /** RAG query with LLM synthesis and current-turn citation-boundary checks. */
    async queryWithCitations(question, questionEmbedding, options = {}) {
        const base = await this.query(question, questionEmbedding, options);
        const answer = base?.answer || base?.text || '';
        const sources = base?.sources || base?.results || [];
        const checked = sanitizeCitationBounds(answer, sources.length);
        const claims = checked.answer ? [{
            claim: checked.answer,
            supporting_doc_indices: checked.citations.map(number => {
                const source = sources[number - 1];
                return source?.index ?? source?.documentIndex ?? number - 1;
            }),
            confidence: null,
            verdict: 'not_evaluated',
            reasons: ['llm_synthesis_citation_bounds_only'],
            evidence_excerpts: []
        }] : [];

        return {
            answer: checked.answer,
            claims,
            sources,
            metadata: {
                ...(base?.metadata || {}),
                groundingState: 'llm_synthesis',
                validation_method: 'citation_bounds_only',
                invalidCitationCount: checked.invalidCitations.length,
                removedClaimCount: 0,
                fallbackUsed: false
            }
        };
    }
}
