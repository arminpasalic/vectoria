/**
 * Browser-based RAG (Retrieval-Augmented Generation) using WebLLM
 * Model: gemma-2-2b-it-q4f32_1-MLC-1k
 * Runs Gemma 2 2B locally in browser via WebGPU
 */

import { CreateWebWorkerMLCEngine, prebuiltAppConfig } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.84/+esm";
import { getModelConstraints } from "../model-constraints.js";

// Load cached real download sizes from previous downloads
try {
    const cached = localStorage.getItem('vectoria_model_download_sizes');
    window.__webllmRealDownloadSizes = cached ? JSON.parse(cached) : {};
} catch (_) {
    window.__webllmRealDownloadSizes = {};
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
    constructor(vectorSearch, chunkVectorSearch = null, bm25Search = null) {
        this.vectorSearch = vectorSearch;        // Parent document index (not used for retrieval)
        this.chunkVectorSearch = chunkVectorSearch; // Chunk index for RAG retrieval
        this.bm25Search = bm25Search;            // BM25 keyword search for hybrid retrieval
        this.engine = null;
        this.worker = null;

        // Abort control for stopping generation
        this.shouldAbort = false;
        this.currentGenerationReject = null;
        this.needsReinit = false; // Flag to track if engine needs reinitialization after abort
        this.workerUnloaded = false; // Set true by unloadWorker() to permanently block re-init

        // Soft suspension state (distinct from unloadWorker which is permanent).
        // suspended = worker terminated to free RAM, but allowed to lazily re-init on next use.
        this.isSuspended = false;
        this._idleTimer = null;
        this._idleTimeoutMs = 5 * 60 * 1000; // 5 min idle → auto-suspend

        // Load saved configuration
        const savedConfig = this.loadSavedConfig();

        // Load model ID from saved config or use default
        this.modelId = savedConfig.model_id || "gemma-2-2b-it-q4f16_1-MLC";

        // Get model constraints
        this.modelConstraints = getModelConstraints(this.modelId);

        this.isInitialized = false;

        // Load LLM generation parameters from saved config
        this.temperature = savedConfig.temperature || 0.5;
        this.maxTokens = savedConfig.max_tokens || 1024;
        this.topP = savedConfig.top_p || 0.9;
        this.repeatPenalty = savedConfig.repeat_penalty || 1.25;
        this.maxContextLength = savedConfig.context_window_size || this.modelConstraints.contextWindow || 2048;

        // Load RAG parameters
        this.numResults = savedConfig.num_results || 5;
        this.similarityThreshold = savedConfig.similarity_threshold || 0.7;  // e5-base-v2 range: 0.7-1.0
        this.retrievalK = savedConfig.retrieval_k || 60;
        this.vectorWeight = savedConfig.vector_weight !== undefined ? savedConfig.vector_weight : 0.6;
        this.maxChunksPerParent = savedConfig.max_chunks_per_parent || 5;  // Limit chunks per parent to prevent context overflow

        // Conversation history for export
        this.conversationHistory = [];

        // Load RAG prompts
        this.systemPrompt = savedConfig.system_prompt ||
`You are a helpful assistant answering questions based on provided documents.
Use [Doc N] to cite sources. If information is missing, say so. Keep answers clear and focused.`;

        this.userTemplate = savedConfig.user_template ||
`Documents:
{context}

Question: {question}

Answer based on the documents above:`;

        // Load HyDE prompts and settings
        this.hydePrompt = savedConfig.hyde_prompt ||
`Write a short factual paragraph that could answer this question:`;

        this.hydeTemperature = savedConfig.hyde_temperature !== undefined ? savedConfig.hyde_temperature : 0.2;
        this.hydeMaxTokens = savedConfig.hyde_max_tokens !== undefined ? savedConfig.hyde_max_tokens : 256;

    }

    /**
     * Get model constraints for UI validation
     * @returns {Object} Model constraints
     */
    getModelConstraints() {
        return this.modelConstraints;
    }

    /**
     * Abort the current RAG generation
     * Sets a flag that the streaming loop checks to stop gracefully
     */
    abort() {
        this.shouldAbort = true;
        this._clearIdleTimer();

        // Try to interrupt the WebLLM engine directly
        if (this.engine && typeof this.engine.interruptGenerate === 'function') {
            this.engine.interruptGenerate();
        }

        if (this.currentGenerationReject) {
            this.currentGenerationReject(new Error('Generation stopped by user'));
        }
    }

    unloadWorker() {
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
    }

    /**
     * Soft-suspend the LLM worker to free GPU/RAM during heavy stages.
     * Unlike unloadWorker(), this allows lazy re-initialization on next query.
     * Safe to call when worker is not loaded (no-op).
     */
    suspendWorker(reason = 'idle') {
        this._clearIdleTimer();
        if (!this.worker && !this.isInitialized) {
            return;
        }
        if (this.workerUnloaded) {
            return; // Permanent unload already in effect
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

    /**
     * Reset abort state (called before starting new generation)
     */
    resetAbort() {
        const wasAborted = this.shouldAbort;
        this.shouldAbort = false;
        this.currentGenerationReject = null;
    }

    /**
     * Reinitialize the engine (needed after abort corrupts the engine state)
     */
    async reinitializeEngine() {
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
        await this.initialize();
    }

    /**
     * Check if engine needs reinitialization and do it if needed
     */
    async ensureEngineReady() {
        if (this.workerUnloaded) {
            throw new Error('Local LLM is unloaded. Use query_rag_external (Claude as RAG) instead, or reload the page to re-enable the local LLM.');
        }
        // If soft-suspended, transparently resume (model files are still cached in IndexedDB)
        if (this.isSuspended && !this.isInitialized) {
            console.log('🤖 LLM was suspended — resuming for query...');
            this.isSuspended = false;
            await this.initialize();
        }
        if (this.needsReinit) {
            await this.reinitializeEngine();
        }
        if (!this.isInitialized) {
            throw new Error('LLM not initialized. Call initialize() first.');
        }
        // Verify engine can actually respond — ModelNotLoadedError means the worker
        // process was created but reload() never completed (e.g. after IDB cache wipe)
        try {
            await this.engine.runtimeStatsText();
        } catch (e) {
            const msg = e?.message || '';
            if (msg.includes('ModelNotLoaded') || msg.includes('not loaded')) {
                console.warn('⚠️ Engine exists but model not loaded, reinitializing...');
                this.isInitialized = false;
                this._idbRetried = false;
                await this.reinitializeEngine();
            }
        }
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
                    context_window_size: config.llm?.context_window_size,
                    // RAG settings
                    num_results: config.search?.num_results,
                    similarity_threshold: config.search?.similarity_threshold,
                    retrieval_k: config.search?.retrieval_k,
                    vector_weight: config.search?.vector_weight,
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

            const captureModelId = this.modelId;
            const stall = makeStallTimeout(120000, 'LLM model download');
            const enginePromise = CreateWebWorkerMLCEngine(
                this.worker,
                this.modelId,
                {
                    initProgressCallback: (progress) => {
                        stall.reset();
                        // Capture real total download size from progress text (e.g. "3.2GB/7.1GB")
                        if (progress.text) {
                            const sizeMatch = progress.text.match(/\/([\d.]+)\s*(GB|MB)/i);
                            if (sizeMatch) {
                                const val = parseFloat(sizeMatch[1]);
                                const unit = sizeMatch[2].toUpperCase();
                                const sizeStr = val + ' ' + unit;
                                if (!window.__webllmRealDownloadSizes[captureModelId] || window.__webllmRealDownloadSizes[captureModelId] !== sizeStr) {
                                    window.__webllmRealDownloadSizes[captureModelId] = sizeStr;
                                    try { localStorage.setItem('vectoria_model_download_sizes', JSON.stringify(window.__webllmRealDownloadSizes)); } catch (_) {}
                                }
                            }
                        }
                        // Only send to UI callback for modal display
                        if (onProgress) {
                            onProgress({
                                status: 'loading',
                                text: progress.text,
                                progress: progress.progress || 0
                            });
                        }
                    },
                    appConfig: { ...prebuiltAppConfig, cacheBackend: "indexeddb" }
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

            console.error('❌ Failed to initialize LLM:', error);
            let msg = rawMsg;
            if (rawMsg.includes('ArtifactIndexedDBCache')) {
                msg = `Model cache is corrupted. Please go to Advanced Settings → Language Model → "Clear Model Cache" and try again.`;
            } else if (rawMsg.includes('Cache') && rawMsg.includes('network')) {
                msg = `Model download failed (network error). This usually means:\n` +
                    `• The model files couldn't be fetched from HuggingFace\n` +
                    `• A firewall or VPN is blocking the download\n` +
                    `• HuggingFace is temporarily unavailable\n\n` +
                    `Try refreshing the page or switching to a different model in Advanced Settings.`;
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
    async generateHyDE(question) {
        // Ensure engine is ready (reinitialize if needed after abort)
        await this.ensureEngineReady();

        let maxTokens = this.hydeMaxTokens;

        // Think-mode models need extra token budget for reasoning + answer
        if (this.modelConstraints?.hasThinkMode) {
            const boosted = maxTokens * 3;
            maxTokens = Math.min(boosted, this.modelConstraints.maxTokens[1]);
        }

        const userPrompt = `${this.hydePrompt}

${question}`;

        try {
            // Use streaming API for abort support
            const completion = await this.engine.chat.completions.create({
                messages: [
                    {
                        role: "user",
                        content: userPrompt
                    }
                ],
                temperature: this.hydeTemperature,
                max_tokens: maxTokens,
                top_p: 0.9,
                stream: true
            });

            let hydeText = '';
            let wasStopped = false;
            const thinkFilter = this._createThinkFilter();

            for await (const chunk of completion) {
                // Check abort flag between chunks
                if (this.shouldAbort) {
                    wasStopped = true;
                    break;
                }
                const content = chunk.choices[0]?.delta?.content || '';
                hydeText += thinkFilter.push(content);
            }

            hydeText += thinkFilter.flush();

            // If we aborted, mark engine for reinitialization
            if (wasStopped) {
                this.needsReinit = true;
                throw new Error('HyDE generation stopped by user');
            }

            hydeText = this._stripThinkingTokens(hydeText);
            return hydeText;
        } catch (error) {
            console.error('❌ HyDE generation failed:', error);
            throw new Error(`Failed to generate HyDE: ${error.message}`);
        }
    }

    /**
     * Perform RAG query
     * @param {string} question - User question
     * @param {number[]} questionEmbedding - Embedding of the question
     * @param {Object} options - Query options
     * @returns {Promise<Object>} RAG response with answer and sources
     */
    async query(question, questionEmbedding, options = {}) {
        // Ensure engine is ready (reinitialize if needed after abort)
        await this.ensureEngineReady();

        // Reload config before each query to pick up setting changes
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.temperature !== undefined) this.temperature = freshConfig.temperature;
        if (freshConfig.max_tokens) this.maxTokens = freshConfig.max_tokens;
        if (freshConfig.top_p !== undefined) this.topP = freshConfig.top_p;
        if (freshConfig.repeat_penalty !== undefined) this.repeatPenalty = freshConfig.repeat_penalty;
        if (freshConfig.num_results) this.numResults = freshConfig.num_results;
        if (freshConfig.similarity_threshold !== undefined) this.similarityThreshold = freshConfig.similarity_threshold;
        if (freshConfig.retrieval_k !== undefined) this.retrievalK = freshConfig.retrieval_k;
        if (freshConfig.vector_weight !== undefined) this.vectorWeight = freshConfig.vector_weight;
        if (freshConfig.max_chunks_per_parent !== undefined) this.maxChunksPerParent = freshConfig.max_chunks_per_parent;
        if (freshConfig.system_prompt) this.systemPrompt = freshConfig.system_prompt;
        if (freshConfig.user_template) this.userTemplate = freshConfig.user_template;

        const {
            numResults = this.numResults,
            temperature: rawTemperature = this.temperature,
            maxTokens: rawMaxTokens = this.maxTokens,
            includeMetadata = true,
            similarityThreshold = this.similarityThreshold,
            allowedDocIds = null,
            retrievalK = this.retrievalK ?? this.numResults * 3
        } = options;

        // Clamp parameters to model constraints
        const temperature = Math.max(
            this.modelConstraints.temp[0],
            Math.min(this.modelConstraints.temp[1], rawTemperature)
        );
        let maxTokens = Math.max(
            this.modelConstraints.maxTokens[0],
            Math.min(this.modelConstraints.maxTokens[1], rawMaxTokens)
        );

        // Think-mode models need extra token budget for reasoning + answer
        if (this.modelConstraints?.hasThinkMode) {
            const boosted = maxTokens * 3;
            maxTokens = Math.min(boosted, this.modelConstraints.maxTokens[1]);
        }

        if (temperature !== rawTemperature) {
            console.warn(`⚠️ Temperature ${rawTemperature} clamped to ${temperature} (model range: ${this.modelConstraints.temp[0]}-${this.modelConstraints.temp[1]})`);
        }
        if (maxTokens !== rawMaxTokens && !this.modelConstraints?.hasThinkMode) {
            console.warn(`⚠️ MaxTokens ${rawMaxTokens} clamped to ${maxTokens} (model range: ${this.modelConstraints.maxTokens[0]}-${this.modelConstraints.maxTokens[1]})`);
        }

        const requestedSearchType = options.searchType ? String(options.searchType).toLowerCase() : 'semantic';
        if (requestedSearchType !== 'semantic' && requestedSearchType) {
            console.warn(`Keyword retrieval mode is no longer supported for RAG. Using semantic vectors instead (requested: ${requestedSearchType}).`);
        }
        const normalizedSearchType = 'semantic';

        const allowedDocIdSet = this._normalizeDocScope(allowedDocIds);
        if (allowedDocIdSet) {
        }
        const scopeMetadata = allowedDocIdSet ? { type: 'doc_filter', size: allowedDocIdSet.size } : null;
        const allowDoc = (candidate) => {
            if (!allowedDocIdSet) {
                return true;
            }
            if (candidate === undefined || candidate === null) {
                return false;
            }
            return allowedDocIdSet.has(String(candidate));
        };
        const chunkFilter = allowedDocIdSet
            ? (metadata) => allowDoc(metadata?.parent_id ?? metadata?.doc_id ?? metadata?.id)
            : null;
        const docFilter = allowedDocIdSet
            ? (metadata) => allowDoc(metadata?.doc_id ?? metadata?.id)
            : null;

        // 1. Retrieve relevant chunks or documents using HYBRID SEARCH
        let results;
        let isChunkBased = false;
        let retrievalMetrics = {};

        if (this.chunkVectorSearch && this.chunkVectorSearch.isBuilt) {
            // TIER 3: Use chunk-based retrieval with HYBRID SEARCH
            const chunkRetrievalK = Math.max(numResults, retrievalK || this.retrievalK || numResults * 3);

            // Determine search strategy based on vectorWeight FIRST
            const vectorWeight = options.vectorWeight !== undefined ? options.vectorWeight : this.vectorWeight;
            const useBM25 = vectorWeight < 1.0 && this.bm25Search && this.bm25Search.isBuilt;
            const useVector = vectorWeight > 0.0;

            // Log search mode
            if (vectorWeight >= 1.0) {
            } else if (vectorWeight <= 0.0) {
            } else {
            }

            // Vector search for chunks (only if weight > 0)
            let vectorChunkResults = [];
            if (useVector) {
                vectorChunkResults = this.chunkVectorSearch.search(questionEmbedding, chunkRetrievalK, {
                    minScore: similarityThreshold,
                    includeMetadata: true,
                    filter: chunkFilter
                });
                if (vectorChunkResults.length > 0) {
                }
            } else {
            }

            // BM25 search for chunks (if weight < 100% and index available)
            let bm25ChunkResults = [];
            if (useBM25) {
                bm25ChunkResults = this.bm25Search.search(question, chunkRetrievalK);
                if (allowedDocIdSet) {
                    bm25ChunkResults = bm25ChunkResults.filter(result =>
                        allowDoc(result.parent_id ?? result.metadata?.parent_id ?? result.metadata?.doc_id)
                    );
                }
                if (bm25ChunkResults.length > 0) {
                }
            } else if (useVector) {
            } else if (!this.bm25Search || !this.bm25Search.isBuilt) {
            }

            // Fuse results using Reciprocal Rank Fusion (RRF) or use single-source results
            let fusedChunks;
            if (vectorWeight >= 1.0) {
                // 100% vector - use vector results directly
                fusedChunks = vectorChunkResults;
            } else if (vectorWeight <= 0.0 && bm25ChunkResults.length > 0) {
                // 100% BM25 - use BM25 results directly
                fusedChunks = bm25ChunkResults;
            } else if (bm25ChunkResults.length > 0) {
                // Hybrid - fuse with RRF
                fusedChunks = this._fuseResults(vectorChunkResults, bm25ChunkResults, {
                    k: 60,
                    vectorWeight: vectorWeight,
                    topK: chunkRetrievalK
                });
            } else {
                // Fallback to vector results
                fusedChunks = vectorChunkResults;
            }

            // Pre-limit total chunks before grouping to prevent excessive processing
            const maxTotalChunks = numResults * this.maxChunksPerParent * 2;
            if (fusedChunks.length > maxTotalChunks) {
                fusedChunks = fusedChunks.slice(0, maxTotalChunks);
            }

            // Group chunks by parent document
            const parentGroups = this._groupChunksByParent(fusedChunks, numResults, this.maxChunksPerParent);
            if (allowedDocIdSet && parentGroups.length === 0) {
                console.warn('    No scoped chunks matched; falling back to scoped parent document search');
                results = this.vectorSearch.search(questionEmbedding, numResults, {
                    minScore: similarityThreshold,
                    includeMetadata: true,
                    filter: docFilter
                }).map(result => ({
                    ...result,
                    text: result.text || result.metadata?.text || ''
                }));
                isChunkBased = false;

                retrievalMetrics.parent_count = results.length;
                retrievalMetrics.scope_size = allowedDocIdSet.size;
                retrievalMetrics.fallback = 'parent_scope';
            } else {
                results = parentGroups;
                isChunkBased = true;
            }

            // Store retrieval metrics
            retrievalMetrics = {
                vector_count: vectorChunkResults.length,
                bm25_count: bm25ChunkResults.length,
                fused_count: fusedChunks.length,
                parent_count: parentGroups.length,
                fusion_method: vectorWeight >= 1.0 ? 'vector-only' :
                               vectorWeight <= 0.0 ? 'bm25-only' :
                               bm25ChunkResults.length > 0 ? 'RRF' : 'vector-only',
                scope_size: allowedDocIdSet ? allowedDocIdSet.size : null,
                requested_k: chunkRetrievalK
            };
        } else {
            // Fallback: Use parent document search (original behavior)
            results = this.vectorSearch.search(questionEmbedding, numResults, {
                minScore: similarityThreshold,
                includeMetadata: true,
                filter: docFilter
            });
            // Ensure text is present on all results
            results = results.map(result => ({
                ...result,
                text: result.text || result.metadata?.text || ''
            }));
            isChunkBased = false;

            retrievalMetrics = {
                vector_count: results.length,
                bm25_count: 0,
                fused_count: results.length,
                parent_count: results.length,
                fusion_method: 'vector-only',
                scope_size: allowedDocIdSet ? allowedDocIdSet.size : null,
                requested_k: retrievalK || this.retrievalK || numResults * 3
            };
        }

        // 2. Build context from retrieved results
        if (includeMetadata) {
        } else {
        }
        const contextResult = isChunkBased
            ? this._buildChunkedContext(results, includeMetadata, options.metadataFields)
            : this._buildContext(results, includeMetadata, options.metadataFields);

        const { context, contextLimited } = contextResult;

        // 3. Create RAG prompt
        const prompt = this._buildRAGPrompt(question, context);

        // 4. Generate answer using LLM (streaming internally for abort support)
        const startTime = Date.now();

        try {
            // Use streaming API internally to support abort
            const completion = await this.engine.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: this.systemPrompt
                    },
                    {
                        role: "user",
                        content: prompt
                    }
                ],
                temperature: temperature || this.temperature,
                max_tokens: maxTokens || this.maxTokens,
                top_p: this.topP,
                repetition_penalty: this.repeatPenalty,
                stream: true  // Enable streaming for abort support
            });

            let answer = '';
            let wasStopped = false;
            const thinkFilter = this._createThinkFilter();

            for await (const chunk of completion) {
                // Check abort flag between chunks
                if (this.shouldAbort) {
                    wasStopped = true;
                    break;
                }
                const content = chunk.choices[0]?.delta?.content || '';
                answer += thinkFilter.push(content);
            }

            // Flush any remaining buffered content
            answer += thinkFilter.flush();

            // If we aborted, mark engine for reinitialization
            if (wasStopped) {
                this.needsReinit = true;
            }

            answer = this._stripThinkingTokens(answer);

            const generationTime = (Date.now() - startTime) / 1000;

            if (wasStopped) {
            } else {
            }

            const result = {
                answer: answer,
                sources: results,
                metadata: {
                    numSources: results.length,
                    searchType: 'hybrid',
                    generationTime: generationTime,
                    model: this.modelId,
                    temperature: temperature,
                    retrieval: retrievalMetrics,  // Expose hybrid search details
                    scope: scopeMetadata,
                    retrieval_time_ms: generationTime * 1000,
                    wasStopped: wasStopped,
                    contextLimited: contextLimited
                }
            };

            // Store in conversation history for export
            this.conversationHistory.push({
                timestamp: new Date().toISOString(),
                query: question,
                answer: answer,
                sources: results.map(r => ({
                    id: r.id || r.index,
                    text: r.text || '',
                    score: r.score || 0,
                    metadata: r.metadata || {}
                })),
                metadata: result.metadata
            });

            // Schedule idle auto-suspend (5 min) to free GPU/RAM if user goes idle
            this._scheduleIdleSuspend();

            return result;
        } catch (error) {
            console.error('❌ Answer generation failed:', error);
            const msg = error?.message || '';
            // ModelNotLoadedError: engine exists but model wasn't loaded — force full reinit next call
            if (msg.includes('ModelNotLoaded') || msg.includes('not loaded before')) {
                this.isInitialized = false;
                this._idbRetried = false;
                this.needsReinit = true;
            }
            throw new Error(`Failed to generate answer: ${msg}`);
        }
    }

    /**
     * Stream RAG response (for real-time display)
     * @param {string} question
     * @param {number[]} questionEmbedding
     * @param {Function} onChunk - Callback for each generated chunk
     * @param {Object} options
     */
    async queryStream(question, questionEmbedding, onChunk, options = {}) {
        // Ensure engine is ready (reinitialize if needed after abort)
        await this.ensureEngineReady();

        // Reload config before streaming query
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.temperature !== undefined) this.temperature = freshConfig.temperature;
        if (freshConfig.max_tokens) this.maxTokens = freshConfig.max_tokens;
        if (freshConfig.top_p !== undefined) this.topP = freshConfig.top_p;
        if (freshConfig.repeat_penalty !== undefined) this.repeatPenalty = freshConfig.repeat_penalty;
        if (freshConfig.num_results) this.numResults = freshConfig.num_results;
        if (freshConfig.similarity_threshold !== undefined) this.similarityThreshold = freshConfig.similarity_threshold;
        if (freshConfig.retrieval_k !== undefined) this.retrievalK = freshConfig.retrieval_k;
        if (freshConfig.system_prompt) this.systemPrompt = freshConfig.system_prompt;
        if (freshConfig.user_template) this.userTemplate = freshConfig.user_template;

        const {
            numResults = this.numResults,
            temperature = this.temperature,
            maxTokens: rawMaxTokens = this.maxTokens,
            includeMetadata = true,
            metadataFields = undefined,
            similarityThreshold = this.similarityThreshold,
            allowedDocIds = null
        } = options;

        // Think-mode models need extra token budget for reasoning + answer
        let maxTokens = rawMaxTokens;
        if (this.modelConstraints?.hasThinkMode) {
            const boosted = maxTokens * 3;
            maxTokens = Math.min(boosted, this.modelConstraints.maxTokens[1]);
        }

        const requestedSearchType = options.searchType ? String(options.searchType).toLowerCase() : 'semantic';
        if (requestedSearchType !== 'semantic' && requestedSearchType) {
            console.warn(`Keyword retrieval mode is no longer supported for RAG streaming. Using semantic vectors instead (requested: ${requestedSearchType}).`);
        }
        const allowedDocIdSet = this._normalizeDocScope(allowedDocIds);
        if (allowedDocIdSet) {
        }
        const scopeMetadata = allowedDocIdSet ? { type: 'doc_filter', size: allowedDocIdSet.size } : null;
        const allowDoc = (candidate) => {
            if (!allowedDocIdSet) return true;
            if (candidate === undefined || candidate === null) return false;
            return allowedDocIdSet.has(String(candidate));
        };
        const docFilter = allowedDocIdSet
            ? (metadata) => allowDoc(metadata?.doc_id ?? metadata?.id)
            : null;

        const normalizedSearchType = 'semantic';

        // Retrieve documents
        let results = this.vectorSearch.search(questionEmbedding, numResults, {
            minScore: similarityThreshold,
            includeMetadata: true,
            filter: docFilter
        });

        results = results.map(result => ({
            ...result,
            text: result.text || result.metadata?.text || ''
        }));

        const contextResult = this._buildContext(results, includeMetadata, metadataFields);
        const { context, contextLimited } = contextResult;
        const prompt = this._buildRAGPrompt(question, context);

        // Stream completion
        const completion = await this.engine.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: this.systemPrompt
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: temperature || this.temperature,
            max_tokens: maxTokens || this.maxTokens,
            top_p: this.topP,
            repetition_penalty: this.repeatPenalty,
            stream: true
        });

        let fullAnswer = '';
        let visibleAnswer = '';
        let wasStopped = false;
        const thinkFilter = this._createThinkFilter();

        for await (const chunk of completion) {
            // Check abort flag between chunks
            if (this.shouldAbort) {
                wasStopped = true;
                break;
            }

            const content = chunk.choices[0]?.delta?.content || '';
            fullAnswer += content;
            const visible = thinkFilter.push(content);
            visibleAnswer += visible;
            if (onChunk && visible) {
                onChunk(visible, visibleAnswer);
            }
        }

        // Flush any remaining buffered content
        const flushed = thinkFilter.flush();
        visibleAnswer += flushed;

        // Final cleanup pass for any remaining think tags
        visibleAnswer = this._stripThinkingTokens(visibleAnswer);

        // Schedule idle auto-suspend after streaming completes
        this._scheduleIdleSuspend();

        return {
            answer: visibleAnswer,
            sources: results,
            metadata: {
                numSources: results.length,
                searchType: normalizedSearchType,
                model: this.modelId,
                scope: scopeMetadata,
                wasStopped: wasStopped,
                contextLimited: contextLimited
            }
        };
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

        if (cleaned.length !== text.length) {
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

            let contextItem = `[${i + 1}] ${text}`;

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
            let parentContext = `\n[Document ${i + 1}]`;

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
        await this.ensureEngineReady();
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

        const completion = await this.engine.chat.completions.create({
            messages,
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
    }

    /**
     * RAG query with per-sentence provenance. Runs the standard `query()` flow,
     * then splits the generated answer into sentence-level "claims" and links
     * each one back to the retrieved sources by maximum text overlap. Confidence
     * is the best per-source overlap score in [0, 1].
     *
     * Returns: { answer, claims:[{claim, supporting_doc_indices, confidence}], sources, metadata }
     */
    async queryWithCitations(question, questionEmbedding, options = {}) {
        const base = await this.query(question, questionEmbedding, options);
        const answer = base?.answer || base?.text || '';
        const sources = base?.sources || base?.results || [];
        const confidenceThreshold = options.confidenceThreshold ?? 0.0;

        const sourceTexts = sources.map(s => s?.text || s?.metadata?.text || '');
        const sourceTokens = sourceTexts.map(t => tokenize(t));

        const sentences = splitSentences(answer);
        const claims = sentences.map(sentence => {
            const claimTokens = tokenize(sentence);
            const supporting = [];
            for (let i = 0; i < sourceTokens.length; i++) {
                const overlap = jaccard(claimTokens, sourceTokens[i]);
                if (overlap > 0) supporting.push({ index: i, overlap });
            }
            supporting.sort((a, b) => b.overlap - a.overlap);
            const top = supporting.slice(0, 3);
            const confidence = top[0]?.overlap || 0;
            return {
                claim: sentence,
                supporting_doc_indices: top
                    .filter(t => t.overlap >= confidenceThreshold)
                    .map(t => sources[t.index]?.index ?? t.index),
                confidence
            };
        }).filter(c => c.confidence >= confidenceThreshold || confidenceThreshold === 0);

        return {
            answer,
            claims,
            sources,
            metadata: base?.metadata || null
        };
    }
}

function splitSentences(text) {
    if (!text) return [];
    // Simple splitter: punctuation followed by whitespace+capital, or newlines.
    const raw = text.split(/(?<=[.!?])\s+(?=[A-ZÆØÅ])|\n+/);
    return raw.map(s => s.trim()).filter(s => s.length > 0);
}

function tokenize(text) {
    if (!text) return new Set();
    return new Set(
        text.toLowerCase()
            .replace(/[^a-z0-9æøåäöü\s]/gi, ' ')
            .split(/\s+/)
            .filter(t => t.length > 2)
    );
}

function jaccard(a, b) {
    if (!a.size || !b.size) return 0;
    let inter = 0;
    for (const t of a) if (b.has(t)) inter++;
    const union = a.size + b.size - inter;
    return union === 0 ? 0 : inter / union;
}
