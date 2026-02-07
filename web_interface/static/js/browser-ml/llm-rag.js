/**
 * Browser-based RAG (Retrieval-Augmented Generation) using WebLLM
 * Model: gemma-2-2b-it-q4f32_1-MLC-1k
 * Runs Gemma 2 2B locally in browser via WebGPU
 */

import { CreateWebWorkerMLCEngine, prebuiltAppConfig } from "https://esm.run/@mlc-ai/web-llm";
import { getModelConstraints } from "../model-constraints.js";

// Load cached real download sizes from previous downloads
try {
    const cached = localStorage.getItem('vectoria_model_download_sizes');
    window.__webllmRealDownloadSizes = cached ? JSON.parse(cached) : {};
} catch (_) {
    window.__webllmRealDownloadSizes = {};
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

        // Load saved configuration
        const savedConfig = this.loadSavedConfig();

        // Load model ID from saved config or use default
        this.modelId = savedConfig.model_id || "gemma-2-2b-it-q4f32_1-MLC";

        // Get model constraints
        this.modelConstraints = getModelConstraints(this.modelId);

        this.isInitialized = false;

        // Load LLM generation parameters from saved config
        this.temperature = savedConfig.temperature || 0.5;
        this.maxTokens = savedConfig.max_tokens || 1024;
        this.topP = savedConfig.top_p || 0.9;
        this.repeatPenalty = savedConfig.repeat_penalty || 1.15;
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

        // Try to interrupt the WebLLM engine directly
        if (this.engine && typeof this.engine.interruptGenerate === 'function') {
            this.engine.interruptGenerate();
        }

        if (this.currentGenerationReject) {
            this.currentGenerationReject(new Error('Generation stopped by user'));
        }
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
        if (this.needsReinit) {
            await this.reinitializeEngine();
        }
        if (!this.isInitialized) {
            throw new Error('LLM not initialized. Call initialize() first.');
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
            this.engine = await CreateWebWorkerMLCEngine(this.worker, this.modelId, {
                initProgressCallback: (progress) => {
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
                // Set context window size
                context_window_size: this.maxContextLength,
                appConfig: {
                    ...prebuiltAppConfig,
                    cache: {
                        ...(prebuiltAppConfig?.cache || {}),
                        enabled: true,
                        storageType: "indexeddb"
                    }
                }
            });

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
                try {
                    this.worker.terminate();
                } catch (_) { /* noop */ }
                this.worker = null;
            }
            console.error('❌ Failed to initialize LLM:', error);
            throw new Error(`LLM initialization failed: ${error.message}`);
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

            return result;
        } catch (error) {
            console.error('❌ Answer generation failed:', error);
            throw new Error(`Failed to generate answer: ${error.message}`);
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
}
