/**
 * Browser ML Orchestration Module
 * Main entry point for browser-based ML pipeline
 * Coordinates all browser ML modules and provides unified API
 */

import { BrowserEmbeddings } from './embeddings.js';
import { BrowserVectorSearch, BM25Search } from './vector-search.js';
import { BrowserRAG } from './llm-rag.js';
import { BrowserFileProcessor } from './file-processor.js';
import { BrowserClustering, terminatePyodideWorker } from './clustering.js';
import { BrowserStorage } from './storage.js';
import { chunkDocuments } from './chunking/chonkieChunker.js';
import { embedChunks, buildChunkIndex, groupChunksByParent } from './embedding/tier3ChunkEmbeddings.js';
import { AnalysisService, cleanLabel } from './analysis.js';
import { AnnotationsStore } from './annotations-store.js';
import { MetricsRegistry } from './metrics-registry.js';
import { SessionsStore, hashCanonical as _hashCanonical } from './sessions-store.js';
import { ChatConversationStore } from './chat-store.js';
import {
    buildDocumentHelperReply,
    buildDocumentChatPrompt,
    buildChatRetrievalQueries,
    normalizeLocalAIError,
    runChatGenerationWithRecovery,
    routeChatTurn,
    sanitizeCitationBounds
} from './chat-context.js';
import {
    createMetadataFilterScope,
    mergeMetadataFilters,
    normalizeMetadataFilters,
    serializeMetadataFilterScope
} from './metadata-filters.js';
import { BrowserReranker } from './reranker.js';
import { rerankAndDiversify } from './retrieval-ranking.js';

function applyCitationSafety(result, sources) {
    if (result?.metadata?.wasStopped) return result;
    const checked = sanitizeCitationBounds(result?.answer || '', sources?.length || 0);
    return {
        ...result,
        answer: checked.answer,
        citations: checked.citations,
        metadata: {
            ...(result?.metadata || {}),
            groundingMode: 'llm',
            validationMethod: 'citation_bounds_only',
            invalidCitationCount: checked.invalidCitations.length
        }
    };
}

export class BrowserMLPipeline {
    constructor() {
        // Initialize all modules
        this.embeddings = new BrowserEmbeddings();
        this.vectorSearch = new BrowserVectorSearch(384); // paraphrase-multilingual-MiniLM-L12-v2 dimension
        this.bm25Search = new BM25Search();
        this.reranker = new BrowserReranker();
        this.lastSearchDiagnostics = null;
        this.rag = null; // Will be initialized after vector search is ready
        this.fileProcessor = new BrowserFileProcessor();
        this.clustering = new BrowserClustering();
        this.storage = new BrowserStorage();

        // Tier 3: Chunk-based retrieval for RAG
        this.chunkVectorSearch = null; // Initialized during processFile
        this.chunkBM25Search = null;
        this.chunkToParentMap = null;

        // State
        this.isInitialized = false;
        this.currentDataset = null;
        this.currentDatasetId = null;
        this.mcpMetadataFilters = {};

        // Processing state
        this.isProcessing = false;
        this.processingProgress = 0;

        // Anti-throttle state
        this._wakeLock = null;
        this._keepaliveInterval = null;

        // Analysis-layer state (annotations, custom labels, metrics, sessions, subsets)
        this.annotations = new Map();
        this.customClusterLabels = new Map(); // clusterId(int) → {label, source, updated_at}
        this.registeredMetrics = new Map();
        this.analysisSessions = [];
        this.subsets = new Map();

        this.analysis       = new AnalysisService(this);
        this.annotationsApi = new AnnotationsStore(this);
        this.metrics        = new MetricsRegistry(this);
        this.sessions       = new SessionsStore(this);
        this.chat           = new ChatConversationStore(this.storage.chatStore);
    }

    /**
     * Set or update a cluster's display label. Mirrors into the existing
     * window.setClusterNames map so the WebGL viz tooltip / legend pick it up
     * for free, and dispatches a DOM event so other UI can react.
     */
    setCustomClusterLabel(clusterId, label, source = 'mcp') {
        const cid = Number(clusterId);
        if (!Number.isFinite(cid) || typeof label !== 'string') return false;
        // Strip markdown/quotes an AI may wrap the label in (e.g. "**Topic**").
        const cleaned = cleanLabel(label);
        if (!cleaned) return false;
        const entry = { label: cleaned, source, updated_at: Date.now() };
        this.customClusterLabels.set(cid, entry);

        if (typeof window !== 'undefined') {
            try {
                const current = typeof window.getClusterNames === 'function' ? (window.getClusterNames() || {}) : {};
                const next = { ...current, [cid]: entry.label };
                if (typeof window.setClusterNames === 'function') {
                    window.setClusterNames(next);
                }
                window.__clusterLabelCache = null;
                document.dispatchEvent(new CustomEvent('vectoria:cluster-labels-changed', {
                    detail: { cluster_id: cid, label: entry.label, source }
                }));
            } catch (e) {
                console.warn('cluster label propagation failed:', e.message);
            }
        }
        return true;
    }

    getCustomClusterLabel(clusterId) {
        return this.customClusterLabels.get(Number(clusterId)) || null;
    }

    async hashCanonical(obj) { return _hashCanonical(obj); }

    setMcpMetadataFilters(filters = {}) {
        if (!this.currentDataset) throw new Error('No dataset loaded');
        this.mcpMetadataFilters = normalizeMetadataFilters(filters, this.currentDataset.documents);
        if (typeof window !== 'undefined') window.currentMetadataFilters = this.mcpMetadataFilters;
        return this.createMetadataFilterScope({}, { includePersistent: true });
    }

    clearMcpMetadataFilters() {
        this.mcpMetadataFilters = {};
        if (typeof window !== 'undefined') window.currentMetadataFilters = {};
    }

    resolveMetadataFilters(inlineFilters = {}, { includePersistent = false } = {}) {
        const documents = this.currentDataset?.documents || [];
        const inline = normalizeMetadataFilters(inlineFilters || {}, documents);
        return includePersistent
            ? mergeMetadataFilters(this.mcpMetadataFilters, inline)
            : inline;
    }

    createMetadataFilterScope(inlineFilters = {}, { includePersistent = false } = {}) {
        const documents = this.currentDataset?.documents || [];
        const effectiveFilters = this.resolveMetadataFilters(inlineFilters, { includePersistent });
        return createMetadataFilterScope(documents, effectiveFilters);
    }

    serializeMetadataFilterScope(scope) {
        return serializeMetadataFilterScope(scope);
    }

    combineAllowedDocIds(allowedDocIds, filterScope) {
        const requested = allowedDocIds === null || allowedDocIds === undefined
            ? null
            : new Set((allowedDocIds instanceof Set ? [...allowedDocIds] : [].concat(allowedDocIds)).map(String));
        const filtered = filterScope?.applied
            ? new Set(filterScope.indices.map(index => {
                const document = this.currentDataset?.documents?.[index];
                return String(document?.id ?? document?.doc_id ?? document?.metadata?.doc_id ?? document?.metadata?.id ?? index);
            }))
            : null;

        if (!requested && !filtered) return null;
        if (!requested) return [...filtered];
        if (!filtered) return [...requested];
        return [...requested].filter(id => filtered.has(id));
    }

    /**
     * ANTI-THROTTLE: Request Screen Wake Lock to prevent browser throttling
     * This keeps the screen/GPU active even in fullscreen mode on macOS
     */
    async _requestWakeLock() {
        if (this._wakeLock) return;

        try {
            if ('wakeLock' in navigator) {
                this._wakeLock = await navigator.wakeLock.request('screen');
                this._wakeLock.addEventListener('release', () => {
                    this._wakeLock = null;
                });
            } else {
                this._startKeepalivePing();
            }
        } catch (err) {
            console.warn('⚠️ Wake lock request failed:', err.message);
            this._startKeepalivePing();
        }
    }

    /**
     * ANTI-THROTTLE: Release wake lock when processing is complete
     */
    _releaseWakeLock() {
        if (this._wakeLock) {
            this._wakeLock.release();
            this._wakeLock = null;
        }
        this._stopKeepalivePing();
    }

    /**
     * ANTI-THROTTLE: Keepalive ping fallback for browsers without Wake Lock API
     * Uses periodic DOM access to prevent throttling
     */
    _startKeepalivePing() {
        if (this._keepaliveInterval) return;

        // Perform minimal DOM operation every 500ms to prevent throttling
        this._keepaliveInterval = setInterval(() => {
            // Minimal DOM read to keep the main thread "active"
            if (typeof document !== 'undefined') {
                const _ = document.hidden;
            }
        }, 500);

    }

    /**
     * ANTI-THROTTLE: Stop keepalive ping
     */
    _stopKeepalivePing() {
        if (this._keepaliveInterval) {
            clearInterval(this._keepaliveInterval);
            this._keepaliveInterval = null;
        }
    }

    /**
     * Initialize all ML models
     * @param {Object} callbacks - Progress callbacks
     */
    async initialize(callbacks = {}, options = {}) {
        if (this.isInitialized) {
            return;
        }

        const {
            onEmbeddingsProgress = null,
            onLLMProgress = null,
            onComplete = null
        } = callbacks;
        const {
            deferModels = false,
            releaseAfterInitialize = false
        } = options;

        try {
            // The RAG coordinator is cheap and must exist even while the actual
            // model runtime remains cold. This lets cached installations restore
            // instantly without allocating WebLLM GPU/RAM on every page load.
            if (!this.rag) this.rag = new BrowserRAG(this.vectorSearch, null, null, this.reranker);
            else this.rag.setReranker(this.reranker);
            if (this.chunkVectorSearch) {
                this.rag.setChunkVectorSearch(this.chunkVectorSearch);
            }
            if (this.chunkBM25Search) {
                this.rag.setBM25Search(this.chunkBM25Search);
            }

            if (!deferModels) {
                // Initial setup has to instantiate each runtime once so its files
                // are downloaded into browser storage. Callers may release both
                // runtimes immediately afterward while keeping those files cached.
                await this.embeddings.initialize(onEmbeddingsProgress);
                // Release embeddings before WebLLM starts so even the one-time
                // download flow never keeps both model runtimes resident together.
                if (releaseAfterInitialize) this.embeddings.suspendWorker('models-cached');

                await this.rag.initialize(onLLMProgress);
                if (releaseAfterInitialize && !this.rag.workerUnloaded) {
                    this.rag.suspendWorker('models-cached');
                }
            }

            this.isInitialized = true;

            if (onComplete) {
                onComplete();
            }

            return true;
        } catch (error) {
            console.error('❌ Pipeline initialization failed:', error);
            throw error;
        }
    }

    /**
     * Process uploaded file end-to-end
     * @param {File} file - File to process
     * @param {string} textColumn - Column containing text data
     * @param {Object} options - Processing options
     * @param {Function} onProgress - Progress callback
     */
    async processFile(file, textColumn, options = {}, onProgress = null) {
        if (!this.isInitialized) {
            throw new Error('Pipeline not initialized. Call initialize() first.');
        }
        if (this.rag?.activeOperation) {
            const active = this.rag.activeOperation;
            throw new Error(`Cannot process a new file while local AI is ${this.rag._operationLabel(active.owner)}. Stop or finish that task first.`);
        }

        this.isProcessing = true;
        this.processingProgress = 0;

        // Track timing for each stage
        const timings = {
            start: Date.now(),
            parsing: 0,
            embedding: 0,
            indexing: 0,
            umap: 0,
            clustering: 0,
            saving: 0,
            total: 0
        };

        const updateProgress = (stage, progress, message) => {
            this.processingProgress = progress;
            if (onProgress) {
                onProgress({ stage, progress, message });
            }
        };

        // ANTI-THROTTLE: Request wake lock for entire processing pipeline
        await this._requestWakeLock();

        // A connected client may have requested generation while the wake-lock
        // promise was pending. Never terminate that request underneath it.
        if (this.rag?.activeOperation) {
            const active = this.rag.activeOperation;
            this.isProcessing = false;
            this._releaseWakeLock();
            throw new Error(`Cannot process a new file while local AI is ${this.rag._operationLabel(active.owner)}. Stop or finish that task first.`);
        }

        // MEMORY: Suspend LLM worker during heavy processing (frees 1-3GB GPU/RAM).
        // Worker will lazy-reload on first RAG/HyDE query after processing.
        // Skip if LLM was permanently unloaded (workerUnloaded flag).
        if (this.rag && typeof this.rag.suspendWorker === 'function' && !this.rag.workerUnloaded) {
            this.rag.suspendWorker('processing-file');
        }

        try {
            // Cached setup deliberately leaves model runtimes out of memory.
            // Processing needs embeddings, but the much larger chat LLM stays
            // cold for the entire processing pipeline and loads on first use.
            updateProgress('models', 0.01, 'Preparing cached embeddings model...');
            await this.embeddings.initialize((modelProgress) => {
                const ratio = Math.min(1, Math.max(0, Number(modelProgress?.progress) || 0));
                updateProgress('models', 0.01 + (ratio * 0.03), 'Preparing cached embeddings model...');
            });

            // 1. Parse file (10%)
            const parseStart = Date.now();
            updateProgress('parsing', 0.05, 'Parsing file...');
            const parsedData = await this.fileProcessor.parseFile(file);
            timings.parsing = (Date.now() - parseStart) / 1000;
            updateProgress('parsing', 0.10, `Parsed ${parsedData.rowCount} rows`);

            // 2. Extract documents (15%)
            updateProgress('extracting', 0.12, 'Extracting documents...');
            const { documents: allDocuments, emptyRowCount } = this.fileProcessor.extractDocuments(parsedData.data, textColumn);

            // Filter out documents with empty text
            let documents = allDocuments.filter(doc => !doc.hasEmptyText);
            const { documents: uniqueDocuments, duplicateCount } = this._deduplicateDocuments(documents);
            documents = uniqueDocuments;
            const duplicatesRemoved = duplicateCount;

            const emptyMessage = `${emptyRowCount} dropped due to empty text`;
            const duplicateMessage = duplicatesRemoved > 0 ? `, ${duplicatesRemoved} duplicates removed` : '';
            updateProgress('extracting', 0.15, `Prepared ${documents.length} documents (${emptyMessage}${duplicateMessage})`);

            // 3. Generate embeddings - 3-TIER STRATEGY (20% → 50%)
            const embeddingStart = Date.now();
            let texts = documents.map(doc => doc.text);

            // TIER 1: Parent summaries for clustering/visualization (query mode)
            updateProgress('embedding', 0.20, 'Generating parent summaries for visualization...');
            let documentSummaries = texts.map(text => {
                const tokens = text.split(/\s+/);
                if (tokens.length <= 256) return text;
                return tokens.slice(0, 256).join(' ') + '...';
            });
            // Free `texts` — summaries are now derived and we don't need full doc text array here
            // (full text still lives on each `documents[i].text`)
            texts = null;

            // Ensure anti-throttle hacks are active before heavy lifting
            await this.embeddings._requestWakeLock();

            const parentEmbeddings = await this.embeddings.embed(documentSummaries, {
                showProgress: true,
                useCache: true,
                mode: 'query',  // Symmetric similarity for clustering
                maxLength: 256,
                maxTokensPerBatch: options.maxTokensPerBatch,
                onProgress: (embProgress) => {
                    const overallProgress = 0.20 + (embProgress.progress * 0.10);
                    const percent = Math.round((embProgress.progress || 0) * 100);
                    const message = `Parent embeddings batch ${embProgress.batch}/${embProgress.totalBatches} (${percent}%)`;
                    updateProgress('embedding', overallProgress, message);
                }
            });
            updateProgress('embedding', 0.30, 'Parent embeddings complete');

            // MEMORY: Free document summaries — parent embeddings already generated.
            // For 10K docs at ~256 tokens, this releases ~10-20MB of string data.
            documentSummaries = null;

            // --- STAGE BREAK: PREVENT WORKER OVERLOAD ---
            await new Promise(resolve => setTimeout(resolve, 2000)); // 2s pause
            
            // Force worker restart if it seems groggy (preventive maintenance)
            // This is handled automatically by the robust _embedWithWorker now,
            // but a pause here helps the browser reclaim resources.

            // TIER 2: Full document text stored (no embedding needed for display)
            // Documents already have full text - will be used for text list/viewer

            // TIER 3: Chunk documents for RAG retrieval
            // Read chunking config from ConfigManager
            const config = window.ConfigManager ? window.ConfigManager.getConfig() : {};
            const chunkConfig = config.chunking || {};

            let chunks, chunkToParentMap;

            if (chunkConfig.enabled === false) {
                // Skip chunking - treat each document as single chunk
                updateProgress('embedding', 0.30, 'Chunking disabled - using full documents...');
                chunks = documents.map(doc => ({
                    chunk_id: `${doc.id}_chunk_0`,
                    parent_id: doc.id,
                    text: doc.text,
                    position: 0,
                    totalChunks: 1,
                    metadata: { ...doc.metadata, parent_id: doc.id, chunk_position: '1/1' }
                }));
                chunkToParentMap = {};
                chunks.forEach(c => chunkToParentMap[c.chunk_id] = c.parent_id);
                this.chunkToParentMap = chunkToParentMap;
            } else {
                // Use configured chunking options
                updateProgress('embedding', 0.30, 'Chunking documents with ChonkieJS...');
                const chunkingOptions = {
                    strategy: chunkConfig.strategy || 'token',
                    chunkSize: chunkConfig.chunk_size ?? 512,
                    chunkOverlap: chunkConfig.chunk_overlap ?? 128,
                    minChunkSize: chunkConfig.min_chunk_size ?? 50,
                    sentenceMinSentences: chunkConfig.sentence_min_sentences || 1,
                    sentenceMinCharacters: chunkConfig.sentence_min_characters || 12,
                    sentenceDelimiters: chunkConfig.sentence_delimiters || ['. ', '! ', '? ', '\n'],
                    sentenceIncludeDelimiter: chunkConfig.sentence_include_delimiter || 'prev',
                    semanticEmbeddings: textsToEmbed => this.embeddings.embed(textsToEmbed, {
                        mode: 'passage',
                        showProgress: false,
                        useCache: true,
                        maxLength: config.embeddings?.max_length || 256
                    }),
                    semanticThreshold: chunkConfig.semantic_threshold ?? 0.8,
                    semanticSimilarityWindow: chunkConfig.semantic_similarity_window ?? 3,
                    semanticFilterWindow: chunkConfig.semantic_filter_window ?? 5,
                    semanticFilterPolyorder: chunkConfig.semantic_filter_polyorder ?? 3,
                    semanticFilterTolerance: chunkConfig.semantic_filter_tolerance ?? 0.2,
                    semanticSkipWindow: chunkConfig.semantic_skip_window ?? 0,
                    codeLanguage: chunkConfig.code_language || 'auto',
                    tableMode: chunkConfig.table_mode || 'row',
                    tableRowsPerChunk: chunkConfig.table_rows_per_chunk ?? 10,
                    fastDelimiters: chunkConfig.fast_delimiters || '\n.?',
                    fastPrefix: chunkConfig.fast_prefix === true,
                    fastConsecutive: chunkConfig.fast_consecutive === true,
                    fastForwardFallback: chunkConfig.fast_forward_fallback !== false
                };
                const result = await chunkDocuments(documents, chunkingOptions);
                chunks = result.chunks;
                chunkToParentMap = result.chunkToParentMap;
                this.chunkToParentMap = chunkToParentMap;
            }
            updateProgress('embedding', 0.35, `Created ${chunks.length} chunks from ${documents.length} documents`);

            // Validate chunk size vs embedding max_length
            const embeddingMaxLength = config.embeddings?.max_length || 256;
            const effectiveChunkSize = chunkConfig.chunk_size || 512;
            const estimatedTokens = Math.ceil(effectiveChunkSize / 4); // ~4 chars per token
            if (chunkConfig.enabled !== false && estimatedTokens > embeddingMaxLength) {
                console.warn(`⚠️ Chunk size (${effectiveChunkSize} chars ≈ ${estimatedTokens} tokens) exceeds embedding max_length (${embeddingMaxLength}). Some chunk text may be truncated.`);
            }

            // Embed chunks in passage mode for asymmetric RAG retrieval
            updateProgress('embedding', 0.35, 'Generating chunk embeddings (passage mode)...');
            const embeddedChunks = await embedChunks(chunks, this.embeddings, {
                onProgress: (embProgress) => {
                    const overallProgress = 0.35 + (embProgress.progress * 0.15);
                    updateProgress('embedding', overallProgress, embProgress.message);
                }
            });

            timings.embedding = (Date.now() - embeddingStart) / 1000;
            updateProgress('embedding', 0.50, 'All embeddings complete');

            const embeddings = {
                parent: parentEmbeddings,           // Tier 1: for viz/clustering (query mode)
                chunks: embeddedChunks,             // Tier 3: for RAG retrieval (passage mode)
                chunkToParentMap: chunkToParentMap,
                model: this.embeddings.modelName,
                dimension: this.embeddings.dimension,
                schema: 'three-tier-v1',
                modes: {
                    parent: 'query',    // Symmetric similarity
                    chunks: 'passage'   // Asymmetric retrieval
                }
            };

            // 4. Build vector indexes
            const indexingStart = Date.now();
            updateProgress('indexing', 0.50, 'Processing file...');
            const docIds = documents.map(doc => doc.id);

            // Build parent document index for visualization (not used for search)
            await this.vectorSearch.buildIndex(embeddings.parent, docIds, documents);
            updateProgress('indexing', 0.51, 'Processing file...');

            // Build chunk index for RAG retrieval
            this.chunkVectorSearch = buildChunkIndex(embeddings.chunks, BrowserVectorSearch);
            updateProgress('indexing', 0.51, 'Processing file...');

            // Update RAG with chunk index
            if (this.rag) {
                this.rag.setChunkVectorSearch(this.chunkVectorSearch);
            }

            // 5. Build BM25 indexes
            // Parent documents BM25 (for UI search)
            this.bm25Search.buildIndex(documents, docIds);
            updateProgress('indexing', 0.52, 'Processing file...');

            // Chunk BM25 for hybrid RAG retrieval
            const chunkBM25Search = new BM25Search();
            let chunkDocs = chunks.map(chunk => ({
                id: chunk.chunk_id,
                text: chunk.text,
                metadata: chunk.metadata
            }));
            let chunkIds = chunks.map(c => c.chunk_id);
            chunkBM25Search.buildIndex(chunkDocs, chunkIds);
            this.chunkBM25Search = chunkBM25Search;
            updateProgress('indexing', 0.52, 'Processing file...');

            // Update RAG with BM25 chunk search for hybrid retrieval
            if (this.rag) {
                this.rag.setBM25Search(chunkBM25Search);
            }

            // MEMORY: Free intermediate chunk references — they've been absorbed by
            // chunkVectorSearch (embeddings + text) and chunkBM25Search (text + ids).
            // Raw `chunks` array and the derived chunkDocs/chunkIds are no longer needed.
            // `embeddings.chunks` (the EmbeddedChunkRecord array) is still referenced — we keep
            // it because storage.saveDataset persists it and getVisualizationData reads it.
            chunkDocs = null;
            chunkIds = null;
            chunks = null;

            timings.indexing = (Date.now() - indexingStart) / 1000;

            // 6. Transition to UMAP stage
            updateProgress('umap', 0.52, 'Starting dimensionality reduction...');

            // Brief pause to ensure UI shows the transition
            await new Promise(resolve => setTimeout(resolve, 100));

            // 7. Compute UMAP (clustering dimensions + 2D for visualization)
            const umapStart = Date.now();

            // Force immediate display of 0s
            updateProgress('umap', 0.54, 'Computing UMAP (0s)');
            await new Promise(resolve => setTimeout(resolve, 50));

            // Set up a timer to force elapsed time updates every 300ms
            let umapElapsedTimer = setInterval(() => {
                const elapsed = Math.floor((Date.now() - umapStart) / 1000);
                const progress = Math.min(0.87, 0.54 + (elapsed / 10.0) * 0.34);
                updateProgress('umap', progress, `Computing UMAP (${elapsed}s)`);
            }, 300);

            // First: ND UMAP for clustering - dimensions configurable via settings
            // Use parent embeddings (query mode) which capture document-level themes
            const clusteringProjection = await this.clustering.computeClusteringUMAP(embeddings.parent, {
                nNeighbors: 15,
                minDist: 0.0,  // Dense clusters
                // nComponents is read from config inside computeClusteringUMAP
                onProgress: (prog) => {
                    const totalElapsed = Math.floor((Date.now() - umapStart) / 1000);
                    const progress = 0.54 + (prog.progress || 0) * 0.18;
                    updateProgress('umap', progress, `Computing UMAP (${totalElapsed}s)`);
                }
            });

            const umapClusteringTime = (Date.now() - umapStart) / 1000;

            // Second: 2D UMAP for visualization (72% -> 88%)
            const projection = await this.clustering.computeVisualizationUMAP(embeddings.parent, {
                nNeighbors: 15,
                minDist: 0.1,
                nComponents: 2,
                onProgress: (prog) => {
                    const totalElapsed = Math.floor((Date.now() - umapStart) / 1000);
                    const progress = 0.72 + (prog.progress || 0) * 0.16;
                    updateProgress('umap', progress, `Computing UMAP (${totalElapsed}s)`);
                }
            });

            // Clear the elapsed timer
            clearInterval(umapElapsedTimer);

            const totalUmapTime = (Date.now() - umapStart) / 1000;
            const totalUmapElapsed = Math.floor(totalUmapTime);
            timings.umap = totalUmapTime;
            updateProgress('umap', 0.88, `UMAP complete (${totalUmapElapsed}s)`);

            // 9. Compute HDBSCAN clusters on ND projection (88% → 92%)
            const clusteringStart = Date.now();
            updateProgress('clustering', 0.89, 'Running HDBSCAN clustering (0s)');
            const clusteringDims = clusteringProjection[0]?.length || 0;
            // Pass HDBSCAN options if provided, otherwise let clustering module use saved config
            const clusteringOptions = {};
            if (options.minClusterSize !== undefined) {
                clusteringOptions.minClusterSize = options.minClusterSize;
            }
            if (options.minSamples !== undefined) {
                clusteringOptions.minSamples = options.minSamples;
            }

            const documentTexts = documents.map((doc) => {
                if (!doc) return '';
                if (typeof doc === 'string') return doc;
                if (typeof doc.text === 'string') return doc.text;
                if (doc.content && typeof doc.content === 'string') return doc.content;
                if (doc.metadata) {
                    if (typeof doc.metadata.cleaned_text === 'string') return doc.metadata.cleaned_text;
                    if (typeof doc.metadata.text === 'string') return doc.metadata.text;
                }
                return '';
            });
            const hasAnyText = documentTexts.some(text => typeof text === 'string' && text.trim().length > 0);
            if (hasAnyText) {
                clusteringOptions.documents = documentTexts;
                clusteringOptions.keywordOptions = {
                    metadata_top_n: 10,
                    viz_top_n: 3,
                    min_df: 1
                };
            }

            // Since computeClusters doesn't have progress callbacks, track elapsed time
            const clusteringPromise = this.clustering.computeClusters(clusteringProjection, clusteringOptions);

            // Real-time elapsed time updates for HDBSCAN (every 500ms)
            let clusteringProgressSimulator = setInterval(() => {
                const elapsed = Math.floor((Date.now() - clusteringStart) / 1000);
                // Smooth progress curve based on elapsed time
                const estimatedProgress = Math.min(0.91, 0.89 + (elapsed / 10.0) * 0.03);
                updateProgress('clustering', estimatedProgress, `Running HDBSCAN clustering (${elapsed}s)`);
            }, 500);

            const clusters = await clusteringPromise;
            clearInterval(clusteringProgressSimulator);

            timings.clustering = (Date.now() - clusteringStart) / 1000;
            const clusteringElapsed = Math.floor(timings.clustering);
            updateProgress('clustering', 0.92, `HDBSCAN complete (${clusteringElapsed}s)`);

            // 9b. Add cluster information and probabilities to document metadata
            const probabilities = this.clustering.getProbabilities();
            const clusterKeywordsMap = this.clustering.getClusterKeywords();
            const clusterKeywordScoresMap = this.clustering.getClusterKeywordScores();
            const clusterKeywordsVizMap = this.clustering.getClusterKeywordsViz();

            documents.forEach((doc, idx) => {
                if (!doc.metadata) {
                    doc.metadata = {};
                }

                const clusterId = clusters[idx];
                doc.metadata.cluster = clusterId;
                doc.metadata.cluster_label = clusterId === -1 ? 'Outlier' : `Cluster ${clusterId}`;

                if (probabilities && probabilities[idx] !== undefined) {
                    doc.metadata.cluster_probability = probabilities[idx];
                }

                let clusterKeywords = [];
                let clusterKeywordScores = [];
                let clusterKeywordsViz = [];

                if (clusterId !== -1) {
                    if (clusterKeywordsMap && clusterKeywordsMap.has(clusterId)) {
                        clusterKeywords = clusterKeywordsMap.get(clusterId).slice(0, 10);
                    }

                    if (clusterKeywordScoresMap && clusterKeywordScoresMap.has(clusterId)) {
                        clusterKeywordScores = clusterKeywordScoresMap
                            .get(clusterId)
                            .slice(0, 10)
                            .map(item => ({
                                keyword: item.keyword,
                                score: item.score
                            }));
                    }

                    if (clusterKeywordsVizMap && clusterKeywordsVizMap.has(clusterId)) {
                        clusterKeywordsViz = clusterKeywordsVizMap.get(clusterId).slice(0, 3);
                    } else if (clusterKeywords.length > 0) {
                        clusterKeywordsViz = clusterKeywords.slice(0, 3);
                    }
                }

                doc.metadata.cluster_keywords = clusterKeywords;
                doc.metadata.cluster_keyword_scores = clusterKeywordScores;
                doc.metadata.cluster_keywords_viz = clusterKeywordsViz;

                doc.cluster_keywords = clusterKeywords;
                doc.cluster_keyword_scores = clusterKeywordScores;
                doc.cluster_keywords_viz = clusterKeywordsViz;
            });
            // Log cluster statistics
            const clusterStats = this.clustering.getClusterStats(clusters);
            const clusterKeywordData = this.clustering.getClusterKeywordData();

            // 10. Save to storage (95%)
            const savingStart = Date.now();
            updateProgress('saving', 0.93, 'Processing file...');
            const datasetId = `dataset_${Date.now()}`;
            await this.storage.saveDataset(datasetId, {
                embeddings: embeddings,
                vectorIndex: this.vectorSearch.serialize(),
                documents: documents,
                projection: projection,  // 2D for visualization
                clusteringProjection: clusteringProjection,  // ND for clustering
                clusters: clusters,
                fileName: file.name,
                fileType: parsedData.fileType,
                textColumn: textColumn,
                emptyRowCount: emptyRowCount,
                duplicateCount: duplicatesRemoved,
                clusterKeywords: clusterKeywordData
            });
            timings.saving = (Date.now() - savingStart) / 1000;
            updateProgress('saving', 0.95, 'Processing file...');

            // 11. Store current dataset
            if (typeof window !== 'undefined') window.clearActiveClusterLabels?.();
            this.clearMcpMetadataFilters();
            this.currentDataset = {
                id: datasetId,
                fileName: file.name,
                fileType: parsedData.fileType,
                textColumn: textColumn,
                documents: documents,
                embeddings: embeddings,
                retrievalEmbeddings: embeddings.retrieval || null,
                clusteringEmbeddings: embeddings.clustering || embeddings.retrieval || null,
                projection: projection,  // 2D for visualization
                clusteringProjection: clusteringProjection,  // ND for clustering
                clusters: clusters,
                numDocuments: documents.length,
                emptyRowCount: emptyRowCount,
                duplicateCount: duplicatesRemoved,
                clusterKeywords: clusterKeywordData
            };
            this.currentDatasetId = datasetId;

            if (typeof document !== 'undefined') {
                document.dispatchEvent(new CustomEvent('vectoria:dataset-changed', {
                    detail: { datasetId, reason: 'processed' }
                }));
            }

            updateProgress('complete', 1.0, 'Processing complete!');

            // Calculate total time
            timings.total = (Date.now() - timings.start) / 1000;

            this.isProcessing = false;
            // Release wake lock on successful completion
            this._releaseWakeLock();

            // MEMORY: Terminate Pyodide worker now that HDBSCAN is done.
            // Python runtime + numpy/scipy/sklearn holds ~200-500MB. Worker will lazy-reload
            // on next HDBSCAN call (only happens when processing a new file).
            // NOTE: Embedding worker is NOT suspended here — it's needed for every search
            // and RAG query that follows, so re-initializing would add 3-8s latency per query.
            // Auto-idle-suspend handles longer idle periods.
            try { terminatePyodideWorker(); } catch (e) { console.warn('Pyodide teardown failed:', e); }

            // MEMORY: Free the clustering instance's intermediate arrays. The N×15 ND
            // projection and N×2 viz projection (plus labels/probabilities) are already
            // captured on currentDataset and persisted to IndexedDB. The clustering
            // instance survives across runs and would otherwise keep these forever.
            try {
                this.clustering.clusteringProjection = null;
                this.clustering.visualizationProjection = null;
                this.clustering.labels = null;
                this.clustering.probabilities = null;
            } catch (_) {}

            // MEMORY: Clear the embedding cache — it accumulated keys + Float32Arrays
            // during corpus embedding. The cache only helps repeated identical queries
            // (rare) and is opportunistic; clearing forces at most one re-embed on
            // future duplicate queries.
            try { this.embeddings.clearCache(); } catch (_) {}
            const result = {
                datasetId: datasetId,
                numDocuments: documents.length,
                numClusters: this.clustering.getClusterStats(clusters)?.numClusters || 0,
                emptyRowCount: emptyRowCount,
                duplicateCount: duplicatesRemoved,
                fileName: file.name,
                textColumn: textColumn,
                timings: timings,
                visualization: {
                    projection: projection,
                    clusters: clusters
                },
                clusterKeywords: clusterKeywordData.cluster_keywords
            };

            return result;

        } catch (error) {
            this.isProcessing = false;
            // Release wake lock on error
            this._releaseWakeLock();
            // MEMORY: tear down Pyodide on error path too — it may be partially loaded
            try { terminatePyodideWorker(); } catch (_) {}
            console.error('❌ File processing failed:', error);
            throw error;
        }
    }

    /**
     * Perform search
     * @param {string} query - Search query
     * @param {Object} options - Search options
     */
    async search(query, options = {}) {
        if (!this.currentDataset) {
            throw new Error('No dataset loaded');
        }

        const {
            searchType = 'fast', // 'fast' (keyword/BM25) or 'semantic'
            k = 10,
            minScore = 0.0,
            filter = null,
            vectorWeight = 0.6,
            rerankerEnabled = undefined,
            signal = null,
            onRerankerProgress = null
        } = options;

        const normalizedType = (searchType || 'fast').toLowerCase();
        let results;
        const configRerankerEnabled = typeof window !== 'undefined'
            && window.ConfigManager?.getConfig?.()?.search?.reranker_enabled === true;
        const useReranker = rerankerEnabled === undefined ? configRerankerEnabled : rerankerEnabled === true;

        if (normalizedType === 'keyword' || normalizedType === 'bm25' || normalizedType === 'normal' || normalizedType === 'fast') {
            // Keyword/BM25 search
            results = this.bm25Search.search(query, k, { filter });
        } else if (normalizedType === 'hybrid') {
            const queryEmbedding = await this.embeddings.embedSingle(query, { mode: 'query' });
            const candidateK = useReranker ? Math.max(k * 4, 40) : k;
            const vectorResults = this.vectorSearch.search(queryEmbedding, candidateK, {
                minScore,
                includeMetadata: true,
                filter
            });
            const bm25Results = this.bm25Search.search(query, candidateK, { filter });
            results = fuseSearchResults(vectorResults, bm25Results, {
                k: useReranker ? candidateK : k,
                vectorWeight
            });
        } else {
            // Semantic vector search (default)
            const queryEmbedding = await this.embeddings.embedSingle(query, { mode: 'query' });
            results = this.vectorSearch.search(queryEmbedding, useReranker ? Math.max(k * 4, 40) : k, {
                minScore,
                includeMetadata: true,
                filter
            });
        }

        if (useReranker && normalizedType !== 'fast' && normalizedType !== 'keyword'
            && normalizedType !== 'bm25' && normalizedType !== 'normal') {
            const neural = await this.reranker.rerank(query, results, { signal, onProgress: onRerankerProgress });
            results = neural.results;
            this.lastSearchDiagnostics = neural.diagnostics;
            if (neural.diagnostics.reranker_applied) {
                results = rerankAndDiversify(results, query, { maxResults: k });
                this.lastSearchDiagnostics.post_fusion = 'neural_rrf_coverage_mmr_v1';
            }
        } else {
            this.lastSearchDiagnostics = {
                reranker_applied: false,
                reranker_model: null,
                reranker_candidates: 0,
                reranker_latency_ms: 0,
                reranker_fallback_reason: null
            };
        }

        return results.slice(0, k);
    }

    /**
     * Perform RAG query
     * @param {string} question - Question to answer
     * @param {Object} options - RAG options
     */
    async queryLocalGrounded(question, options = {}) {
        return this.queryRAG(question, options);
    }

    async queryRAG(question, options = {}) {
        if (!this.currentDataset) {
            throw new Error('No dataset loaded');
        }
        if (this.isProcessing) throw new Error('Wait for data processing to finish before using local AI.');

        if (!this.rag) {
            throw new Error('RAG not initialized');
        }

        const datasetRef = this.currentDataset;
        const datasetId = this.currentDatasetId || datasetRef?.id || null;
        const suppliedOperation = options.operationToken || null;
        const operationToken = suppliedOperation || this.rag.beginOperation(options.owner || 'rag', { datasetId });
        const ownsOperation = !suppliedOperation;
        this.rag.assertOperation(operationToken);

        try {
        const {
            numResults = 5,
            searchType = 'semantic',
            temperature,
            maxTokens,
            stream = false,
            onChunk = null,
            includeMetadata = false,
            metadataFields = undefined,
            hydeText = null,  // Optional HyDE text for embedding
            allowedDocIds = null,
            retrievalK = null,
            vectorWeight = undefined,
            similarityThreshold = undefined,
            metadataFilters = {},
            includePersistentFilters = false
        } = options;
        const filterScope = this.createMetadataFilterScope(metadataFilters, {
            includePersistent: includePersistentFilters
        });
        const effectiveAllowedDocIds = this.combineAllowedDocIds(allowedDocIds, filterScope);

        // Generate question embedding (use HyDE text if provided, otherwise original question)
        const textToEmbed = hydeText || question;
        const questionEmbedding = await this.embeddings.embedSingle(textToEmbed, { mode: 'query' });

        if (filterScope.applied && filterScope.matchedDocuments === 0) {
            return {
                answer: '',
                sources: [],
                metadata: {
                    generationProvider: 'local',
                    filter: this.serializeMetadataFilterScope(filterScope)
                }
            };
        }

        let result;
        if (stream && onChunk) {
            result = await this.rag.queryStream(question, questionEmbedding, onChunk, {
                numResults,
                searchType,
                temperature,
                maxTokens,
                includeMetadata,
                metadataFields,
                allowedDocIds: effectiveAllowedDocIds,
                retrievalK,
                vectorWeight,
                similarityThreshold,
                operationToken,
                owner: options.owner || 'rag',
                onStatus: options.onStatus
            });
        } else {
            // Regular response
            result = await this.rag.query(question, questionEmbedding, {
                numResults,
                searchType,
                temperature,
                maxTokens,
                includeMetadata,
                metadataFields,
                allowedDocIds: effectiveAllowedDocIds,
                retrievalK,
                vectorWeight,
                similarityThreshold,
                operationToken,
                owner: options.owner || 'rag',
                onStatus: options.onStatus
            });
        }
        if (this.currentDataset !== datasetRef || String(this.currentDatasetId || '') !== String(datasetId || '')) {
            throw new Error('The active dataset changed while the RAG question was running.');
        }
        result = applyCitationSafety(result, result.sources || []);
        result.metadata = {
            ...(result.metadata || {}),
            filter: this.serializeMetadataFilterScope(filterScope),
            groundingMode: 'llm'
        };
        return result;
        } finally {
            if (ownsOperation) this.rag.endOperation(operationToken);
        }
    }

    /**
     * Conversational RAG for Vectoria's local browser chat. This path remains
     * intentionally separate from MCP's one-shot query tools.
     */
    async queryChat(question, options = {}) {
        if (!this.currentDataset) throw new Error('No dataset loaded');
        if (this.isProcessing) throw new Error('Wait for data processing to finish before using Ask.');
        const datasetRef = this.currentDataset;
        const datasetId = this.currentDatasetId || datasetRef?.id || null;
        const startedAt = Date.now();
        const config = typeof window !== 'undefined' && window.ConfigManager
            ? window.ConfigManager.getConfig()
            : {};
        const history = options.history || [];
        const route = routeChatTurn(question, { history });
        options.onRouteResolved?.(route);

        if (route.resolved === 'helper') {
            return {
                answer: buildDocumentHelperReply(question, history),
                sources: [],
                citations: [],
                route,
                metadata: {
                    generationProvider: 'router',
                    retrievalPerformed: false,
                    route,
                    topicEligible: false,
                    durationMs: Date.now() - startedAt,
                    wasStopped: false,
                    finishReason: 'stop'
                }
            };
        }

        if (!this.rag) throw new Error('Set up the local AI model before asking a substantive document question.');

        const operationToken = this.rag.beginOperation('chat', { datasetId });
        try {
        const configuredMetadataFields = config.chat?.metadata_fields;
        const scope = options.scope || 'all';
        const numResults = options.numResults ?? config.search?.num_results ?? 5;
        const retrievalK = options.retrievalK ?? config.search?.retrieval_k ?? null;
        const vectorWeight = options.vectorWeight ?? config.search?.vector_weight;
        const similarityThreshold = options.similarityThreshold ?? config.search?.similarity_threshold;
        const configuredMetadataMode = ['off', 'selected', 'all'].includes(config.chat?.metadata_mode)
            ? config.chat.metadata_mode
            : (config.chat?.include_metadata === true
                ? Array.isArray(configuredMetadataFields) && configuredMetadataFields.length ? 'selected' : 'all'
                : 'off');
        const metadataMode = ['off', 'selected', 'all'].includes(options.metadataMode)
            ? options.metadataMode
            : options.includeMetadata !== undefined
                ? (options.includeMetadata ? (Array.isArray(options.metadataFields) ? 'selected' : 'all') : 'off')
                : configuredMetadataMode;
        const includeMetadata = metadataMode !== 'off';
        const metadataFields = metadataMode === 'selected'
            ? (options.metadataFields !== undefined ? options.metadataFields : (configuredMetadataFields || []))
            : undefined;
        const metadataFilters = options.metadataFilters || {};
        const includePersistentFilters = options.includePersistentFilters === true;
        const allowedDocIds = options.allowedDocIds ?? null;
        const onRetrievalComplete = options.onRetrievalComplete || null;
        const onChunk = options.onChunk || null;
        const onStatus = options.onStatus || null;
        const useHyDE = options.useHyDE ?? (config.ui_preferences?.hyde_enabled === true);
        const memoryMode = options.memoryMode || config.chat?.memory_mode || 'adaptive';
        const maxMemoryTurns = options.maxMemoryTurns ?? config.chat?.max_memory_turns ?? 8;
        const temperature = options.temperature ?? config.llm?.temperature;
        const topP = options.topP ?? config.llm?.top_p;
        const repeatPenalty = options.repeatPenalty ?? config.llm?.repeat_penalty;
        const maxOutputTokens = options.maxTokens ?? config.llm?.max_tokens ?? this.rag.maxTokens;
        const contextWindow = options.contextWindow ?? config.llm?.context_window_size ?? this.rag.maxContextLength;
        const systemPrompt = options.systemPrompt ?? config.rag_prompts?.system_prompt ?? this.rag.systemPrompt;
        const userTemplate = options.userTemplate ?? config.rag_prompts?.user_template ?? this.rag.userTemplate;
        const hydePrompt = options.hydePrompt ?? config.hyde?.prompt;
        const hydeTemperature = options.hydeTemperature ?? config.hyde?.temperature;
        const hydeMaxTokens = options.hydeMaxTokens ?? config.hyde?.max_tokens;
        const effectiveSettings = {
            configVersion: options.configVersion ?? config.version ?? null,
            route: { requested: route.requested, resolved: route.resolved },
            retrieval: { numResults, retrievalK, vectorWeight, similarityThreshold, useHyDE },
            generation: { temperature, topP, repeatPenalty, maxOutputTokens, contextWindow },
            memory: { mode: memoryMode, maxTurns: maxMemoryTurns },
            evidence: { metadataMode, includeMetadata, metadataFields: metadataFields || [] }
        };
        let hydeUsed = false;
        let filterScope = null;
        let hydeProvenance = null;
        const responseMetadata = (extra = {}) => ({
            ...extra,
            generationProvider: 'local-chat',
            retrievalPerformed: route.resolved === 'documents',
            route,
            hydeUsed,
            hydeEdited: Boolean(hydeProvenance?.edited),
            hyde: hydeProvenance,
            includeMetadata,
            metadataMode,
            metadataFields: metadataFields || [],
            settings: effectiveSettings,
            durationMs: Date.now() - startedAt,
            filter: filterScope ? this.serializeMetadataFilterScope(filterScope) : null
        });

        onStatus?.(useHyDE ? 'hyde' : 'retrieving');

        const streamState = { emitted: false };
        const streamChunk = (chunk, fullText) => {
            if (chunk) streamState.emitted = true;
            onChunk?.(chunk, fullText);
        };
        const generatePlanned = async (buildPrompt, onPlanned = null) => {
            return runChatGenerationWithRecovery({
                buildPrompt,
                onPlanned,
                onStatus,
                hasVisibleOutput: () => streamState.emitted,
                diagnosticContext: () => ({
                    phase: this.rag.activeOperation?.phase,
                    retrievalCompleted: true,
                    outputStreamed: streamState.emitted,
                    model: this.rag.modelId
                }),
                generate: prompt => this.rag.generateFromMessages(prompt.messages, {
                    maxTokens: prompt.telemetry.outputTokens,
                    temperature,
                    topP,
                    repeatPenalty,
                    onChunk: streamChunk,
                    operationToken,
                    owner: 'chat',
                    datasetId,
                    onStatus
                }),
                recoverEngine: async diagnostic => {
                    console.warn('Recovering local AI before visible output:', diagnostic);
                    await this.rag.recoverEngine(operationToken, onStatus);
                }
            });
        };

        filterScope = this.createMetadataFilterScope(metadataFilters, {
            includePersistent: includePersistentFilters
        });
        let scopeDocIds = allowedDocIds;
        if (scope === 'current' && scopeDocIds === null && typeof window !== 'undefined'
            && typeof window.getCurrentRAGScope === 'function') {
            const activeScope = window.getCurrentRAGScope('current');
            scopeDocIds = activeScope?.scopeType === 'all' ? null : activeScope?.docIds;
        }
        const effectiveAllowedDocIds = this.combineAllowedDocIds(scopeDocIds, filterScope);
        if (scope === 'current' && Array.isArray(effectiveAllowedDocIds) && effectiveAllowedDocIds.length === 0) {
            return {
                answer: 'No documents are available in the captured scope.',
                sources: [],
                citations: [],
                route,
                metadata: responseMetadata({ finishReason: 'stop' })
            };
        }
        if (filterScope.applied && filterScope.matchedDocuments === 0) {
            return {
                answer: 'No documents match the selected scope.',
                sources: [],
                citations: [],
                route,
                metadata: responseMetadata()
            };
        }

        let retrievalQueries = buildChatRetrievalQueries(question, history);
        if (useHyDE) {
            const hypothetical = await this.rag.generateHyDE(retrievalQueries.contextualSemanticQuery, {
                prompt: hydePrompt,
                temperature: hydeTemperature,
                maxTokens: hydeMaxTokens,
                operationToken,
                owner: 'chat',
                datasetId,
                onStatus
            });
            this.rag.throwIfOperationCancelled(operationToken);
            this.rag.setOperationPhase(operationToken, 'awaiting-input');
            onStatus?.('awaiting-hyde');
            const review = typeof options.onHyDEReview === 'function'
                ? await options.onHyDEReview({
                    question,
                    contextualQuery: retrievalQueries.contextualSemanticQuery,
                    generatedText: hypothetical
                })
                : { action: 'without_hyde' };
            this.rag.throwIfOperationCancelled(operationToken);
            const action = review?.action === 'use' ? 'use'
                : review?.action === 'cancel' ? 'cancel' : 'without_hyde';
            const approvedText = action === 'use' ? String(review?.text || '').trim() : '';
            hydeProvenance = {
                generated: hypothetical,
                approved: approvedText || null,
                edited: action === 'use' && approvedText !== String(hypothetical || '').trim(),
                action,
                contextualQuery: retrievalQueries.contextualSemanticQuery,
                semanticQuery: action === 'use' && approvedText
                    ? approvedText
                    : retrievalQueries.contextualSemanticQuery,
                keywordQuery: retrievalQueries.keywordQuery
            };
            if (action === 'cancel') {
                onStatus?.('stopped');
                return {
                    answer: 'HyDE review cancelled.',
                    sources: [],
                    route,
                    metadata: responseMetadata({ wasStopped: true, finishReason: 'abort' })
                };
            }
            if (action === 'use' && approvedText) {
                retrievalQueries = buildChatRetrievalQueries(question, history, approvedText);
                hydeUsed = true;
            }
            this.rag.setOperationPhase(operationToken, 'retrieving');
            onStatus?.('retrieving');
        }
        const questionEmbedding = await this.embeddings.embedSingle(retrievalQueries.semanticQuery, { mode: 'query' });
        this.rag.throwIfOperationCancelled(operationToken);
        if (this.currentDataset !== datasetRef || String(this.currentDatasetId || '') !== String(datasetId || '')) {
            throw new Error('The active dataset changed while this chat turn was running.');
        }
        const retrieval = await this.rag.retrieveContext(question, questionEmbedding, {
            numResults,
            retrievalK,
            vectorWeight,
            includeMetadata,
            metadataFields,
            similarityThreshold,
            allowedDocIds: effectiveAllowedDocIds,
            keywordQuery: retrievalQueries.keywordQuery,
            semanticQuery: retrievalQueries.semanticQuery,
            signal: operationToken.abortController?.signal || null,
            onRerankerProgress: options.onRerankerProgress,
            onStatus
        });
        this.rag.throwIfOperationCancelled(operationToken);

        if (!retrieval.sources.length) {
            return {
                answer: 'No relevant documents were found for this question.',
                sources: [],
                citations: [],
                route,
                metadata: responseMetadata({
                    ...(retrieval.metadata || {}),
                    topicEligible: false,
                    resolvedQuery: retrievalQueries.resolvedQuery,
                    topicAnchorQuestion: retrievalQueries.anchorQuestion
                })
            };
        }

        this.reranker.releaseForGeneration();
        onStatus?.('generating');
        const planned = await generatePlanned(additionalSafetyPercent => buildDocumentChatPrompt({
            question,
            history,
            sources: retrieval.sources,
            systemPrompt,
            userTemplate,
            contextWindow,
            maxOutputTokens,
            includeMetadata,
            metadataFields,
            maxSources: numResults,
            memoryMode,
            maxMemoryTurns,
            additionalSafetyPercent
        }), prompt => onRetrievalComplete?.(prompt.includedSources, prompt.telemetry));
        if (this.currentDataset !== datasetRef || String(this.currentDatasetId || '') !== String(datasetId || '')) {
            throw new Error('The active dataset changed while this chat turn was running.');
        }
        let answerResult = {
            answer: planned.generated.answer,
            sources: planned.prompt.includedSources,
            route,
            metadata: responseMetadata({
                ...(retrieval.metadata || {}),
                ...(planned.generated.metadata || {}),
                ...planned.prompt.telemetry,
                contextRetry: planned.contextRetry,
                recoveryAttempts: planned.recoveryAttempts,
                recoveryDiagnostics: planned.recoveryDiagnostics,
                semanticQueryKind: hydeUsed ? 'hyde' : 'contextual',
                semanticQuery: retrievalQueries.semanticQuery,
                keywordQuery: retrievalQueries.keywordQuery,
                retrievalAnchorUsed: retrievalQueries.anchorUsed,
                retrievalAnchorQuestion: retrievalQueries.anchorQuestion,
                resolvedQuery: retrievalQueries.resolvedQuery,
                topicTurnId: retrievalQueries.topicTurnId
            })
        };
        answerResult = applyCitationSafety(answerResult, planned.prompt.includedSources);
        const topicEligible = !planned.generated.wasStopped;
        answerResult.metadata.topicEligible = topicEligible;
        answerResult.metadata.topicAnchorQuestion = topicEligible
            ? retrievalQueries.resolvedQuery
            : retrievalQueries.anchorQuestion;
        onStatus?.(planned.generated.wasStopped ? 'stopped' : 'complete');
        return answerResult;
        } finally {
            this.rag.endOperation(operationToken);
        }
    }

    /** Retrieve RAG sources/context without requiring the local generation model. */
    async retrieveRAGContext(question, options = {}) {
        if (!this.currentDataset) throw new Error('No dataset loaded');
        if (this.isProcessing) throw new Error('Wait for data processing to finish before retrieving document context.');
        if (!this.rag) throw new Error('RAG retrieval is not initialized');

        const metadataFilters = options.metadataFilters || {};
        const filterScope = this.createMetadataFilterScope(metadataFilters, {
            includePersistent: options.includePersistentFilters === true
        });
        const allowedDocIds = this.combineAllowedDocIds(options.allowedDocIds, filterScope);
        if (filterScope.applied && filterScope.matchedDocuments === 0) {
            return {
                context: '',
                contextPrompt: `No relevant documents found for: "${question}".`,
                sources: [],
                metadata: {
                    generationProvider: 'none',
                    filter: this.serializeMetadataFilterScope(filterScope)
                }
            };
        }

        const textToEmbed = options.hydeText || question;
        const questionEmbedding = await this.embeddings.embedSingle(textToEmbed, { mode: 'query' });
        const result = await this.rag.retrieveContext(question, questionEmbedding, {
            ...options,
            allowedDocIds
        });
        result.metadata = {
            ...(result.metadata || {}),
            filter: this.serializeMetadataFilterScope(filterScope)
        };
        return result;
    }

    /**
     * Get visualization data
     */
    getVisualizationData() {
        if (!this.currentDataset) {
            return null;
        }

        const dataset = this.currentDataset;
        const embeddings = dataset.embeddings || null;
        const chunkRecords = embeddings?.chunks || null;
        const chunkMap = this.chunkToParentMap || embeddings?.chunkToParentMap || null;

        return {
            projection: dataset.projection,
            clusters: dataset.clusters,
            documents: dataset.documents,
            numDocuments: dataset.numDocuments,
            embeddings: embeddings
                ? {
                    parent: embeddings.parent || dataset.retrievalEmbeddings || null,
                    chunks: chunkRecords || null,
                    chunkToParentMap: chunkMap,
                    model: embeddings.model || dataset.embeddingModel || null,
                    dimension: embeddings.dimension || this.embeddings.dimension || null,
                    schema: embeddings.schema || dataset.embeddingSchema || null
                }
                : null,
            chunkToParentMap: chunkMap,
            clusterKeywords: dataset.clusterKeywords || null,
            metadataSchema: dataset.metadataSchema || null
        };
    }

    /**
     * Load dataset from storage
     */
    async loadDataset(datasetId) {
        if (this.rag?.activeOperation) {
            const active = this.rag.activeOperation;
            throw new Error(`Cannot switch datasets while local AI is ${this.rag._operationLabel(active.owner)}. Stop or finish that task first.`);
        }
        const data = await this.storage.loadDataset(datasetId);
        if (typeof window !== 'undefined') window.clearActiveClusterLabels?.();
        this.clearMcpMetadataFilters();

        // Restore cluster metadata if not already present
        if (data.clusters && data.documents) {
            data.documents.forEach((doc, idx) => {
                if (!doc.metadata) {
                    doc.metadata = {};
                }
                if (doc.metadata.cluster === undefined && data.clusters[idx] !== undefined) {
                    doc.metadata.cluster = data.clusters[idx];
                    doc.metadata.cluster_label = data.clusters[idx] === -1 ? 'Outlier' : `Cluster ${data.clusters[idx]}`;
                }
            });
        }

        if (data.clusterKeywords) {
            this.clustering.hydrateClusterKeywords(data.clusterKeywords);
        } else {
            this.clustering.hydrateClusterKeywords();
        }

        const resolvedEmbeddings = data.embeddings;
        const retrievalEmbeddings = Array.isArray(resolvedEmbeddings)
            ? resolvedEmbeddings
            : (resolvedEmbeddings?.parent ?? resolvedEmbeddings?.retrieval);
        const clusteringEmbeddings = Array.isArray(resolvedEmbeddings)
            ? resolvedEmbeddings
            : (resolvedEmbeddings?.clustering ?? resolvedEmbeddings?.parent ?? resolvedEmbeddings?.retrieval);

        if (!retrievalEmbeddings) {
            throw new Error('Stored dataset is missing retrieval embeddings');
        }

        const docIds = data.documents.map(d => d.id);

        // Rebuild indexes
        await this.vectorSearch.buildIndex(
            retrievalEmbeddings,
            docIds,
            data.documents
        );

        this.bm25Search.buildIndex(data.documents, docIds);

        const storedChunks = Array.isArray(resolvedEmbeddings?.chunks) ? resolvedEmbeddings.chunks : [];
        this.chunkToParentMap = resolvedEmbeddings?.chunkToParentMap || null;
        if (storedChunks.length > 0) {
            this.chunkVectorSearch = buildChunkIndex(storedChunks, BrowserVectorSearch);
            this.chunkBM25Search = new BM25Search();
            const chunkDocuments = storedChunks.map(chunk => ({
                id: chunk.chunkId,
                text: chunk.text,
                metadata: {
                    ...(chunk.metadata || {}),
                    parent_id: chunk.docId,
                    chunk_index: chunk.chunkIndex
                }
            }));
            this.chunkBM25Search.buildIndex(chunkDocuments, chunkDocuments.map(chunk => chunk.id));
        } else {
            this.chunkVectorSearch = null;
            this.chunkBM25Search = null;
        }
        if (this.rag) {
            this.rag.setChunkVectorSearch(this.chunkVectorSearch);
            this.rag.setBM25Search(this.chunkBM25Search || this.bm25Search);
        }

        this.currentDataset = {
            id: datasetId,
            ...data.metadata,
            documents: data.documents,
            embeddings: resolvedEmbeddings,
            retrievalEmbeddings,
            clusteringEmbeddings,
            embeddingSchema: Array.isArray(resolvedEmbeddings) ? 'single-embedding-legacy' : resolvedEmbeddings?.schema,
            embeddingModel: Array.isArray(resolvedEmbeddings) ? 'unknown' : resolvedEmbeddings?.model,
            projection: data.projection,  // 2D for visualization
            clusteringProjection: data.clusteringProjection,  // ND for clustering (if available)
            clusters: data.clusters,
            numDocuments: data.documents.length,
            emptyRowCount: data.emptyRowCount || 0,
            duplicateCount: data.duplicateCount || 0,
            clusterKeywords: data.clusterKeywords || null,
            metadataSchema: data.metadataSchema || data.metadata?.metadataSchema || null
        };
        this.currentDatasetId = datasetId;

        if (typeof document !== 'undefined') {
            document.dispatchEvent(new CustomEvent('vectoria:dataset-changed', {
                detail: { datasetId, reason: 'loaded' }
            }));
        }

        return this.currentDataset;
    }

    /**
     * Get pipeline statistics
     */
    getStats() {
        return {
            isInitialized: this.isInitialized,
            currentDataset: this.currentDatasetId,
            embeddings: this.embeddings.getCacheStats(),
            vectorSearch: this.vectorSearch.getStats(),
            rag: this.rag?.getStats() || null
        };
    }

    /**
     * Remove duplicate documents by exact text match, keeping the first occurrence.
     */
    _deduplicateDocuments(documents) {
        const uniqueDocuments = [];
        const seenTexts = new Set();
        let duplicateCount = 0;

        for (const doc of documents) {
            const key = doc.text;
            if (seenTexts.has(key)) {
                duplicateCount++;
                continue;
            }
            seenTexts.add(key);
            uniqueDocuments.push(doc);
        }

        return { documents: uniqueDocuments, duplicateCount };
    }

    /**
     * Clear current dataset
     */
    clearDataset() {
        if (typeof window !== 'undefined') window.clearActiveClusterLabels?.();
        this.clearMcpMetadataFilters();
        this.currentDataset = null;
        this.currentDatasetId = null;
        this.vectorSearch.clear();
        this.chunkVectorSearch = null;
        this.chunkBM25Search = null;
        this.chunkToParentMap = null;
        if (typeof document !== 'undefined') {
            document.dispatchEvent(new CustomEvent('vectoria:dataset-changed', {
                detail: { datasetId: null, reason: 'cleared' }
            }));
        }
    }

    /**
     * Abort the current RAG generation
     */
    abortRAG(ownerOrOperation = null) {
        if (this.rag) {
            return this.rag.abort(ownerOrOperation);
        }
        return false;
    }
}

// Export singleton instance
export const pipeline = new BrowserMLPipeline();

function fuseSearchResults(vectorResults, bm25Results, { k, vectorWeight }) {
    const rrfK = 60;
    const entries = new Map();
    const add = (result, rank, type) => {
        const key = String(result.doc_id ?? result.index);
        const current = entries.get(key) || {
            ...result,
            vector_score: null,
            bm25_score: null,
            score: 0
        };
        const weight = type === 'vector' ? vectorWeight : (1 - vectorWeight);
        current.score += weight / (rrfK + rank + 1);
        if (type === 'vector') current.vector_score = result.score;
        else current.bm25_score = result.score;
        if (!current.text && result.text) current.text = result.text;
        if ((!current.metadata || !Object.keys(current.metadata).length) && result.metadata) {
            current.metadata = result.metadata;
        }
        entries.set(key, current);
    };

    vectorResults.forEach((result, rank) => add(result, rank, 'vector'));
    bm25Results.forEach((result, rank) => add(result, rank, 'bm25'));
    return [...entries.values()]
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
}
