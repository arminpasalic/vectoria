/**
 * Browser ML Orchestration Module
 * Main entry point for browser-based ML pipeline
 * Coordinates all browser ML modules and provides unified API
 */

import { BrowserEmbeddings } from './embeddings.js';
import { BrowserVectorSearch, BM25Search } from './vector-search.js';
import { BrowserRAG } from './llm-rag.js';
import { BrowserFileProcessor } from './file-processor.js';
import { BrowserClustering } from './clustering.js';
import { BrowserStorage } from './storage.js';
import { chunkDocuments } from './chunking/chonkieChunker.js';
import { embedChunks, buildChunkIndex, groupChunksByParent } from './embedding/tier3ChunkEmbeddings.js';

export class BrowserMLPipeline {
    constructor() {
        // Initialize all modules
        this.embeddings = new BrowserEmbeddings();
        this.vectorSearch = new BrowserVectorSearch(384); // paraphrase-multilingual-MiniLM-L12-v2 dimension
        this.bm25Search = new BM25Search();
        this.rag = null; // Will be initialized after vector search is ready
        this.fileProcessor = new BrowserFileProcessor();
        this.clustering = new BrowserClustering();
        this.storage = new BrowserStorage();

        // Tier 3: Chunk-based retrieval for RAG
        this.chunkVectorSearch = null; // Initialized during processFile
        this.chunkToParentMap = null;

        // State
        this.isInitialized = false;
        this.currentDataset = null;
        this.currentDatasetId = null;

        // Processing state
        this.isProcessing = false;
        this.processingProgress = 0;

        // Anti-throttle state
        this._wakeLock = null;
        this._keepaliveInterval = null;
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
    async initialize(callbacks = {}) {
        if (this.isInitialized) {
            return;
        }

        const {
            onEmbeddingsProgress = null,
            onLLMProgress = null,
            onComplete = null
        } = callbacks;

        try {
            // 1. Initialize embeddings model
            await this.embeddings.initialize(onEmbeddingsProgress);

            // 2. Initialize LLM (RAG will be created after we have documents)
            // Create RAG handler (it will initialize engine on first use)
            this.rag = new BrowserRAG(this.vectorSearch);
            await this.rag.initialize(onLLMProgress);

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

        try {

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
            const texts = documents.map(doc => doc.text);

            // TIER 1: Parent summaries for clustering/visualization (query mode)
            updateProgress('embedding', 0.20, 'Generating parent summaries for visualization...');
            const documentSummaries = texts.map(text => {
                const tokens = text.split(/\s+/);
                if (tokens.length <= 256) return text;
                return tokens.slice(0, 256).join(' ') + '...';
            });
            
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
                    chunkSize: chunkConfig.chunk_size || 512,
                    chunkOverlap: chunkConfig.chunk_overlap || 128,
                    minChunkSize: chunkConfig.min_chunk_size || 50
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
            const chunkDocs = chunks.map(chunk => ({
                id: chunk.chunk_id,
                text: chunk.text,
                metadata: chunk.metadata
            }));
            const chunkIds = chunks.map(c => c.chunk_id);
            chunkBM25Search.buildIndex(chunkDocs, chunkIds);
            updateProgress('indexing', 0.52, 'Processing file...');

            // Update RAG with BM25 chunk search for hybrid retrieval
            if (this.rag) {
                this.rag.setBM25Search(chunkBM25Search);
            }

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
                doc.metadata.cluster_label = clusterId === -1 ? 'Noise' : `Cluster ${clusterId}`;

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

            updateProgress('complete', 1.0, 'Processing complete!');

            // Calculate total time
            timings.total = (Date.now() - timings.start) / 1000;

            this.isProcessing = false;
            // Release wake lock on successful completion
            this._releaseWakeLock();
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
            minScore = 0.0
        } = options;

        const normalizedType = (searchType || 'fast').toLowerCase();
        let results;

        if (normalizedType === 'keyword' || normalizedType === 'bm25' || normalizedType === 'normal' || normalizedType === 'fast') {
            // Keyword/BM25 search
            results = this.bm25Search.search(query, k);
        } else {
            // Semantic vector search (default)
            const queryEmbedding = await this.embeddings.embedSingle(query, { mode: 'query' });
            results = this.vectorSearch.search(queryEmbedding, k, { minScore, includeMetadata: true });
        }

        return results;
    }

    /**
     * Perform RAG query
     * @param {string} question - Question to answer
     * @param {Object} options - RAG options
     */
    async queryRAG(question, options = {}) {
        if (!this.currentDataset) {
            throw new Error('No dataset loaded');
        }

        if (!this.rag) {
            throw new Error('RAG not initialized');
        }

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
            retrievalK = null
        } = options;

        // Generate question embedding (use HyDE text if provided, otherwise original question)
        const textToEmbed = hydeText || question;
        const questionEmbedding = await this.embeddings.embedSingle(textToEmbed, { mode: 'query' });

        if (stream && onChunk) {
            // Streaming response
            return await this.rag.queryStream(question, questionEmbedding, onChunk, {
                numResults,
                searchType,
                temperature,
                maxTokens,
                includeMetadata,
                metadataFields,
                allowedDocIds,
                retrievalK
            });
        } else {
            // Regular response
            return await this.rag.query(question, questionEmbedding, {
                numResults,
                searchType,
                temperature,
                maxTokens,
                includeMetadata,
                metadataFields,
                allowedDocIds,
                retrievalK
            });
        }
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
        const data = await this.storage.loadDataset(datasetId);

        // Restore cluster metadata if not already present
        if (data.clusters && data.documents) {
            data.documents.forEach((doc, idx) => {
                if (!doc.metadata) {
                    doc.metadata = {};
                }
                if (doc.metadata.cluster === undefined && data.clusters[idx] !== undefined) {
                    doc.metadata.cluster = data.clusters[idx];
                    doc.metadata.cluster_label = data.clusters[idx] === -1 ? 'Noise' : `Cluster ${data.clusters[idx]}`;
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
            : resolvedEmbeddings?.retrieval;
        const clusteringEmbeddings = Array.isArray(resolvedEmbeddings)
            ? resolvedEmbeddings
            : (resolvedEmbeddings?.clustering ?? resolvedEmbeddings?.retrieval);

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
            clusterKeywords: data.clusterKeywords || null
        };
        this.currentDatasetId = datasetId;

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
        this.currentDataset = null;
        this.currentDatasetId = null;
        this.vectorSearch.clear();
    }

    /**
     * Abort the current RAG generation
     */
    abortRAG() {
        if (this.rag) {
            this.rag.abort();
        }
    }
}

// Export singleton instance
export const pipeline = new BrowserMLPipeline();
