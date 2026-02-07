/**
 * Data Export/Import Module
 * Handles saving and loading processed datasets to/from JSON
 */

export class DataExporter {
    constructor(pipeline) {
        this.pipeline = pipeline;
    }

    /**
     * Export current dataset to JSON format
     * @returns {Object} Complete dataset export
     */
    async exportToJSON() {
        if (!this.pipeline.currentDataset || !this.pipeline.currentDataset.documents) {
            throw new Error('No dataset available to export');
        }

        const dataset = this.pipeline.currentDataset;
        const documents = dataset.documents;
        const embeddings = dataset.embeddings;
        const metadataSchema = dataset.metadataSchema
            || (typeof window !== 'undefined' ? window.metadataSchema : null)
            || null;

        // Build export object
        const exportData = {
            metadata: {
                version: "1.0",
                created: new Date().toISOString(),
                model: this.pipeline.embeddings.modelName,
                dimension: this.pipeline.embeddings.dimension,
                num_documents: documents.length,
                num_chunks: embeddings?.chunks?.length || 0,
                schema: "three-tier-v1"
            },
            documents: documents.map(doc => ({
                id: doc.id,
                text: doc.text,
                metadata: doc.metadata || {}
            })),
            embeddings: {
                parent: {
                    vectors: embeddings?.parent ? embeddings.parent.map(e => Array.from(e)) : [],
                    mode: "query",
                    description: "Parent summaries for clustering"
                },
                chunks: {
                    vectors: embeddings?.chunks ? embeddings.chunks.map(c => Array.from(c.embedding)) : [],
                    mode: "passage",
                    description: "Chunk embeddings for RAG"
                },
                chunk_map: this.pipeline.chunkToParentMap || embeddings?.chunkToParentMap || null
            },
            chunks: embeddings?.chunks ? embeddings.chunks.map(c => ({
                chunk_id: c.chunkId,
                parent_id: c.docId,
                text: c.text,
                position: c.chunkIndex,
                total_chunks: c.totalChunks ?? null,
                metadata: c.metadata || {}
            })) : [],
            visualization: {
                projection_2d: dataset.projection ? dataset.projection.map(p => Array.from(p)) : [],
                clusters: dataset.clusters ? Array.from(dataset.clusters) : []
            },
            indexes: {
                vector_index_type: "flat",
                bm25_available: this.pipeline.bm25Search?.isBuilt || false,
                chunk_index_size: embeddings?.chunks?.length || 0
            },
            metadata_schema: metadataSchema
        };

        return exportData;
    }

    /**
     * Download JSON export as file
     * @param {Object} data - Export data
     * @param {string} filename - Optional custom filename
     */
    downloadJSON(data, filename = null) {
        const defaultFilename = `vectoria_export_${Date.now()}.json`;
        const finalFilename = filename || defaultFilename;

        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = finalFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    }

    /**
     * Import dataset from JSON
     * @param {Object} jsonData - Parsed JSON data
     * @returns {Promise<boolean>} Success status
     */
    async importFromJSON(jsonData) {
        // 1. Validate JSON structure
        if (!jsonData.metadata || !jsonData.documents || !jsonData.embeddings) {
            throw new Error('Invalid Vectoria export format - missing required fields');
        }

        // 2. Restore documents
        const documents = jsonData.documents.map(doc => ({
            id: doc.id,
            text: doc.text,
            metadata: doc.metadata || {}
        }));

        // 3. Convert embeddings to Float32Array
        const parentEmbeddings = jsonData.embeddings.parent.vectors.map(v =>
            new Float32Array(v)
        );

        // 4. Rebuild vector indexes from saved embeddings
        const docIds = documents.map(d => d.id);
        this.pipeline.vectorSearch.buildIndex(parentEmbeddings, docIds, documents);

        // 5. Rebuild chunk indexes
        if (jsonData.chunks && jsonData.embeddings.chunks) {
            const chunks = jsonData.chunks.map((c, i) => ({
                chunkId: c.chunk_id,
                docId: c.parent_id,
                text: c.text,
                chunkIndex: c.position,
                totalChunks: c.total_chunks,
                embedding: new Float32Array(jsonData.embeddings.chunks.vectors[i])
            }));

            // Build chunk vector search (reusing buildChunkIndex logic)
            const { BrowserVectorSearch } = await import('./vector-search.js');
            const chunkVectorSearch = new BrowserVectorSearch(this.pipeline.embeddings.dimension);
            const chunkEmbeddings = chunks.map(c => c.embedding);
            const chunkIds = chunks.map(c => c.chunkId);
            chunkVectorSearch.buildIndex(chunkEmbeddings, chunkIds, chunks);

            this.pipeline.chunkVectorSearch = chunkVectorSearch;
            this.pipeline.rag.setChunkVectorSearch(chunkVectorSearch);

            // Rebuild chunk-to-parent map
            const chunkToParentMap = jsonData.embeddings.chunk_map || {};
            if (!chunkToParentMap || Object.keys(chunkToParentMap).length === 0) {
                chunks.forEach(chunk => {
                    chunkToParentMap[chunk.chunkId] = chunk.docId;
                });
            }
            this.pipeline.chunkToParentMap = chunkToParentMap;

            // Store chunks in dataset
            if (!this.pipeline.currentDataset) this.pipeline.currentDataset = {};
            if (!this.pipeline.currentDataset.embeddings) this.pipeline.currentDataset.embeddings = {};
            this.pipeline.currentDataset.embeddings.chunks = chunks;
        }

        // 6. Rebuild BM25 indexes
        if (jsonData.indexes.bm25_available) {
            // Parent BM25
            this.pipeline.bm25Search.buildIndex(documents.map((d, i) => ({
                doc_id: d.id,
                text: d.text,
                metadata: d.metadata
            })));

            // Chunk BM25 (if chunks exist)
            if (jsonData.chunks && this.pipeline.currentDataset.embeddings.chunks) {
                const { BM25Search } = await import('./vector-search.js');
                const chunkBM25 = new BM25Search();
                chunkBM25.buildIndex(this.pipeline.currentDataset.embeddings.chunks.map(c => ({
                    doc_id: c.chunkId,
                    text: c.text,
                    metadata: { parent_id: c.docId, chunk_index: c.chunkIndex }
                })));

                this.pipeline.rag.setBM25Search(chunkBM25);
            }
        }

        // 7. Restore visualization
        const projection = jsonData.visualization.projection_2d.map(p => new Float32Array(p));
        const clusters = new Int32Array(jsonData.visualization.clusters);

        // 8. Update pipeline state
        this.pipeline.currentDataset = {
            documents: documents,
            embeddings: {
                parent: parentEmbeddings,
                chunks: this.pipeline.currentDataset?.embeddings?.chunks || []
            },
            projection: projection,
            clusters: clusters,
            clusterLabels: this._extractClusterLabels(clusters, documents),
            metadataSchema: jsonData.metadata_schema || null,
            numDocuments: documents.length
        };

        this.pipeline.isProcessing = false;
        this.pipeline.processingProgress = 1.0;

        return true;
    }

    /**
     * Extract cluster labels (top keywords per cluster)
     */
    _extractClusterLabels(clusters, documents) {
        const clusterDocs = new Map();

        clusters.forEach((cluster, idx) => {
            if (cluster < 0) return; // Outlier
            if (!clusterDocs.has(cluster)) {
                clusterDocs.set(cluster, []);
            }
            clusterDocs.get(cluster).push(documents[idx].text);
        });

        const labels = {};
        clusterDocs.forEach((texts, cluster) => {
            // Simple keyword extraction: most common words
            const words = {};
            texts.forEach(text => {
                const tokens = text.toLowerCase().split(/\W+/).filter(w => w.length > 3);
                tokens.forEach(token => {
                    words[token] = (words[token] || 0) + 1;
                });
            });

            const topWords = Object.entries(words)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([word]) => word);

            labels[cluster] = topWords.join(', ');
        });

        return labels;
    }
}
