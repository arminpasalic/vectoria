/**
 * Browser-based Storage using IndexedDB
 * Persistent storage for embeddings, vector index, and processed data
 */

import localforage from 'https://cdn.jsdelivr.net/npm/localforage@1.10.0/+esm';

export class BrowserStorage {
    constructor() {
        // Create separate stores for different data types
        this.embeddingsStore = localforage.createInstance({
            name: 'vectoria',
            storeName: 'embeddings',
            description: 'Cached embeddings and vectors'
        });

        this.indexStore = localforage.createInstance({
            name: 'vectoria',
            storeName: 'vectorIndex',
            description: 'Vector search index'
        });

        this.documentsStore = localforage.createInstance({
            name: 'vectoria',
            storeName: 'documents',
            description: 'Processed documents and metadata'
        });

        this.visualizationStore = localforage.createInstance({
            name: 'vectoria',
            storeName: 'visualization',
            description: 'UMAP projections and cluster data'
        });

        this.metadataStore = localforage.createInstance({
            name: 'vectoria',
            storeName: 'metadata',
            description: 'System metadata and settings'
        });
    }

    /**
     * Infer vector dimension from stored vectors
     * @param {Array|TypedArray} vectors
     * @returns {number}
     */
    _inferVectorDimension(vectors) {
        if (!vectors) return 0;
        if (Array.isArray(vectors)) {
            if (vectors.length === 0) return 0;
            const first = vectors[0];
            if (!first) return 0;
            return typeof first.length === 'number' ? first.length : 0;
        }
        if (ArrayBuffer.isView(vectors)) {
            return vectors.length || 0;
        }
        return 0;
    }

    /**
     * Normalize embeddings payload to the new dual-embedding schema.
     * Supports legacy datasets where a single embedding set was stored.
     * @param {any} raw
     * @returns {Object|null}
     */
    _normalizeEmbeddingsPayload(raw) {
        if (!raw) return null;

        const wrapLegacy = (vectors, schema = 'single-embedding-legacy') => {
            const dimension = this._inferVectorDimension(vectors);
            return {
                retrieval: vectors,
                clustering: vectors,
                model: 'unknown',
                dimension,
                schema,
                modes: {
                    retrieval: schema === 'dual-embedding-v1' ? 'passage' : 'unspecified',
                    clustering: schema === 'dual-embedding-v1' ? 'query' : 'unspecified'
                },
                legacy: schema !== 'dual-embedding-v1'
            };
        };

        if (Array.isArray(raw) || ArrayBuffer.isView(raw)) {
            return wrapLegacy(raw);
        }

        if (typeof raw === 'object') {
            if (Array.isArray(raw.retrieval) || Array.isArray(raw.clustering) || ArrayBuffer.isView(raw.retrieval) || ArrayBuffer.isView(raw.clustering)) {
                const retrieval = raw.retrieval ?? raw.passages ?? raw.vectors ?? raw.documents ?? null;
                const clustering = raw.clustering ?? raw.queries ?? retrieval ?? null;
                const schema = raw.schema || (retrieval && clustering && retrieval !== clustering ? 'dual-embedding-v1' : 'single-embedding-legacy');
                const dimension = raw.dimension
                    || this._inferVectorDimension(retrieval)
                    || this._inferVectorDimension(clustering);
                const modes = raw.modes || {
                    retrieval: raw.retrievalMode || (schema === 'dual-embedding-v1' ? 'passage' : 'unspecified'),
                    clustering: raw.clusteringMode || (schema === 'dual-embedding-v1' ? 'query' : raw.retrievalMode || 'unspecified')
                };

                return {
                    retrieval,
                    clustering: clustering ?? retrieval,
                    model: raw.model || 'unknown',
                    dimension,
                    schema,
                    modes,
                    legacy: raw.legacy ?? schema !== 'dual-embedding-v1'
                };
            }

            if (Array.isArray(raw.embeddings) || raw.embeddings) {
                return this._normalizeEmbeddingsPayload(raw.embeddings);
            }
        }

        return null;
    }

    /**
     * Estimate memory footprint (bytes) for stored embeddings
     * @param {Object|null} embeddings
     * @returns {number}
     */
    _estimateEmbeddingBytes(embeddings) {
        if (!embeddings) return 0;
        const dim = embeddings.dimension
            || this._inferVectorDimension(embeddings.retrieval)
            || this._inferVectorDimension(embeddings.clustering)
            || 0;
        if (!dim) return 0;

        const seen = new Set();
        let vectorCount = 0;

        const consume = (vectors) => {
            if (!vectors || seen.has(vectors)) {
                return;
            }
            seen.add(vectors);

            if (Array.isArray(vectors)) {
                vectorCount += vectors.length;
            } else if (ArrayBuffer.isView(vectors)) {
                vectorCount += 1;
            }
        };

        consume(embeddings.retrieval);
        consume(embeddings.clustering);

        return vectorCount * dim * 4;
    }

    /**
     * Save processed dataset
     * @param {Object} data - Complete dataset with all components
     */
    async saveDataset(datasetId, data) {
        try {
            const {
                embeddings,
                vectorIndex,
                documents,
                projection,
                clusteringProjection,
                clusters,
                fileName,
                fileType,
                textColumn
            } = data;

            const normalizedEmbeddings = this._normalizeEmbeddingsPayload(embeddings);

            // Save each component
            await Promise.all([
                this.embeddingsStore.setItem(datasetId, {
                    embeddings: normalizedEmbeddings,
                    timestamp: Date.now()
                }),
                this.indexStore.setItem(datasetId, {
                    index: vectorIndex,
                    timestamp: Date.now()
                }),
                this.documentsStore.setItem(datasetId, {
                    documents: documents,
                    timestamp: Date.now()
                }),
                this.visualizationStore.setItem(datasetId, {
                    projection: projection,  // 2D for visualization
                    clusteringProjection: clusteringProjection,  // ND for clustering
                    clusters: clusters,
                    timestamp: Date.now()
                }).then(() => {
                    const clusteringDims = clusteringProjection?.[0]?.length || 0;
                }),
                this.metadataStore.setItem(datasetId, {
                    fileName: fileName,
                    fileType: fileType,
                    textColumn: textColumn,
                    numDocuments: documents.length,
                    embeddingModel: normalizedEmbeddings?.model || 'unknown',
                    embeddingSchema: normalizedEmbeddings?.schema || (normalizedEmbeddings ? 'single-embedding-legacy' : 'none'),
                    timestamp: Date.now()
                })
            ]);

            if (normalizedEmbeddings) {
                const retrievalCount = Array.isArray(normalizedEmbeddings.retrieval) ? normalizedEmbeddings.retrieval.length : 0;
                const clusteringCount = Array.isArray(normalizedEmbeddings.clustering) ? normalizedEmbeddings.clustering.length : 0;
            }

            return true;
        } catch (error) {
            console.error('❌ Failed to save dataset:', error);
            // Rollback: remove any partially saved data
            try {
                await Promise.all([
                    this.embeddingsStore.removeItem(datasetId),
                    this.indexStore.removeItem(datasetId),
                    this.documentsStore.removeItem(datasetId),
                    this.visualizationStore.removeItem(datasetId),
                    this.metadataStore.removeItem(datasetId)
                ]);
            } catch (rollbackError) {
                console.error('❌ Rollback also failed:', rollbackError);
            }
            throw new Error(`Save failed: ${error.message}`);
        }
    }

    /**
     * Load dataset
     * @param {string} datasetId
     */
    async loadDataset(datasetId) {
        try {
            const [embeddingsData, indexData, documentsData, visualizationData, metadata] = await Promise.all([
                this.embeddingsStore.getItem(datasetId),
                this.indexStore.getItem(datasetId),
                this.documentsStore.getItem(datasetId),
                this.visualizationStore.getItem(datasetId),
                this.metadataStore.getItem(datasetId)
            ]);

            if (!embeddingsData || !documentsData) {
                throw new Error('Dataset not found');
            }

            const normalizedEmbeddings = this._normalizeEmbeddingsPayload(embeddingsData.embeddings);

            const clusteringDims = visualizationData?.clusteringProjection?.[0]?.length || 0;
            if (normalizedEmbeddings) {
                const retrievalCount = Array.isArray(normalizedEmbeddings.retrieval) ? normalizedEmbeddings.retrieval.length : 0;
                const clusteringCount = Array.isArray(normalizedEmbeddings.clustering) ? normalizedEmbeddings.clustering.length : 0;
            }

            return {
                embeddings: normalizedEmbeddings,
                vectorIndex: indexData?.index,
                documents: documentsData.documents,
                projection: visualizationData?.projection,  // 2D for visualization
                clusteringProjection: visualizationData?.clusteringProjection,  // ND for clustering
                clusters: visualizationData?.clusters,
                metadata: metadata
            };
        } catch (error) {
            console.error('❌ Failed to load dataset:', error);
            throw new Error(`Load failed: ${error.message}`);
        }
    }

    /**
     * Check if dataset exists
     */
    async hasDataset(datasetId) {
        const metadata = await this.metadataStore.getItem(datasetId);
        return metadata !== null;
    }

    /**
     * List all saved datasets
     */
    async listDatasets() {
        const keys = await this.metadataStore.keys();
        const datasets = [];

        for (const key of keys) {
            const metadata = await this.metadataStore.getItem(key);
            if (metadata) {
                datasets.push({
                    id: key,
                    ...metadata
                });
            }
        }

        // Sort by timestamp (newest first)
        datasets.sort((a, b) => b.timestamp - a.timestamp);

        return datasets;
    }

    /**
     * Delete dataset
     */
    async deleteDataset(datasetId) {
        try {
            await Promise.all([
                this.embeddingsStore.removeItem(datasetId),
                this.indexStore.removeItem(datasetId),
                this.documentsStore.removeItem(datasetId),
                this.visualizationStore.removeItem(datasetId),
                this.metadataStore.removeItem(datasetId)
            ]);

            return true;
        } catch (error) {
            console.error('❌ Failed to delete dataset:', error);
            throw new Error(`Delete failed: ${error.message}`);
        }
    }

    /**
     * Clear all data
     */
    async clearAll() {
        await Promise.all([
            this.embeddingsStore.clear(),
            this.indexStore.clear(),
            this.documentsStore.clear(),
            this.visualizationStore.clear(),
            this.metadataStore.clear()
        ]);

    }

    /**
     * Get storage statistics
     */
    async getStats() {
        const [embKeys, idxKeys, docKeys, vizKeys, metaKeys] = await Promise.all([
            this.embeddingsStore.keys(),
            this.indexStore.keys(),
            this.documentsStore.keys(),
            this.visualizationStore.keys(),
            this.metadataStore.keys()
        ]);

        // Estimate storage usage (rough approximation)
        let totalSize = 0;
        const datasets = await this.listDatasets();

        for (const dataset of datasets) {
            const embData = await this.embeddingsStore.getItem(dataset.id);
            if (embData && embData.embeddings) {
                const normalized = this._normalizeEmbeddingsPayload(embData.embeddings);
                totalSize += this._estimateEmbeddingBytes(normalized);
            }
        }

        return {
            numDatasets: metaKeys.length,
            numEmbeddings: embKeys.length,
            numIndexes: idxKeys.length,
            numDocuments: docKeys.length,
            numVisualizations: vizKeys.length,
            estimatedSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
            datasets: datasets
        };
    }

    /**
     * Save user settings
     */
    async saveSettings(settings) {
        await this.metadataStore.setItem('__settings__', {
            ...settings,
            timestamp: Date.now()
        });
    }

    /**
     * Load user settings
     */
    async loadSettings() {
        const settings = await this.metadataStore.getItem('__settings__');
        return settings || {};
    }

    /**
     * Save current session state
     */
    async saveSession(sessionData) {
        await this.metadataStore.setItem('__current_session__', {
            ...sessionData,
            timestamp: Date.now()
        });
    }

    /**
     * Load current session state
     */
    async loadSession() {
        const session = await this.metadataStore.getItem('__current_session__');
        return session || null;
    }

    /**
     * Clear current session
     */
    async clearSession() {
        await this.metadataStore.removeItem('__current_session__');
    }

    /**
     * Export dataset as JSON
     */
    async exportDataset(datasetId) {
        const data = await this.loadDataset(datasetId);
        const exportData = {
            version: '2.0',
            datasetId: datasetId,
            ...data,
            exportedAt: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });

        return blob;
    }

    /**
     * Import dataset from JSON
     */
    async importDataset(jsonData) {
        try {
            const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;

            if (data.version !== '2.0') {
                console.warn('Warning: Importing data from different version');
            }

            const datasetId = data.datasetId || `import_${Date.now()}`;

            await this.saveDataset(datasetId, {
                embeddings: data.embeddings,
                vectorIndex: data.vectorIndex,
                documents: data.documents,
                projection: data.projection,
                clusteringProjection: data.clusteringProjection,
                clusters: data.clusters,
                fileName: data.metadata?.fileName,
                fileType: data.metadata?.fileType,
                textColumn: data.metadata?.textColumn
            });

            return datasetId;
        } catch (error) {
            console.error('❌ Import failed:', error);
            throw new Error(`Import failed: ${error.message}`);
        }
    }
}

// Export singleton instance
export const storage = new BrowserStorage();
