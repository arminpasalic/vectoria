/**
 * Unified Export/Import System for Vectoria (Version 3.0)
 *
 * Single .vectoria.json format that handles:
 * - Full dataset exports (all documents)
 * - Filtered selection exports (subset based on filters)
 * - Lasso selection exports (manually selected subset)
 * - RAG results exports (documents from RAG query)
 *
 * All exports are importable as full datasets with complete functionality:
 * - Visualization (UMAP projection, clusters)
 * - Search (parent embeddings, chunk embeddings)
 * - RAG/HYDE (chunk vectors, metadata)
 * - Filtering (metadata schema)
 */

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Download content as a file
 */
function downloadFile(content, filename) {
    const blob = typeof content === 'string'
        ? new Blob([content], {type: 'application/json;charset=utf-8'})
        : content;

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Generate filename with timestamp
 */
function generateFilename(base, ext) {
    const date = new Date().toISOString().split('T')[0];
    const time = new Date().toTimeString().split(' ')[0].replace(/:/g, '-');
    return `${base}-${date}-${time}.${ext}`;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    if (typeof window.showToast === 'function') {
        window.showToast(message, type);
    } else {
    }
}

/**
 * Get document key from various possible fields
 */
function getDocumentKey(doc, fallback = null) {
    if (!doc) {
        return fallback !== null && fallback !== undefined ? String(fallback) : null;
    }

    const candidates = [
        doc.id,
        doc.doc_id,
        doc.docId,
        doc.metadata?.doc_id,
        doc.metadata?.id
    ];

    for (const candidate of candidates) {
        if (candidate !== undefined && candidate !== null && candidate !== '') {
            return String(candidate);
        }
    }

    if (Number.isFinite(doc.index)) {
        return String(doc.index);
    }

    return fallback !== null && fallback !== undefined ? String(fallback) : null;
}

// ============================================================================
// UNIFIED EXPORT FORMAT (Version 3.0)
// ============================================================================

/**
 * Build unified export payload from documents
 * @param {Array} documents - Documents to export
 * @param {string} exportType - 'full', 'filtered', 'lasso', or 'rag_results'
 * @param {Object} metadata - Additional metadata about the export
 * @returns {Object} Unified export payload
 */
async function buildUnifiedExportPayload(documents, exportType = 'full', metadata = {}) {
    const pipeline = window.browserML?.pipeline;
    if (!pipeline) {
        throw new Error('Pipeline not available');
    }

    const dataset = pipeline.currentDataset;
    if (!dataset) {
        throw new Error('No dataset available');
    }

    // Get embeddings for the selected documents
    const { parentVectors, chunkVectors, chunkMap } = await getEmbeddingsForDocuments(documents, dataset);

    // Get UMAP projection and clusters from dataset (pipeline stores these separately)
    const projection = [];
    const clusters = [];
    documents.forEach((doc, idx) => {
        if (doc.x !== undefined && doc.y !== undefined) {
            projection.push([doc.x, doc.y]);
        } else if (dataset.projection && dataset.projection[idx]) {
            const p = dataset.projection[idx];
            projection.push([p[0], p[1]]);
        } else {
            projection.push([0, 0]);
        }

        if (doc.cluster !== undefined) {
            clusters.push(doc.cluster);
        } else if (dataset.clusters && dataset.clusters[idx] !== undefined) {
            clusters.push(dataset.clusters[idx]);
        } else {
            clusters.push(-1);
        }
    });

    // Get metadata schema
    const metadataSchema = dataset.metadataSchema
        || (typeof window !== 'undefined' ? window.metadataSchema : null)
        || {};

    // Build unified payload
    const payload = {
        version: '3.0',
        export_type: exportType,
        exported_at: new Date().toISOString(),

        metadata: {
            document_count: documents.length,
            embedding_model: pipeline.embeddings?.modelName || 'unknown',
            embedding_dimension: pipeline.embeddings?.dimension || 384,
            text_column: dataset.textColumn || null,

            // Selection-specific metadata
            ...(exportType !== 'full' && {
                selection_info: {
                    type: exportType,
                    filters_applied: metadata.filters || [],
                    original_dataset_size: dataset.documents?.length || null,
                    query: metadata.query || null
                }
            })
        },

        documents: documents.map((doc, idx) => {
            const cluster = doc.cluster !== undefined ? doc.cluster
                : (dataset.clusters && dataset.clusters[idx] !== undefined ? dataset.clusters[idx] : null);
            const coords = projection[idx] || [0, 0];
            return {
                id: getDocumentKey(doc, idx),
                text: doc.text || doc.content || '',
                cluster: cluster,
                cluster_probability: doc.cluster_probability !== undefined ? doc.cluster_probability
                    : (doc.metadata?.cluster_probability !== undefined ? doc.metadata.cluster_probability : null),
                umap_coords: { x: coords[0], y: coords[1] },
                metadata: doc.metadata || {}
            };
        }),

        embeddings: {
            parent: {
                vectors: parentVectors,
                mode: 'query',
                description: 'Parent document embeddings for semantic search'
            },
            chunks: {
                vectors: chunkVectors,
                mode: 'passage',
                description: 'Chunk embeddings for RAG/HYDE'
            },
            chunk_map: chunkMap
        },

        visualization: {
            projection_2d: projection,
            clusters: clusters
        },

        metadata_schema: metadataSchema
    };

    // Add custom cluster names if any exist
    if (typeof window.getClusterNames === 'function') {
        const clusterNames = window.getClusterNames();
        if (clusterNames && Object.keys(clusterNames).length > 0) {
            payload.custom_cluster_names = clusterNames;
        }
    }

    return payload;
}

/**
 * Get embeddings for specific documents from the dataset
 */
async function getEmbeddingsForDocuments(documents, dataset) {
    const embeddings = dataset.embeddings;
    if (!embeddings || !embeddings.parent) {
        console.warn('No embeddings available');
        return {
            parentVectors: [],
            chunkVectors: [],
            chunkMap: {}
        };
    }

    // Build document index map
    const docIndexMap = new Map();
    const docKeyMap = new Map();

    if (Array.isArray(dataset.documents)) {
        dataset.documents.forEach((doc, idx) => {
            const key = getDocumentKey(doc, idx);
            if (key !== null) {
                docIndexMap.set(String(key), idx);
                docKeyMap.set(idx, String(key));
            }
        });
    }

    // Extract parent vectors for selected documents
    const parentVectors = [];
    const selectedDocKeys = new Set();

    documents.forEach((doc, idx) => {
        const key = getDocumentKey(doc);
        const docIndex = Number.isFinite(doc.index) ? doc.index : (key ? docIndexMap.get(String(key)) : null);

        if (docIndex !== undefined && docIndex !== null) {
            const vector = embeddings.parent[docIndex];
            if (vector) {
                const vectorArray = Array.from(vector);
                parentVectors.push({
                    doc_id: String(getDocumentKey(doc, idx)),
                    doc_index: docIndex,
                    vector: vectorArray
                });
                selectedDocKeys.add(docKeyMap.get(docIndex) || String(docIndex));
            }
        }
    });

    // Extract chunk vectors for selected documents
    const chunkVectors = [];
    const chunkMap = {};
    const chunks = embeddings.chunks || [];

    chunks.forEach(chunk => {
        const parentKey = String(chunk.docId ?? chunk.parent_id ?? chunk.doc_id);

        if (selectedDocKeys.has(parentKey)) {
            const chunkData = {
                chunk_id: chunk.chunkId,
                doc_id: parentKey,
                text: chunk.text,
                chunk_index: chunk.chunkIndex,
                vector: Array.from(chunk.embedding || []),
                metadata: chunk.metadata || {}
            };

            chunkVectors.push(chunkData);
            chunkMap[chunk.chunkId] = parentKey;
        }
    });

    return { parentVectors, chunkVectors, chunkMap };
}

// ============================================================================
// EXPORT FUNCTIONS
// ============================================================================

/**
 * Export full dataset
 */
async function exportFullDataset() {
    const pipeline = window.browserML?.pipeline;
    if (!pipeline || !pipeline.currentDataset) {
        showToast('No dataset available to export', 'warning');
        return;
    }

    showToast('Preparing full dataset export...', 'info');

    try {
        const documents = pipeline.currentDataset.documents || [];
        const payload = await buildUnifiedExportPayload(documents, 'full');

        const jsonString = JSON.stringify(payload, null, 2);
        const filename = generateFilename('vectoria-full', 'vectoria.json');
        downloadFile(jsonString, filename);

        const sizeKB = (jsonString.length / 1024).toFixed(2);
        showToast(`Exported ${documents.length} documents (${sizeKB} KB) to ${filename}`, 'success');

    } catch (error) {
        console.error('❌ Export failed:', error);
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

/**
 * Export filtered/lasso selection
 */
async function exportSelection(documents, selectionType = 'filtered', metadata = {}) {
    if (!documents || documents.length === 0) {
        showToast('No documents to export', 'warning');
        return;
    }

    showToast(`Preparing ${selectionType} selection export...`, 'info');

    try {
        const payload = await buildUnifiedExportPayload(documents, selectionType, metadata);

        const jsonString = JSON.stringify(payload, null, 2);
        const filename = generateFilename(`vectoria-${selectionType}`, 'vectoria.json');
        downloadFile(jsonString, filename);

        const sizeKB = (jsonString.length / 1024).toFixed(2);
        showToast(`Exported ${documents.length} documents (${sizeKB} KB) to ${filename}`, 'success');

    } catch (error) {
        console.error('❌ Export failed:', error);
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

/**
 * Export RAG results
 */
async function exportRAGResults(documents, query = '') {
    if (!documents || documents.length === 0) {
        showToast('No RAG results to export', 'warning');
        return;
    }

    showToast('Preparing RAG results export...', 'info');

    try {
        const payload = await buildUnifiedExportPayload(documents, 'rag_results', { query });

        const jsonString = JSON.stringify(payload, null, 2);
        const filename = generateFilename('vectoria-rag-results', 'vectoria.json');
        downloadFile(jsonString, filename);

        const sizeKB = (jsonString.length / 1024).toFixed(2);
        showToast(`Exported ${documents.length} RAG results (${sizeKB} KB) to ${filename}`, 'success');

    } catch (error) {
        console.error('❌ Export failed:', error);
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// IMPORT FUNCTION
// ============================================================================

/**
 * Import dataset from .vectoria.json file
 */
async function importDataset(file) {
    if (!file) {
        throw new Error('No file provided');
    }

    const pipeline = window.browserML?.pipeline;
    if (!pipeline) {
        throw new Error('Pipeline not available');
    }

    showToast('Loading dataset...', 'info');

    try {
        // Read file
        const text = await file.text();
        const data = JSON.parse(text);

        // Validate format
        if (data.version !== '3.0') {
            throw new Error(`Unsupported format version: ${data.version}. Please use version 3.0.`);
        }

        if (!data.documents || !Array.isArray(data.documents)) {
            throw new Error('Invalid dataset: missing documents array');
        }

        // Reconstruct dataset in pipeline format
        showToast('Rebuilding embeddings and indexes...', 'info');

        const reconstructedData = await reconstructDataset(data, pipeline);

        // Restore custom cluster names if present in export
        if (data.custom_cluster_names && typeof window.setClusterNames === 'function') {
            window.setClusterNames(data.custom_cluster_names);
        }

        // Return visualization data
        const vizData = {
            documents: reconstructedData.documents,
            projection: reconstructedData.projection,
            clusters: reconstructedData.clusters,
            metadataSchema: data.metadata_schema || {}
        };

        showToast(`Loaded ${data.documents.length} documents`, 'success');

        return {
            success: true,
            data: vizData,
            metadata: {
                export_type: data.export_type,
                exported_at: data.exported_at,
                document_count: data.documents.length
            }
        };

    } catch (error) {
        console.error('❌ Import failed:', error);
        showToast(`Import failed: ${error.message}`, 'error');
        throw error;
    }
}

/**
 * Reconstruct dataset from imported data
 */
async function reconstructDataset(data, pipeline) {
    // Rebuild documents array
    const documents = data.documents.map((doc, idx) => ({
        id: doc.id || idx,
        text: doc.text,
        metadata: doc.metadata || {},
        index: idx
    }));

    // Rebuild embeddings
    const parentEmbeddings = data.embeddings?.parent?.vectors?.map(v =>
        new Float32Array(v.vector || v)
    ) || [];

    const chunks = data.embeddings?.chunks?.vectors?.map(chunk => ({
        chunkId: chunk.chunk_id,
        docId: chunk.doc_id,
        text: chunk.text,
        chunkIndex: chunk.chunk_index,
        embedding: new Float32Array(chunk.vector || chunk.embedding || []),
        metadata: chunk.metadata || {}
    })) || [];

    // Rebuild visualization data
    const projection = data.visualization?.projection_2d || [];
    const clusters = data.visualization?.clusters || [];

    // Prepare documents with visualization coords
    const documentsWithViz = documents.map((doc, idx) => ({
        ...doc,
        x: projection[idx] ? projection[idx][0] : 0,
        y: projection[idx] ? projection[idx][1] : 0,
        cluster: clusters[idx] !== undefined ? clusters[idx] : -1,
        cluster_probability: 1.0
    }));

    // Store in pipeline
    pipeline.currentDataset = {
        documents: documentsWithViz,
        embeddings: {
            parent: parentEmbeddings,
            chunks: chunks,
            dimension: data.metadata?.embedding_dimension || 384,
            chunkToParentMap: data.embeddings?.chunk_map || {}
        },
        projection: projection.map(p => new Float32Array(p)),
        clusters: new Int32Array(clusters),
        metadataSchema: data.metadata_schema || {},
        embeddingModel: data.metadata?.embedding_model || 'unknown',
        textColumn: data.metadata?.text_column || null
    };

    // Rebuild search indexes
    if (pipeline.vectorSearch && parentEmbeddings.length > 0) {
        // Build doc IDs for vector search
        const docIds = documentsWithViz.map(d => String(d.id));
        await pipeline.vectorSearch.buildIndex(parentEmbeddings, docIds, documentsWithViz);
    }

    if (pipeline.chunkSearch && chunks.length > 0) {
        // Build chunk IDs and embeddings for chunk search
        const chunkIds = chunks.map(c => String(c.chunkId));
        const chunkEmbeddings = chunks.map(c => c.embedding);
        await pipeline.chunkSearch.buildIndex(chunkEmbeddings, chunkIds, chunks);
    }

    if (pipeline.bm25Search) {
        const docIds = documentsWithViz.map(d => String(d.id));
        pipeline.bm25Search.buildIndex(documentsWithViz, docIds);
    }

    return {
        documents: documentsWithViz,
        projection: projection,
        clusters: clusters
    };
}

// ============================================================================
// LEGACY RAG CONVERSATION EXPORT (separate format)
// ============================================================================

/**
 * Export RAG conversation history (separate from dataset exports)
 */
function exportRAGConversation(conversationHistory, format = 'json') {
    if (!conversationHistory || conversationHistory.length === 0) {
        showToast('No conversation history to export', 'warning');
        return;
    }

    try {
        if (format === 'json') {
            const exportData = {
                conversation: conversationHistory,
                exported_at: new Date().toISOString(),
                total_queries: conversationHistory.length
            };

            const json = JSON.stringify(exportData, null, 2);
            const filename = generateFilename('vectoria-conversation', 'json');
            downloadFile(json, filename);
            showToast(`Exported ${conversationHistory.length} Q&A pairs`, 'success');
        } else if (format === 'csv') {
            if (typeof Papa === 'undefined') {
                throw new Error('Papa Parse library not loaded');
            }

            const csvData = conversationHistory.map(entry => ({
                timestamp: entry.timestamp,
                query: entry.query,
                answer: entry.answer,
                num_sources: entry.sources?.length || 0,
                source_ids: entry.sources?.map(s => s.id || s.index).join('|') || '',
                model: entry.metadata?.model || '',
                temperature: entry.metadata?.temperature || ''
            }));

            const csv = Papa.unparse(csvData, { quotes: true, header: true });
            const filename = generateFilename('vectoria-conversation', 'csv');
            downloadFile(csv, filename);
            showToast(`Exported ${conversationHistory.length} Q&A pairs`, 'success');
        }
    } catch (error) {
        console.error('Export failed:', error);
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// GLOBAL EXPORTS
// ============================================================================

window.exportFullDataset = exportFullDataset;
window.exportSelection = exportSelection;
window.exportRAGResults = exportRAGResults;
window.importDataset = importDataset;
window.exportRAGConversation = exportRAGConversation;

// Backwards compatibility aliases
window.exportProcessedDataset = exportFullDataset;
window.importProcessedDataset = importDataset;

