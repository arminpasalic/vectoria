/**
 * Browser Integration Layer
 * Connects browser ML pipeline with existing Vectoria UI
 * Maintains all existing UI functionality while using browser-based processing
 */

import { pipeline } from './browser-ml/index.js';

// Global state management
window.browserML = {
    pipeline: pipeline,
    isReady: false,
    currentFile: null,
    processingStatus: null
};

function normalizeIndexValue(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === 'string' && value.trim() !== '') {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
}

function getDocIdFromIndex(index, documents) {
    if (!Array.isArray(documents) || !Number.isFinite(index) || index < 0 || index >= documents.length) {
        return null;
    }

    const doc = documents[index];
    if (!doc) {
        return null;
    }

    let docId = doc.id ?? doc.doc_id ?? doc.metadata?.doc_id ?? doc.metadata?.id ?? null;

    if (!docId && window.currentVisualizationData && Array.isArray(window.currentVisualizationData.points)) {
        const point = window.currentVisualizationData.points[index];
        if (point && point.doc_id !== undefined && point.doc_id !== null) {
            docId = point.doc_id;
        }
    }

    if (docId === undefined || docId === null) {
        return null;
    }

    return String(docId);
}

function collectDocIdsFromIndexCollection(collection, documents) {
    if (!collection) {
        return [];
    }

    const indices = [];

    if (typeof collection.forEach === 'function') {
        collection.forEach((value, key) => {
            let candidate = normalizeIndexValue(value);
            if (candidate === null) {
                candidate = normalizeIndexValue(key);
            }
            if (candidate !== null) {
                indices.push(candidate);
            }
        });
    } else if (Array.isArray(collection)) {
        collection.forEach(value => {
            const candidate = normalizeIndexValue(value);
            if (candidate !== null) {
                indices.push(candidate);
            }
        });
    } else if (typeof collection === 'object') {
        Object.values(collection).forEach(value => {
            const candidate = normalizeIndexValue(value);
            if (candidate !== null) {
                indices.push(candidate);
            }
        });
    }

    const docIds = [];
    const seen = new Set();

    indices.forEach(idx => {
        const docId = getDocIdFromIndex(idx, documents);
        if (docId && !seen.has(docId)) {
            seen.add(docId);
            docIds.push(docId);
        }
    });

    return docIds;
}

function matchesFiltersFallback(point, metadataFilters) {
    if (!metadataFilters || Object.keys(metadataFilters).length === 0) {
        return true;
    }

    const resolveValue = (fieldName) => {
        if (!point) return undefined;
        if (Object.prototype.hasOwnProperty.call(point, fieldName)) {
            return point[fieldName];
        }
        if (point.metadata && Object.prototype.hasOwnProperty.call(point.metadata, fieldName)) {
            return point.metadata[fieldName];
        }
        if (point.data && Object.prototype.hasOwnProperty.call(point.data, fieldName)) {
            return point.data[fieldName];
        }
        return undefined;
    };

    for (const [fieldName, filterConfig] of Object.entries(metadataFilters)) {
        if (!filterConfig || filterConfig.value === undefined || filterConfig.value === null || filterConfig.value === '') {
            continue;
        }

        const actualValue = resolveValue(fieldName);
        if (actualValue === undefined || actualValue === null || actualValue === '') {
            return false;
        }

        switch (filterConfig.type) {
            case 'category': {
                if (Array.isArray(filterConfig.value)) {
                    const match = filterConfig.value.some(val => String(actualValue) === String(val));
                    if (!match) {
                        return false;
                    }
                } else if (String(actualValue) !== String(filterConfig.value)) {
                    return false;
                }
                break;
            }
            case 'number': {
                const numericValue = Number(actualValue);
                if (!Number.isFinite(numericValue)) {
                    return false;
                }
                if (filterConfig.value.min !== undefined && numericValue < Number(filterConfig.value.min)) {
                    return false;
                }
                if (filterConfig.value.max !== undefined && numericValue > Number(filterConfig.value.max)) {
                    return false;
                }
                break;
            }
            case 'boolean': {
                const expected = filterConfig.value === true || String(filterConfig.value).toLowerCase() === 'true';
                const actual = actualValue === true ||
                    actualValue === 1 ||
                    String(actualValue).toLowerCase() === 'true' ||
                    String(actualValue).toLowerCase() === '1';
                if (expected !== actual) {
                    return false;
                }
                break;
            }
            case 'text': {
                const searchText = String(filterConfig.value).toLowerCase();
                const actualText = String(actualValue).toLowerCase();
                if (!actualText.includes(searchText)) {
                    return false;
                }
                break;
            }
            case 'date': {
                try {
                    const actualDate = new Date(actualValue);
                    if (filterConfig.value.min && actualDate < new Date(filterConfig.value.min)) {
                        return false;
                    }
                    if (filterConfig.value.max && actualDate > new Date(filterConfig.value.max)) {
                        return false;
                    }
                } catch (error) {
                    return false;
                }
                break;
            }
            default:
                break;
        }
    }

    return true;
}

function collectDocIdsFromActiveFilters(filters, documents) {
    if (!filters || Object.keys(filters).length === 0) {
        return [];
    }

    const points = window.currentVisualizationData && Array.isArray(window.currentVisualizationData.points)
        ? window.currentVisualizationData.points
        : null;

    const docIds = [];
    const seen = new Set();

    const matcher = (point) => {
        if (window.globalSearchInterface && typeof window.globalSearchInterface.matchesMetadataFilters === 'function') {
            try {
                return window.globalSearchInterface.matchesMetadataFilters(point, filters);
            } catch (error) {
                console.warn('matchesMetadataFilters failed, falling back to basic matcher:', error);
            }
        }
        return matchesFiltersFallback(point, filters);
    };

    if (points && points.length > 0) {
        points.forEach(point => {
            if (!matcher(point)) {
                return;
            }
            let docId = point.doc_id ? String(point.doc_id) : null;
            if (!docId && typeof point.index === 'number') {
                docId = getDocIdFromIndex(point.index, documents);
            }
            if (docId && !seen.has(docId)) {
                seen.add(docId);
                docIds.push(docId);
            }
        });
        return docIds;
    }

    documents.forEach((doc, index) => {
        const pseudoPoint = {
            ...doc.metadata,
            text: doc.text,
            metadata: doc.metadata,
            data: doc.metadata,
            index,
            doc_id: doc.id ?? doc.doc_id
        };

        if (matchesFiltersFallback(pseudoPoint, filters)) {
            const docId = getDocIdFromIndex(index, documents);
            if (docId && !seen.has(docId)) {
                seen.add(docId);
                docIds.push(docId);
            }
        }
    });

    return docIds;
}

function summarizeFilterValue(fieldName, filterConfig) {
    if (!filterConfig || filterConfig.value === undefined || filterConfig.value === null || filterConfig.value === '') {
        return null;
    }

    const value = filterConfig.value;
    if (Array.isArray(value)) {
        const sanitized = value.map(v => String(v)).filter(v => v.trim() !== '');
        if (!sanitized.length) return null;
        const preview = sanitized.slice(0, 3).join(', ');
        const suffix = sanitized.length > 3 ? ` +${sanitized.length - 3}` : '';
        return `${fieldName}: ${preview}${suffix ? ` (${suffix} more)` : ''}`;
    }

    if (value && typeof value === 'object') {
        const parts = [];
        if (value.min !== undefined && value.min !== null && value.min !== '') {
            parts.push(`≥ ${value.min}`);
        }
        if (value.max !== undefined && value.max !== null && value.max !== '') {
            parts.push(`≤ ${value.max}`);
        }
        if (parts.length) {
            return `${fieldName}: ${parts.join(' & ')}`;
        }
    }

    return `${fieldName}: ${value}`;
}

function summarizeFiltersForScope(filters) {
    if (!filters || Object.keys(filters).length === 0) {
        return [];
    }

    const summaries = [];
    Object.entries(filters).forEach(([field, filterConfig]) => {
        const summary = summarizeFilterValue(field, filterConfig);
        if (summary) {
            summaries.push(summary);
        }
    });

    if (!summaries.length) {
        return [];
    }

    const limited = summaries.slice(0, 3);
    if (summaries.length > 3) {
        limited.push(`+${summaries.length - 3} more filter${summaries.length - 3 === 1 ? '' : 's'}`);
    }

    return [`Filters: ${limited.join(' • ')}`];
}

function determineRAGScope() {
    const documents = Array.isArray(pipeline?.currentDataset?.documents)
        ? pipeline.currentDataset.documents
        : [];
    const totalDocuments = documents.length;
    const activeFilters = window.getActiveMetadataFilters ? window.getActiveMetadataFilters() : null;
    const filterDetailLines = summarizeFiltersForScope(activeFilters);

    const createScope = ({ type, docIds, label, details = [] }) => {
        const normalizedDocIds = Array.isArray(docIds) ? docIds : null;
        const scopedCount = normalizedDocIds ? normalizedDocIds.length : totalDocuments;
        return {
            scopeType: type,
            docIds: normalizedDocIds,
            scopedCount,
            totalDocuments,
            label,
            details: details.filter(Boolean)
        };
    };

    const baseScope = createScope({
        type: 'all',
        docIds: null,
        label: totalDocuments > 0 ? `All ${totalDocuments} documents` : 'No documents loaded',
        details: ['Full dataset']
    });

    if (!totalDocuments) {
        return baseScope;
    }

    const lassoSet = window.mainVisualization ? window.mainVisualization.lassoSelectedIndices : null;
    if (lassoSet && typeof lassoSet.size === 'number' && lassoSet.size > 0) {
        const docIds = collectDocIdsFromIndexCollection(lassoSet, documents);
        return createScope({
            type: 'lasso',
            docIds,
            label: `Lasso selection (${docIds.length})`,
            details: ['Selection: Lasso on visualization', ...filterDetailLines]
        });
    }

    const metadataFiltered = window.mainVisualization ? window.mainVisualization.metadataFilteredIndices : null;
    if (metadataFiltered) {
        const docIds = collectDocIdsFromIndexCollection(metadataFiltered, documents);
        return createScope({
            type: 'filters',
            docIds,
            label: `Filtered view (${docIds.length})`,
            details: filterDetailLines.length ? filterDetailLines : ['Active metadata filters']
        });
    }

    if (activeFilters && Object.keys(activeFilters).length > 0) {
        const docIds = collectDocIdsFromActiveFilters(activeFilters, documents);
        return createScope({
            type: 'filters',
            docIds,
            label: `Filtered view (${docIds.length})`,
            details: filterDetailLines.length ? filterDetailLines : ['Active metadata filters']
        });
    }

    return baseScope;
}

window.getCurrentRAGScope = determineRAGScope;

/**
 * Initialize browser ML system with progress UI
 */
export async function initializeBrowserML() {
    // Clear any existing progress tracking from previous loads
    modelProgressTracking.clear();

    // Update button to show initialization started
    updateUploadButtonStatus('Initializing AI models...');

    // Show loading modal
    showModelLoadingModal();
    try {
        await pipeline.initialize({
            onEmbeddingsProgress: (progress) => {
                updateModelLoadingProgress('embeddings', progress);
                // progress.progress is a DECIMAL (0-1), not a percentage
                const percent = Math.min(Math.floor((progress.progress || 0) * 100), 100);
                updateUploadButtonStatus(`Loading embeddings: ${percent}%`);
                // Update size badge dynamically
                if (progress.total && progress.total > 0) {
                    updateModelSizeBadge('embeddings', progress.loaded, progress.total);
                }
            },
            onLLMProgress: (progress) => {
                updateModelLoadingProgress('llm', progress);
                // progress.progress is already a percentage (0-100), not a decimal
                const percent = Math.min(Math.floor(progress.progress || 0), 100);
                updateUploadButtonStatus(`Loading LLM: ${percent}%`);
                // Parse size from WebLLM progress text - multiple patterns
                if (progress.text) {
                    // Pattern 1: "500MB/1.4GB" or "500 MB / 1.4 GB"
                    let sizeMatch = progress.text.match(/(\d+(?:\.\d+)?)\s*(MB|GB)\s*\/\s*(\d+(?:\.\d+)?)\s*(MB|GB)/i);
                    // Pattern 2: "Fetching param cache[0/14]:" followed by percentage
                    if (!sizeMatch) {
                        // Try to extract from "[X/Y]" pattern for chunk counting
                        const chunkMatch = progress.text.match(/\[(\d+)\/(\d+)\]/);
                        if (chunkMatch) {
                            const current = parseInt(chunkMatch[1]);
                            const total = parseInt(chunkMatch[2]);
                            // Estimate ~100MB per chunk for LLM
                            updateModelSizeBadge('llm', current * 100 * 1024 * 1024, total * 100 * 1024 * 1024);
                        }
                    }
                    if (sizeMatch) {
                        const loadedVal = parseFloat(sizeMatch[1]);
                        const loadedUnit = sizeMatch[2].toUpperCase();
                        const totalVal = parseFloat(sizeMatch[3]);
                        const totalUnit = sizeMatch[4].toUpperCase();
                        const loaded = loadedUnit === 'GB' ? loadedVal * 1024 * 1024 * 1024 : loadedVal * 1024 * 1024;
                        const total = totalUnit === 'GB' ? totalVal * 1024 * 1024 * 1024 : totalVal * 1024 * 1024;
                        updateModelSizeBadge('llm', loaded, total);
                    }
                }
            },
            onComplete: () => {
                // Manually set both models to 100% since we filtered "ready" signals
                updateModelLoadingProgress('embeddings', { status: 'loading', progress: 1 });
                updateModelLoadingProgress('llm', { status: 'loading', progress: 1 });

                // Set final badge sizes from tracking data
                if (downloadTracking.embeddings.totalBytes > 0) {
                    setModelSizeBadgeFinal('embeddings', downloadTracking.embeddings.totalBytes);
                }
                if (downloadTracking.llm.totalBytes > 0) {
                    setModelSizeBadgeFinal('llm', downloadTracking.llm.totalBytes);
                    // Cache real LLM download size for model-change modal
                    try {
                        const llmModelId = window.ConfigManager?.getConfig()?.llm?.model_id || localStorage.getItem('vectoria_llm_model') || 'gemma-2-2b-it-q4f32_1-MLC';
                        const cached = JSON.parse(localStorage.getItem('vectoria_model_download_sizes') || '{}');
                        cached[llmModelId] = formatBytes(downloadTracking.llm.totalBytes);
                        localStorage.setItem('vectoria_model_download_sizes', JSON.stringify(cached));
                        window.__webllmRealDownloadSizes = cached;
                    } catch (_) {}
                }

                window.browserML.isReady = true;
                hideModelLoadingModal();
                enableUploadUI();
                showToast('AI models loaded successfully! You can now upload files.', 'success');
            }
        });

        return true;
    } catch (error) {
        console.error('Browser ML initialization failed:', error);
        console.error('Error stack:', error.stack);
        hideModelLoadingModal();
        updateUploadButtonStatus('Model loading failed');
        showToast(`Failed to load AI models: ${error.message}`, 'error');
        throw error;
    }
}

/**
 * Update upload button status text
 */
function updateUploadButtonStatus(text) {
    const uploadBtnText = document.getElementById('upload-btn-text');
    const uploadBtn = document.getElementById('upload-btn');

    if (uploadBtnText) {
        uploadBtnText.textContent = text;
    } else if (uploadBtn) {
        uploadBtn.textContent = text;
    }

    // console.log(`Button status: ${text}`);
}

/**
 * Enable upload UI after models are loaded
 */
function enableUploadUI() {
    const uploadBtn = document.getElementById('upload-btn');
    const uploadBtnText = document.getElementById('upload-btn-text');
    const fileInput = document.getElementById('csv-file');
    const modelLoadingStatus = document.getElementById('model-loading-status');

    if (uploadBtn) {
        const hasFile = !!(fileInput && fileInput.files && fileInput.files.length > 0);
        uploadBtn.disabled = !hasFile;
        if (uploadBtnText) {
            uploadBtnText.textContent = 'Upload';
        } else {
            uploadBtn.textContent = 'Upload';
        }
    }

    if (fileInput) {
        fileInput.disabled = false;
        // Also enable the custom file button
        const customFileBtn = document.getElementById('custom-file-btn');
        if (customFileBtn) {
            customFileBtn.disabled = false;
        }
        fileInput.addEventListener('change', () => {
            if (!uploadBtn) return;
            const hasFile = !!(fileInput.files && fileInput.files.length > 0);
            uploadBtn.disabled = !hasFile;
        });
    }

    if (modelLoadingStatus) {
        modelLoadingStatus.style.display = 'none';
    }

}

/**
 * Handle file upload (replaces Flask POST /upload)
 */
export async function handleBrowserFileUpload(file) {
    try {
        // Parse file to get columns
        const parsedData = await pipeline.fileProcessor.parseFile(file);

        window.browserML.currentFile = file;
        window.browserML.parsedData = parsedData;

        // Show column selection UI (keep existing UI)
        displayColumnSelection(parsedData.columns, parsedData.data);

        return {
            success: true,
            fileName: file.name,
            columns: parsedData.columns,
            rowCount: parsedData.rowCount
        };

    } catch (error) {
        console.error('File upload failed:', error);
        throw error;
    }
}

/**
 * Process file with selected text column (replaces Flask POST /process)
 */
export async function handleBrowserFileProcessing(textColumn, options = {}) {
    if (!window.browserML.currentFile) {
        throw new Error('No file uploaded');
    }

    // Show processing progress modal
    showProcessingModal();

    try {
        const result = await pipeline.processFile(
            window.browserML.currentFile,
            textColumn,
            options,
            (progress) => {
                updateProcessingProgress(progress);
            }
        );

        // Hide processing modal
        hideProcessingModal();

        // Update visualization and metadata system
        await updateVisualizationWithBrowserData(result);

        // Load metadata schema and populate filter UI
        if (window.loadMetadataFromProcessedData) {
            // console.log('Loading metadata for filters and RAG...');
            await window.loadMetadataFromProcessedData();

            // Explicitly call RAG metadata population after loading
            if (window.populateRAGMetadataFields) {
                await window.populateRAGMetadataFields();
            } else {
                console.warn('window.populateRAGMetadataFields not available yet');
            }
        } else {
            console.warn('window.loadMetadataFromProcessedData not available');
        }

        // Show processing summary modal with timing info
        if (window.showProcessingSummaryModal) {
            // console.log('Showing processing summary modal with result:', result);
            // console.log('Result has timings?', !!result.timings);
            // console.log('Timings data:', result.timings);
            window.showProcessingSummaryModal(result);
        } else {
            // Fallback: Show success message and switch to explore tab
            console.warn('window.showProcessingSummaryModal not found - using fallback');
            showToast(`Processed ${result.numDocuments} documents successfully!`, 'success');
            activateTab('explore-tab');
        }

        return result;

    } catch (error) {
        hideProcessingModal();
        console.error('Processing failed:', error);
        throw error;
    }
}

/**
 * Update visualization with processed data
 */
async function updateVisualizationWithBrowserData(result) {
    try {
        // Get visualization data from pipeline
        const vizData = pipeline.getVisualizationData();

        if (!vizData) {
            console.warn('No visualization data available');
            return;
        }

        // Format points with metadata spread at top level for UI components
        const canvasPoints = vizData.documents.map((doc, index) => ({
            index: index,
            x: vizData.projection ? vizData.projection[index][0] : 0,
            y: vizData.projection ? vizData.projection[index][1] : 0,
            cluster: vizData.clusters ? vizData.clusters[index] : -1,
            text: doc.text || doc.metadata?.text || `Document ${index + 1}`,
            // Use actual cluster probability from metadata, fallback to 1.0 if not available
            cluster_probability: doc.metadata?.cluster_probability ?? 1.0,
            ...doc.metadata // Spread all metadata at top level
        }));

        // Debug: Log sample point with metadata
        // if (canvasPoints.length > 0) {
        //     console.log('Sample point with metadata:', canvasPoints[0]);
        //     const metadataKeys = Object.keys(canvasPoints[0]).filter(k =>
        //         !['index', 'x', 'y', 'cluster', 'text', 'cluster_probability'].includes(k)
        //     );
        //     console.log(`Metadata fields found (${metadataKeys.length}):`, metadataKeys);
        // }

        // Update the main visualization canvas if it exists
        if (window.mainVisualization && vizData.documents) {
            window.mainVisualization.loadData(canvasPoints);
        }

        // Store formatted data globally for other components
        // IMPORTANT: Use canvasPoints (with spread metadata) not vizData.documents
        // This allows metadata extraction to find fields at the top level
        window.currentVisualizationData = {
            points: canvasPoints,  // Fixed: use formatted points with spread metadata
            projection: vizData.projection,
            clusters: vizData.clusters,
            numDocuments: vizData.numDocuments
        };

        // Mirror historic globals so export/filter utilities can read the dataset
        window.canvasData = canvasPoints;
        window.currentDataset = [...canvasPoints];

    } catch (error) {
        console.error('Failed to update visualization:', error);
    }
}

/**
 * Perform search (replaces Flask POST /search)
 */
export async function handleBrowserSearch(query, searchType = 'fast', numResults = 10, metadataOptions = {}) {
    const {
        includeMetadata = false,
        metadataFields = undefined,
        metadataFilters = {}
    } = metadataOptions;
    const hasMetadataFilters = metadataFilters && Object.keys(metadataFilters).length > 0;
    if (searchType === 'semantic') {
        const allowed = [5, 10, 20, 50];
        if (!allowed.includes(numResults)) {
            numResults = 10;
        }
    }

    if (includeMetadata) {
    }

    try {
        const results = await pipeline.search(query, {
            searchType: searchType,
            k: numResults
        });

        // Format results with metadata if requested
        const formattedResults = results.map(result => {
            if (!includeMetadata) {
                return result;
            }

            // Build metadata string for display
            const doc = pipeline.currentDataset?.documents?.[result.index];
            let docMetadata = {};
            if (doc && typeof doc === 'object' && doc.metadata && typeof doc.metadata === 'object') {
                docMetadata = doc.metadata;
            } else if (result.metadata && typeof result.metadata === 'object') {
                docMetadata = result.metadata;
            }

            if (!docMetadata || Object.keys(docMetadata).length === 0) {
                return result;
            }

            // If specific fields requested, filter metadata
            let metadataToInclude = docMetadata;
            if (metadataFields && Array.isArray(metadataFields) && metadataFields.length > 0) {
                metadataToInclude = {};
                metadataFields.forEach(field => {
                    if (docMetadata && docMetadata[field] !== undefined) {
                        metadataToInclude[field] = docMetadata[field];
                    }
                });
            }

            const sanitizedMetadata = {};
            Object.entries(metadataToInclude || {}).forEach(([key, value]) => {
                if (typeof key === 'string' && key.toLowerCase().includes('color')) {
                    return;
                }
                sanitizedMetadata[key] = value;
            });

            const metadataString = Object.entries(sanitizedMetadata)
                .map(([key, value]) => `${key}: ${value}`)
                .join(' | ');

            const baseText = (typeof result.text === 'string' && result.text.length > 0)
                ? result.text
                : (doc.text || docMetadata.text || '');

            return {
                ...result,
                text: baseText + (metadataString ? `\n[Metadata: ${metadataString}]` : ''),
                metadata: sanitizedMetadata
            };
        });

        const payload = {
            results: formattedResults,
            search_type: searchType,
            query,
            metadata_filters_applied: hasMetadataFilters,
            include_metadata: includeMetadata,
            active_filters: hasMetadataFilters ? Object.keys(metadataFilters) : [],
            metadata_filters: metadataFilters
        };

        // Update search results UI (keep existing UI)
        displaySearchResults(payload);

        // Highlight results on visualization if enabled
        if (document.getElementById('highlight-results')?.checked) {
            if (typeof highlightSearchResultsInVisualization === 'function') {
                highlightSearchResultsInVisualization(payload.results, query);
            }
        }

        return payload;

    } catch (error) {
        console.error('Search failed:', error);
        throw error;
    }
}

/**
 * Perform RAG query (replaces Flask POST /query)
 */
export async function handleBrowserRAGQuery(question, options = {}) {
    // Reset abort state at the very start of a new query
    if (pipeline && pipeline.rag) {
        pipeline.rag.resetAbort();
    }

    // Load RAG settings from ConfigManager (centralized config)
    const config = window.ConfigManager ? window.ConfigManager.getConfig() : {};
    const searchConfig = config.search || {};

    const {
        searchType: requestedSearchType = 'semantic',
        numResults = searchConfig.num_results || 5,
        vectorWeight = searchConfig.vector_weight !== undefined ? searchConfig.vector_weight : 0.6,
        enableBM25 = true,
        retrievalK = searchConfig.retrieval_k || 60,
        stream = false,
        includeMetadata = false,
        metadataFields = undefined
    } = options;

    if (requestedSearchType && requestedSearchType.toLowerCase() !== 'semantic') {
        console.warn(`Keyword retrieval mode is no longer supported for RAG API requests. Using semantic instead (requested: ${requestedSearchType}).`);
    }

    if (includeMetadata) {
    }

    const scopeInfo = determineRAGScope();
    if (!scopeInfo.totalDocuments) {
        showToast('No data available. Please upload and process a dataset first.', 'warning');
        return {
            answer: '',
            sources: [],
            metadata: { error: true, message: 'No data available' }
        };
    }

    const scopedDocIds = Array.isArray(scopeInfo.docIds) ? scopeInfo.docIds : null;
    const allowedDocIds = scopedDocIds && scopedDocIds.length > 0 ? scopedDocIds : null;

    if (scopeInfo.scopeType !== 'all' && (!allowedDocIds || allowedDocIds.length === 0)) {
        showToast('No documents match your current selection or filters. Clear the scope and try again.', 'warning');
        return {
            answer: '',
            sources: [],
            metadata: { error: true, message: 'No documents match filters' }
        };
    }

    // Show loading indicator early (so stop button works during HyDE too)
    showRAGLoading(scopeInfo);

    try {
        // Check if HyDE mode is enabled
        let searchText = question;
        let useHyDE = false;

        if (window.hydeMode && window.hydeMode.enabled) {
            try {
                // Generate HyDE text
                const hydeText = await pipeline.rag.generateHyDE(question);

                // Show review modal
                const approvedText = await window.showHyDEReviewModal(question, hydeText);

                searchText = approvedText;
                useHyDE = true;

                // Store HyDE answer for export and viewing
                window.lastHyDEAnswer = searchText;
                // Show HyDE viewer button
                const hydeViewerBtn = document.getElementById('hyde-viewer-btn');
                if (hydeViewerBtn) {
                    hydeViewerBtn.style.display = 'inline-block';
                } else {
                    console.error('❌ HyDE viewer button not found in DOM!');
                }

            } catch (hydeError) {
                // Check if this was a user cancellation or abort
                const isCancelled = hydeError.message.includes('cancelled') ||
                                   hydeError.message.includes('stopped by user');

                hideRAGLoading();

                // Show the AI Answer card with stopped/error message
                const answerCard = document.getElementById('rag-answer-card');
                const answerText = document.getElementById('rag-answer-text');
                const durationEl = document.getElementById('rag-duration');

                if (answerCard) answerCard.style.display = 'block';
                if (typeof updateExportButtonVisibility === 'function') updateExportButtonVisibility();

                if (isCancelled) {
                    if (answerText) answerText.textContent = 'Generation stopped.';
                    showToast('Search cancelled', 'info');
                } else {
                    console.error('❌ HyDE generation failed:', hydeError.message);
                    if (answerText) answerText.textContent = 'Generation failed.';
                    showToast(`HyDE failed: ${hydeError.message}`, 'error');
                }

                // Show duration
                if (durationEl && ragGenerationStartTime) {
                    const duration = ((Date.now() - ragGenerationStartTime) / 1000).toFixed(1);
                    durationEl.textContent = isCancelled ? `Stopped after ${duration}s` : `Failed after ${duration}s`;
                }

                // Return result with null sources to prevent caller from highlighting
                return {
                    answer: '',
                    sources: null,
                    metadata: isCancelled
                        ? { cancelled: true, wasStopped: true }
                        : { error: true, message: hydeError.message }
                };
            }
        }

        // Clear HyDE data if not using HyDE
        if (!useHyDE) {
            window.lastHyDEAnswer = null;
            const hydeViewerBtn = document.getElementById('hyde-viewer-btn');
            if (hydeViewerBtn) {
                hydeViewerBtn.style.display = 'none';
            }
        }

        let result;

        if (stream) {
            // Streaming response for real-time display
            result = await pipeline.queryRAG(question, {
                searchType: 'semantic',
                numResults: numResults,
                vectorWeight: vectorWeight,
                enableBM25: enableBM25,
                hydeText: useHyDE ? searchText : null,  // Pass HyDE text if enabled
                stream: true,
                includeMetadata: includeMetadata,
                metadataFields: metadataFields,
                retrievalK: retrievalK,
                allowedDocIds: allowedDocIds,
                onChunk: (chunk, fullText) => {
                    // Update answer card in real-time
                    updateRAGAnswerCard(fullText, null, true, scopeInfo);
                }
            });

            // Final update after streaming completes (to show context warning, duration, etc.)
            const streamWasStopped = result.metadata?.wasStopped || false;
            updateRAGAnswerCard(result.answer, result.sources, false, scopeInfo, streamWasStopped, result.metadata);

            // Display sources for streaming too
            if (result.sources && result.sources.length > 0) {
                displaySearchResults({
                    results: result.sources,
                    search_type: searchType,
                    query: question
                });

                // Store RAG sources and answer for export
                window.lastRAGSources = result.sources;
                window.lastRAGQuery = question;
                window.lastRAGAnswer = result.answer || '';

                // Highlight sources on visualization
                if (document.getElementById('highlight-results')?.checked) {
                    if (typeof highlightSearchResultsInVisualization === 'function') {
                        highlightSearchResultsInVisualization(result.sources, question);
                    }
                }
            }
        } else {
            // Regular response
            result = await pipeline.queryRAG(question, {
                searchType: 'semantic',
                numResults: numResults,
                vectorWeight: vectorWeight,
                enableBM25: enableBM25,
                hydeText: useHyDE ? searchText : null,  // Pass HyDE text if enabled
                includeMetadata: includeMetadata,
                metadataFields: metadataFields,
                retrievalK: retrievalK,
                allowedDocIds: allowedDocIds
            });
        }

        // Hide loading, show answer
        hideRAGLoading();

        // Check if this was a cancelled/error result
        const wasCancelled = result.metadata?.cancelled || result.metadata?.error;
        const wasStopped = result.metadata?.wasStopped || false;

        if (wasCancelled || (wasStopped && (!result.answer || result.answer.trim() === ''))) {
            // Show the AI Answer card with "Generation stopped" message
            const answerCard = document.getElementById('rag-answer-card');
            const answerText = document.getElementById('rag-answer-text');
            const durationEl = document.getElementById('rag-duration');

            if (answerCard) answerCard.style.display = 'block';
            if (typeof updateExportButtonVisibility === 'function') updateExportButtonVisibility();
            if (answerText) answerText.textContent = 'Generation stopped.';

            // Show duration if we have timing
            if (durationEl && ragGenerationStartTime) {
                const duration = ((Date.now() - ragGenerationStartTime) / 1000).toFixed(1);
                durationEl.textContent = `Stopped after ${duration}s`;
            }

            // Set sources to null to prevent caller from highlighting
            result.sources = null;
            return result;
        }

        updateRAGAnswerCard(result.answer, result.sources, false, scopeInfo, wasStopped, result.metadata);

        // Display sources - wrap in object format expected by displaySearchResults
        if (result.sources && result.sources.length > 0) {
            displaySearchResults({
                results: result.sources,
                search_type: 'semantic',
                query: question
            });

            // Store RAG sources and answer for export
            window.lastRAGSources = result.sources;
            window.lastRAGQuery = question;
            window.lastRAGAnswer = result.answer || '';

            // Highlight sources on visualization
            if (document.getElementById('highlight-results')?.checked) {
                if (typeof highlightSearchResultsInVisualization === 'function') {
                    highlightSearchResultsInVisualization(result.sources, question);
                }
            }
        }

        if (result && typeof result === 'object') {
            result.scope = scopeInfo;
        }

        return result;

    } catch (error) {
        hideRAGLoading();
        console.error('RAG query failed:', error);
        showToast(`RAG query failed: ${error.message}`, 'error');
        throw error;
    }
}

/**
 * Get visualization data (replaces Flask GET /visualization_data)
 */
export function getBrowserVisualizationData() {
    const vizData = pipeline.getVisualizationData();

    if (!vizData) {
        return null;
    }

    // Format data for existing visualization code (vectoria.js expects "points" array)
    const points = vizData.projection.map((coord, i) => {
        const doc = vizData.documents[i];
        return {
            index: i,
            x: coord[0],
            y: coord[1],
            doc_id: doc.id,
            chunk_id: doc.id, // Same as doc_id for browser version
            cluster: vizData.clusters[i],
            cluster_color: getClusterColor(vizData.clusters[i]),
            cluster_name: `Cluster ${vizData.clusters[i]}`,
            cluster_probability: 1.0, // Browser clustering doesn't have probabilities
            text: doc.text,
            // Spread all metadata fields at top level (vectoria.js expects this)
            ...doc.metadata,
            // Also keep metadata object for compatibility
            metadata: doc.metadata
        };
    });

    // Calculate clustering stats safely
    const uniqueClusters = [...new Set(vizData.clusters)].filter(c => c !== -1);
    const nClusters = uniqueClusters.length;
    const noisePoints = vizData.clusters.filter(c => c === -1).length;

    return {
        points: points, // vectoria.js expects "points" not "coordinates"
        coordinates: points, // Keep for backward compatibility
        clusters: vizData.clusters,
        num_documents: vizData.numDocuments,
        doc_count: vizData.numDocuments, // Alias
        clustering_stats: {
            n_clusters: nClusters,
            noise_points: noisePoints,
            total_points: vizData.clusters.length
        },
        reduction_stats: {
            method: 'UMAP',
            n_components: 2
        }
    };
}

/**
 * Get cluster color - delegates to global color manager
 * Uses the centralized Tailwind palette from vectoria.js
 */
function getClusterColor(clusterNum) {
    // Use the global color manager if available
    if (window.getClusterColor) {
        return window.getClusterColor(clusterNum);
    }

    // Fallback to VectoriaColorManager if available
    if (window.VectoriaColorManager) {
        return window.VectoriaColorManager.getColor(clusterNum);
    }

    // Last resort fallback (should never happen)
    console.warn('Color manager not available, using fallback color');
    return clusterNum === -1 ? '#9CA3AF' : '#3B82F6';
}

// ============================================================================
// UI Helper Functions (keep existing UI behavior)
// ============================================================================

// Download tracking for ETA and speed calculations
const downloadTracking = {
    embeddings: { startTime: null, bytesLoaded: 0, totalBytes: 50 * 1024 * 1024 },
    llm: { startTime: null, bytesLoaded: 0, totalBytes: 1.4 * 1024 * 1024 * 1024 }
};

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Update the model size badge dynamically during download
 * @param {string} modelType - 'embeddings' or 'llm'
 * @param {number} loaded - Bytes loaded so far
 * @param {number} total - Total bytes
 */
function updateModelSizeBadge(modelType, loaded, total) {
    const badge = document.getElementById(`${modelType}-size-badge`);
    if (!badge) return;

    // Update download tracking with actual total
    if (total > 0) {
        downloadTracking[modelType].totalBytes = total;
    }

    if (total > 0) {
        const loadedStr = formatBytes(loaded);
        const totalStr = formatBytes(total);
        // Show progress during download, just total when complete
        if (loaded >= total) {
            badge.textContent = totalStr;
            badge.classList.remove('downloading');
        } else {
            badge.textContent = `${loadedStr} / ${totalStr}`;
            badge.classList.add('downloading');
        }
    }
}

/**
 * Set the final model size badge when loading is complete
 * @param {string} modelType - 'embeddings' or 'llm'
 * @param {number} totalBytes - Total size in bytes
 */
function setModelSizeBadgeFinal(modelType, totalBytes) {
    const badge = document.getElementById(`${modelType}-size-badge`);
    if (!badge || !totalBytes) return;

    badge.textContent = formatBytes(totalBytes);
    badge.classList.remove('downloading');
}

function formatTime(seconds) {
    if (!seconds || seconds === Infinity || seconds < 0) return '--';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function showModelLoadingModal() {
    // Remove existing modal if any
    const existingModal = document.getElementById('model-loading-modal');
    if (existingModal) {
        existingModal.remove();
    }

    // Check if first visit (no cached models)
    let isFirstVisit = true;
    try { isFirstVisit = !localStorage.getItem('vectoria_models_cached'); } catch (_) {}

    const modal = document.createElement('div');
    modal.id = 'model-loading-modal';
    modal.className = 'modal-overlay ml-modal-overlay';
    modal.style.display = 'flex'; // Ensure it's visible
    modal.innerHTML = `
        <div class="modal-content ml-modal-content model-loading-content">
            <header class="processing-modal-header">
                <span class="processing-modal-icon" aria-hidden="true">
                    <svg class="processing-modal-icon-graphic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 3v12"></path>
                        <path d="M8 11l4 4 4-4"></path>
                        <path d="M6 19h12"></path>
                    </svg>
                </span>
                <div class="processing-modal-heading">
                    <h2>${isFirstVisit ? 'Downloading AI Models' : 'Loading AI Models'}</h2>
                    <p class="processing-modal-subtitle">${isFirstVisit ?
                        'First-time setup requires downloading AI models.' :
                        'Loading cached models.'}</p>
                    <p class="processing-modal-subtext" id="loading-subtext">${isFirstVisit ?
                        'This only happens once - models are cached for future visits. You can always change AI models in the advanced settings.' :
                        'This should only take a few seconds.'}</p>
                </div>
            </header>

            <div class="model-progress-container">
                <div class="model-progress-item" id="embeddings-progress-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <label>
                            <span class="model-loading-icon pending" id="embeddings-icon">
                                <i class="fas fa-circle-notch"></i>
                            </span>
                            Embeddings Model
                            <span class="model-size-badge" id="embeddings-size-badge">Loading...</span>
                        </label>
                        <span class="model-eta" id="embeddings-eta"></span>
                    </div>
                    <div class="download-progress-wrapper">
                        <div class="progress-bar">
                            <div id="embeddings-progress-bar" class="progress-fill"></div>
                        </div>
                        <div class="download-stats">
                            <span id="embeddings-progress-text">Waiting<span class="loading-dots"></span></span>
                            <span id="embeddings-speed"></span>
                        </div>
                    </div>
                </div>

                <div class="model-progress-item" id="llm-progress-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <label>
                            <span class="model-loading-icon pending" id="llm-icon">
                                <i class="fas fa-circle-notch"></i>
                            </span>
                            Large Language Model (LLM)
                            <span class="model-size-badge" id="llm-size-badge">Loading...</span>
                        </label>
                        <span class="model-eta" id="llm-eta"></span>
                    </div>
                    <div class="download-progress-wrapper">
                        <div class="progress-bar">
                            <div id="llm-progress-bar" class="progress-fill"></div>
                        </div>
                        <div class="download-stats">
                            <span id="llm-progress-text">Waiting<span class="loading-dots"></span></span>
                            <span id="llm-speed"></span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="model-loading-note">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-info-circle" style="color: var(--text-muted);"></i>
                    <span>${isFirstVisit ?
                        'Models are stored in your browser cache. Clear browser data to remove them.' :
                        'Data and models are cached locally. Your files and queries never leave your computer.'}</span>
                </div>
            </div>

            ${window.browserCapabilities && !window.browserCapabilities.isFullySupported ? `
                <div class="model-loading-note" style="margin-top: 0.75rem; border-left-color: #f59e0b;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <i class="fas fa-exclamation-triangle" style="color: #f59e0b;"></i>
                        <span>Some features may be limited. <a href="#" onclick="showCapabilityDetails(); return false;" style="color: var(--accent-color);">Check compatibility</a></span>
                    </div>
                </div>
            ` : ''}
        </div>
    `;

    document.body.appendChild(modal);
    // Initialize download tracking
    downloadTracking.embeddings.startTime = null;
    downloadTracking.llm.startTime = null;

    // Scroll to top of page in background
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Prevent background scrolling while modal is open
    document.body.classList.add('ml-modal-open');
}

// Track maximum progress per model type to prevent backwards movement
const modelProgressTracking = new Map();

function updateModelLoadingProgress(modelType, progress) {
    const progressBar = document.getElementById(`${modelType}-progress-bar`);
    const progressText = document.getElementById(`${modelType}-progress-text`);
    const iconEl = document.getElementById(`${modelType}-icon`);
    const speedEl = document.getElementById(`${modelType}-speed`);
    const etaEl = document.getElementById(`${modelType}-eta`);

    if (progressBar && progressText) {
        // Ignore "ready" status - only process actual progress updates during download
        if (progress.status === 'ready') {
            return;
        }

        let progressValue = progress.progress || 0;
        let percent = 0;

        // Progress is now normalized to 0-1 by the worker aggregation logic
        const text = progress.text || '';
        if (text.match(/(\d+(?:\.\d+)?)\s*%/)) {
            // Text contains explicit percentage - extract it
            const match = text.match(/(\d+(?:\.\d+)?)\s*%/);
            percent = parseFloat(match[1]);
        } else {
            // Convert 0-1 decimal to percentage
            percent = progressValue * 100;
        }

        // Clamp to valid range
        percent = Math.min(Math.max(percent, 0), 100);

        // Prevent backwards movement - only increase progress
        const tracking = modelProgressTracking.get(modelType) || { maxPercent: 0, lastLoggedPercent: -1, lastUpdateTime: 0 };
        if (percent < tracking.maxPercent) {
            percent = tracking.maxPercent;
        } else {
            tracking.maxPercent = percent;
        }

        // Throttle updates - only update if progress increased by at least 0.5% or 100ms passed
        const now = Date.now();
        const timeSinceLastUpdate = now - tracking.lastUpdateTime;
        const percentIncrease = percent - (tracking.lastDisplayedPercent || 0);

        if (percentIncrease < 0.5 && timeSinceLastUpdate < 100 && percent < 100) {
            // Skip this update - not enough change
            return;
        }

        tracking.lastDisplayedPercent = percent;
        tracking.lastUpdateTime = now;

        // Start tracking time on first real progress
        const dlTrack = downloadTracking[modelType];
        if (!dlTrack.startTime && percent > 0) {
            dlTrack.startTime = now;
        }

        // Update icon state
        if (iconEl) {
            iconEl.classList.remove('pending', 'loading', 'complete');
            if (percent >= 100) {
                iconEl.classList.add('complete');
                iconEl.innerHTML = '<i class="fas fa-check-circle"></i>';
            } else if (percent > 0) {
                iconEl.classList.add('loading');
                iconEl.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i>';
            } else {
                iconEl.classList.add('pending');
                iconEl.innerHTML = '<i class="fas fa-circle-notch"></i>';
            }
        }

        // Apply smooth CSS transition
        if (!progressBar.style.transition) {
            progressBar.style.transition = 'width 0.3s ease-out';
        }

        const displayPercent = Math.floor(percent);
        progressBar.style.width = `${displayPercent}%`;

        // Calculate speed and ETA
        if (dlTrack.startTime && percent > 0 && percent < 100) {
            const elapsedSeconds = (now - dlTrack.startTime) / 1000;
            const bytesLoaded = (percent / 100) * dlTrack.totalBytes;
            const speed = bytesLoaded / elapsedSeconds; // bytes per second
            const remainingBytes = dlTrack.totalBytes - bytesLoaded;
            const etaSeconds = speed > 0 ? remainingBytes / speed : 0;

            if (speedEl && speed > 0) {
                speedEl.textContent = `${formatBytes(speed)}/s`;
            }
            if (etaEl && etaSeconds > 0) {
                etaEl.textContent = `~${formatTime(etaSeconds)} remaining`;
            }
        }

        // Update progress text
        if (percent >= 100) {
            progressText.textContent = 'Complete';
            progressText.classList.remove('loading-dots');
            if (speedEl) speedEl.textContent = '';
            if (etaEl) etaEl.textContent = '';

            // Mark models as cached on complete
            if (modelType === 'llm') {
                try { localStorage.setItem('vectoria_models_cached', 'true'); } catch (_) {}
            }
        } else if (percent > 0) {
            progressText.textContent = `${displayPercent}%`;
            progressText.classList.remove('loading-dots');
        } else {
            progressText.innerHTML = 'Waiting<span class="loading-dots"></span>';
        }

        // Minimal logging - only log start and complete
        if (modelType === 'embeddings') {
            if (!tracking.hasLoggedStart) {
                tracking.hasLoggedStart = true;
            } else if (displayPercent === 100 && !tracking.hasLoggedComplete) {
                tracking.hasLoggedComplete = true;
            }
        }

        modelProgressTracking.set(modelType, tracking);
    }
}

function hideModelLoadingModal() {
    const modal = document.getElementById('model-loading-modal');
    if (modal) {
        modal.remove();
        // Clear all progress tracking when modal closes
        modelProgressTracking.clear();
    }

    document.body.classList.remove('ml-modal-open');
}

function showProcessingModal() {
    stageStatusMap.clear();
    lastStage = '';
    lastBatch = 0;
    lastStatus = '';
    lastProgressUpdate = 0;

    // Clear cached elements and RAF state for new modal
    cachedStageItems = null;
    cachedProgressBar = null;
    cachedMessageEl = null;
    rafPending = false;
    pendingProgress = null;

    const modal = document.createElement('div');
    modal.id = 'processing-modal';
    modal.className = 'modal-overlay ml-modal-overlay';
    modal.innerHTML = `
        <div class="modal-content ml-modal-content processing-content">
            <header class="processing-modal-header">
                <span class="processing-modal-icon" aria-hidden="true">
                    <svg class="processing-modal-icon-graphic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="9"></circle>
                        <path d="M12 7v5l3 2"></path>
                    </svg>
                </span>
                <div class="processing-modal-heading">
                    <h2>Processing Your Data</h2>
                    <p class="processing-modal-subtitle">Tracking each stage of processing.</p>
                </div>
            </header>
            <div class="processing-stages">
                <div class="processing-stage-list">
                    ${VISIBLE_STAGES.map((stage, index) => `
                        <div class="processing-stage-item${index === 0 ? ' active' : ''}" data-stage="${stage.id}">
                            <div class="stage-header">
                                <span class="stage-index">${index + 1}</span>
                                <span class="stage-label">${stage.label}</span>
                            </div>
                            <div class="stage-status">Waiting...</div>
                        </div>
                    `).join('')}
                </div>
                <div class="progress-bar">
                    <div id="processing-progress-bar" class="progress-fill"></div>
                </div>
                <div id="processing-message" class="processing-message">Starting...</div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Scroll to top of page in background
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Prevent background scrolling while processing modal is open
    document.body.classList.add('ml-modal-open');
}

// Smart throttling for smooth updates without blocking important changes
let lastProgressUpdate = 0;
let lastStage = '';
let lastBatch = 0;
let lastStatus = '';
const PROGRESS_UPDATE_THROTTLE = 16; // ~60fps (16ms)
const PROGRESS_UPDATE_THROTTLE_UMAP = 200; // 200ms for UMAP/clustering (real-time seconds, 5 updates/sec)

const PROCESSING_STAGES = [
    { id: 'parsing', label: 'Parsing file', show: true },
    { id: 'embedding', label: 'Generating embeddings', show: true },
    { id: 'umap', label: 'UMAP dimensionality reduction', show: true },
    { id: 'clustering', label: 'HDBSCAN clustering', show: true },
    { id: 'complete', label: 'Finished', show: false }
];

const VISIBLE_STAGES = PROCESSING_STAGES.filter(stage => stage.show !== false);
const STAGE_ORDER = PROCESSING_STAGES.map(stage => stage.id);
const VISIBLE_STAGE_IDS = VISIBLE_STAGES.map(stage => stage.id);

const stageStatusMap = new Map();

// Cache DOM elements to avoid repeated queries
let cachedStageItems = null;
let cachedProgressBar = null;
let cachedMessageEl = null;

// RAF batching to prevent blocking main thread with too many updates
let rafPending = false;
let pendingProgress = null;

function markStageCompleted(stageId) {
    if (!stageId) return;
    const item = document.querySelector(`.processing-stage-item[data-stage="${stageId}"]`);
    if (!item) return;

    const statusElement = item.querySelector('.stage-status');
    item.classList.add('completed');
    item.classList.remove('active');

    if (statusElement) {
        statusElement.textContent = 'Completed';
    }

    stageStatusMap.delete(stageId);
    stageStatusMap.set(stageId, { state: 'completed' });
}

function updateProcessingProgress(progress) {
    // Store latest progress and schedule RAF update if not already pending
    pendingProgress = progress;

    if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(() => {
            rafPending = false;
            if (pendingProgress) {
                updateProcessingProgressImpl(pendingProgress);
            }
        });
    }
}

function updateProcessingProgressImpl(progress) {
    // Cache DOM elements on first call to avoid repeated queries
    if (!cachedProgressBar) {
        cachedProgressBar = document.getElementById('processing-progress-bar');
        cachedMessageEl = document.getElementById('processing-message');
        cachedStageItems = document.querySelectorAll('.processing-stage-item');
    }

    const progressBar = cachedProgressBar;
    const messageEl = cachedMessageEl;

    const now = Date.now();
    const currentStageId = progress.stage || 'parsing';
    const stageDef = PROCESSING_STAGES.find(stage => stage.id === currentStageId);

    // If stage is not tracked in UI (e.g. extracting, indexing, saving), ignore it
    if (!stageDef) return;
    const stageName = stageDef.label;
    const displayIndex = getVisibleIndexForStage(stageDef.id);
    const isVisibleStage = stageDef.show !== false;
    const stageChanged = stageDef.id !== lastStage;
    const batchChanged = progress.batch && progress.batch !== lastBatch;
    const statusChanged = progress.status && progress.status !== lastStatus;
    const progressChanged = progress.progress !== undefined; // If progress value exists, it's changing

    // Use longer throttle for UMAP/clustering stages (to show elapsed seconds)
    const isTimedStage = currentStageId === 'umap' || currentStageId === 'clustering';
    const throttleTime = isTimedStage ? PROGRESS_UPDATE_THROTTLE_UMAP : PROGRESS_UPDATE_THROTTLE;

    // Allow update if: forceUpdate flag, OR important changes, OR enough time has passed
    const shouldForceUpdate = progress.forceUpdate === true;
    const hasImportantChanges = stageChanged || batchChanged || statusChanged;
    const timeElapsed = now - lastProgressUpdate >= throttleTime;

    // Always respect throttle unless forceUpdate is true or stage/batch/status changed
    if (!shouldForceUpdate && !hasImportantChanges && !timeElapsed) {
        return;
    }

    lastProgressUpdate = now;

    if (stageChanged && lastStage) {
        markStageCompleted(lastStage);
    }

    lastStage = stageDef.id;
    if (progress.batch) {
        lastBatch = progress.batch;
    }
    if (progress.status) {
        lastStatus = progress.status;
    }

    // Use cached stage items instead of querying DOM every time
    const stageItems = cachedStageItems;
    if (stageItems) {
        stageItems.forEach((item, index) => {
        const stageId = item.getAttribute('data-stage');
        const statusElement = item.querySelector('.stage-status');

        const shouldMarkCompleted = index < displayIndex || (!isVisibleStage && index === displayIndex);

        if (shouldMarkCompleted) {
            markStageCompleted(stageId);
        } else if (index === displayIndex) {
            item.classList.add('active');
            item.classList.remove('completed');
            if (statusElement) {
                let message = progress.message || (progress.batch && progress.totalBatches
                    ? `Batch ${progress.batch}/${progress.totalBatches}`
                    : 'In progress...');

                // Add device indicator for embedding stage in stage status
                if (progress.stage === 'embedding' && progress.deviceLabel && progress.batch) {
                    const deviceIcon = progress.device === 'webgpu' ? '🚀' : '⚙️';
                    message = `${deviceIcon} Batch ${progress.batch}/${progress.totalBatches} [${progress.deviceLabel}]`;
                }

                statusElement.textContent = message;
                stageStatusMap.set(stageId, { state: 'active', message });
            }
        } else {
            item.classList.remove('active', 'completed');
            const record = stageStatusMap.get(stageId);
            if (statusElement && (!record || record.state !== 'completed')) {
                statusElement.textContent = 'Waiting...';
                stageStatusMap.set(stageId, { state: 'idle' });
            }
        }
        });
    }

    if (progressBar) {
        const rawPercent = typeof progress.progress === 'number'
            ? progress.progress * 100
            : 0;
        const clampedPercent = Math.min(100, Math.max(0, rawPercent));
        const percent = Math.round(clampedPercent * 10) / 10;

        if (!progressBar.style.transition) {
            progressBar.style.transition = 'width 150ms ease-out';
        }

        progressBar.style.width = `${percent}%`;
    }

    if (messageEl) {
        const rawPercent = typeof progress.progress === 'number' ? progress.progress * 100 : 0;
        const percent = Math.round(rawPercent);

        if (progress.status === 'completed' || stageDef.id === 'complete') {
            messageEl.textContent = 'Completed (100%)';
            messageEl.classList.add('processing-message-complete');
        } else if (progress.message) {
            // Add device indicator for embedding stage
            let message = progress.message;
            if (progress.stage === 'embedding' && progress.deviceLabel) {
                const deviceIcon = progress.device === 'webgpu' ? '🚀' : '⚙️';
                message = `${deviceIcon} ${message} [${progress.deviceLabel}]`;
            }
            // Append percentage if not already in message
            if (!message.includes('%')) {
                message = `${message} (${percent}%)`;
            }
            messageEl.textContent = message;
            messageEl.classList.remove('processing-message-complete');
        } else if (progress.batch && progress.totalBatches) {
            let message = `${stageName}: Batch ${progress.batch}/${progress.totalBatches} (${percent}%)`;
            if (progress.stage === 'embedding' && progress.deviceLabel) {
                const deviceIcon = progress.device === 'webgpu' ? '🚀' : '⚙️';
                message = `${deviceIcon} ${message} [${progress.deviceLabel}]`;
            }
            messageEl.textContent = message;
            messageEl.classList.remove('processing-message-complete');
        } else {
            messageEl.textContent = `${stageName}`;
            messageEl.classList.remove('processing-message-complete');
        }
    }
}

function hideProcessingModal() {
    const modal = document.getElementById('processing-modal');
    if (modal) {
        modal.remove();
    }
    stageStatusMap.clear();

    // Clear cached DOM elements and RAF state when modal is closed
    cachedStageItems = null;
    cachedProgressBar = null;
    cachedMessageEl = null;
    rafPending = false;
    pendingProgress = null;

    document.body.classList.remove('ml-modal-open');
}

function getVisibleIndexForStage(stageId) {
    const orderIndex = STAGE_ORDER.indexOf(stageId);
    if (orderIndex === -1) {
        return 0;
    }
    for (let i = orderIndex; i >= 0; i--) {
        const candidateId = STAGE_ORDER[i];
        const visibleIndex = VISIBLE_STAGE_IDS.indexOf(candidateId);
        if (visibleIndex !== -1) {
            return visibleIndex;
        }
    }
    return 0;
}

// RAG generation timing
let ragGenerationStartTime = null;

function showRAGLoading(scopeInfo = null) {
    const answerCard = document.getElementById('rag-answer-card');
    const answerText = document.getElementById('rag-answer-text');
    const stopBtn = document.getElementById('stop-generation-btn');
    const durationEl = document.getElementById('rag-duration');
    const contextWarning = document.getElementById('rag-context-warning');
    const scopeContext = scopeInfo || determineRAGScope();

    // Reset abort state and start timing
    if (pipeline && pipeline.rag) {
        pipeline.rag.resetAbort();
    } else {
        console.warn('⚠️ pipeline.rag not available to reset abort state');
    }
    ragGenerationStartTime = Date.now();

    if (answerCard) answerCard.style.display = 'block';
    if (typeof updateExportButtonVisibility === 'function') updateExportButtonVisibility();
    if (answerText) answerText.innerHTML = '<div class="loading-spinner"></div> Generating answer...';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
    if (durationEl) durationEl.textContent = '';
    if (contextWarning) contextWarning.style.display = 'none';

    updateRAGScopeIndicators(scopeContext);
}

function hideRAGLoading() {
    const stopBtn = document.getElementById('stop-generation-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

/**
 * Setup stop generation button handler
 * Should be called once after DOM is loaded
 */
export function setupStopGenerationHandler() {
    const stopBtn = document.getElementById('stop-generation-btn');
    if (!stopBtn) {
        console.warn('Stop generation button not found');
        return;
    }

    stopBtn.addEventListener('click', () => {
        const answerText = document.getElementById('rag-answer-text');
        if (answerText && answerText.innerHTML.includes('Generating answer')) {
            answerText.innerHTML = '<div class="loading-spinner"></div> Stopping...';
        }
        if (pipeline && pipeline.rag) {
            pipeline.abortRAG();
        }
        stopBtn.style.display = 'none';
    });

}

function updateRAGScopeIndicators(scopeInfo) {
    const scopeText = document.getElementById('rag-scope-text');

    if (scopeText) {
        if (scopeInfo && scopeInfo.label) {
            const count = typeof scopeInfo.scopedCount === 'number'
                ? `${scopeInfo.scopedCount.toLocaleString()} document${scopeInfo.scopedCount === 1 ? '' : 's'}`
                : 'your selection';
            const from = scopeInfo.scopeType === 'all'
                ? 'the entire dataset'
                : scopeInfo.label.toLowerCase();
            scopeText.textContent = `Analyzing ${count} from ${from}`;
        } else {
            scopeText.textContent = 'Analyzing your entire dataset';
        }
    }
}

function updateRAGAnswerCard(answer, sources, isStreaming, scopeInfo = null, wasStopped = false, metadata = null) {
    const answerCard = document.getElementById('rag-answer-card');
    const answerText = document.getElementById('rag-answer-text');
    const answerMeta = document.getElementById('rag-answer-meta');
    const stopBtn = document.getElementById('stop-generation-btn');
    const durationEl = document.getElementById('rag-duration');
    const contextWarning = document.getElementById('rag-context-warning');

    if (!answerCard || !answerText) return;

    answerCard.style.display = 'block';
    if (typeof updateExportButtonVisibility === 'function') updateExportButtonVisibility();
    answerText.textContent = answer;

    // Hide stop button when not streaming
    if (!isStreaming && stopBtn) {
        stopBtn.style.display = 'none';
    }

    // Update duration when generation completes (not streaming anymore)
    if (!isStreaming && ragGenerationStartTime && durationEl) {
        const durationMs = Date.now() - ragGenerationStartTime;
        const durationSec = (durationMs / 1000).toFixed(1);
        if (wasStopped) {
            durationEl.textContent = `Stopped after ${durationSec}s`;
        } else {
            durationEl.textContent = `Generated in ${durationSec}s`;
        }
        ragGenerationStartTime = null;
    }

    // Show context limited warning if applicable
    if (contextWarning) {
        if (metadata?.contextLimited) {
            contextWarning.textContent = 'Note: Context window limit reached. Some retrieved content was truncated.';
            contextWarning.style.display = 'block';
        } else {
            contextWarning.style.display = 'none';
        }
    }

    if (answerMeta) {
        const metaParts = [];
        if (Array.isArray(sources)) {
            metaParts.push(`${sources.length} sources`);
        } else if (isStreaming) {
            metaParts.push('Streaming answer…');
        }
        if (scopeInfo && scopeInfo.scopeType && scopeInfo.scopeType !== 'all' && scopeInfo.label) {
            metaParts.push(scopeInfo.label);
        }
        answerMeta.textContent = metaParts.join(' • ');
    }

    updateRAGScopeIndicators(scopeInfo);
}

function showToast(message, type = 'info') {
    // Use existing toast notification system if available
    if (window.showToast) {
        window.showToast(message, type);
    } else {
    }
}

// ============================================================================
// Fetch Handler - Routes Flask API calls to browser ML functions
// ============================================================================

/**
 * Helper to safely read request body from various formats
 */
async function getRequestBody(options) {
    if (!options.body) {
        return null;
    }

    // If body is already a string
    if (typeof options.body === 'string') {
        return options.body;
    }

    // If body is FormData, we can't parse it as JSON
    if (options.body instanceof FormData) {
        return null; // FormData is handled separately
    }

    // If body has a text() method (ReadableStream/Blob)
    if (typeof options.body.text === 'function') {
        return await options.body.text();
    }

    // Try to convert to string
    return String(options.body);
}

/**
 * Browser ML fetch handler - called by the interceptor in index.html
 */
async function browserMLFetchHandler(url, options = {}) {
    // Extract path from URL (handle both relative and absolute URLs)
    let urlPath = url;
    if (typeof url === 'string') {
        try {
            const urlObj = new URL(url, window.location.href);
            urlPath = urlObj.pathname;
        } catch (e) {
            // Already a relative path
            urlPath = url;
        }
    }

    // Handle API routes (with or without /api/ prefix)
    if (typeof urlPath === 'string' && (
        urlPath.includes('/api/') ||
        urlPath.includes('/query') ||
        urlPath.includes('/search') ||
        urlPath.includes('/metadata_schema') ||
        urlPath.includes('/visualization_data') ||
        urlPath.includes('/config')
    )) {
        try {
            // Route to appropriate browser ML function
            if (urlPath.includes('/api/csv_columns')) {
                return await handleCSVColumnsAPI(options);
            }
            else if (urlPath.includes('/api/process_csv')) {
                return await handleProcessCSVAPI(options);
            }
            else if (urlPath.includes('/api/visualization_data') || urlPath.includes('/visualization_data')) {
                return await handleVisualizationDataAPI();
            }
            else if (urlPath.includes('/api/delete_all_data')) {
                return await handleDeleteAllDataAPI();
            }
            else if (urlPath.includes('/search')) {
                return await handleSearchAPI(options);
            }
            else if (urlPath.includes('/query')) {
                return await handleRAGQueryAPI(options);
            }
            else if (urlPath.includes('/api/set-metadata-filters')) {
                return await handleSetMetadataFiltersAPI(options);
            }
            else if (urlPath.includes('/config/update') || urlPath.includes('/config')) {
                return await handleConfigAPI(options);
            }
            else if (urlPath.includes('/metadata_schema')) {
                return await handleMetadataSchemaAPI();
            }
            else {
                // Unknown API endpoint - return 404
                console.warn(`Unknown API endpoint: ${urlPath}`);
                return new Response(JSON.stringify({ error: `Not implemented in browser version: ${urlPath}` }), {
                    status: 404,
                    headers: { 'Content-Type': 'application/json' }
                });
            }
        } catch (error) {
            console.error(`API handler error for ${urlPath}:`, error);
            return new Response(JSON.stringify({ error: error.message, stack: error.stack }), {
                status: 500,
                headers: { 'Content-Type': 'application/json' }
            });
        }
    }

    // For non-API calls, return error (shouldn't happen)
    return new Response(JSON.stringify({ error: 'Not an API call' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
    });
}

// Register the fetch handler globally so the interceptor can use it
window.browserMLFetch = browserMLFetchHandler;
// Signal that browser ML integration is ready
if (window.resolveBrowserML) {
    window.resolveBrowserML();
} else {
    console.error('window.resolveBrowserML not found!');
}

/**
 * Handle /api/csv_columns - Return columns and preview after file upload
 */
async function handleCSVColumnsAPI(options) {
    // Check if browser ML is ready
    if (!window.browserML?.isReady) {
        console.error('Browser ML not ready');
        return new Response(JSON.stringify({
            error: 'AI models are still loading. Please wait a moment and try again.'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    // File should be in the file input element
    const fileInput = document.getElementById('csv-file');
    const file = fileInput?.files[0];

    if (!file) {
        console.error('No file found');
        return new Response(JSON.stringify({ error: 'No file selected' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    try {
        const parsedData = await pipeline.fileProcessor.parseFile(file);

        // Store for later processing
        window.browserML.parsedData = parsedData;
        window.browserML.currentFile = file;

        // Detect file type from extension
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const fileTypeMap = {
            'csv': 'csv',
            'xlsx': 'excel',
            'xls': 'excel',
            'json': 'json',
            'txt': 'txt'
        };
        const fileType = fileTypeMap[fileExtension] || parsedData.fileType || 'unknown';

        // Return columns and preview (matching Flask response format)
        const response = {
            success: true,
            columns: parsedData.columns,
            sample_data: parsedData.data.slice(0, 5), // First 5 rows
            filename: file.name,
            file_type: fileType,  // Add file_type to response (vectoria.js expects this)
            num_rows: parsedData.rowCount
        };

        return new Response(JSON.stringify(response), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        console.error('File parsing error:', error);
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle /api/process_csv - Process file with selected text column
 */
async function handleProcessCSVAPI(options) {
    // Check if browser ML is ready
    if (!window.browserML?.isReady) {
        console.error('Browser ML not ready');
        return new Response(JSON.stringify({
            error: 'AI models are still loading. Please wait a moment and try again.'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    const bodyText = await getRequestBody(options);
    if (!bodyText) {
        return new Response(JSON.stringify({ error: 'No request body provided' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    const body = JSON.parse(bodyText);
    const textColumn = body.text_column;

    if (!window.browserML.parsedData) {
        return new Response(JSON.stringify({ error: 'No file data available' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    try {
        // Show processing modal (defined below)
        showProcessingModal();

        const result = await pipeline.processFile(
            window.browserML.currentFile,
            textColumn,
            {},
            (progress) => {
                updateProcessingProgress(progress);
            }
        );

        hideProcessingModal();

        // Show save dataset button now that processing is complete
        const saveBtn = document.getElementById('save-dataset-json');
        if (saveBtn) {
            saveBtn.style.display = 'inline-block';
        }

        // Return success response with ALL data including timings
        const response = {
            success: true,
            message: 'Processing complete',
            num_documents: result.numDocuments,
            num_clusters: result.numClusters,
            numDocuments: result.numDocuments,  // Add both formats for compatibility
            numClusters: result.numClusters,
            emptyRowCount: result.emptyRowCount || 0,  // Include empty row count
            duplicateCount: result.duplicateCount || 0,
            fileName: result.fileName,
            textColumn: result.textColumn,
            timings: result.timings  // Include timings
        };

        if (response.duplicateCount > 0) {
        }

        return new Response(JSON.stringify(response), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        hideProcessingModal();
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle /api/visualization_data - Return visualization coordinates
 */
async function handleVisualizationDataAPI() {
    const vizData = getBrowserVisualizationData();

    if (!vizData) {
        return new Response(JSON.stringify({ error: 'No visualization data available' }), {
            status: 404,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    return new Response(JSON.stringify(vizData), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
    });
}

/**
 * Handle /api/delete_all_data - Clear all browser storage
 */
async function handleDeleteAllDataAPI() {
    try {
        await pipeline.storage.clearAll();
        pipeline.clearDataset();

        return new Response(JSON.stringify({ success: true, message: 'All data cleared' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle /search - Perform search
 */
async function handleSearchAPI(options) {
    try {
        const bodyText = await getRequestBody(options);
        const body = JSON.parse(bodyText);

        // Check if data is processed
        if (!pipeline.currentDataset || pipeline.currentDataset.length === 0) {
            return new Response(JSON.stringify({
                error: 'No data available. Please upload and process a file first.',
                results: []
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        const payload = await handleBrowserSearch(
            body.query,
            body.search_type || 'fast',
            body.k || 10,
            {
                includeMetadata: body.include_metadata,
                metadataFields: body.metadata_fields,
                metadataFilters: body.metadata_filters || {}
            }
        );

        return new Response(JSON.stringify(payload), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        console.error('Search API error:', error);
        return new Response(JSON.stringify({
            error: error.message,
            results: []
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle /query - RAG query
 */
async function handleRAGQueryAPI(options) {
    try {
        const bodyText = await getRequestBody(options);
        const body = JSON.parse(bodyText);

        // Check if data is processed
        if (!pipeline.currentDataset || pipeline.currentDataset.length === 0) {
            return new Response(JSON.stringify({
                error: 'No data available. Please upload and process a file first.',
                answer: 'No data has been processed yet. Please upload and process a file before asking questions.',
                sources: []
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        const result = await handleBrowserRAGQuery(body.question, {
            searchType: body.search_type || 'semantic',
            numResults: body.num_results || 5,
            includeMetadata: body.include_metadata,
            metadataFields: body.metadata_fields
        });

        return new Response(JSON.stringify(result), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        console.error('RAG API error:', error);
        return new Response(JSON.stringify({
            error: error.message,
            answer: `Error: ${error.message}`,
            sources: []
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle /api/set-metadata-filters - Set metadata filters
 */
async function handleSetMetadataFiltersAPI(options) {
    const bodyText = await getRequestBody(options);
    const body = JSON.parse(bodyText);
    // Store filters globally for now (can be enhanced later)
    window.currentMetadataFilters = body.filters || {};

    return new Response(JSON.stringify({
        success: true,
        message: 'Filters applied successfully'
    }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
    });
}

/**
 * Handle /metadata_schema - Get metadata schema from processed data
 */
async function handleMetadataSchemaAPI() {
    try {
        const vizData = pipeline.getVisualizationData();

        if (!vizData || !vizData.documents || vizData.documents.length === 0) {
            // Return empty schema instead of 404 to prevent errors in UI
            return new Response(JSON.stringify({
                metadata_fields: {},
                message: 'No data processed yet. Please upload and process a file first.'
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        // Analyze metadata to infer schema
        const schema = {};
        const fieldSamples = {};

        vizData.documents.forEach(doc => {
            if (!doc.metadata) return;

            Object.entries(doc.metadata).forEach(([key, value]) => {
                if (!schema[key]) {
                    schema[key] = {
                        type: inferFieldType(value),
                        values: new Set(),
                        count: 0
                    };
                    fieldSamples[key] = [];
                }

                schema[key].count++;

                // Add to unique values (convert to string for Set)
                const strValue = value !== null && value !== undefined ? String(value) : 'null';
                schema[key].values.add(strValue);

                // Keep sample values
                if (fieldSamples[key].length < 10) {
                    fieldSamples[key].push(value);
                }
            });
        });

        // Format schema for response
        const schemaData = {};
        Object.entries(schema).forEach(([key, info]) => {
            schemaData[key] = {
                type: info.type,
                unique_values: info.values.size,
                sample_values: Array.from(info.values).slice(0, 10),
                count: info.count
            };
        });

        return new Response(JSON.stringify(schemaData), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });

    } catch (error) {
        console.error('Metadata schema error:', error);
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Infer field type from value
 */
function inferFieldType(value) {
    if (value === null || value === undefined || value === '' ||
        String(value).toLowerCase() === 'nan' || String(value).toLowerCase() === 'null') {
        return 'text';
    }

    if (typeof value === 'boolean') {
        return 'boolean';
    }

    if (typeof value === 'number' && isFinite(value)) {
        return 'number';
    }

    // Try to parse as number
    if (typeof value === 'string') {
        const num = parseFloat(value);
        if (!isNaN(num) && isFinite(num) && value.trim() === String(num)) {
            return 'number';
        }

        // Try to parse as date
        if (value.length > 8) {
            // ISO date pattern
            if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
                return 'date';
            }
        }

        // Check if it looks like a category (short text)
        if (value.length < 50 && !value.includes('\n')) {
            return 'category';
        }
    }

    return 'text';
}

/**
 * Handle /config - Get/Set configuration
 * Now proxies to ConfigManager for unified settings management
 */
async function handleConfigAPI(options) {
    const method = options.method || 'GET';

    if (method === 'GET') {
        // Get config from ConfigManager (uses localStorage)
        const config = window.ConfigManager ? window.ConfigManager.getConfig() : {};

        return new Response(JSON.stringify(config), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });

    } else if (method === 'POST') {
        // Save config via ConfigManager
        const bodyText = await getRequestBody(options);
        const config = JSON.parse(bodyText);

        if (window.ConfigManager) {
            window.ConfigManager.saveConfig(config);
        } else {
            // Fallback to direct localStorage if ConfigManager not available
            try {
                localStorage.setItem('vectoria_config', JSON.stringify(config));
            } catch (e) {
                console.warn('⚠️ Failed to save config to localStorage:', e);
            }
            console.warn('⚠️ ConfigManager not available, using direct localStorage');
        }

        return new Response(JSON.stringify({
            success: true,
            message: 'Configuration saved successfully'
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json' }
    });
}

/**
 * Setup Save/Load Dataset UI Handlers
 */
export function setupDatasetSaveLoadHandlers() {
    const saveBtn = document.getElementById('save-dataset-json');
    const loadBtn = document.getElementById('load-dataset-json');
    const fileInput = document.getElementById('load-dataset-file-input');

    if (!saveBtn || !loadBtn || !fileInput) {
        console.warn('⚠️ Save/Load dataset buttons not found');
        return;
    }

    fileInput.setAttribute('accept', '.vectoria.gz,.gz');

    // Save Dataset Handler
    saveBtn.addEventListener('click', async () => {
        if (!pipeline.currentDataset || !pipeline.currentDataset.documents) {
            if (typeof showToast === 'function') {
                showToast('No dataset to save. Process data first.', 'error');
            }
            return;
        }

        try {
            saveBtn.disabled = true;
            saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

            const { exportProcessedDataset } = await import('./export-import.js');
            await exportProcessedDataset(pipeline);

            if (typeof showToast === 'function') {
                showToast('Dataset saved successfully!', 'success');
            }

        } catch (error) {
            console.error('❌ Save dataset failed:', error);
            if (typeof showToast === 'function') {
                showToast(`Save failed: ${error.message}`, 'error');
            }
        } finally {
            saveBtn.disabled = false;
            saveBtn.innerHTML = '<i class="fas fa-save"></i> Save Dataset';
        }
    });

    // Load Dataset Handler
    loadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File Input Handler
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            loadBtn.disabled = true;
            loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';

            const { importProcessedDataset } = await import('./export-import.js');
            const result = await importProcessedDataset(file, pipeline);

            if (result?.success) {
                const vizData = {
                    coordinates: pipeline.currentDataset.projection,
                    clusters: pipeline.currentDataset.clusters,
                    labels: pipeline.currentDataset.clusterLabels || {},
                    documents: pipeline.currentDataset.documents
                };

                if (typeof updateVisualization === 'function') {
                    updateVisualization(vizData);
                }

                if (saveBtn) saveBtn.style.display = 'inline-block';

                const navButtons = document.querySelectorAll('[data-view]');
                navButtons.forEach(btn => {
                    if (btn.dataset.view === 'explore') {
                        btn.click();
                    }
                });

                if (typeof showToast === 'function') {
                    showToast(`Dataset loaded successfully! ${vizData.documents.length} documents ready.`, 'success');
                }
            }

        } catch (error) {
            console.error('❌ Load dataset failed:', error);
            if (typeof showToast === 'function') {
                showToast(`Load failed: ${error.message}`, 'error');
            }
        } finally {
            loadBtn.disabled = false;
            loadBtn.innerHTML = '<i class="fas fa-folder-open"></i> Load Dataset';
            fileInput.value = ''; // Reset input
        }
    });

}

// Export all handler functions
// Note: All functions are already exported inline with 'export async function ...'
// - initializeBrowserML (line 20)
// - handleBrowserFileUpload (line 84)
// - handleBrowserFileProcessing (line 113)
// - handleBrowserSearch (line 157)
// - handleBrowserRAGQuery (line 185)
// - getBrowserVisualizationData (line 244)
// - setupDatasetSaveLoadHandlers (NEW)
// No need for additional export statement
