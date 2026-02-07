/**
 * RAG-Vectoria Explorer - Enhanced Web Interface JavaScript
 * 
 * This file handles all client-side functionality including:
 * - Tab navigation and UI state management
 * - File upload with drag & drop support
 * - Real-time progress tracking
 * - Interactive visualization with Plotly
 * - Search interface and result handling
 * - Configuration management
 * - Toast notifications and error handling
 */

class RAGVectoriaApp {
    constructor() {
        this.currentTab = 'upload-tab';
        this.processingSession = null;
        this.visualizationData = null;
        this.searchResults = null;
        this.progressInterval = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.loadSystemConfig();
        this.showToast('Welcome to RAG-Vectoria Explorer!', 'info');
    }
    
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.activateTab(e.target.dataset.tab);
            });
        });
        
        // File upload form
        const uploadForm = document.getElementById('file-upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFileUpload();
            });
        }
        
        // File input change
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.handleFileSelection(e.target.files[0]);
            });
        }
        
        // Search functionality
        const searchBtn = document.getElementById('search-btn');
        const searchInput = document.getElementById('search-input');
        
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.performSearch());
        }
        
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.performSearch();
                }
            });
        }
        
        // Visualization controls
        const colorBySelect = document.getElementById('color-by-select');
        if (colorBySelect) {
            colorBySelect.addEventListener('change', () => this.updateVisualizationColors());
        }
        
        const resetZoomBtn = document.getElementById('reset-zoom-btn');
        if (resetZoomBtn) {
            resetZoomBtn.addEventListener('click', () => this.resetVisualizationZoom());
        }
        
        const clearHighlightsBtn = document.getElementById('clear-highlights-btn');
        if (clearHighlightsBtn) {
            clearHighlightsBtn.addEventListener('click', () => this.clearHighlights());
        }
    }
    
    setupDragAndDrop() {
        const fileLabel = document.querySelector('.file-input-label');
        if (!fileLabel) return;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileLabel.addEventListener(eventName, this.preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileLabel.addEventListener(eventName, () => {
                fileLabel.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileLabel.addEventListener(eventName, () => {
                fileLabel.classList.remove('dragover');
            }, false);
        });
        
        fileLabel.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
                document.getElementById('file-input').files = files;
            }
        }, false);
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    activateTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tab === tabId) {
                btn.classList.add('active');
            }
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        const targetTab = document.getElementById(tabId);
        if (targetTab) {
            targetTab.classList.add('active');
            this.currentTab = tabId;
            
            // Load tab-specific content
            if (tabId === 'explore-tab' && !this.visualizationData) {
                this.loadVisualizationData();
            } else if (tabId === 'settings-tab') {
                this.loadSystemConfig();
            }
        }
    }
    
    handleFileSelection(file) {
        if (!file) return;
        
        // Update UI to show file info
        const label = document.querySelector('.file-input-label span');
        if (label) {
            label.textContent = `Selected: ${file.name}`;
        }
        
        // Validate file type
        const allowedTypes = ['.csv', '.xlsx', '.xls', '.json', '.txt'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExt)) {
            this.showToast(`Unsupported file type: ${fileExt}`, 'error');
            return;
        }
        
        // Check file size (assuming 100MB limit from config)
        const maxSizeMB = 100;
        if (file.size > maxSizeMB * 1024 * 1024) {
            this.showToast(`File too large. Maximum size is ${maxSizeMB}MB`, 'error');
            return;
        }
        
        this.showToast(`File selected: ${file.name} (${this.formatFileSize(file.size)})`, 'success');
    }
    
    async handleFileUpload() {
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showToast('Please select a file first', 'error');
            return;
        }
        
        const createViz = document.getElementById('create-visualization').checked;
        
        try {
            this.showProcessingState('uploading');
            
            // Upload file
            const formData = new FormData();
            formData.append('file', file);
            
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const uploadResult = await uploadResponse.json();
            
            if (!uploadResponse.ok) {
                throw new Error(uploadResult.error || 'Upload failed');
            }
            
            // Process file
            const processFormData = new FormData();
            processFormData.append('file', file);
            processFormData.append('create_visualization', createViz);
            
            const processResponse = await fetch('/process', {
                method: 'POST',
                body: processFormData
            });
            
            const processResult = await processResponse.json();
            
            if (!processResponse.ok) {
                throw new Error(processResult.error || 'Processing failed');
            }
            
            this.showProcessingComplete(processResult);
            
            // Enable navigation tabs
            this.enableTab('explore-tab');
            this.enableTab('search-tab');
            
            // Update header stats
            this.updateHeaderStats();
            
        } catch (error) {
            console.error('Upload/Processing error:', error);
            this.showProcessingError(error.message);
        }
    }
    
    showProcessingState(state) {
        const sections = {
            upload: document.getElementById('upload-section'),
            progress: document.getElementById('progress-section'),
            results: document.getElementById('results-section'),
            error: document.getElementById('error-section')
        };
        
        // Hide all sections
        Object.values(sections).forEach(section => {
            if (section) section.style.display = 'none';
        });
        
        if (state === 'uploading' || state === 'processing') {
            sections.progress.style.display = 'block';
            this.startProgressPolling();
        }
        
        // Disable process button
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = true;
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }
    }
    
    showProcessingComplete(result) {
        this.stopProgressPolling();
        
        // Hide progress, show results
        document.getElementById('progress-section').style.display = 'none';
        document.getElementById('results-section').style.display = 'block';
        
        // Update result information
        document.getElementById('result-filename').textContent = result.filename;
        document.getElementById('result-chunks').textContent = `${result.num_chunks} chunks`;
        document.getElementById('result-embeddings').textContent = `${result.num_embeddings} embeddings`;
        
        if (result.has_visualization && result.visualization) {
            document.getElementById('result-clusters-card').style.display = 'block';
            document.getElementById('result-clusters').textContent = 
                `${result.visualization.num_clusters} clusters`;
        }
        
        // Re-enable process button
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-play"></i> Process Document';
        }
        
        this.showToast('Document processed successfully!', 'success');
        
        // Show warnings if any
        if (result.warnings && result.warnings.length > 0) {
            result.warnings.forEach(warning => {
                this.showToast(warning, 'warning');
            });
        }
    }
    
    showProcessingError(error) {
        this.stopProgressPolling();
        
        // Hide progress, show error
        document.getElementById('progress-section').style.display = 'none';
        document.getElementById('error-section').style.display = 'block';
        
        // Update error message
        document.getElementById('error-message').textContent = error;
        
        // Re-enable process button
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-play"></i> Process Document';
        }
        
        this.showToast(`Processing error: ${error}`, 'error');
    }
    
    startProgressPolling() {
        this.progressInterval = setInterval(async () => {
            try {
                const response = await fetch('/progress');
                const progress = await response.json();
                
                this.updateProgressDisplay(progress);
                
                if (progress.status === 'complete' || progress.status === 'error') {
                    this.stopProgressPolling();
                }
            } catch (error) {
                console.error('Progress polling error:', error);
            }
        }, 1000);
    }
    
    stopProgressPolling() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }
    
    updateProgressDisplay(progress) {
        document.getElementById('progress-percentage').textContent = `${progress.progress}%`;
        document.getElementById('progress-fill').style.width = `${progress.progress}%`;
        document.getElementById('progress-status').textContent = progress.current_step;
        
        if (progress.file_info && progress.file_info.filename) {
            document.getElementById('detail-filename').textContent = progress.file_info.filename;
        }
    }
    
    enableTab(tabId) {
        const tab = document.querySelector(`[data-tab="${tabId}"]`);
        if (tab) {
            tab.disabled = false;
        }
    }
    
    async updateHeaderStats() {
        try {
            const response = await fetch('/system/stats');
            const stats = await response.json();
            
            if (stats.storage && stats.storage.storage) {
                document.getElementById('docs-count').innerHTML = 
                    `<i class="fas fa-file-text"></i><span>${stats.storage.storage.num_documents} docs</span>`;
                document.getElementById('chunks-count').innerHTML = 
                    `<i class="fas fa-cubes"></i><span>${stats.storage.storage.num_chunks} chunks</span>`;
            }
        } catch (error) {
            console.error('Error updating header stats:', error);
        }
    }
    
    async loadVisualizationData() {
        try {
            this.showLoadingOverlay(true);
            
            const response = await fetch('/visualization_data');
            if (!response.ok) {
                throw new Error('No visualization data available');
            }
            
            this.visualizationData = await response.json();
            this.renderVisualization();
            
        } catch (error) {
            console.error('Visualization data error:', error);
            this.showVisualizationPlaceholder();
        } finally {
            this.showLoadingOverlay(false);
        }
    }
    
    renderVisualization() {
        // Skip Plotly visualization in favor of Canvas visualization for better performance
        // This method is kept for compatibility but redirects to Canvas visualization
        if (window.mainVisualization && this.visualizationData && this.visualizationData.documents) {
            // Convert data format for Canvas visualization
            const canvasPoints = this.visualizationData.documents.map((doc, index) => ({
                index: index,
                x: doc.x,
                y: doc.y,
                cluster: doc.cluster,
                text: doc.metadata?.text || doc.metadata?.filename || `Document ${index + 1}`,
                cluster_probability: 1.0, // Default if not available
                ...doc.metadata // Include all metadata
            }));
            
            window.mainVisualization.loadData(canvasPoints);
        }
        
        // Show cluster information
        this.displayClusterInformation();
    }
    
    displayClusterInformation() {
        if (!this.visualizationData.clustering_stats) return;
        
        const stats = this.visualizationData.clustering_stats;
        const clusterInfo = document.getElementById('cluster-info');
        const clusterDetails = document.getElementById('cluster-details');
        
        clusterDetails.innerHTML = `
            <div class="cluster-stat">
                <label>Total Clusters:</label>
                <span>${stats.n_clusters}</span>
            </div>
            <div class="cluster-stat">
                <label>Noise Points:</label>
                <span>${stats.noise_points}</span>
            </div>
            ${stats.silhouette_score ? `
                <div class="cluster-stat">
                    <label>Silhouette Score:</label>
                    <span>${stats.silhouette_score.toFixed(3)}</span>
                </div>
            ` : ''}
            <div class="cluster-stat">
                <label>Clustering Method:</label>
                <span>${stats.method}</span>
            </div>
        `;
        
        clusterInfo.style.display = 'block';
    }
    
    showVisualizationPlaceholder() {
        const plotDiv = document.getElementById('visualization-plot');
        plotDiv.innerHTML = `
            <div class="plot-placeholder">
                <i class="fas fa-chart-scatter"></i>
                <p>No visualization data available. Process a document first.</p>
            </div>
        `;
    }
    
    handleVisualizationClick(point) {
        const docId = point.ids;
        const metadata = point.customdata;
        
        this.showToast(`Clicked document: ${metadata.filename || docId}`, 'info');
        
        // Could implement document detail view here
    }
    
    async performSearch() {
        // Delegate to vectoria.js search functionality to avoid duplication
        if (window.performSearch) {
            window.performSearch();
        } else {
            console.error('Search functionality not available from vectoria.js');
            this.showToast('Search functionality not loaded', 'error');
        }
    }
    
    displaySearchResults(results) {
        const searchResults = document.getElementById('search-results');
        const searchPlaceholder = document.getElementById('search-placeholder');
        const resultsList = document.getElementById('results-list');
        const resultsTitle = document.getElementById('results-title');
        const resultsCount = document.getElementById('results-count');
        
        if (results.results.length === 0) {
            searchPlaceholder.style.display = 'block';
            searchResults.style.display = 'none';
            this.showToast('No results found for your query', 'warning');
            return;
        }
        
        searchPlaceholder.style.display = 'none';
        searchResults.style.display = 'block';
        
        resultsTitle.textContent = `Search Results for "${results.query}"`;
        resultsCount.textContent = `${results.results.length} results`;
        
        resultsList.innerHTML = results.results.map((result, index) => `
            <div class="result-item" data-doc-id="${result.doc_id}" data-chunk-id="${result.chunk_id}">
                <div class="result-header">
                    <span class="result-score">Score: ${result.score.toFixed(3)}</span>
                </div>
                <div class="result-text">${result.text}</div>
                <div class="result-metadata">
                    <span>Type: ${result.search_type}</span>
                    <span>Rank: ${result.rank + 1}</span>
                    ${result.metadata.metadata_title ? `<span>Title: ${result.metadata.metadata_title}</span>` : ''}
                </div>
            </div>
        `).join('');
        
        // Add click handlers to result items
        document.querySelectorAll('.result-item').forEach(item => {
            item.addEventListener('click', () => {
                const docId = item.dataset.docId;
                const chunkId = item.dataset.chunkId;
                this.handleResultClick(docId, chunkId);
            });
        });
        
        this.searchResults = results;
    }
    
    async highlightSearchResults(query, searchType, k) {
        try {
            const response = await fetch('/highlight_search_results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    search_type: searchType,
                    k: k
                })
            });
            
            if (!response.ok) return;
            
            const highlightData = await response.json();
            
            // Update visualization with highlights
            if (this.visualizationData) {
                this.updateVisualizationHighlights(highlightData.highlighted_docs);
            }
            
        } catch (error) {
            console.error('Highlight error:', error);
        }
    }
    
    updateVisualizationHighlights(highlightedDocs) {
        // Implementation would update the Plotly visualization
        // to highlight the specified documents
    }
    
    clearHighlights() {
        // Clear any visualization highlights
        if (this.visualizationData) {
            this.renderVisualization(); // Re-render without highlights
        }
    }
    
    handleResultClick(docId, chunkId) {
        this.showToast(`Viewing document: ${docId}`, 'info');
        // Could implement detailed document view
    }
    
    // Removed unused Plotly methods - using Canvas visualization instead
    
    async loadSystemConfig() {
        try {
            const response = await fetch('/config');
            const config = await response.json();
            
            // Update configuration display
            this.updateConfigDisplay(config);
            
        } catch (error) {
            console.error('Config loading error:', error);
        }
    }
    
    updateConfigDisplay(config) {
        const configElements = {
            'config-embedding-model': config.embedding_model,
            'config-batch-size': config.batch_size,
            'config-chunk-size': config.chunk_size,
            'config-chunk-overlap': config.chunk_overlap,
            'config-hybrid-search': 'Semantic (Vector)',
            'config-cosine-weight': config.cosine_weight ?? 'N/A',
            'config-bm25-weight': config.bm25_weight ?? 'N/A',
            'config-umap-neighbors': config.umap_n_neighbors,
            'config-umap-min-dist': config.umap_min_dist,
            'config-hdbscan-min-size': config.hdbscan_min_cluster_size
        };
        
        Object.entries(configElements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element && value !== undefined) {
                element.textContent = value;
            }
        });
    }
    
    resetUpload() {
        // Reset upload form
        document.getElementById('file-upload-form').reset();
        document.querySelector('.file-input-label span').textContent = 'Choose file or drag & drop';
        
        // Hide all processing sections
        ['progress-section', 'results-section', 'error-section'].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
        
        // Reset button
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-play"></i> Process Document';
        }
    }
    
    showLoadingOverlay(show) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = {
            'info': 'fas fa-info-circle',
            'success': 'fas fa-check-circle',
            'warning': 'fas fa-exclamation-triangle',
            'error': 'fas fa-times-circle'
        }[type] || 'fas fa-info-circle';
        
        toast.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Click to dismiss
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Global functions for template access
function activateTab(tabId) {
    if (window.app) {
        window.app.activateTab(tabId);
    }
}

function resetUpload() {
    if (window.app) {
        window.app.resetUpload();
    }
}

function loadSystemConfig() {
    if (window.app) {
        window.app.loadSystemConfig();
    }
}

function loadSystemStats() {
    if (window.app) {
        window.app.updateHeaderStats();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGVectoriaApp();
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.app) {
        window.app.showToast('An unexpected error occurred', 'error');
    }
});
