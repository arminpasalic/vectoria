/**
 * Browser-based Clustering and Dimensionality Reduction
 * Uses UMAP for dimensionality reduction and HDBSCAN for clustering
 *
 * Two-stage process:
 * 1. 15D UMAP (minDist=0.0) for clustering with HNSW-accelerated KNN
 * 2. 2D UMAP for visualization
 *
 * Performance notes:
 * - UMAP: HNSW-accelerated KNN for datasets >10k points (O(n log n) vs O(n²))
 * - HDBSCAN: scikit-learn (via Pyodide) runs entirely in-browser
 * - NO SUBSAMPLING: All points are processed in full for accurate clustering
 *
 * Complexity:
 * - UMAP: O(n k log n) where k = nNeighbors (typically 15)
 * - HDBSCAN: depends on selected scikit-learn backend (auto -> kd-tree / ball-tree / brute)
 * - Total: dominated by HDBSCAN for very large n; UMAP remains O(n k log n)
 */

import { createUMAP } from './umap-wasm-adapter.js';
import { runHDBSCANPyodide } from './hdbscan-pyodide.js';

export class BrowserClustering {
    constructor(options = {}) {
        // Load saved configuration
        const savedConfig = this.loadSavedConfig();

        this.nNeighbors = options.nNeighbors || savedConfig.umap_n_neighbors || 15;
        this.minDist = options.minDist || savedConfig.umap_min_dist || 0.1;
        this.nComponents = options.nComponents || 2;
        this.metric = options.metric || savedConfig.umap_metric || 'cosine';
        this.clusteringDimensions = savedConfig.umap_clustering_dimensions || 15;  // Configurable clustering dimensions
        this.minClusterSize = savedConfig.hdbscan_min_cluster_size || 5;
        this.minSamples = savedConfig.hdbscan_min_samples || 5;
        this.hdbscanMetric = savedConfig.hdbscan_metric || 'euclidean';
        this.umapSampleSize = savedConfig.umap_sample_size || 10000;  // Default 10K sample

        this.umapClustering = null;  // ND UMAP for clustering (default 15D)
        this.umapViz = null;          // 2D UMAP for visualization
        this.clusteringProjection = null;  // ND projection for clustering
        this.visualizationProjection = null; // 2D projection
        this.labels = null;
        this.probabilities = null;  // HDBSCAN cluster membership probabilities
        this.clusterKeywords = new Map();
        this.clusterKeywordScores = new Map();
        this.clusterKeywordsViz = new Map();
    }

    /**
     * Load saved configuration from localStorage via ConfigManager
     */
    loadSavedConfig() {
        try {
            // Use ConfigManager if available (centralized config system)
            const config = window.ConfigManager ? window.ConfigManager.getConfig() : null;

            if (config) {
                return config.clustering || {};
            }
        } catch (error) {
            console.warn('Failed to load saved config:', error);
        }
        return {};
    }

    /**
     * Compute UMAP projection for embeddings
     * @param {number[][]} embeddings - Array of embedding vectors
     * @param {Object} options - UMAP parameters
     * @returns {Promise<number[][]>} 2D coordinates
     */
    async computeUMAP(embeddings, options = {}) {
        // Reload config before UMAP to pick up setting changes
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.umap_n_neighbors) this.nNeighbors = freshConfig.umap_n_neighbors;
        if (freshConfig.umap_min_dist !== undefined) this.minDist = freshConfig.umap_min_dist;
        if (freshConfig.umap_metric) this.metric = freshConfig.umap_metric;

        const {
            nNeighbors = this.nNeighbors,
            minDist = this.minDist,
            nComponents = this.nComponents,
            metric = this.metric,
            onProgress = null
        } = options;

        try {
            // Initialize UMAP
            this.umap = new UMAP({
                nComponents: nComponents,
                nNeighbors: nNeighbors,
                minDist: minDist,
                metric: metric,
                random: Math.random // Use deterministic seed if needed
            });

            // Report progress
            if (onProgress) {
                onProgress({ status: 'fitting', progress: 0.1 });
            }

            // Fit and transform
            const startTime = Date.now();
            this.projection = await this.umap.fitAsync(embeddings, (epochNumber) => {
                // UMAP progress callback (number of epochs completed)
                if (onProgress) {
                    const progress = Math.min(0.9, 0.1 + (epochNumber / 500) * 0.8);
                    onProgress({
                        status: 'fitting',
                        progress: progress,
                        epoch: epochNumber
                    });
                }
            });

            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
            if (onProgress) {
                onProgress({ status: 'complete', progress: 1.0 });
            }

            return this.projection;
        } catch (error) {
            console.error('❌ UMAP projection failed:', error);
            throw new Error(`UMAP failed: ${error.message}`);
        }
    }

    /**
     * Custom UMAP implementation that uses precomputed HNSW KNN graph
     * This avoids the O(n²) neighbor computation that causes stack overflow on large datasets
     * @param {number[][]} embeddings - Array of embedding vectors
     * @param {number[][]} knnIndices - Precomputed KNN indices
     * @param {number[][]} knnDistances - Precomputed KNN distances
     * @param {Object} options - UMAP parameters
     * @returns {Promise<number[][]>} Projected coordinates
     */
    async _customUMAPWithPrecomputedKNN(embeddings, knnIndices, knnDistances, options = {}) {
        const {
            nComponents = 15,
            minDist = 0.0,
            metric = 'cosine',
            onProgress = null
        } = options;

        try {
            const numPoints = embeddings.length;
            const nNeighbors = knnIndices[0].length;

            // Step 1: Initialize random projection
            if (onProgress) onProgress(0.1);
            const projection = this._initializeRandomProjection(numPoints, nComponents);
            // Step 2: Convert KNN graph to UMAP format
            if (onProgress) onProgress(0.2);
            const umapGraph = this._convertKNNToUMAPGraph(knnIndices, knnDistances, minDist);
            // Step 3: Run UMAP optimization with precomputed graph
            if (onProgress) onProgress(0.3);
            const optimizedProjection = await this._optimizeUMAPProjection(
                projection,
                umapGraph,
                {
                    nComponents,
                    onProgress: (progress) => {
                        if (onProgress) {
                            // Progress from 0.3 to 0.9
                            onProgress(0.3 + progress * 0.6);
                        }
                    }
                }
            );
            if (onProgress) onProgress(1.0);
            
            // Validate projection quality before returning
            this._validateProjectionQuality(optimizedProjection, 'Custom UMAP 15D');
            
            return optimizedProjection;

        } catch (error) {
            console.error('❌ Custom UMAP with precomputed KNN failed:', error);
            throw new Error(`Custom UMAP failed: ${error.message}`);
        }
    }

    /**
     * Initialize random projection for UMAP
     */
    _initializeRandomProjection(numPoints, nComponents) {
        const projection = [];
        for (let i = 0; i < numPoints; i++) {
            const point = [];
            for (let j = 0; j < nComponents; j++) {
                point.push((Math.random() - 0.5) * 0.1); // Small random initialization
            }
            projection.push(point);
        }
        return projection;
    }

    /**
     * Convert KNN graph to UMAP-compatible format
     */
    _convertKNNToUMAPGraph(knnIndices, knnDistances, minDist) {
        const numPoints = knnIndices.length;
        const nNeighbors = knnIndices[0].length;
        const graph = [];

        for (let i = 0; i < numPoints; i++) {
            const neighbors = [];
            for (let j = 0; j < nNeighbors; j++) {
                const neighborIdx = knnIndices[i][j];
                const distance = knnDistances[i][j];
                
                // Convert distance to UMAP weight (higher distance = lower weight)
                // UMAP uses a smooth approximation to the heaviside step function
                const weight = 1.0 / (1.0 + distance);
                neighbors.push({
                    index: neighborIdx,
                    weight: weight
                });
            }
            graph.push(neighbors);
        }

        return graph;
    }

    /**
     * Optimize UMAP projection using gradient descent
     */
    async _optimizeUMAPProjection(projection, graph, options = {}) {
        const { nComponents, onProgress = null } = options;
        const numPoints = projection.length;
        let learningRate = 1.0;
        const epochs = 500; // Industry standard for quality embeddings
        const minLearningRate = 0.01;

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Compute gradients and update positions
            const gradients = this._computeUMAPGradients(projection, graph);

            // Adaptive learning rate (decreases over time)
            const adaptiveLR = Math.max(minLearningRate, learningRate * (1 - epoch / epochs));

            // Adaptive gradient clipping (larger movements in early epochs)
            const maxGradient = 4.0;  // Allow larger movements
            const adaptiveClip = maxGradient * (1 - epoch / epochs * 0.75);  // Decay over time

            // Update projection with gradients
            for (let i = 0; i < numPoints; i++) {
                for (let j = 0; j < nComponents; j++) {
                    // Clip gradient with adaptive threshold
                    const clippedGradient = Math.max(-adaptiveClip, Math.min(adaptiveClip, gradients[i][j]));
                    projection[i][j] -= adaptiveLR * clippedGradient;

                    // Ensure values stay finite
                    if (!Number.isFinite(projection[i][j])) {
                        projection[i][j] = 0;
                    }
                }
            }

            // Report progress more frequently for better UX
            if (onProgress && epoch % 10 === 0) {
                onProgress(epoch / epochs);
            }
        }

        return projection;
    }

    /**
     * Compute UMAP gradients for optimization
     */
    _computeUMAPGradients(projection, graph) {
        const numPoints = projection.length;
        const nComponents = projection[0].length;
        const gradients = Array(numPoints).fill().map(() => Array(nComponents).fill(0));

        // Attractive forces (pull neighbors together)
        for (let i = 0; i < numPoints; i++) {
            for (const neighbor of graph[i]) {
                const j = neighbor.index;
                const weight = neighbor.weight;
                
                for (let d = 0; d < nComponents; d++) {
                    const diff = projection[i][d] - projection[j][d];
                    const attractiveForce = -2 * weight * diff;
                    gradients[i][d] += attractiveForce;
                    gradients[j][d] -= attractiveForce;
                }
            }
        }

        // Repulsive forces (push non-neighbors apart) - simplified
        // For large datasets, we use a sampling approach to avoid O(n²) computation
        // 40% coverage for n < 10k, capped at 5k for performance
        const repulsiveSampleSize = Math.min(
            Math.floor(numPoints * 0.4),
            5000
        );
        for (let i = 0; i < numPoints; i++) {
            const sampleIndices = this._getRandomSampleIndices(numPoints, repulsiveSampleSize, i);
            
            for (const j of sampleIndices) {
                // Skip if j is a neighbor of i
                const isNeighbor = graph[i].some(n => n.index === j);
                if (isNeighbor) continue;

                const dist = this._euclideanDistanceND(projection[i], projection[j]);
                if (dist > 0) {
                    const repulsiveForce = 1.0 / (1.0 + dist * dist);
                    for (let d = 0; d < nComponents; d++) {
                        const diff = projection[i][d] - projection[j][d];
                        const force = repulsiveForce * diff / dist;
                        gradients[i][d] += force;
                        gradients[j][d] -= force;
                    }
                }
            }
        }

        return gradients;
    }

    /**
     * Compute projection statistics
     */
    _computeProjectionStats(projection) {
        if (!projection || projection.length === 0) {
            return { mean: 0, std: 0, min: 0, max: 0 };
        }

        const numPoints = projection.length;
        const numDims = projection[0]?.length || 0;
        const values = [];

        for (let i = 0; i < numPoints; i++) {
            for (let j = 0; j < numDims; j++) {
                const value = projection[i][j];
                if (Number.isFinite(value)) {
                    values.push(value);
                }
            }
        }

        if (values.length === 0) {
            return { mean: 0, std: 0, min: 0, max: 0 };
        }

        let min = Infinity;
        let max = -Infinity;
        let sum = 0;

        for (const v of values) {
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }

        const mean = sum / values.length;
        let variance = 0;
        for (const v of values) {
            variance += Math.pow(v - mean, 2);
        }
        const std = Math.sqrt(variance / values.length);

        return { mean, std, min, max };
    }

    /**
     * Validate projection quality and check for numerical anomalies
     */
    _validateProjectionQuality(projection, label) {
        if (!projection || projection.length === 0) return false;


        const numPoints = projection.length;
        const numDims = projection[0]?.length || 0;
        
        // Check for NaN, Infinity, and extreme values
        let nanCount = 0;
        let infinityCount = 0;
        let extremeCount = 0;
        const values = [];
        const maxValue = 1e6; // Threshold for "extreme" values
        
        for (let i = 0; i < numPoints; i++) {
            if (!projection[i] || !Array.isArray(projection[i])) {
                console.error(`❌ ${label}: Invalid point at index ${i}`);
                return false;
            }
            
            for (let j = 0; j < numDims; j++) {
                const value = projection[i][j];
                values.push(value);
                
                if (!Number.isFinite(value)) {
                    if (Number.isNaN(value)) {
                        nanCount++;
                    } else {
                        infinityCount++;
                    }
                } else if (Math.abs(value) > maxValue) {
                    extremeCount++;
                }
            }
        }
        
        // Calculate statistics (avoid spread operator for large arrays - causes stack overflow)
        const validValues = values.filter(v => Number.isFinite(v));

        let min = Infinity;
        let max = -Infinity;
        for (const v of validValues) {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const std = Math.sqrt(validValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / validValues.length);
        
        // Log quality report
        // Check for problems
        const hasProblems = nanCount > 0 || infinityCount > 0 || extremeCount > 0;
        if (hasProblems) {
            console.warn(`⚠️ ${label}: Numerical anomalies detected!`);
            if (nanCount > 0) console.warn(`   - ${nanCount} NaN values found`);
            if (infinityCount > 0) console.warn(`   - ${infinityCount} Infinity values found`);
            if (extremeCount > 0) console.warn(`   - ${extremeCount} extreme values (>${maxValue}) found`);
        } else {
        }
        
        return !hasProblems;
    }

    /**
     * Get random sample indices excluding a specific index
     */
    _getRandomSampleIndices(totalSize, sampleSize, excludeIndex) {
        const indices = [];
        const used = new Set([excludeIndex]);
        
        while (indices.length < sampleSize && indices.length < totalSize - 1) {
            const randomIndex = Math.floor(Math.random() * totalSize);
            if (!used.has(randomIndex)) {
                indices.push(randomIndex);
                used.add(randomIndex);
            }
        }
        
        return indices;
    }

    /**
     * Build HNSW index for fast KNN search
     * @param {Float32Array} embeddingsFlat - Flattened embeddings array
     * @param {number} numPoints - Number of points
     * @param {number} dimensions - Dimension of each vector
     * @param {number} nNeighbors - Number of neighbors to find
     * @returns {Promise<{indices: number[][], distances: number[][]}>} KNN graph
     */
    async _buildHNSWIndex(embeddingsFlat, numPoints, dimensions, nNeighbors) {
        try {
            // Load WASM module
            const hnswlib = await loadHnswlib();

            // Initialize HNSW index
            // HierarchicalNSW(space, dim, indexPath)
            // indexPath: empty string for in-memory index, or path for IDBFS persistence
            const index = new hnswlib.HierarchicalNSW('cosine', dimensions, '');

            // initIndex(maxElements, M, efConstruction, randomSeed)
            // maxElements = maximum number of points in the index
            // M = max number of connections per layer (16 is default, higher = better recall but more memory)
            // efConstruction = size of dynamic candidate list (200 is default, higher = better quality but slower build)
            // randomSeed = seed for reproducibility
            index.initIndex(numPoints, 16, 200, 100);

            // Add all vectors to the index
            for (let i = 0; i < numPoints; i++) {
                const start = i * dimensions;
                const vector = embeddingsFlat.slice(start, start + dimensions);
                // addPoint(vector, label, replaceDeleted)
                // replaceDeleted: false = don't replace deleted points (standard add)
                index.addPoint(vector, i, false);

                if ((i + 1) % 1000 === 0) {
                }
            }

            // Set ef parameter for search (higher = better recall but slower search)
            // Good rule of thumb: ef >= k, typically 2*k for good quality
            // Try different method names for hnswlib-wasm
            const efSearch = Math.max(nNeighbors * 2, 50);
            if (typeof index.setEfSearch === 'function') {
                index.setEfSearch(efSearch);
            } else if (typeof index.setEf === 'function') {
                index.setEf(efSearch);
            } else {
                console.warn(`⚠️ Could not set ef search parameter (method not found), using default`);
            }

            // Query KNN for all points
            const indices = [];
            const distances = [];

            for (let i = 0; i < numPoints; i++) {
                const start = i * dimensions;
                const vector = embeddingsFlat.slice(start, start + dimensions);
                // searchKnn(vector, k, filter)
                // vector: query vector
                // k: number of neighbors to find
                // filter: optional filter function (null = no filter)
                const result = index.searchKnn(vector, nNeighbors + 1, null); // +1 because point finds itself

                // result format: { neighbors: [labels...], distances: [dists...] }
                // Remove self from neighbors
                const neighborIndices = result.neighbors.filter(idx => idx !== i).slice(0, nNeighbors);
                const neighborDistances = result.distances.filter((_, idx) => result.neighbors[idx] !== i).slice(0, nNeighbors);

                indices.push(neighborIndices);
                distances.push(neighborDistances);

                if ((i + 1) % 1000 === 0) {
                }
            }

            return { indices, distances };

        } catch (error) {
            console.error(`❌ HNSW index build failed: ${error.message}`);
            throw new Error(`HNSW KNN graph construction failed: ${error.message}`);
        }
    }

    /**
     * Compute ND UMAP projection for clustering using Apple's WASM UMAP (with built-in HNSW)
     * Dimensions configurable via settings (default: 15D)
     * @param {number[][]} embeddings - Array of embedding vectors
     * @param {Object} options - UMAP parameters
     * @returns {Promise<number[][]>} ND coordinates for clustering
     */
    async computeClusteringUMAP(embeddings, options = {}) {
        // Reload config before clustering to pick up setting changes
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.umap_n_neighbors) this.nNeighbors = freshConfig.umap_n_neighbors;
        if (freshConfig.umap_min_dist !== undefined) this.minDist = freshConfig.umap_min_dist;
        if (freshConfig.umap_metric) this.metric = freshConfig.umap_metric;
        // Always update clusteringDimensions from config
        this.clusteringDimensions = freshConfig.umap_clustering_dimensions || this.clusteringDimensions || 15;
        if (freshConfig.hdbscan_min_cluster_size) this.minClusterSize = freshConfig.hdbscan_min_cluster_size;
        if (freshConfig.hdbscan_min_samples) this.minSamples = freshConfig.hdbscan_min_samples;
        if (freshConfig.hdbscan_metric) this.hdbscanMetric = freshConfig.hdbscan_metric;
        if (freshConfig.umap_sample_size) this.umapSampleSize = freshConfig.umap_sample_size;

        // Use options.nComponents if provided, otherwise use config value
        const nComponents = options.nComponents || this.clusteringDimensions || 15;
        const {
            nNeighbors = this.nNeighbors,
            minDist = 0.0,  // Dense clusters for better separation
            metric = 'cosine',
            onProgress = null
        } = options;

        try {
            const startTime = Date.now();

            // Validate and flatten embeddings
            const embeddingsArray = embeddings.map(emb => {
                if (Array.isArray(emb)) return [...emb];
                if (emb instanceof Float32Array || emb instanceof Float64Array) return Array.from(emb);
                return Array.from(emb);
            });

            const expectedDim = embeddingsArray[0]?.length;
            const invalidIndices = [];
            for (let i = 0; i < embeddingsArray.length; i++) {
                if (!embeddingsArray[i] || embeddingsArray[i].length !== expectedDim) {
                    invalidIndices.push(i);
                }
            }

            if (invalidIndices.length > 0) {
                console.error(`❌ Found ${invalidIndices.length} embeddings with invalid dimensions`);
                throw new Error(`Invalid embedding dimensions: ${invalidIndices.length} embeddings don't match expected size ${expectedDim}`);
            }

            const numPoints = embeddingsArray.length;

            // Flatten embeddings for WASM UMAP
            const flatData = new Float32Array(numPoints * expectedDim);
            for (let i = 0; i < numPoints; i++) {
                for (let j = 0; j < expectedDim; j++) {
                    flatData[i * expectedDim + j] = embeddingsArray[i][j];
                }
            }

            if (onProgress) {
                onProgress({ status: 'umap_fitting', progress: 0.10, elapsed: 0 });
            }

            // Set up periodic elapsed time updates (every 500ms)
            let elapsedTimer = null;
            let lastElapsed = -1;
            if (onProgress) {
                elapsedTimer = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    if (elapsed !== lastElapsed) {
                        lastElapsed = elapsed;
                        // Trigger a progress update with current elapsed time
                        onProgress({
                            status: 'umap_fitting',
                            progress: 0.5,  // Will be overridden by actual progress
                            elapsed: elapsed,
                            forceUpdate: true  // Flag to bypass some throttling
                        });
                    }
                }, 500);
            }

            const umapInstance = await createUMAP(numPoints, expectedDim, nComponents, flatData, {
                n_neighbors: nNeighbors,
                min_dist: minDist,
                distance: metric,
                onProgress: (progress) => {
                    if (onProgress) {
                        const elapsed = Math.floor((Date.now() - startTime) / 1000);
                        lastElapsed = elapsed;
                        const adjustedProgress = Math.min(0.95, 0.10 + progress * 0.85);
                        onProgress({
                            status: 'umap_fitting',
                            progress: adjustedProgress,
                            elapsed: elapsed
                        });
                    }
                }
            });

            await umapInstance.run();

            // Clear the elapsed time timer
            if (elapsedTimer) {
                clearInterval(elapsedTimer);
            }

            // Get embedding and convert back to 2D array
            const flatResult = umapInstance.embedding;
            this.clusteringProjection = [];
            for (let i = 0; i < numPoints; i++) {
                const row = [];
                for (let j = 0; j < nComponents; j++) {
                    row.push(flatResult[i * nComponents + j]);
                }
                this.clusteringProjection.push(row);
            }

            umapInstance.destroy();

            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
            if (onProgress) {
                onProgress({ status: 'complete', progress: 1.0 });
            }

            return this.clusteringProjection;

        } catch (error) {
            console.error('❌ UMAP clustering projection failed:', error);
            throw new Error(`UMAP clustering failed: ${error.message}`);
        }
    }

    /**
     * Compute 2D UMAP projection for visualization using Apple's WASM UMAP (with built-in HNSW)
     * @param {number[][]} embeddings - Array of embedding vectors
     * @param {Object} options - UMAP parameters
     * @returns {Promise<number[][]>} 2D coordinates for visualization
     */
    async computeVisualizationUMAP(embeddings, options = {}) {
        const {
            nNeighbors = 15,
            minDist = 0.1,
            nComponents = 2,
            metric = 'cosine',
            onProgress = null
        } = options;

        try {
            const startTime = Date.now();

            // Convert embeddings to regular 2D array
            const embeddingsArray = embeddings.map(emb => {
                if (Array.isArray(emb)) return [...emb];
                if (emb instanceof Float32Array || emb instanceof Float64Array) return Array.from(emb);
                return Array.from(emb);
            });

            // Validate all embeddings have the same dimension
            const expectedDim = embeddingsArray[0]?.length;
            const invalidIndices = [];
            for (let i = 0; i < embeddingsArray.length; i++) {
                if (!embeddingsArray[i] || embeddingsArray[i].length !== expectedDim) {
                    invalidIndices.push(i);
                }
            }

            if (invalidIndices.length > 0) {
                console.error(`❌ Found ${invalidIndices.length} embeddings with invalid dimensions (2D viz)`);
                throw new Error(`Invalid embedding dimensions: ${invalidIndices.length} embeddings don't match expected size ${expectedDim}`);
            }

            const numPoints = embeddingsArray.length;

            // Flatten embeddings for WASM UMAP
            const flatData = new Float32Array(numPoints * expectedDim);
            for (let i = 0; i < numPoints; i++) {
                for (let j = 0; j < expectedDim; j++) {
                    flatData[i * expectedDim + j] = embeddingsArray[i][j];
                }
            }

            if (onProgress) {
                onProgress({ status: 'umap_fitting_viz', progress: 0.10, elapsed: 0 });
            }

            // Set up periodic elapsed time updates (every 500ms)
            let elapsedTimer2D = null;
            let lastElapsed2D = -1;
            if (onProgress) {
                elapsedTimer2D = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    if (elapsed !== lastElapsed2D) {
                        lastElapsed2D = elapsed;
                        onProgress({
                            status: 'umap_fitting_viz',
                            progress: 0.5,
                            elapsed: elapsed,
                            forceUpdate: true
                        });
                    }
                }, 500);
            }

            const umapInstance = await createUMAP(numPoints, expectedDim, nComponents, flatData, {
                n_neighbors: nNeighbors,
                min_dist: minDist,
                distance: metric,
                onProgress: (progress) => {
                    if (onProgress) {
                        const elapsed = Math.floor((Date.now() - startTime) / 1000);
                        lastElapsed2D = elapsed;
                        const adjustedProgress = Math.min(0.95, 0.10 + progress * 0.85);
                        onProgress({
                            status: 'umap_fitting_viz',
                            progress: adjustedProgress,
                            elapsed: elapsed
                        });
                    }
                }
            });

            await umapInstance.run();

            // Clear the elapsed time timer
            if (elapsedTimer2D) {
                clearInterval(elapsedTimer2D);
            }

            // Get embedding and convert back to 2D array
            const flatResult = umapInstance.embedding;
            this.visualizationProjection = [];
            for (let i = 0; i < numPoints; i++) {
                const row = [];
                for (let j = 0; j < nComponents; j++) {
                    row.push(flatResult[i * nComponents + j]);
                }
                this.visualizationProjection.push(row);
            }

            umapInstance.destroy();

            const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
            // Validate 2D projection quality
            const isValid = this._validateProjectionQuality(this.visualizationProjection, '2D UMAP Viz');
            if (!isValid) {
                console.error('❌ 2D UMAP failed quality checks - projection may be degenerate');
                throw new Error('UMAP 2D projection failed quality validation');
            }

            // Check for degenerate embedding (collapsed dimensions)
            const stats = this._computeProjectionStats(this.visualizationProjection);
            if (stats.std < 0.1) {
                console.warn('⚠️ 2D projection appears collapsed (std < 0.1)');
                console.warn('   This usually indicates insufficient repulsive forces or epochs');
                console.warn(`   Projection stats: mean=${stats.mean.toFixed(4)}, std=${stats.std.toFixed(4)}, range=[${stats.min.toFixed(4)}, ${stats.max.toFixed(4)}]`);
            } else if (stats.std < 1.0) {
                console.warn(`⚠️ 2D projection has low spread (std=${stats.std.toFixed(2)}). Consider adjusting UMAP parameters.`);
            }

            if (onProgress) {
                onProgress({ status: 'complete', progress: 1.0 });
            }

            return this.visualizationProjection;
        } catch (error) {
            console.error('❌ UMAP visualization projection failed:', error);
            throw new Error(`UMAP visualization failed: ${error.message}`);
        }
    }

    /**
     * HDBSCAN clustering algorithm using the WASM backend with automatic fallback
     * @param {number[][]} points - Array of points (typically 15D UMAP projection)
     * @param {Object} options - Clustering parameters
     * @returns {Promise<number[]>} Cluster labels (-1 for noise/outliers)
     */
    async computeClusters(points, options = {}) {
        // Reload config before clustering to pick up setting changes
        const freshConfig = this.loadSavedConfig();
        if (freshConfig.hdbscan_min_cluster_size !== undefined) {
            this.minClusterSize = freshConfig.hdbscan_min_cluster_size;
        }
        if (freshConfig.hdbscan_min_samples !== undefined) {
            this.minSamples = freshConfig.hdbscan_min_samples;
        }
        if (freshConfig.hdbscan_metric) {
            this.hdbscanMetric = freshConfig.hdbscan_metric;
        }

        const {
            minClusterSize = this.minClusterSize,
            minSamples = this.minSamples,
            documents = null,
            keywordOptions = null
        } = options;

        // Edge case 1: Handle empty arrays
        if (!points || points.length === 0) {
            console.warn('⚠️ Empty array provided to clustering - returning empty labels');
            this.labels = [];
            this.probabilities = [];
            return this.labels;
        }

        // Edge case 2: Handle data length < minClusterSize
        if (points.length < minClusterSize) {
            console.warn(`⚠️ Dataset size (${points.length}) is smaller than minClusterSize (${minClusterSize}) - all points marked as noise`);
            this.labels = new Array(points.length).fill(-1);
            this.probabilities = new Array(points.length).fill(this._formatProbability(0));
            return this.labels;
        }

        const dimensions = points[0]?.length || 0;
        try {
            const startTime = Date.now();

            // Reload config to get latest hdbscan_metric setting
            const freshConfig = this.loadSavedConfig();
            const hdbscanMetric = freshConfig.hdbscan_metric || this.hdbscanMetric || 'euclidean';
            // Fit the model to the data using Pyodide (scikit-learn)
            const clusteringResult = await runHDBSCANPyodide(points, {
                minClusterSize,
                minSamples,
                metric: hdbscanMetric,
                documents,
                keywordOptions
            });

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            // Access labels and probabilities
            this.labels = Array.from(clusteringResult.labels || []);

            // Debug: Log probability values from HDBSCAN
            if (clusteringResult.probabilities && clusteringResult.probabilities.length > 0) {
                const sample = clusteringResult.probabilities.slice(0, 10);
                const stats = {
                    min: Math.min(...clusteringResult.probabilities),
                    max: Math.max(...clusteringResult.probabilities),
                    avg: clusteringResult.probabilities.reduce((a, b) => a + b, 0) / clusteringResult.probabilities.length
                };
                // Count how many are exactly 1.0
                const exactOnes = clusteringResult.probabilities.filter(p => p === 1.0).length;
                const percentage = (exactOnes / clusteringResult.probabilities.length * 100).toFixed(1);
            }

            // Debug: Log outlier scores if available
            if (clusteringResult.outlier_scores && clusteringResult.outlier_scores.length > 0) {
                const sample = clusteringResult.outlier_scores.slice(0, 10);
                const stats = {
                    min: Math.min(...clusteringResult.outlier_scores),
                    max: Math.max(...clusteringResult.outlier_scores),
                    avg: clusteringResult.outlier_scores.reduce((a, b) => a + b, 0) / clusteringResult.outlier_scores.length
                };
            }

            this.probabilities = this._prepareProbabilityArray(
                clusteringResult.probabilities,
                this.labels,
                points.length
            );

            this._storeClusterKeywords(clusteringResult);

            if (this.labels.length !== points.length) {
                console.warn(`⚠️ Label length (${this.labels.length}) does not match dataset size (${points.length})`);
                this.labels = this._resizeArray(this.labels, points.length, -1);
                this.probabilities = this._resizeArray(this.probabilities, points.length, this._formatProbability(0));
            }

            this._logClusterStats(points.length);
            return this.labels;

        } catch (error) {
            console.error('❌ HDBSCAN clustering failed:', error);
            this.labels = new Array(points.length).fill(-1);
            this.probabilities = new Array(points.length).fill(this._formatProbability(0));
            console.warn('⚠️ HDBSCAN clustering failed, all points assigned to noise:', error.message || error);
            throw new Error(`HDBSCAN clustering failed: ${error.message}`);
        }
    }


    /**
     * Log cluster statistics
     */
    _logClusterStats(totalPoints) {
        const uniqueLabels = new Set(this.labels);
        const numClusters = uniqueLabels.size - (uniqueLabels.has(-1) ? 1 : 0);
        const noiseCount = this.labels.filter(l => l === -1).length;
        const noisePercentage = ((noiseCount / totalPoints) * 100).toFixed(1);

        // Log cluster size distribution
        const clusterSizes = {};
        for (const label of this.labels) {
            if (label !== -1) {
                clusterSizes[label] = (clusterSizes[label] || 0) + 1;
            }
        }

        if (Object.keys(clusterSizes).length > 0) {
            const sizes = Object.values(clusterSizes).sort((a, b) => b - a);
        }
    }

    /**
     * Get cluster membership probabilities
     * @returns {number[]} Array of probabilities (0-1) for each point
     */
    getProbabilities() {
        return this.probabilities;
    }

    /**
     * Identify outliers based on probability threshold
     * @param {number} threshold - Probability threshold (default: 0.5)
     * @returns {number[]} Indices of outlier points
     */
    getOutliers(threshold = 0.5) {
        if (!this.probabilities || !this.labels) {
            console.warn('⚠️ No clustering results available');
            return [];
        }

        const outliers = [];
        for (let i = 0; i < this.labels.length; i++) {
            // Consider both explicit noise points and low-probability cluster members
            if (this.labels[i] === -1 || this.probabilities[i] < threshold) {
                outliers.push(i);
            }
        }

        return outliers;
    }

    /**
     * Group data points by cluster label
     * @param {Array} data - Original data array (e.g., texts)
     * @returns {Object} Dictionary mapping cluster labels to data arrays
     */
    groupByCluster(data) {
        if (!this.labels || this.labels.length !== data.length) {
            console.error('❌ Labels not available or length mismatch');
            return {};
        }

        const grouped = {};
        for (let i = 0; i < this.labels.length; i++) {
            const label = this.labels[i];
            if (!grouped[label]) {
                grouped[label] = [];
            }
            grouped[label].push({
                index: i,
                data: data[i],
                probability: this.probabilities ? this.probabilities[i] : null
            });
        }

        // Log summary
        const clusterCount = Object.keys(grouped).filter(k => k !== '-1').length;
        const noiseCount = grouped['-1'] ? grouped['-1'].length : 0;
        return grouped;
    }

    /**
     * Euclidean distance between two points (N-dimensional)
     */
    _euclideanDistanceND(p1, p2) {
        let sum = 0;
        for (let i = 0; i < p1.length; i++) {
            const diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Ensure probability array matches dataset length and precision requirements
     */
    _prepareProbabilityArray(rawProbabilities, labels, expectedLength = 0) {
        const targetLength = Array.isArray(labels) ? labels.length : expectedLength || 0;
        if (!targetLength) {
            return [];
        }

        const hasRaw = rawProbabilities && typeof rawProbabilities.length === 'number';
        const formatted = new Array(targetLength);
        let defaultedCount = 0;
        let noiseCount = 0;
        let validCount = 0;

        for (let i = 0; i < targetLength; i++) {
            let value;

            if (hasRaw && rawProbabilities[i] !== undefined) {
                value = rawProbabilities[i];
                validCount++;
            } else if (labels && labels[i] === -1) {
                value = 0.5;
                noiseCount++;
            } else {
                value = 1;
                defaultedCount++;
            }

            formatted[i] = this._formatProbability(value);
        }

        if (defaultedCount > 0) {
            console.warn(`⚠️ ${defaultedCount} probabilities defaulted (missing from HDBSCAN output)`);
        }

        return formatted;
    }

    /**
     * Resize array to expected length while preserving existing entries
     */
    _resizeArray(arr, targetLength, fillValue) {
        const result = new Array(targetLength);
        for (let i = 0; i < targetLength; i++) {
            result[i] = i < arr.length ? arr[i] : fillValue;
        }
        return result;
    }

    /**
     * Normalize probability to [0,1] range
     * @returns {number} Probability as float in [0,1]
     */
    _formatProbability(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return 0.0;
        }
        return Math.min(1, Math.max(0, numeric));  // Return number, not string
    }

    /**
     * Format probability for display as percentage string
     * @returns {string} Probability as percentage (e.g., "85.32%")
     */
    _formatProbabilityDisplay(value) {
        const numeric = typeof value === 'number' ? value : this._formatProbability(value);
        return (numeric * 100).toFixed(2) + "%";
    }

    _storeClusterKeywords(result) {
        this.clusterKeywords = new Map();
        this.clusterKeywordScores = new Map();
        this.clusterKeywordsViz = new Map();

        if (!result) {
            return;
        }

        const assignKeywords = (targetMap, entries, transformValue = (value) => value) => {
            if (!entries || typeof entries !== 'object') {
                return;
            }
            Object.entries(entries).forEach(([key, value]) => {
                const numericKey = Number(key);
                if (Number.isNaN(numericKey)) {
                    return;
                }
                targetMap.set(numericKey, transformValue(value));
            });
        };

        assignKeywords(this.clusterKeywords, result.cluster_keywords, (value) => {
            if (Array.isArray(value)) {
                return value.slice();
            }
            return [];
        });

        assignKeywords(this.clusterKeywordScores, result.cluster_keyword_scores, (value) => {
            if (!Array.isArray(value)) {
                return [];
            }
            return value.map((item) => {
                if (item && typeof item === 'object') {
                    const keyword = typeof item.keyword === 'string' ? item.keyword : '';
                    const score = typeof item.score === 'number' ? item.score : Number(item.score) || 0;
                    return { keyword, score };
                }
                return { keyword: '', score: 0 };
            });
        });

        assignKeywords(this.clusterKeywordsViz, result.cluster_keywords_viz, (value) => {
            if (Array.isArray(value)) {
                return value.slice();
            }
            return [];
        });
    }

    getClusterKeywords() {
        return this.clusterKeywords;
    }

    getClusterKeywordScores() {
        return this.clusterKeywordScores;
    }

    getClusterKeywordsViz() {
        return this.clusterKeywordsViz;
    }

    hydrateClusterKeywords(data = {}) {
        if (!data || typeof data !== 'object') {
            this.clusterKeywords = new Map();
            this.clusterKeywordScores = new Map();
            this.clusterKeywordsViz = new Map();
            return;
        }

        this._storeClusterKeywords({
            cluster_keywords: data.cluster_keywords || data.clusterKeywords,
            cluster_keyword_scores: data.cluster_keyword_scores || data.clusterKeywordScores,
            cluster_keywords_viz: data.cluster_keywords_viz || data.clusterKeywordsViz
        });
    }

    getClusterKeywordData() {
        const toObject = (map) => Object.fromEntries(Array.from(map.entries()).map(([key, value]) => {
            if (Array.isArray(value)) {
                return [key, value.map((item) => {
                    if (item && typeof item === 'object' && 'keyword' in item && 'score' in item) {
                        return { keyword: item.keyword, score: item.score };
                    }
                    return item;
                })];
            }
            return [key, value];
        }));

        return {
            cluster_keywords: toObject(this.clusterKeywords),
            cluster_keyword_scores: toObject(this.clusterKeywordScores),
            cluster_keywords_viz: toObject(this.clusterKeywordsViz)
        };
    }

    /**
     * Get cluster statistics with probability information
     */
    getClusterStats(labels = null) {
        const clustLabels = labels || this.labels;
        if (!clustLabels) {
            return null;
        }

        const clusterCounts = new Map();
        const clusterProbabilities = new Map();

        for (let i = 0; i < clustLabels.length; i++) {
            const label = clustLabels[i];
            clusterCounts.set(label, (clusterCounts.get(label) || 0) + 1);

            // Track probabilities per cluster
            if (this.probabilities && this.probabilities[i] !== undefined) {
                if (!clusterProbabilities.has(label)) {
                    clusterProbabilities.set(label, []);
                }
                clusterProbabilities.get(label).push(this.probabilities[i]);
            }
        }

        // Calculate average probabilities per cluster
        const avgProbabilities = {};
        for (const [label, probs] of clusterProbabilities.entries()) {
            if (label !== -1) {
                const avg = probs.reduce((sum, p) => sum + p, 0) / probs.length;
                avgProbabilities[label] = this._formatProbability(avg);  // Store as number
            }
        }

        const stats = {
            numClusters: clusterCounts.size - (clusterCounts.has(-1) ? 1 : 0),
            numNoise: clusterCounts.get(-1) || 0,
            clusterSizes: Object.fromEntries(
                Array.from(clusterCounts.entries())
                    .filter(([label]) => label !== -1)
                    .sort((a, b) => b[1] - a[1])
            ),
            avgProbabilities: avgProbabilities,
            totalPoints: clustLabels.length,
            noisePercentage: parseFloat(
                ((clusterCounts.get(-1) || 0) / clustLabels.length * 100).toFixed(1)
            )
        };

        return stats;
    }

    /**
     * Normalize 2D coordinates to range [0, 1]
     */
    normalizeProjection(projection = null) {
        const proj = projection || this.projection;
        if (!proj) return null;

        // Find bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (const point of proj) {
            minX = Math.min(minX, point[0]);
            maxX = Math.max(maxX, point[0]);
            minY = Math.min(minY, point[1]);
            maxY = Math.max(maxY, point[1]);
        }

        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;

        // Normalize
        return proj.map(point => [
            (point[0] - minX) / rangeX,
            (point[1] - minY) / rangeY
        ]);
    }

}

// Export singleton instance
export const clustering = new BrowserClustering();
