/**
 * Apple's UMAP-WASM adapter
 * Uses production-grade C++ UMAP implementation (umappp by Aaron Lun)
 * Compiled to WebAssembly with built-in HNSW from hnswlib
 *
 * Source: https://github.com/apple/embedding-atlas
 *
 * Key features:
 * - Native C++ performance via WASM
 * - Built-in HNSW approximate nearest neighbor search
 * - Proper progress callbacks
 * - Memory-efficient
 */

import { createUMAP as createUMAPWasm } from './umap-wasm/index.js';

export class UMAPAdapter {
    /**
     * Create UMAP instance using Apple's WASM implementation
     * @param {number} count - Number of data points
     * @param {number} inputDim - Input dimensionality
     * @param {number} outputDim - Output dimensionality
     * @param {Float32Array|number[][]} data - Input data (flat or 2D)
     * @param {Object} options - UMAP parameters
     */
    static async createUMAP(count, inputDim, outputDim, data, options = {}) {
        const {
            n_neighbors = 15,
            min_dist = 0.1,
            distance = 'cosine',
            onProgress = null
        } = options;

        // Convert data to Float32Array if needed
        const flatData = this._prepareData(data, count, inputDim);

        // Create UMAP instance using Apple's WASM implementation
        const umapOptions = {
            metric: distance,
            knnMethod: 'hnsw',  // Use HNSW for fast KNN
            nNeighbors: n_neighbors,
            minDist: min_dist,
            // Spectral initialization triggers an IRLBA run that is fragile in WASM
            // for higher-dimensional embeddings; prefer random init for stability.
            initializeMethod: outputDim > 2 ? 'random' : 'spectral'
        };

        const inputMb = (flatData.byteLength / (1024 * 1024)).toFixed(1);
        const outputMb = ((count * outputDim * 4) / (1024 * 1024)).toFixed(1);

        let umap;
        try {
            umap = await createUMAPWasm(count, inputDim, outputDim, flatData, umapOptions);
        } catch (error) {
            if (error && typeof error.message === 'string' && error.message.includes('memory access out of bounds')) {
                const enhanced = new Error(
                    `Apple UMAP WASM exhausted its heap while preparing the embedding (${inputMb} MB input + ${outputMb} MB output). ` +
                    `Try reducing the dataset size/dimensionality or rebuild the wasm runtime with a larger memory budget (see umap-wasm/REBUILD.md).`
                );
                enhanced.cause = error;
                throw enhanced;
            }
            throw error;
        }

        const heapMb = ((umap.heapBytes ?? 0) / (1024 * 1024)).toFixed(1);
        // Progress monitoring wrapper
        const wrapper = {
            _umap: umap,
            _count: count,
            _outputDim: outputDim,

            /**
             * Run UMAP optimization
             * @param {number} epochs - Number of epochs to run (default: all)
             */
            async run(epochs = null) {
                const nEpochs = umap.nEpochs;
                const targetEpoch = epochs ?? nEpochs;

                if (onProgress) {
                    // Run with progress monitoring and UI yielding
                    // Update every ~100-200ms for smooth real-time feedback
                    const progressInterval = Math.max(1, Math.floor(nEpochs / 50));
                    let lastUpdate = Date.now();

                    for (let epoch = 0; epoch <= targetEpoch; epoch += progressInterval) {
                        umap.run(epoch);
                        const progress = epoch / nEpochs;
                        onProgress(progress);

                        // Yield to browser event loop so UI can update
                        // Force a longer yield every update to ensure rendering
                        const now = Date.now();
                        if (now - lastUpdate > 100) {  // At least 100ms between renders
                            await new Promise(resolve => {
                                if (typeof requestAnimationFrame === 'function') {
                                    requestAnimationFrame(() => setTimeout(resolve, 0));
                                } else {
                                    setTimeout(resolve, 10);
                                }
                            });
                            lastUpdate = now;
                        } else {
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                    }
                    // Ensure completion
                    if (targetEpoch >= nEpochs) {
                        umap.run(nEpochs);
                        onProgress(1.0);
                    }
                } else {
                    // Run without progress monitoring
                    umap.run(targetEpoch);
                }
            },

            /**
             * Get current embedding as Float32Array
             */
            get embedding() {
                return this._umap.embedding;
            },

            /**
             * Clean up resources
             */
            destroy() {
                if (this._umap) {
                    this._umap.destroy();
                    this._umap = null;
                }
            }
        };

        return wrapper;
    }

    /**
     * Prepare data for UMAP (ensure Float32Array format)
     */
    static _prepareData(data, count, inputDim) {
        // If already Float32Array, return as-is
        if (data instanceof Float32Array) {
            return data;
        }

        // Convert 2D array to flat Float32Array
        const flat = new Float32Array(count * inputDim);
        if (Array.isArray(data) && Array.isArray(data[0])) {
            for (let i = 0; i < count; i++) {
                for (let j = 0; j < inputDim; j++) {
                    flat[i * inputDim + j] = data[i][j];
                }
            }
        } else {
            // Assume already flat array (TypedArray or Array)
            flat.set(data);
        }
        return flat;
    }
}

// Export createUMAP function for compatibility
export const createUMAP = UMAPAdapter.createUMAP.bind(UMAPAdapter);
