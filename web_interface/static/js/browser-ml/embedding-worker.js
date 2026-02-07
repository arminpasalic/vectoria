/**
 * Embedding Worker - Runs embeddings in background thread
 * Allows UI to stay responsive during processing
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5/+esm';

// Configure transformers.js environment
env.allowLocalModels = false;
env.useBrowserCache = true;

// Enable all available accelerations
try {
    const cores = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
    if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
        env.backends.onnx.wasm.numThreads = cores;
        env.backends.onnx.wasm.simd = true;
        env.backends.onnx.wasm.proxy = false;
    }
} catch (_) { /* best-effort tuning */ }

// Worker state
let embedder = null;
let modelName = 'Xenova/multilingual-e5-small';
let dimension = 384;
let isInitialized = false;
let activeDevice = null;

/**
 * Check if WebGPU is available
 */
function isWebGPUAvailable() {
    try {
        return typeof navigator !== 'undefined' && 'gpu' in navigator;
    } catch (_) {
        return false;
    }
}

/**
 * Resolve device target based on preference
 */
function resolveDeviceTarget(preference) {
    if (preference === 'cpu' || preference === 'wasm') {
        return 'cpu';
    }
    if (preference === 'webgpu' || preference === 'gpu') {
        return isWebGPUAvailable() ? 'webgpu' : 'cpu';
    }
    // Auto
    return isWebGPUAvailable() ? 'webgpu' : 'cpu';
}

/**
 * Map device to transformers.js parameter
 */
function deviceParameterForPipeline(device) {
    if (device === 'webgpu') return 'webgpu';
    if (device === 'cpu') return 'wasm';
    return null;
}

/**
 * Initialize the embedding model
 */
async function initializeModel(devicePreference = 'auto') {
    const targetDevice = resolveDeviceTarget(devicePreference);

    if (embedder && isInitialized && activeDevice === targetDevice) {
        self.postMessage({
            type: 'init_complete',
            device: activeDevice
        });
        return;
    }

    self.postMessage({
        type: 'init_start',
        targetDevice: targetDevice
    });

    const options = {};
    const deviceParam = deviceParameterForPipeline(targetDevice);
    if (deviceParam) {
        options.device = deviceParam;
    }

    // Track overall progress across multiple files
    const fileProgress = new Map();
    const fileWeights = {
        'tokenizer_config.json': 0.02,  // 2% - very small file
        'config.json': 0.02,            // 2% - very small file
        'tokenizer.json': 0.10,         // 10% - tokenizer file
        'onnx/model.onnx': 0.86         // 86% - main model file
    };
    const extraFileWeight = 0.05;
    let lastOverallProgress = 0;

    const clampRatio = (value) => Math.min(Math.max(value, 0), 1);

    const computeOverallProgress = () => {
        let weightedSum = 0;
        let totalWeight = 0;
        const details = [];

        for (const [fileName, weight] of Object.entries(fileWeights)) {
            const completion = fileProgress.get(fileName) || 0;
            const contribution = completion * weight;
            weightedSum += contribution;
            totalWeight += weight;
            details.push(`${fileName}=${(completion * 100).toFixed(1)}% (${(contribution * 100).toFixed(1)}%)`);
        }

        for (const [fileName, completion] of fileProgress.entries()) {
            if (fileWeights[fileName] !== undefined) continue;
            const contribution = completion * extraFileWeight;
            weightedSum += contribution;
            totalWeight += extraFileWeight;
            details.push(`${fileName}=${(completion * 100).toFixed(1)}% (${(contribution * 100).toFixed(1)}%)`);
        }

        if (totalWeight === 0) {
            return { overall: 0, details };
        }

        const normalized = weightedSum / totalWeight;
        if (normalized > lastOverallProgress) {
            lastOverallProgress = normalized;
        }

        return { overall: lastOverallProgress, details };
    };

    const normalizeFileProgress = (progress, currentValue) => {
        if (progress.status === 'done') {
            return 1;
        }

        const { loaded, total } = progress;
        if (typeof loaded === 'number' && typeof total === 'number' && total > 0) {
            return Math.max(currentValue, clampRatio(loaded / total));
        }

        if (typeof progress.progress === 'number') {
            const normalized = progress.progress / 100;
            return Math.max(currentValue, clampRatio(normalized));
        }

        return currentValue;
    };

    // Track total bytes across all files
    const fileBytesLoaded = new Map();
    const fileBytesTotal = new Map();

    // Progress callback - aggregate progress across all files
    options.progress_callback = (progress) => {
        const { file, status, loaded, total } = progress;

        if (!file || (status !== 'progress' && status !== 'done')) {
            return;
        }

        // Track bytes for this file
        if (typeof loaded === 'number') {
            fileBytesLoaded.set(file, loaded);
        }
        if (typeof total === 'number' && total > 0) {
            fileBytesTotal.set(file, total);
        }

        const current = fileProgress.get(file) || 0;
        const updated = normalizeFileProgress(progress, current);
        fileProgress.set(file, clampRatio(updated));

        const { overall, details } = computeOverallProgress();

        // Sum up all bytes
        let totalLoaded = 0;
        let totalSize = 0;
        for (const bytes of fileBytesLoaded.values()) {
            totalLoaded += bytes;
        }
        for (const bytes of fileBytesTotal.values()) {
            totalSize += bytes;
        }

        // Send aggregated progress with byte info
        self.postMessage({
            type: 'init_progress',
            progress: overall,
            file: file,
            fileStatus: status,
            loaded: totalLoaded,
            total: totalSize
        });
    };

    try {
        // Request high-performance GPU adapter for WebGPU
        if (targetDevice === 'webgpu' && navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter({
                    powerPreference: 'high-performance'
                });
                if (adapter) {
                    self.postMessage({
                        type: 'log',
                        message: 'Requested high-performance GPU adapter'
                    });
                }
            } catch (gpuError) {
                self.postMessage({
                    type: 'warning',
                    message: 'Could not request high-performance GPU adapter'
                });
            }
        }

        embedder = await pipeline('feature-extraction', modelName, options);
        isInitialized = true;
        activeDevice = targetDevice;

        self.postMessage({
            type: 'init_complete',
            device: activeDevice,
            modelName: modelName,
            dimension: dimension
        });

    } catch (error) {
        // If WebGPU failed, try fallback to CPU
        if (targetDevice === 'webgpu') {
            self.postMessage({
                type: 'warning',
                message: 'WebGPU initialization failed, falling back to CPU...'
            });

            try {
                const cpuOptions = { device: 'wasm' };
                cpuOptions.progress_callback = options.progress_callback;

                embedder = await pipeline('feature-extraction', modelName, cpuOptions);
                isInitialized = true;
                activeDevice = 'cpu';

                self.postMessage({
                    type: 'init_complete',
                    device: 'cpu',
                    modelName: modelName,
                    dimension: dimension,
                    fallback: true
                });
                return;
            } catch (fallbackError) {
                self.postMessage({
                    type: 'error',
                    error: `CPU fallback also failed: ${fallbackError.message}`
                });
                return;
            }
        }

        self.postMessage({
            type: 'error',
            error: `Embeddings initialization failed: ${error.message}`
        });
    }
}

/**
 * Apply E5-style prefix
 */
function prepareE5Input(text, mode) {
    if (!text) return '';
    const prefix = mode === 'query' ? 'query: ' : 'passage: ';
    return `${prefix}${text}`;
}

/**
 * Generate embeddings for a batch of texts
 */
async function embedBatch(texts, options = {}) {
    if (!isInitialized || !embedder) {
        self.postMessage({
            type: 'error',
            error: 'Model not initialized'
        });
        return;
    }

    const {
        normalize = true,
        pooling = 'mean',
        maxLength = 256,
        mode = 'passage',
        batchId = null
    } = options;

    try {
        // Prepare texts with E5 prefix
        const preparedTexts = texts.map(t => prepareE5Input(t, mode));

        // Generate embeddings
        const output = await embedder(preparedTexts, {
            pooling: pooling,
            normalize: normalize,
            max_length: maxLength,
            truncate: true
        });

        // Extract embeddings
        const batchSize = texts.length;
        const embeddings = [];

        for (let j = 0; j < batchSize; j++) {
            const start = j * dimension;
            const end = start + dimension;
            const embedding = Array.from(output.data.slice(start, end));
            embeddings.push(embedding);
        }

        self.postMessage({
            type: 'embed_complete',
            embeddings: embeddings,
            batchId: batchId,
            device: activeDevice
        });

    } catch (error) {
        self.postMessage({
            type: 'error',
            error: `Embedding generation failed: ${error.message}`,
            batchId: batchId
        });
    }
}

/**
 * Handle messages from main thread
 */
self.addEventListener('message', async (e) => {
    const { type, ...data } = e.data;

    switch (type) {
        case 'init':
            await initializeModel(data.devicePreference);
            break;

        case 'embed':
            await embedBatch(data.texts, {
                normalize: data.normalize,
                pooling: data.pooling,
                maxLength: data.maxLength,
                mode: data.mode,
                batchId: data.batchId
            });
            break;

        case 'ping':
            self.postMessage({ type: 'pong' });
            break;

        default:
            self.postMessage({
                type: 'error',
                error: `Unknown message type: ${type}`
            });
    }
});

// Signal worker is ready
self.postMessage({ type: 'ready' });
