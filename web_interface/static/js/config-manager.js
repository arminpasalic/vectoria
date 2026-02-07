/**
 * Centralized Configuration Manager for Vectoria
 *
 * Single source of truth for ALL application settings
 * - Stores configuration in localStorage under 'vectoria_config' key
 * - Provides reactive updates via observers
 * - Handles defaults, migrations, and resets
 *
 * Usage:
 *   import { getConfig, updateConfig, resetConfig, observeConfig } from './config-manager.js';
 *
 *   const config = getConfig();
 *   updateConfig({ llm: { temperature: 0.8 } });
 *   observeConfig((newConfig) => console.log('Config changed:', newConfig));
 */

const STORAGE_KEY = 'vectoria_config';
const STORAGE_VERSION = 2;

// Complete default configuration
export const DEFAULT_CONFIG = {
    version: STORAGE_VERSION,

    // LLM / Language Model Settings
    llm: {
        model_id: 'gemma-2-2b-it-q4f32_1-MLC',
        temperature: 0.5,
        max_tokens: 768,
        top_p: 0.9,
        repeat_penalty: 1.15,
        context_window_size: 2048
    },

    // Embeddings Settings
    embeddings: {
        model_name: 'intfloat/multilingual-e5-small',
        batch_size: null,  // null = auto-detect
        device: 'auto',    // 'auto', 'cpu', 'webgpu'
        max_length: 256,
        tokens_per_batch: null,  // null = auto
        use_worker: true,        // Use Web Worker for background processing (prevents MacOS tab throttling)
        aggressive_mode: true    // Skip UI yields for max speed (prevents setTimeout throttling)
    },

    // Text Chunking Settings
    chunking: {
        enabled: true,
        chunk_size: 512,
        chunk_overlap: 128,
        min_chunk_size: 50
    },

    // RAG Prompts
    rag_prompts: {
        system_prompt: 'You are a helpful assistant answering questions based on provided documents.\nUse [Doc N] to cite sources. If information is missing, say so. Keep answers clear and focused.',
        user_template: 'Question: {question}\n\nDocuments:\n{context}\n\nAnswer based on the documents above:'
    },

    // HyDE Settings
    hyde: {
        prompt: 'Write a short factual paragraph that could answer this question:',
        temperature: 0.2,
        max_tokens: 256
    },

    // Search / RAG Retrieval Settings
    search: {
        num_results: 5,           // Number of context documents to retrieve
        retrieval_k: 60,          // Initial retrieval pool size
        vector_weight: 0.6,       // Balance between vector (0.6) and BM25 (0.4)
        similarity_threshold: 0.7, // Minimum similarity score
        retrieval_mode: 'semantic',
        quick_mode: 'vector'
    },

    // Clustering Settings (UMAP + HDBSCAN)
    clustering: {
        umap_n_neighbors: 15,
        umap_min_dist: 0.0,
        umap_metric: 'cosine',
        umap_clustering_dimensions: 15,  // Intermediate dimensions for clustering (before 2D viz)
        umap_sample_size: 10000,
        hdbscan_min_cluster_size: 5,
        hdbscan_min_samples: 5,
        hdbscan_metric: 'euclidean'
    },

    // Visualization Settings
    visualization: {
        point_size: 4,
        opacity: 0.8,
        show_cluster_hulls: true,
        enhanced_tooltips: true
    },

    // UI Preferences
    ui_preferences: {
        search_type: 'fast',   // 'fast', 'semantic', 'rag'
        hyde_enabled: false,
        highlight_results: true,
        result_count: 5            // Default result count for searches
    }
};

// Config change observers
const observers = new Set();

/**
 * Deep merge two objects
 * @param {Object} target - Target object
 * @param {Object} source - Source object to merge
 * @returns {Object} Merged object
 */
function deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
            result[key] = deepMerge(target[key] || {}, source[key]);
        } else {
            result[key] = source[key];
        }
    }

    return result;
}

/**
 * Get current configuration from localStorage or defaults
 * @returns {Object} Current configuration
 */
export function getConfig() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);

        if (stored) {
            const parsed = JSON.parse(stored);

            // Merge with defaults to handle new fields added in updates
            const merged = deepMerge(DEFAULT_CONFIG, parsed);

            // Update version if needed
            if (merged.version !== STORAGE_VERSION) {
                const parsedVersion = Number(parsed.version || 0);
                if (parsedVersion < 2) {
                    merged.ui_preferences = {
                        ...merged.ui_preferences,
                        search_type: 'fast'
                    };
                }
                merged.version = STORAGE_VERSION;
                saveConfig(merged);
            }

            return merged;
        }
    } catch (error) {
        console.error('❌ Error loading config from localStorage:', error);
    }

    // Return defaults if no stored config or error
    return JSON.parse(JSON.stringify(DEFAULT_CONFIG));
}

/**
 * Save configuration to localStorage
 * @param {Object} config - Configuration object to save
 * @returns {boolean} Success status
 */
export function saveConfig(config) {
    try {
        const toSave = {
            ...config,
            version: STORAGE_VERSION,
            lastUpdated: new Date().toISOString()
        };

        localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));

        // Notify all observers
        notifyObservers(toSave);

        return true;
    } catch (error) {
        console.error('❌ Error saving config to localStorage:', error);
        return false;
    }
}

/**
 * Update configuration with partial changes
 * @param {Object} updates - Partial configuration updates (supports nested)
 * @returns {Object} Updated configuration
 */
export function updateConfig(updates) {
    const current = getConfig();
    const updated = deepMerge(current, updates);

    saveConfig(updated);

    return updated;
}

/**
 * Reset configuration to defaults
 * @returns {Object} Default configuration
 */
export function resetConfig() {
    const defaults = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
    saveConfig(defaults);

    return defaults;
}

/**
 * Register observer callback for config changes
 * @param {Function} callback - Callback function(config)
 * @returns {Function} Unsubscribe function
 */
export function observeConfig(callback) {
    observers.add(callback);

    // Return unsubscribe function
    return () => observers.delete(callback);
}

/**
 * Notify all observers of config changes
 * @param {Object} config - New configuration
 */
function notifyObservers(config) {
    observers.forEach(callback => {
        try {
            callback(config);
        } catch (error) {
            console.error('❌ Error in config observer:', error);
        }
    });
}

/**
 * Migrate old localStorage keys to new config structure
 * Called automatically on first load
 */
export function migrateOldConfig() {
    const oldKeys = {
        'vectoria_rag_num_results': 'search.num_results',
        'vectoria_context_window': 'llm.context_window_size',
        'vectoria_llm_model': 'llm.model_id'
    };

    let needsMigration = false;
    const updates = {};

    for (const [oldKey, newPath] of Object.entries(oldKeys)) {
        let oldValue = null;
        try {
            oldValue = localStorage.getItem(oldKey);
        } catch (_) {
            continue;
        }

        if (oldValue !== null) {
            needsMigration = true;

            // Parse value (handle numbers)
            let parsedValue = oldValue;
            const numValue = parseInt(oldValue);
            if (!isNaN(numValue) && String(numValue) === oldValue) {
                parsedValue = numValue;
            }

            // Set nested path
            const pathParts = newPath.split('.');
            let current = updates;

            for (let i = 0; i < pathParts.length - 1; i++) {
                if (!current[pathParts[i]]) {
                    current[pathParts[i]] = {};
                }
                current = current[pathParts[i]];
            }

            current[pathParts[pathParts.length - 1]] = parsedValue;

            // Remove old key
            try { localStorage.removeItem(oldKey); } catch (_) {}
        }
    }

    if (needsMigration) {
        updateConfig(updates);
    }
}

/**
 * Get a specific config value by path
 * @param {string} path - Dot-separated path (e.g., 'llm.temperature')
 * @param {*} defaultValue - Default value if path not found
 * @returns {*} Config value
 */
export function getConfigValue(path, defaultValue = undefined) {
    const config = getConfig();
    const parts = path.split('.');
    let current = config;

    for (const part of parts) {
        if (current && typeof current === 'object' && part in current) {
            current = current[part];
        } else {
            return defaultValue;
        }
    }

    return current;
}

/**
 * Set a specific config value by path
 * @param {string} path - Dot-separated path (e.g., 'llm.temperature')
 * @param {*} value - Value to set
 */
export function setConfigValue(path, value) {
    const parts = path.split('.');
    const updates = {};
    let current = updates;

    for (let i = 0; i < parts.length - 1; i++) {
        current[parts[i]] = {};
        current = current[parts[i]];
    }

    current[parts[parts.length - 1]] = value;

    updateConfig(updates);
}

// Automatically migrate old config on module load
migrateOldConfig();

// Export for global access (for debugging)
if (typeof window !== 'undefined') {
    window.ConfigManager = {
        getConfig,
        saveConfig,
        updateConfig,
        resetConfig,
        observeConfig,
        getConfigValue,
        setConfigValue,
        DEFAULT_CONFIG
    };
}
