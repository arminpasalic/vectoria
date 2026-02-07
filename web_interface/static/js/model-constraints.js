/**
 * Model-specific parameter constraints for WebLLM models
 * Defines valid ranges for temperature, max_tokens, and context windows
 */

// Model constraints (can be imported as ES module or used as global)
const MODEL_CONSTRAINTS = {
    // Gemma models
    'gemma-2-2b-it-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'Gemma 2 2B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        estimatedSize: '~1.3 GB'
    },
    'gemma-2-9b-it-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'Gemma 2 9B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        estimatedSize: '~5.5 GB'
    },
    // Llama models
    'Llama-3.2-3B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'Llama 3.2 3B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        estimatedSize: '~1.8 GB'
    },
    'Llama-3.2-1B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'Llama 3.2 1B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        estimatedSize: '~0.7 GB'
    },
    // Qwen models
    'Qwen3-0.6B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 32768],
        contextWindow: 32768,
        description: 'Qwen 3 0.6B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~0.4 GB'
    },
    'Qwen3-1.7B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 32768],
        contextWindow: 32768,
        description: 'Qwen 3 1.7B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~1.0 GB'
    },
    'Qwen3-4B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 32768],
        contextWindow: 32768,
        description: 'Qwen 3 4B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~2.5 GB'
    },
    'Qwen3-8B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 32768],
        contextWindow: 32768,
        description: 'Qwen 3 8B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~4.5 GB'
    },
    // SmolLM models
    'SmolLM2-1.7B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'SmolLM2 1.7B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        estimatedSize: '~1.0 GB'
    },
    // DeepSeek models
    'DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 32768],
        contextWindow: 32768,
        description: 'DeepSeek R1 Qwen 7B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~4.0 GB'
    },
    'DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 8192],
        contextWindow: 8192,
        description: 'DeepSeek R1 Llama 8B',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 768,
        hasThinkMode: true,
        estimatedSize: '~4.5 GB'
    },
    // Phi models
    'Phi-3.5-mini-instruct-q4f32_1-MLC': {
        temp: [0, 2.0],
        maxTokens: [1, 4096],
        contextWindow: 4096,
        description: 'Phi 3.5 Mini',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 512,
        estimatedSize: '~2.0 GB'
    },
    // Default fallback
    'default': {
        temp: [0, 2.0],
        maxTokens: [1, 4096],
        contextWindow: 4096,
        description: 'Default',
        recommendedTemp: 0.4,
        recommendedMaxTokens: 512,
        estimatedSize: '~2.0 GB'
    }
};

/**
 * Get constraints for a specific model
 * @param {string} modelId - WebLLM model ID
 * @returns {Object} Model constraints
 */
function getModelConstraints(modelId) {
    return MODEL_CONSTRAINTS[modelId] || MODEL_CONSTRAINTS['default'];
}

// Make available globally for non-module scripts
if (typeof window !== 'undefined') {
    window.MODEL_CONSTRAINTS = MODEL_CONSTRAINTS;
    window.getModelConstraints = getModelConstraints;
}

// Export for ES modules
export { MODEL_CONSTRAINTS, getModelConstraints };
