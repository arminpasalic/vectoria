/**
 * Model-specific parameter constraints for WebLLM models
 * Defines valid ranges for temperature, max_tokens, and context windows
 */

const MODEL_CONSTRAINTS = {
    // --- Qwen 3 (latest generation, 2025) ---
    'Qwen3-0.6B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3 0.6B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~1.4 GB'
    },
    'Qwen3-1.7B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3 1.7B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~2.0 GB'
    },
    'Qwen3-4B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3 4B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~3.2 GB'
    },
    'Qwen3-8B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3 8B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~5.6 GB'
    },
    // --- Qwen 3.5 (newest, 2025) ---
    'Qwen3.5-0.8B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3.5 0.8B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~1.6 GB'
    },
    'Qwen3.5-2B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3.5 2B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~2.2 GB'
    },
    'Qwen3.5-4B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3.5 4B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~3.9 GB'
    },
    'Qwen3.5-9B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'Qwen3.5 9B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~6.4 GB'
    },
    // --- Qwen 2.5 ---
    'Qwen2.5-0.5B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Qwen2.5 0.5B', recommendedTemp: 0.4, recommendedMaxTokens: 512,
        estimatedSize: '~0.9 GB'
    },
    'Qwen2.5-1.5B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Qwen2.5 1.5B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~1.6 GB'
    },
    'Qwen2.5-3B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Qwen2.5 3B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~2.5 GB'
    },
    'Qwen2.5-7B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Qwen2.5 7B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~5.1 GB'
    },
    // --- Llama 3.2 / 3.1 ---
    'Llama-3.2-1B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Llama 3.2 1B', recommendedTemp: 0.4, recommendedMaxTokens: 512,
        estimatedSize: '~1.1 GB'
    },
    'Llama-3.2-3B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Llama 3.2 3B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~2.6 GB'
    },
    'Llama-3.1-8B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Llama 3.1 8B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~4.6 GB'
    },
    // --- Gemma 3 ---
    'gemma3-1b-it-q4f16_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Gemma 3 1B', recommendedTemp: 0.4, recommendedMaxTokens: 512,
        estimatedSize: '~2.5 GB'
    },
    // --- Gemma 2 ---
    'gemma-2-2b-it-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Gemma 2 2B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~1.9 GB'
    },
    'gemma-2-9b-it-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Gemma 2 9B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~6.4 GB'
    },
    // --- Phi ---
    'Phi-4-mini-instruct-q4f16_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Phi 4 Mini', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        estimatedSize: '~5.9 GB'
    },
    'Phi-3.5-mini-instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'Phi 3.5 Mini', recommendedTemp: 0.4, recommendedMaxTokens: 512,
        estimatedSize: '~2.0 GB'
    },
    // --- SmolLM2 ---
    'SmolLM2-1.7B-Instruct-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'SmolLM2 1.7B', recommendedTemp: 0.4, recommendedMaxTokens: 512,
        estimatedSize: '~1.8 GB'
    },
    // --- DeepSeek R1 ---
    'DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 32768], contextWindow: 32768,
        description: 'DeepSeek R1 Qwen 7B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~5.1 GB'
    },
    'DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC': {
        temp: [0, 2.0], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'DeepSeek R1 Llama 8B', recommendedTemp: 0.4, recommendedMaxTokens: 768,
        hasThinkMode: true, estimatedSize: '~5.9 GB'
    },
    // Default fallback
    'default': {
        temp: [0, 2.0], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'Default', recommendedTemp: 0.4, recommendedMaxTokens: 512,
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

if (typeof window !== 'undefined') {
    window.MODEL_CONSTRAINTS = MODEL_CONSTRAINTS;
    window.getModelConstraints = getModelConstraints;
}

export { MODEL_CONSTRAINTS, getModelConstraints };
