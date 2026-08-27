/**
 * Authoritative Vectoria catalog for text-only, post-trained instruction models.
 *
 * Model IDs and vramRequiredMB4K come from @mlc-ai/web-llm 0.2.84's
 * prebuiltAppConfig. weightBytes and downloadBytes were audited on 2026-08-27
 * from each Hugging Face tensor/tokenizer manifest plus the exact v0_2_84
 * WebGPU WASM Content-Length. Download bytes and runtime VRAM are deliberately
 * separate: they describe different resources and must never share a UI label.
 */

const MODEL_CONSTRAINTS = {
    'Llama-3.2-3B-Instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'Llama 3.2 3B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'recommended', responseMode: 'direct', isDefault: true,
        weightBytes: 1807423488, downloadBytes: 1822644325, vramRequiredMB4K: 2263.69
    },
    'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Ministral 3 3B Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'recommended', responseMode: 'direct',
        weightBytes: 1929050112, downloadBytes: 1951697417, vramRequiredMB4K: 2863.69
    },
    'SmolLM2-1.7B-Instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'SmolLM2 1.7B Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'recommended', responseMode: 'direct',
        weightBytes: 962793472, downloadBytes: 971950796, vramRequiredMB4K: 1774.19
    },
    'gemma3-1b-it-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 8192,
        description: 'Gemma 3 1B', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'recommended', responseMode: 'direct',
        weightBytes: 562628864, downloadBytes: 607652373, vramRequiredMB4K: 711.07
    },
    'Phi-4-mini-instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'Phi 4 Mini Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'recommended', responseMode: 'direct',
        weightBytes: 2158049280, downloadBytes: 2185443060, vramRequiredMB4K: 3437.58
    },
    'Llama-3.1-8B-Instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'Llama 3.1 8B Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'direct', responseMode: 'direct',
        weightBytes: 4517404672, downloadBytes: 4532810198, vramRequiredMB4K: 5001
    },

    // Qwen3 is post-trained for instruction following. Vectoria can select its
    // documented /think or /no_think soft switch without another download.
    'Qwen3-0.6B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 40960,
        description: 'Qwen3 0.6B', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'switchable', responseMode: 'switchable', hasThinkMode: true,
        thinkSwitch: '/think', noThinkSwitch: '/no_think',
        weightBytes: 335372288, downloadBytes: 356920759, vramRequiredMB4K: 1403.34
    },
    'Qwen3-1.7B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 40960,
        description: 'Qwen3 1.7B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'switchable', responseMode: 'switchable', hasThinkMode: true,
        thinkSwitch: '/think', noThinkSwitch: '/no_think',
        weightBytes: 968001536, downloadBytes: 989585515, vramRequiredMB4K: 2036.66
    },
    'Qwen3-4B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 40960,
        description: 'Qwen3 4B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'switchable', responseMode: 'switchable', hasThinkMode: true,
        thinkSwitch: '/think', noThinkSwitch: '/no_think',
        weightBytes: 2262920192, downloadBytes: 2284830783, vramRequiredMB4K: 3431.59
    },
    'Qwen3-8B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 40960,
        description: 'Qwen3 8B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'switchable', responseMode: 'switchable', hasThinkMode: true,
        thinkSwitch: '/think', noThinkSwitch: '/no_think',
        weightBytes: 4607731712, downloadBytes: 4629660679, vramRequiredMB4K: 5695.78
    },

    // Qwen3.5's hard enable_thinking flag is not exposed by WebLLM 0.2.84's
    // chat-template route, so these are honestly presented as reasoning models.
    'Qwen3.5-0.8B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Qwen3.5 0.8B', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 423937664, downloadBytes: 453205372, vramRequiredMB4K: 1629.49
    },
    'Qwen3.5-2B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Qwen3.5 2B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 1059315328, downloadBytes: 1088608248, vramRequiredMB4K: 2245.44
    },
    'Qwen3.5-4B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Qwen3.5 4B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 2367117312, downloadBytes: 2396760278, vramRequiredMB4K: 3867.82
    },
    'Qwen3.5-9B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Qwen3.5 9B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 5038040064, downloadBytes: 5067721004, vramRequiredMB4K: 6433.01
    },
    'Ministral-3-3B-Reasoning-2512-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 262144,
        description: 'Ministral 3 3B Reasoning', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 1929050112, downloadBytes: 1951695552, vramRequiredMB4K: 2863.69
    },
    'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'DeepSeek R1 Llama 8B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'reasoning', responseMode: 'reasoning', hasThinkMode: true,
        weightBytes: 4517404672, downloadBytes: 4532762359, vramRequiredMB4K: 5001
    },

    'Hermes-3-Llama-3.2-3B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'Hermes 3 Llama 3.2 3B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'advanced', responseMode: 'direct',
        weightBytes: 1807423488, downloadBytes: 1822639792, vramRequiredMB4K: 2263.69
    },
    'Hermes-3-Llama-3.1-8B-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 8192], contextWindow: 131072,
        description: 'Hermes 3 Llama 3.1 8B', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'advanced', responseMode: 'direct',
        weightBytes: 4517404672, downloadBytes: 4532816009, vramRequiredMB4K: 4876.13
    },
    'OLMo-2-0425-1B-Instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'OLMo 2 1B Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'advanced', responseMode: 'direct',
        weightBytes: 835457024, downloadBytes: 851049399, vramRequiredMB4K: 1776.75
    },
    'OLMo-2-1124-7B-Instruct-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'OLMo 2 7B Instruct', recommendedTemp: 0.3, recommendedMaxTokens: 768,
        catalogGroup: 'advanced', responseMode: 'direct',
        weightBytes: 4106231808, downloadBytes: 4123027890, vramRequiredMB4K: 6479.01
    },
    'gemma-2-2b-it-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'Gemma 2 2B', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'legacy', responseMode: 'direct', systemPromptSupport: false,
        weightBytes: 1470915072, downloadBytes: 1498056313, vramRequiredMB4K: 1895.3
    },
    'gemma-2-9b-it-q4f16_1-MLC': {
        temp: [0, 2], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'Gemma 2 9B', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        catalogGroup: 'legacy', responseMode: 'direct', systemPromptSupport: false,
        weightBytes: 5199330304, downloadBytes: 5227099239, vramRequiredMB4K: 6422.01
    },
    default: {
        temp: [0, 2], maxTokens: [1, 4096], contextWindow: 4096,
        description: 'Default', recommendedTemp: 0.3, recommendedMaxTokens: 512,
        responseMode: 'direct', weightBytes: 0, downloadBytes: 0, vramRequiredMB4K: 0
    }
};

const QUALITY_RAG_MODELS = new Set([
    'Qwen3-8B-q4f16_1-MLC', 'Qwen3.5-9B-q4f16_1-MLC',
    'gemma-2-9b-it-q4f16_1-MLC', 'Llama-3.1-8B-Instruct-q4f16_1-MLC',
    'Hermes-3-Llama-3.1-8B-q4f16_1-MLC', 'OLMo-2-1124-7B-Instruct-q4f16_1-MLC',
    'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC'
]);
const LIMITED_RAG_MODELS = new Set([
    'Qwen3-0.6B-q4f16_1-MLC', 'Qwen3-1.7B-q4f16_1-MLC',
    'Qwen3.5-0.8B-q4f16_1-MLC', 'Qwen3.5-2B-q4f16_1-MLC',
    'gemma3-1b-it-q4f16_1-MLC', 'gemma-2-2b-it-q4f16_1-MLC',
    'OLMo-2-0425-1B-Instruct-q4f16_1-MLC', 'SmolLM2-1.7B-Instruct-q4f16_1-MLC'
]);

for (const [modelId, constraints] of Object.entries(MODEL_CONSTRAINTS)) {
    constraints.instructionTuned = modelId !== 'default';
    constraints.modelType = 'text-generation';
    constraints.systemPromptSupport ??= true;
    constraints.ragTier = QUALITY_RAG_MODELS.has(modelId)
        ? 'quality'
        : LIMITED_RAG_MODELS.has(modelId) ? 'limited' : 'recommended';
    constraints.schemaVerified = false;
}

const MODEL_GROUPS = [
    ['recommended', 'Recommended · direct instruction'],
    ['direct', 'Direct instruction'],
    ['switchable', 'Direct or reasoning · one download'],
    ['reasoning', 'Reasoning instruction models'],
    ['advanced', 'Specialized instruction models'],
    ['legacy', 'Legacy instruction models']
];

function getModelConstraints(modelId) {
    return MODEL_CONSTRAINTS[modelId] || MODEL_CONSTRAINTS.default;
}

function formatDecimalBytes(bytes, fractionDigits = 2) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return 'Unknown';
    if (value >= 1e9) return `${(value / 1e9).toFixed(fractionDigits)} GB`;
    return `${Math.round(value / 1e6)} MB`;
}

function formatVRAM(model) {
    const mb = Number(model?.vramRequiredMB4K);
    return Number.isFinite(mb) && mb > 0 ? `~${(mb / 1000).toFixed(2)} GB VRAM at 4K` : 'VRAM unknown';
}

function responseModeLabel(model) {
    if (model.responseMode === 'switchable') return 'direct/reasoning';
    if (model.responseMode === 'reasoning') return 'reasoning';
    return 'direct';
}

function renderModelCatalog(select = document.getElementById('llm-model-id')) {
    if (!select) return;
    const selectedId = window.ConfigManager?.getConfig()?.llm?.model_id
        || window.ConfigManager?.DEFAULT_CONFIG?.llm?.model_id;
    select.replaceChildren();
    for (const [groupId, groupLabel] of MODEL_GROUPS) {
        const models = Object.entries(MODEL_CONSTRAINTS)
            .filter(([, model]) => model.catalogGroup === groupId);
        if (!models.length) continue;
        const group = document.createElement('optgroup');
        group.label = groupLabel;
        for (const [modelId, model] of models) {
            const option = document.createElement('option');
            option.value = modelId;
            option.textContent = `${model.description} · ${formatDecimalBytes(model.downloadBytes)} download · ${responseModeLabel(model)}${model.isDefault ? ' · default' : ''}`;
            option.selected = modelId === selectedId;
            group.appendChild(option);
        }
        select.appendChild(group);
    }
}

function updateReasoningModeControl(modelId, configuredMode) {
    const select = document.getElementById('llm-reasoning-mode');
    const help = document.getElementById('llm-reasoning-mode-help');
    if (!select) return;
    const model = getModelConstraints(modelId);
    const switchable = model.responseMode === 'switchable';
    select.disabled = !switchable;
    select.value = switchable
        ? (configuredMode === 'reasoning' ? 'reasoning' : 'direct')
        : model.responseMode === 'reasoning' ? 'reasoning' : 'direct';
    if (help) {
        help.textContent = switchable
            ? 'Qwen3 can switch modes without downloading another model.'
            : model.responseMode === 'reasoning'
                ? 'This build uses reasoning in Vectoria; internal reasoning is hidden from the final answer.'
                : 'This is a direct instruction model without a reasoning phase.';
    }
}

function getModelDownloadProgress(modelId, progress) {
    const model = getModelConstraints(modelId);
    const total = Number(model.downloadBytes) || 0;
    const weights = Number(model.weightBytes) || 0;
    const overhead = Math.max(total - weights, 0);
    const ratio = Math.min(Math.max(Number(progress) || 0, 0), 1);
    return { loaded: Math.min(overhead + ratio * weights, total), total };
}

if (typeof window !== 'undefined') {
    // Remove values produced by the old 100 MiB-per-shard guess. They are not
    // measurements and must not survive as a source of truth.
    try { localStorage.removeItem('vectoria_model_download_sizes'); } catch (_) {}
    delete window.__webllmRealDownloadSizes;
    Object.assign(window, {
        MODEL_CONSTRAINTS,
        getModelConstraints,
        formatModelDownloadSize: formatDecimalBytes,
        formatModelVRAM: formatVRAM,
        renderModelCatalog,
        updateReasoningModeControl,
        getModelDownloadProgress
    });
    renderModelCatalog();
}

export {
    MODEL_CONSTRAINTS,
    MODEL_GROUPS,
    getModelConstraints,
    formatDecimalBytes,
    formatVRAM,
    renderModelCatalog,
    updateReasoningModeControl,
    getModelDownloadProgress
};
