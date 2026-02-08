// Vectoria - Main JavaScript file for RAG-Vectoria Integration
// Debug: Set a global flag so we can check if this file loaded
window.VECTORIA_JS_LOADED = true;
window.VECTORIA_LOAD_TIME = new Date().toISOString();

/**
 * Update slider constraints based on selected model
 * @param {string} modelId - WebLLM model ID
 */
function updateSlidersForModel(modelId) {
    if (typeof window.getModelConstraints !== 'function') {
        console.warn('⚠️ Model constraints not loaded yet');
        return;
    }

    const constraints = window.getModelConstraints(modelId);
    // Update Advanced Settings sliders
    const tempSlider = document.getElementById('temperature');
    const maxTokensSlider = document.getElementById('max-tokens');

    if (tempSlider) {
        tempSlider.min = constraints.temp[0];
        tempSlider.max = constraints.temp[1];
        // Clamp current value to new range
        if (parseFloat(tempSlider.value) > constraints.temp[1]) {
            tempSlider.value = constraints.recommendedTemp || constraints.temp[1];
        }
        // Update display
        const valueDisplay = tempSlider.nextElementSibling;
        if (valueDisplay && valueDisplay.classList.contains('range-value')) {
            valueDisplay.textContent = tempSlider.value;
        }
    }

    if (maxTokensSlider) {
        maxTokensSlider.min = constraints.maxTokens[0];
        maxTokensSlider.max = constraints.maxTokens[1];
        // Clamp current value to new range
        if (parseInt(maxTokensSlider.value, 10) > constraints.maxTokens[1]) {
            maxTokensSlider.value = constraints.recommendedMaxTokens || Math.min(768, constraints.maxTokens[1]);
        }
    }

    // Update RAG Settings modal sliders
    const quickTempSlider = document.getElementById('quick-temperature');
    const quickMaxTokensSlider = document.getElementById('quick-max-tokens');

    if (quickTempSlider) {
        quickTempSlider.min = constraints.temp[0];
        quickTempSlider.max = constraints.temp[1];
        if (parseFloat(quickTempSlider.value) > constraints.temp[1]) {
            quickTempSlider.value = constraints.recommendedTemp || constraints.temp[1];
        }
        const valueDisplay = document.getElementById('quick-temperature-value');
        if (valueDisplay) valueDisplay.textContent = quickTempSlider.value;
    }

    if (quickMaxTokensSlider) {
        quickMaxTokensSlider.min = constraints.maxTokens[0];
        quickMaxTokensSlider.max = constraints.maxTokens[1];
        if (parseInt(quickMaxTokensSlider.value, 10) > constraints.maxTokens[1]) {
            quickMaxTokensSlider.value = constraints.recommendedMaxTokens || Math.min(768, constraints.maxTokens[1]);
        }
        const valueDisplay = document.getElementById('quick-max-tokens-value');
        if (valueDisplay) valueDisplay.textContent = quickMaxTokensSlider.value;
    }

    showToast(`Sliders updated for ${constraints.description}`, 'info');
}

// Store visualization instances
window.mainVisualization = null;

// Global state
let currentVisualizationData = null;
let currentSession = null;

const METADATA_ORDER_STORAGE_KEY = 'vectoria_metadata_order_v1';
const THEME_STORAGE_KEY = 'vectoria_theme_preference';
const MAX_METADATA_PREVIEW_ITEMS = 5;
const METADATA_EXCLUDED_KEYS = [
    'index', 'x', 'y', 'cluster', 'text', 'cluster_probability', 'cluster_color',
    'cluster_name', 'doc_id', 'chunk_id', 'cluster_keyword_scores', 'cluster_keywords_viz',
    'color', 'filename', 'original_filename', 'processing_type', 'selected_column',
    'num_chunks', 'metadata'
];
const LOW_PRIORITY_METADATA_KEYS = new Set(['cluster_label', 'cluster_keywords', 'cluster_keywords_viz']);
let metadataSortOrder = loadMetadataSortOrder();
let lastSelectedTextPoint = null;
let lastSelectedTextIndex = null;
let _metadataDragAbort = null;
window.__currentTextListPoints = [];
window.__textListLock = null;

// Custom cluster names storage (clusterId -> custom name)
let customClusterNames = new Map();

function unlockTextList(reason = '') {
    if (window.__textListLock) {
    }
    window.__textListLock = null;
    updateExportButtonVisibility();
}

function resolvePointForResult(result) {
    if (!currentVisualizationData || !currentVisualizationData.points) return null;
    const points = currentVisualizationData.points;

    const numericIndex = Number.isInteger(result?.index) ? result.index : Number.parseInt(result?.index, 10);
    if (Number.isInteger(numericIndex)) {
        const indexMatch = points.find(point => point.index === numericIndex);
        if (indexMatch) return indexMatch;
    }

    const candidateCoordinates = Array.isArray(result?.coordinates) && result.coordinates.length >= 2
        ? result.coordinates.map(Number)
        : null;
    if (candidateCoordinates) {
        const [cx, cy] = candidateCoordinates;
        const coordinateMatch = points.find(point => Math.abs(point.x - cx) < 1e-6 && Math.abs(point.y - cy) < 1e-6);
        if (coordinateMatch) return coordinateMatch;
    }

    const candidateChunk = result?.chunk_id ?? result?.metadata?.chunk_id;
    if (candidateChunk !== undefined && candidateChunk !== null) {
        const chunkStr = String(candidateChunk);
        const chunkMatch = points.find(point => String(point.chunk_id) === chunkStr);
        if (chunkMatch) return chunkMatch;
    }

    // Check parent_id first (for BM25 chunk results that have parent_id pointing to visualization point)
    const candidateParent = result?.parent_id ?? result?.metadata?.parent_id;
    if (candidateParent !== undefined && candidateParent !== null) {
        const parentStr = String(candidateParent);
        const parentMatch = points.find(point => String(point.doc_id) === parentStr);
        if (parentMatch) return parentMatch;
    }

    const candidateDoc = result?.doc_id ?? result?.metadata?.doc_id;
    if (candidateDoc !== undefined && candidateDoc !== null) {
        const docStr = String(candidateDoc);
        const docMatch = points.find(point => String(point.doc_id) === docStr);
        if (docMatch) return docMatch;
    }

    const candidateSource = result?.source_id ?? result?.metadata?.source_id;
    if (candidateSource !== undefined && candidateSource !== null) {
        const sourceStr = String(candidateSource);
        const sourceMatch = points.find(point => String(point.source_id) === sourceStr);
        if (sourceMatch) return sourceMatch;
    }

    const baseText = (result?.text || result?.content || '').toLowerCase().trim();
    if (baseText.length) {
        const directMatch = points.find(point => (point.text || '').toLowerCase().trim() === baseText);
        if (directMatch) return directMatch;

        if (baseText.length > 30) {
            const truncated = baseText.substring(0, 120);
            const partialMatch = points.find(point => (point.text || '').toLowerCase().includes(truncated));
            if (partialMatch) return partialMatch;
        }

        if (baseText.length > 20) {
            let bestMatch = null;
            let bestScore = 0;
            points.forEach(point => {
                if (point.text) {
                    const similarity = calculateTextSimilarity(baseText.substring(0, 150), (point.text || '').toLowerCase().substring(0, 150));
                    if (similarity > bestScore) {
                        bestScore = similarity;
                        bestMatch = point;
                    }
                }
            });
            if (bestMatch && bestScore > 0.78) {
                return bestMatch;
            }
        }
    }

    return null;
}

function applyThemePreference(theme) {
    const normalized = theme === 'dark' ? 'dark' : 'light';
    const htmlElement = document.documentElement;
    if (!htmlElement) return;

    htmlElement.classList.remove('light', 'dark'); // Remove both light/dark classes
    htmlElement.classList.add(normalized); // Add the new class
    htmlElement.setAttribute('data-theme', normalized); // Set data-theme attribute

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.classList.toggle('is-dark', normalized === 'dark');
        toggleBtn.setAttribute('aria-label', normalized === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    }

    updateLogoForTheme(normalized);

    if (window.mainVisualization) {
        try {
            window.mainVisualization.centerView({ preferClusterCenter: true, animate: false });
            window.mainVisualization.requestRender();
        } catch (error) {
            console.warn('Unable to re-center visualization after theme change', error);
        }
    }
}

function updateLogoForTheme(theme) {
    const useDark = theme === 'dark';
    const logos = document.querySelectorAll('[data-logo-light][data-logo-dark]');
    logos.forEach((logo) => {
        const target = useDark ? logo.dataset.logoDark : logo.dataset.logoLight;
        if (target && logo.getAttribute('src') !== target) {
            logo.setAttribute('src', target);
        }
    });
}

function updateModelSetupModalModelNames() {
    const embeddingEl = document.getElementById('model-setup-embedding-name');
    const llmEl = document.getElementById('model-setup-llm-name');
    if (!embeddingEl && !llmEl) return;

    const config = window.ConfigManager ? window.ConfigManager.getConfig() : null;
    const defaults = window.ConfigManager ? window.ConfigManager.DEFAULT_CONFIG : null;

    const embeddingName = config?.embeddings?.model_name
        || defaults?.embeddings?.model_name
        || 'intfloat/multilingual-e5-small';
    const llmName = config?.llm?.model_id
        || defaults?.llm?.model_id
        || 'gemma-2-2b-it-q4f32_1-MLC';

    if (embeddingEl) embeddingEl.textContent = embeddingName;
    if (llmEl) llmEl.textContent = llmName;
}

function initThemeToggle() {
    const htmlElement = document.documentElement;

    // Apply initial theme based on the class set by the early script
    const currentTheme = htmlElement.classList.contains('dark') ? 'dark' : 'light';
    applyThemePreference(currentTheme); // This will correctly set the button icon etc.

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const nextTheme = htmlElement.classList.contains('dark') ? 'light' : 'dark';
            applyThemePreference(nextTheme);
            try {
                localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            } catch (error) {
                console.warn('Unable to persist theme preference', error);
            }
        });
    }
}

// Vibrant Tailwind-inspired color palette
const DARK24_BASE_PALETTE = [
    '#3B82F6', // blue-500
    '#EF4444', // red-500
    '#10B981', // green-500
    '#F59E0B', // amber-500
    '#8B5CF6', // violet-500
    '#EC4899', // pink-500
    '#14B8A6', // teal-500
    '#F97316', // orange-500
    '#06B6D4', // cyan-500
    '#6366F1', // indigo-500
    '#84CC16', // lime-500
    '#F43F5E', // rose-500
    '#A855F7', // purple-500
    '#22D3EE', // cyan-400
    '#EAB308', // yellow-500
    '#D946EF', // fuchsia-500
    '#16A34A', // green-600
    '#0EA5E9', // sky-500
    '#DC2626', // red-600
    '#7C3AED', // violet-600
    '#059669', // emerald-600
    '#DB2777', // pink-600
    '#2563EB', // blue-600
    '#EA580C', // orange-600
    '#0D9488'  // teal-600
];
const ACCENT_COLOR_MAP = {
    hover: '#C084FC', // purple-400
    focus: '#3B82F6', // blue-500
    search: '#F43F5E'  // rose-500
};
const GOLDEN_ANGLE = 137.508;
const OUTLIER_COLOR = '#9CA3AF'; // Desaturated gray with slight transparency
const OUTLIER_OPACITY = 0.6;
const OUTLIER_DIM_COLOR = 'rgba(148, 155, 170, 0.65)';
const DIMMED_NEUTRAL_COLOR = 'rgba(82, 86, 94, 0.75)';
const COLOR_STORAGE_KEY = 'vectoria_cluster_topics_v2_tailwind';
function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function hslToHex(h, s, l) {
    const sNorm = clamp(s, 0, 100) / 100;
    const lNorm = clamp(l, 0, 100) / 100;

    const c = (1 - Math.abs(2 * lNorm - 1)) * sNorm;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = lNorm - c / 2;

    let r = 0;
    let g = 0;
    let b = 0;

    if (h >= 0 && h < 60) {
        r = c;
        g = x;
    } else if (h >= 60 && h < 120) {
        r = x;
        g = c;
    } else if (h >= 120 && h < 180) {
        g = c;
        b = x;
    } else if (h >= 180 && h < 240) {
        g = x;
        b = c;
    } else if (h >= 240 && h < 300) {
        r = x;
        b = c;
    } else {
        r = c;
        b = x;
    }

    const toHex = (value) => {
        const scaled = Math.round((value + m) * 255);
        return scaled.toString(16).padStart(2, '0');
    };

    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToHsl(hex) {
    const value = parseInt(hex.replace('#', ''), 16);
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;

    const rNorm = r / 255;
    const gNorm = g / 255;
    const bNorm = b / 255;

    const max = Math.max(rNorm, gNorm, bNorm);
    const min = Math.min(rNorm, gNorm, bNorm);
    let h = 0;
    let s = 0;
    const l = (max + min) / 2;

    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case rNorm:
                h = ((gNorm - bNorm) / d + (gNorm < bNorm ? 6 : 0));
                break;
            case gNorm:
                h = ((bNorm - rNorm) / d + 2);
                break;
            case bNorm:
                h = ((rNorm - gNorm) / d + 4);
                break;
        }
        h *= 60;
    }

    return {
        h: h,
        s: s * 100,
        l: l * 100
    };
}

function adjustPaletteColor(baseHex, variantIndex) {
    const baseHsl = hexToHsl(baseHex);
    const hueOffset = (variantIndex * GOLDEN_ANGLE) % 360;
    const hue = (baseHsl.h + hueOffset) % 360;
    const lightnessShift = ((variantIndex % 3) - 1) * 15; // -15%, 0%, +15%
    const lightness = clamp(baseHsl.l + lightnessShift, 35, 75);
    const saturation = baseHsl.s; // No clamping - preserve original vibrancy
    return hslToHex(hue, saturation, lightness);
}

class VectoriaColorManager {
    constructor() {
        this.fullPalette = [...DARK24_BASE_PALETTE];
        this.basePalette = this.fullPalette.slice(0, Math.max(1, this.fullPalette.length - 3));
        this.accentColors = ACCENT_COLOR_MAP;
        this.goldenAngle = GOLDEN_ANGLE;
        this.outlierColor = OUTLIER_COLOR;
        this.outlierDimColor = OUTLIER_DIM_COLOR;
        this.dimmedNeutralColor = DIMMED_NEUTRAL_COLOR;
        this.topicColorMap = this.loadTopicColorMap();
        this.colorCache = new Map();
    }

    loadTopicColorMap() {
        try {
            if (typeof window !== 'undefined' && window.localStorage) {
                const raw = window.localStorage.getItem(COLOR_STORAGE_KEY);
                if (raw) {
                    const parsed = JSON.parse(raw);
                    if (parsed && typeof parsed === 'object') {
                        return parsed;
                    }
                }
            }
        } catch (error) {
            console.warn('VectoriaColorManager: unable to access localStorage for color persistence', error);
        }
        return {};
    }

    saveTopicColorMap() {
        try {
            if (typeof window !== 'undefined' && window.localStorage) {
                window.localStorage.setItem(COLOR_STORAGE_KEY, JSON.stringify(this.topicColorMap));
            }
        } catch (error) {
            // Silently ignore persistence failures (private mode, etc.)
        }
    }

    normalizeTopic(name) {
        if (!name || typeof name !== 'string') return null;
        const trimmed = name.trim().toLowerCase();
        return trimmed.length ? trimmed : null;
    }

    topicKey(clusterId, clusterName) {
        const normalized = this.normalizeTopic(clusterName);
        if (normalized) return normalized;
        const numericId = Number(clusterId);
        return Number.isFinite(numericId) ? `cluster-${numericId}` : 'cluster-0';
    }

    registerColor(clusterId, clusterName, color) {
        if (!color) return;
        const key = this.topicKey(clusterId, clusterName);
        this.colorCache.set(key, color);
        const normalized = this.normalizeTopic(clusterName);
        if (normalized) {
            this.topicColorMap[normalized] = color;
            this.saveTopicColorMap();
        }
    }

    generateColor(clusterId) {
        const paletteSize = this.basePalette.length;
        const baseIndex = ((clusterId % paletteSize) + paletteSize) % paletteSize;
        const variantIndex = Math.floor(clusterId / paletteSize);
        const baseColor = this.basePalette[baseIndex];
        if (variantIndex === 0) {
            return baseColor;
        }
        return this.adjustPaletteColor(baseColor, variantIndex);
    }

    adjustPaletteColor(baseHex, variantIndex) {
        return adjustPaletteColor(baseHex, variantIndex);
    }

    getColor(clusterId, clusterName = null, providedColor = null) {
        if (Number(clusterId) === -1) {
            return this.outlierColor;
        }

        if (providedColor) {
            this.registerColor(clusterId, clusterName, providedColor);
            return providedColor;
        }

        const normalized = this.normalizeTopic(clusterName);
        if (normalized && this.topicColorMap[normalized]) {
            const stored = this.topicColorMap[normalized];
            this.colorCache.set(normalized, stored);
            return stored;
        }

        const key = this.topicKey(clusterId, clusterName);
        if (this.colorCache.has(key)) {
            return this.colorCache.get(key);
        }

        const numericId = Number(clusterId);
        const color = this.generateColor(Number.isFinite(numericId) ? numericId : 0);
        this.colorCache.set(key, color);
        if (normalized && !this.topicColorMap[normalized]) {
            this.topicColorMap[normalized] = color;
            this.saveTopicColorMap();
        }
        return color;
    }

    getAccent(type) {
        return this.accentColors[type] || this.accentColors.focus;
    }
}

const colorManager = new VectoriaColorManager();
const colorCache = colorManager.colorCache;
window.VectoriaColorManager = colorManager;
window.VECTORIA_COLOR_PALETTE = DARK24_BASE_PALETTE; // Export palette for reference

function generateClusterColor(clusterId) {
    const numericId = Number(clusterId);
    return colorManager.generateColor(Number.isFinite(numericId) ? numericId : 0);
}

function getClusterColor(clusterId, clusterName = null) {
    return colorManager.getColor(clusterId, clusterName);
}


function parseColorToRgb(color) {
    if (typeof color !== 'string') {
        return { r: 120, g: 120, b: 120 };
    }

    const trimmed = color.trim();

    const hexMatch = /^#?([a-f\d]{6})([a-f\d]{2})?$/i.exec(trimmed);
    if (hexMatch) {
        return {
            r: parseInt(hexMatch[1].slice(0, 2), 16),
            g: parseInt(hexMatch[1].slice(2, 4), 16),
            b: parseInt(hexMatch[1].slice(4, 6), 16)
        };
    }

    const rgbaMatch = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+)\s*)?\)$/i.exec(trimmed);
    if (rgbaMatch) {
        return {
            r: Math.round(parseFloat(rgbaMatch[1])),
            g: Math.round(parseFloat(rgbaMatch[2])),
            b: Math.round(parseFloat(rgbaMatch[3]))
        };
    }

    return { r: 120, g: 120, b: 120 };
}

function applyAlphaToColor(color, alpha = 0.2) {
    const { r, g, b } = parseColorToRgb(color);
    const a = clamp(alpha, 0, 1);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function ensureConsistentColor(clusterId, providedColor = null, clusterName = null) {
    return colorManager.getColor(clusterId, clusterName, providedColor);
}

function getClusterAccent(type) {
    return colorManager.getAccent(type);
}

function getClusterName(clusterId) {
    if (clusterId === -1) {
        return 'Outlier';
    }
    // Check for custom name
    if (customClusterNames.has(clusterId)) {
        return customClusterNames.get(clusterId);
    }
    return `Cluster ${clusterId}`;
}

// Cluster renaming persistence
function saveCustomClusterNames() {
    const obj = Object.fromEntries(customClusterNames);
    try { localStorage.setItem('vectoria_cluster_names', JSON.stringify(obj)); } catch (_) {}
}

function loadCustomClusterNames() {
    try {
        const saved = localStorage.getItem('vectoria_cluster_names');
        if (saved) {
            const obj = JSON.parse(saved);
            customClusterNames = new Map(Object.entries(obj).map(([k, v]) => [parseInt(k, 10), v]));
        }
    } catch (e) {
        console.warn('Failed to load custom cluster names:', e);
    }
}

function renameCluster(clusterId, newName) {
    if (clusterId === -1) {
        showToast('Cannot rename outlier cluster', 'warning');
        return;
    }

    const trimmedName = newName.trim();
    if (!trimmedName) {
        // Empty name = reset to default
        customClusterNames.delete(clusterId);
    } else {
        customClusterNames.set(clusterId, trimmedName);
    }

    saveCustomClusterNames();
    refreshClusterDisplays();
}

function promptRenameCluster(clusterId) {
    if (clusterId === -1) {
        showToast('Cannot rename outlier cluster', 'warning');
        return;
    }

    const defaultClusterName = `Cluster ${clusterId}`;
    const hasCustomName = customClusterNames.has(clusterId);

    const modal = document.getElementById('cluster-rename-modal');
    const input = document.getElementById('cluster-rename-input');
    const defaultLabel = document.getElementById('cluster-rename-default');
    const confirmBtn = document.getElementById('cluster-rename-confirm');
    const cancelBtn = document.getElementById('cluster-rename-cancel');
    const cancelXBtn = document.getElementById('cluster-rename-cancel-x');

    defaultLabel.textContent = defaultClusterName;
    input.value = hasCustomName ? customClusterNames.get(clusterId) : '';

    const closeModal = () => {
        modal.style.display = 'none';
        confirmBtn.removeEventListener('click', onConfirm);
        cancelBtn.removeEventListener('click', onCancel);
        cancelXBtn.removeEventListener('click', onCancel);
        modal.removeEventListener('click', onOverlay);
        document.removeEventListener('keydown', onKeydown);
    };

    const onConfirm = () => {
        renameCluster(clusterId, input.value);
        closeModal();
    };

    const onCancel = () => closeModal();

    const onOverlay = (e) => {
        if (e.target === modal) closeModal();
    };

    const onKeydown = (e) => {
        if (e.key === 'Escape') closeModal();
        if (e.key === 'Enter') onConfirm();
    };

    confirmBtn.addEventListener('click', onConfirm);
    cancelBtn.addEventListener('click', onCancel);
    cancelXBtn.addEventListener('click', onCancel);
    modal.addEventListener('click', onOverlay);
    document.addEventListener('keydown', onKeydown);

    modal.style.display = 'flex';
    input.focus();
    input.select();
}

function refreshClusterDisplays() {
    // Refresh text list if visible
    if (typeof updateTextList === 'function' && window.__currentTextListPoints && window.__currentTextListPoints.length > 0) {
        updateTextList(window.__currentTextListPoints, { force: true });
    }

    // Refresh selected text details view if open
    const selectedTextView = document.getElementById('selected-text-view');
    if (selectedTextView && selectedTextView.style.display !== 'none' && lastSelectedTextPoint) {
        showTextDetails(lastSelectedTextPoint, lastSelectedTextIndex || 0, { suppressHighlight: true, preserveScroll: true });
    }

    // Refresh filter panel
    if (typeof updateMetadataUI === 'function') {
        updateMetadataUI();
    }

    // Refresh visualization labels (if renderer has method)
    if (window.renderer && typeof window.renderer.updateClusterLabels === 'function') {
        window.renderer.updateClusterLabels();
    }

    showToast('Cluster renamed', 'success');
}

// Filter by a specific cluster - programmatically select cluster_label filter and apply
function filterByCluster(clusterId) {
    // Build possible cluster label values (metadata stores "Noise" for -1, "Cluster X" for others)
    const clusterLabelValue = clusterId === -1 ? 'Noise' : `Cluster ${clusterId}`;

    // Try to find cluster_label filter first, then fall back to cluster filter
    let filterItem = document.querySelector('.metadata-filter-item[data-field="cluster_label"]');
    if (!filterItem) {
        filterItem = document.querySelector('.metadata-filter-item[data-field="cluster"]');
    }

    if (!filterItem) {
        showToast('Cluster filter not available', 'warning');
        return;
    }

    const filterType = filterItem.dataset.type;
    let found = false;

    // Handle checkbox-based filters (<=20 unique values)
    const checkboxes = filterItem.querySelectorAll('input[type="checkbox"]');
    if (checkboxes.length > 0) {
        // Clear all checkboxes first
        checkboxes.forEach(cb => {
            cb.checked = false;
        });

        // Find and check the matching checkbox
        checkboxes.forEach(cb => {
            const cbValue = cb.value;
            if (cbValue === clusterLabelValue || cbValue === String(clusterId)) {
                cb.checked = true;
                found = true;
            }
        });

        if (!found) {
            const availableValues = Array.from(checkboxes).map(cb => cb.value);
        }
    }

    // Handle select dropdown filters
    if (!found) {
        const selectEl = filterItem.querySelector('select');
        if (selectEl) {
            // Clear previous selection
            Array.from(selectEl.options).forEach(opt => opt.selected = false);

            // Find and select matching option
            Array.from(selectEl.options).forEach(opt => {
                if (opt.value === clusterLabelValue || opt.value === String(clusterId)) {
                    opt.selected = true;
                    found = true;
                }
            });

            if (!found) {
                const availableValues = Array.from(selectEl.options).map(opt => opt.value);
            }
        }
    }

    // Handle text input filters (>20 unique values - uses datalist)
    if (!found) {
        const textInput = filterItem.querySelector('input[type="text"]');
        if (textInput) {
            textInput.value = clusterLabelValue;
            found = true;
        }
    }

    if (!found) {
        showToast(`Cluster "${getClusterName(clusterId)}" not found in filters`, 'warning');
        return;
    }

    // Apply the filter
    if (typeof applyMetadataFilters === 'function') {
        applyMetadataFilters();
    }

    const clusterName = getClusterName(clusterId);
    showToast(`Filtered to: ${clusterName}`, 'success');
}

// Main initialization function
document.addEventListener('DOMContentLoaded', function() {
    initThemeToggle();
    loadCustomClusterNames();

    // Setup bidirectional config sync between duplicate controls
    setupConfigSync();

    // Enforce initial tab state: show Upload only on first load
    try {
        const tabs = document.querySelectorAll('.tab-content');
        tabs.forEach(t => t.classList.remove('active'));
        const up = document.getElementById('upload-tab');
        const ex = document.getElementById('explore-tab');
        if (up) { up.classList.add('active'); up.style.display = 'block'; }
        if (ex) { ex.classList.remove('active'); ex.style.display = 'none'; }
        const sr = document.getElementById('search-results');
        if (sr) sr.style.display = 'none';
    } catch(e) { /* no-op */ }
    
    // Initialize tab navigation
    initTabNavigation();
    
    // Initialize CSV upload functionality
    initCSVUpload();
    
    // Initialize data exploration functionality
    initDataExploration();
    
    // Initialize ENHANCED search functionality with fast search as default
    initEnhancedSearchFunctionality();
    
    // Set up download buttons
    initializeDownloadButtons();
    
    // Set up warning when closing browser with processed data
    setupCloseWarning();
    
    // Initialize settings functionality
    initializeSettingsUI();
    
    // Initialize modal functionality
    initializeModalHandlers();
    
    // Initialize setting restrictions
    initializeSettingRestrictions();
    
    // Initialize enhanced model downloads
    enhanceSettingsWithModelDownloads();
    
    // Initialize metadata detection and filtering
    initializeMetadataSystem();

    // Initialize per-search RAG metadata controls
    initializeRAGMetadataControls();

    // Note: Quick settings modal is initialized in fast-search.js when opened

    // Wire up global filter status bar
    setupFilterStatusBar();

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            const filtersActive = activeMetadataFilters && Object.keys(activeMetadataFilters).length > 0;
            const searchActive = window.__textListLock === 'lasso' || (window.currentSearchResults && Array.isArray(window.currentSearchResults.results) && window.currentSearchResults.results.length > 0);
            if (!filtersActive && !searchActive) {
                return;
            }

            if (filtersActive && typeof clearMetadataFilters === 'function') {
                clearMetadataFilters();
            }

            if (searchActive && typeof clearSearch === 'function') {
                clearSearch();
            }

            event.preventDefault();
        }
    });

    // Lightweight scroll performance toggle to reduce heavy effects during scroll
    initScrollPerfToggle();
});



// Function to check if user has processed data that would be lost
function hasUnsavedData() {
    // Check if there's processed data in the pipeline
    const pipeline = window.browserML?.pipeline;
    if (pipeline?.currentDataset?.documents?.length > 0) {
        return true;
    }
    // Also check for visualization data
    if (window.currentDataset?.length > 0 || window.canvasData?.length > 0) {
        return true;
    }
    return false;
}

// Function to set up warning when closing browser with processed data
function setupCloseWarning() {
    window.addEventListener('beforeunload', function(event) {
        // Only show warning if there's processed data
        if (hasUnsavedData()) {
            // Standard way to trigger the browser's "Leave site?" dialog
            event.preventDefault();
            // For older browsers, return a string (modern browsers show generic message)
            event.returnValue = 'You have processed data that will be lost if you leave. Are you sure?';
            return event.returnValue;
        }
    });
}

// Scroll performance hints centralized in height-matcher.js
function initScrollPerfToggle() {
    let timer = null;
    let active = false;
    let raf = null;

    const onScroll = () => {
        if (raf) return;

        raf = requestAnimationFrame(() => {
            if (!active) {
                active = true;
                document.body.classList.add('scrolling');
            }
            clearTimeout(timer);
            timer = setTimeout(() => {
                active = false;
                document.body.classList.remove('scrolling');
            }, 100);
            raf = null;
        });
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    document.addEventListener('scroll', onScroll, { passive: true, capture: true });
}

// Function to initialize tab navigation
function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            activateTab(tabId);
        });
    });
}

// Function to activate a specific tab (can be called externally)
function activateTab(tabId) {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Remove active class from all tabs and contents
    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabContents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });
    
    // Add active class to selected tab and content
    const targetBtn = document.querySelector(`[data-tab="${tabId}"]`);
    const targetContent = document.getElementById(tabId);
    
    if (targetBtn) {
        targetBtn.classList.add('active');
    }
    if (targetContent) {
        targetContent.classList.add('active');
        targetContent.style.display = 'block';
    }
    
    // Load data for the explore tab even if no button exists
    if (tabId === 'explore-tab') {
        try { loadVisualizationData(); } catch (e) { console.error(e); }

        // IMPORTANT: Attach filter button handlers when explore tab becomes visible
        // This ensures the buttons work even though they're initially hidden
        setTimeout(() => {
            if (typeof attachFilterButtonHandlers === 'function') {
                attachFilterButtonHandlers();
            }
            if (typeof setupFilterStatusBar === 'function') {
                setupFilterStatusBar();
            }

            // Initialize pending visualization if we deferred it earlier
            if (window._pendingVisualizationData) {
                initializeEnhancedVisualization(window._pendingVisualizationData);
            }
        }, 100); // Small delay to ensure DOM is fully visible
    }
}

// Function to initialize CSV upload
function initCSVUpload() {
    const uploadForm = document.getElementById('csv-upload-form');
    const uploadError = document.getElementById('upload-error');
    const uploadSuccess = document.getElementById('upload-success');
    const uploadLoader = document.getElementById('upload-loader');
    const fileInput = document.getElementById('csv-file');
    const columnSelectionDiv = document.getElementById('column-selection');
    const processingSuccess = document.getElementById('processing-success');
    const processingError = document.getElementById('processing-error');
    const processBtn = document.getElementById('process-csv-btn');
    const customFileBtn = document.getElementById('custom-file-btn');
    const customFileName = document.getElementById('custom-file-name');

    // Custom file button click triggers hidden file input
    if (customFileBtn) {
        customFileBtn.addEventListener('click', function() {
            if (!fileInput.disabled) {
                fileInput.click();
            }
        });
    }

    // Update file name display when file is selected
    if (fileInput && customFileName) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                customFileName.textContent = this.files[0].name;
                customFileName.classList.add('has-file');
            } else {
                customFileName.textContent = 'No file selected';
                customFileName.classList.remove('has-file');
            }
        });
    }

    // Drag-and-drop support on the file input wrapper
    const dropZone = document.querySelector('.custom-file-input-wrapper');
    if (dropZone && fileInput) {
        const acceptedExts = ['.csv', '.xlsx', '.xls', '.json', '.txt'];

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragenter', function(e) {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('drag-over');

            if (fileInput.disabled) return;

            const file = e.dataTransfer.files[0];
            if (!file) return;

            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!acceptedExts.includes(ext)) {
                if (typeof showToast === 'function') {
                    showToast('Unsupported file type. Accepted: CSV, Excel, JSON, TXT', 'error');
                }
                return;
            }

            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        });
    }

    // Define the reset function
    function resetUploadForm() {
        fileInput.value = '';
        if (customFileName) {
            customFileName.textContent = 'No file selected';
            customFileName.classList.remove('has-file');
        }
        
        uploadError.style.display = 'none';
        uploadSuccess.style.display = 'none';
        uploadLoader.style.display = 'none';
        columnSelectionDiv.style.display = 'none';
        processingSuccess.style.display = 'none';
        processingError.style.display = 'none';
        
        processBtn.disabled = true;
        delete processBtn.dataset.filepath;
        
        const columnSelect = document.getElementById('text-column-select');
        columnSelect.innerHTML = '<option value="">Select a column...</option>';
        
        const sampleDataTable = document.getElementById('sample-data-table');
        sampleDataTable.innerHTML = '';
        document.getElementById('sample-data-container').style.display = 'none';
    }

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Reset previous states before new upload attempt
        uploadError.style.display = 'none';
        uploadSuccess.style.display = 'none';
        columnSelectionDiv.style.display = 'none';
        processingSuccess.style.display = 'none';
        processingError.style.display = 'none';

        const file = fileInput.files[0];
        
        if (!file) {
            uploadError.textContent = 'Please select a file to upload.';
            uploadError.style.display = 'block';
            return;
        }
        
        // Hide error and show loader
        uploadError.style.display = 'none';
        uploadSuccess.style.display = 'none';
        uploadLoader.style.display = 'block';
        
        // Create FormData and send request
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/csv_columns', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error uploading file');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            uploadLoader.style.display = 'none';
            
            // Show success message
            uploadSuccess.textContent = `File uploaded successfully (${data.file_type.toUpperCase()}). Please select a text column.`;
            uploadSuccess.style.display = 'block';
            
            // Show column selection form
            columnSelectionDiv.style.display = 'block';
            
            // Populate column select dropdown
            populateColumnSelect(data.columns);
            
            // Display sample data
            displaySampleData(data.sample_data);
            
            // Keep process button disabled until column is selected
            processBtn.disabled = true;
        })
        .catch(error => {
            // Hide loader
            uploadLoader.style.display = 'none';
            
            // Show error
            uploadError.textContent = error.message;
            uploadError.style.display = 'block';
            console.error('Upload error:', error);
            processBtn.disabled = true;
        });
    });
    
    // Process CSV button
    processBtn.addEventListener('click', function() {
        const textColumn = document.getElementById('text-column-select').value;
        
        if (!textColumn) {
            processingError.textContent = 'Please select a text column to analyze.';
            processingError.style.display = 'block';
            return;
        }
        
        // Hide previous messages and show loader
        processingError.style.display = 'none';
        processingSuccess.style.display = 'none';
        document.getElementById('processing-loader').style.display = 'block';
        
        // Disable button during processing
        processBtn.disabled = true;
        
        // Get advanced parameters
        const umapParams = {
            n_neighbors: parseInt(document.getElementById('umap-n-neighbors').value, 10) || 15,
            min_dist: parseFloat(document.getElementById('umap-min-dist').value) || 0.1,
            metric: document.getElementById('umap-metric').value || 'cosine'
        };
        
        const hdbscanParams = {
            min_cluster_size: parseInt(document.getElementById('hdbscan-min-cluster-size')?.value, 10) || 5,
            min_samples: parseInt(document.getElementById('hdbscan-min-samples')?.value, 10) || 5
        };
        
        // Send processing request
        fetch('/api/process_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text_column: textColumn,
                umap_params: umapParams,
                hdbscan_params: hdbscanParams
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error processing file');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            document.getElementById('processing-loader').style.display = 'none';
            
            // Show success message
            processingSuccess.innerHTML = `
                <strong>Processing completed successfully!</strong><br>
                • Processed ${data.rows || data.num_documents || 0} rows using column: <strong>${escapeHtml(data.text_column || 'text')}</strong><br>
                • Generated ${data.num_chunks || 0} text chunks<br>
                • Created ${data.has_visualization ? 'interactive visualization' : 'embeddings only'}<br>
                ${data.processing_time ? `• Processing time: ${data.processing_time.toFixed(2)}s` : ''}
            `;
            processingSuccess.style.display = 'block';
            
            // Re-enable button
            processBtn.disabled = false;

            // Enable export dataset button now that data is processed
            const exportDatasetBtn = document.getElementById('export-dataset-btn');
            if (exportDatasetBtn) {
                exportDatasetBtn.disabled = false;
            }

            // Clear current visualization data to force reload
            currentVisualizationData = null;
            window.currentVisualizationData = null;

            // Clear custom cluster names from previous dataset
            customClusterNames.clear();
            saveCustomClusterNames();

            // Show processing summary modal then transition to Explore
            showProcessingSummaryModal(data);
            showToast('File processed successfully! Preparing Explore…', 'success');
            
        })
        .catch(error => {
            // Hide loader
            document.getElementById('processing-loader').style.display = 'none';
            
            // Show error
            processingError.textContent = error.message;
            processingError.style.display = 'block';
            
            // Re-enable button
            processBtn.disabled = false;
            
            console.error('Processing error:', error);
        });
    });
}

// Reveal Explore tab button and navigate with fade-in animation
// Show modal with processing summary and then transition to Explore
function showProcessingSummaryModal(data) {
    const modal = document.getElementById('processing-summary-modal');
    const content = document.getElementById('processing-summary-content');
    if (!modal || !content) {
        // Fallback: transition automatically after 5 seconds
        setTimeout(() => transitionToExplore(), 5000);
        return;
    }

    // Debug: Log the data to see what we're receiving
    // Extract data with fallbacks
    const numDocuments = data.numDocuments || data.num_documents || data.rows || '—';
    const numClusters = data.numClusters || data.num_clusters || '—';
    const emptyRowCount = data.emptyRowCount || 0;
    const duplicateCount = data.duplicateCount || data.duplicatesRemoved || 0;
    const fileName = data.fileName || data.filename || '—';
    const textCol = data.textColumn || data.text_column || '—';
    
    // Extract timing data
    const timings = data.timings || {};
    const totalTime = timings.total || data.processing_time || 0;
    const embeddingTime = timings.embedding || 0;
    const umapTime = timings.umap || 0;
    const clusteringTime = timings.clustering || 0;
    const indexingTime = timings.indexing || 0;
    
    const timingValue = (value) => (value > 0 ? `${value.toFixed(2)}s` : '—');

    content.innerHTML = `
        <div class="processing-summary-card">
            <header class="processing-summary-header">
                <span class="processing-summary-icon" aria-hidden="true">
                    <svg class="processing-summary-icon-graphic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20 6L9 17l-5-5"></path>
                    </svg>
                </span>
                <div class="processing-summary-heading">
                    <h2>Processing Complete</h2>
                    <p class="processing-summary-subtitle">The dataset is structured and ready for exploration.</p>
                </div>
            </header>

            <section class="processing-summary-metrics">
                <article class="processing-summary-stat">
                    <span class="processing-summary-label">Documents</span>
                    <span class="processing-summary-value">${numDocuments}</span>
                </article>
                <article class="processing-summary-stat">
                    <span class="processing-summary-label">Clusters</span>
                    <span class="processing-summary-value">${numClusters}</span>
                </article>
                <article class="processing-summary-stat">
                    <span class="processing-summary-label">Total Time</span>
                    <span class="processing-summary-value">${totalTime.toFixed(2)}s</span>
                </article>
            </section>

            <section class="processing-summary-section">
                <h3 class="processing-summary-section-title">
                    <span class="processing-summary-section-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="9"></circle>
                            <path d="M12 7v5l3 2"></path>
                        </svg>
                    </span>
                    Processing Timeline
                </h3>
                <dl class="processing-summary-timings">
                    <div class="processing-summary-timing">
                        <dt>Embeddings</dt>
                        <dd>${timingValue(embeddingTime)}</dd>
                    </div>
                    <div class="processing-summary-timing">
                        <dt>Indexing</dt>
                        <dd>${timingValue(indexingTime)}</dd>
                    </div>
                    <div class="processing-summary-timing">
                        <dt>UMAP</dt>
                        <dd>${timingValue(umapTime)}</dd>
                    </div>
                    <div class="processing-summary-timing">
                        <dt>HDBSCAN Clustering</dt>
                        <dd>${timingValue(clusteringTime)}</dd>
                    </div>
                </dl>
            </section>

            <section class="processing-summary-meta">
                <div class="processing-summary-meta-row">
                    <span class="processing-summary-meta-label">File</span>
                    <span class="processing-summary-meta-value">${escapeHtml(fileName)}</span>
                </div>
                <div class="processing-summary-meta-row">
                    <span class="processing-summary-meta-label">Text Column</span>
                    <span class="processing-summary-meta-value">${escapeHtml(textCol)}</span>
                </div>
            </section>

            ${emptyRowCount > 0 ? `
                <section class="processing-summary-note processing-summary-note--attention">
                    <span class="processing-summary-note-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 9v4"></path>
                            <path d="M12 17h.01"></path>
                            <path d="M10.29 3.86 1.82 18a1 1 0 0 0 .86 1.5h18.64a1 1 0 0 0 .86-1.5L13.71 3.86a1 1 0 0 0-1.72 0Z"></path>
                        </svg>
                    </span>
                    <div>
                        <p class="processing-summary-note-title">Dropped Rows</p>
                        <p class="processing-summary-note-body">
                            ${emptyRowCount} row${emptyRowCount !== 1 ? 's were' : ' was'} skipped because the selected column contained no text.
                        </p>
                    </div>
                </section>
            ` : ''}

            ${duplicateCount > 0 ? `
                <section class="processing-summary-note">
                    <span class="processing-summary-note-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="7" height="7" rx="1"></rect>
                            <rect x="14" y="14" width="7" height="7" rx="1"></rect>
                            <path d="M7 14h10"></path>
                            <path d="M7 10v4"></path>
                            <path d="M17 10v4"></path>
                        </svg>
                    </span>
                    <div>
                        <p class="processing-summary-note-title">Duplicates Removed</p>
                        <p class="processing-summary-note-body">
                            ${duplicateCount} duplicate entr${duplicateCount !== 1 ? 'ies were' : 'y was'} removed prior to embedding.
                        </p>
                    </div>
                </section>
            ` : ''}

            <footer class="processing-summary-footer">
                <p class="processing-summary-countdown">
                    Continuing to Explore in <span id="countdown-seconds">10</span> seconds...
                </p>
            </footer>
        </div>
    `;

    // Open modal with animation
    modal.style.display = 'flex';
    document.body.classList.add('modal-open');
    setTimeout(() => modal.classList.add('modal-visible'), 10);

    // Countdown timer (10 seconds)
    let countdown = 10;
    const countdownEl = document.getElementById('countdown-seconds');
    
    if (modal.__countdownInterval) clearInterval(modal.__countdownInterval);
    if (modal.__autoTimer) clearTimeout(modal.__autoTimer);
    
    modal.__countdownInterval = setInterval(() => {
        countdown--;
        if (countdownEl) {
            countdownEl.textContent = countdown;
        }
        if (countdown <= 0) {
            clearInterval(modal.__countdownInterval);
        }
    }, 1000);

    // Auto-continue after 10 seconds
    modal.__autoTimer = setTimeout(() => {
        if (modal.classList.contains('modal-visible')) {
            clearInterval(modal.__countdownInterval);
            closeProcessingSummaryModal(() => transitionToExplore());
        }
    }, 10000);
}

function closeProcessingSummaryModal(afterCloseCb = null) {
    const modal = document.getElementById('processing-summary-modal');
    if (!modal) return;
    
    // Clear any running timers
    if (modal.__countdownInterval) {
        clearInterval(modal.__countdownInterval);
        modal.__countdownInterval = null;
    }
    if (modal.__autoTimer) {
        clearTimeout(modal.__autoTimer);
        modal.__autoTimer = null;
    }
    
    modal.classList.remove('modal-visible');
    setTimeout(() => {
        modal.style.display = 'none';
        document.body.classList.remove('modal-open');
        if (typeof afterCloseCb === 'function') afterCloseCb();
    }, 300);
}

function transitionToExplore() {
    const uploadSection = document.getElementById('upload-tab');
    if (!uploadSection) {
        activateTab('explore-tab');
        return;
    }
    // If already not active, just show explore
    if (!uploadSection.classList.contains('active')) {
        activateTab('explore-tab');
        return;
    }
    // Animate upload section out, then show explore (which fades in via CSS)
    let switched = false;
    const onAnimEnd = () => {
        uploadSection.removeEventListener('animationend', onAnimEnd);
        // Now swap to explore
        if (!switched) {
            switched = true;
            activateTab('explore-tab');
        }
        // Clean up the class for next time
        uploadSection.classList.remove('fade-out-up');
    };
    uploadSection.addEventListener('animationend', onAnimEnd);
    uploadSection.classList.add('fade-out-up');
    // Fallback in case animationend doesn't fire
    setTimeout(() => {
        if (!switched) {
            switched = true;
            activateTab('explore-tab');
            uploadSection.classList.remove('fade-out-up');
        }
    }, 700);
}

// Function to populate column select dropdown
function populateColumnSelect(columns) {
    const select = document.getElementById('text-column-select');
    select.innerHTML = '<option value="">Select a column...</option>';
    select.classList.remove('has-selection');

    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        select.appendChild(option);
    });

    // Add change listener for selection styling and process button state
    select.addEventListener('change', function() {
        const processBtn = document.getElementById('process-csv-btn');
        if (this.value) {
            this.classList.add('has-selection');
            processBtn.disabled = false;
            processBtn.classList.add('ready-to-process');
        } else {
            this.classList.remove('has-selection');
            processBtn.disabled = true;
            processBtn.classList.remove('ready-to-process');
        }
    });
}

// Function to display sample data
function displaySampleData(sampleData) {
    const container = document.getElementById('sample-data-container');
    const table = document.getElementById('sample-data-table');
    
    if (!sampleData || sampleData.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    // Create table HTML
    const columns = Object.keys(sampleData[0]);
    let html = '<table class="sample-table-content"><thead><tr>';
    
    // Table headers
    columns.forEach(column => {
        html += `<th>${escapeHtml(column)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Table rows
    sampleData.forEach(row => {
        html += '<tr>';
        columns.forEach(column => {
            let cellValue = row[column] || '';
            // Truncate long values
            if (typeof cellValue === 'string' && cellValue.length > 100) {
                cellValue = cellValue.substring(0, 100) + '...';
            }
            html += `<td>${escapeHtml(String(cellValue))}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    table.innerHTML = html;
    container.style.display = 'block';
}

// Function to initialize data exploration
function initDataExploration() {
    // This will be called when the explore tab is activated
}

// Function to load visualization data
function loadVisualizationData() {
    if (currentVisualizationData) {
        initializeVisualization(currentVisualizationData);
        return;
    }
    
    const loader = document.getElementById('loader');
    const errorDiv = document.getElementById('visualization-error');
    const noDataDiv = document.getElementById('no-data-message');
    
    // Show loader
    loader.style.display = 'block';
    errorDiv.style.display = 'none';
    noDataDiv.style.display = 'none';
    
    fetch('/api/visualization_data')
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to load visualization data');
                });
            }
            return response.json();
        })
        .then(data => {
            loader.style.display = 'none';
        currentVisualizationData = data;
        window.currentVisualizationData = data; // Make available globally for fast search
        unlockTextList('filter update');
        unlockTextList('visualization reset');
            unlockTextList('visualization data loaded');
            initializeVisualization(data);
        })
        .catch(error => {
            loader.style.display = 'none';
            
            if (error.message.includes('No processed data')) {
                noDataDiv.style.display = 'block';
            } else {
                errorDiv.textContent = error.message;
                errorDiv.style.display = 'block';
            }
            
            console.error('Visualization data error:', error);
        });
}

// Enhanced visualization initialization with fast search integration
function initializeVisualization(data) {
    if (!data || !data.points || data.points.length === 0) {
        const noData = document.getElementById('no-data-message');
        if (noData) noData.style.display = 'block';
        return;
    }

    // Route to enhanced WebGL-capable initializer with Canvas fallback
    try {
        initializeEnhancedVisualization(data);

        // Update text list
        updateTextList(data.points);

        // Initialize fast search with visualization data (deferred)
        if (window.initializeFastSearch) {
            setTimeout(() => {
                window.initializeFastSearch(data.points);
            }, 100);
        }
    } catch (error) {
        console.error('Error initializing visualization:', error);
        const vizError = document.getElementById('visualization-error');
        if (vizError) {
            vizError.textContent = 'Error initializing visualization';
            vizError.style.display = 'block';
        }
    }
}

// Function to update text list
function updateTextList(points, options = {}) {
    const { force = false } = options;
    if (!force && window.__textListLock === 'lasso') {
        return;
    }
    if (force) {
        unlockTextList('forced update');
    }

    const container = document.getElementById('text-list');
    if (!container) return;

    updateTextContentCount(points ? points.length : 0);
    window.__currentTextListPoints = Array.isArray(points) ? points : [];

    // Update export button visibility (only shown for full unfiltered dataset)
    updateExportButtonVisibility();

    // Render all items directly for smooth native scrolling
    const fragment = document.createDocumentFragment();
    points.forEach((point, index) => {
        const el = renderTextListItem(point, index);
        fragment.appendChild(el);
    });
    container.innerHTML = '';
    container.appendChild(fragment);
}

function updateTextContentCount(count) {
    const label = document.getElementById('text-content-count');
    if (!label) return;
    const total = typeof count === 'number' && count >= 0 ? count : 0;
    label.textContent = `(${total})`;
}

// Render a single text list item element with improved performance
function renderTextListItem(point, index) {
    const item = document.createElement('div');
    item.className = 'text-item';
    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', '0');
    const originalIndex = (typeof point.index === 'number') ? point.index : index;
    item.dataset.index = originalIndex;
    item.dataset.listIndex = index;

    const clusterColor = ensureConsistentColor(point.cluster, point.cluster_color, point.cluster_name);
    const clusterBadgeColor = applyAlphaToColor(clusterColor, 0.18);
    // Use getClusterName() to support custom names
    const clusterName = getClusterName(point.cluster);
    const clusterId = point.cluster;

    // Add keywords if available
    let keywordText = '';
    const keywordPreview = (point.cluster_keywords_viz && point.cluster_keywords_viz.length > 0)
        ? point.cluster_keywords_viz
        : (point.cluster_keywords || []);
    if (keywordPreview && keywordPreview.length > 0) {
        keywordText = ` • ${keywordPreview.slice(0, 3).join(', ')}`;
    }

    const metadataPreviewItems = buildMetadataPreviewItems(point);
    const metadataPreviewHtml = metadataPreviewItems
        ? `<div class="text-item-metadata-preview">${metadataPreviewItems}</div>`
        : '';

    // Create proper HTML structure with fixed layout
    item.innerHTML = `
        <div class="text-item-header">
            <span class="text-item-number">Item ${index + 1}</span>
            <span class="text-item-cluster cluster-badge-filter"
                  data-cluster-id="${clusterId}"
                  style="background-color: ${clusterBadgeColor}; color: ${clusterColor}; border: 1px solid ${clusterColor}; cursor: pointer;"
                  title="Click to filter by this cluster">
                ${escapeHtml(clusterName)}${escapeHtml(keywordText)}
            </span>
        </div>
        <div class="text-item-content">
            ${point.text ? escapeHtml(point.text.length > 200 ? point.text.substring(0, 200) + '...' : point.text) : 'No text available'}
        </div>
        ${metadataPreviewHtml}
    `;

    // Add click handler for cluster badge to filter by cluster
    const clusterBadge = item.querySelector('.cluster-badge-filter');
    if (clusterBadge) {
        clusterBadge.addEventListener('click', (e) => {
            e.stopPropagation();
            filterByCluster(clusterId);
        });
    }

    // Use event delegation for better performance - attach to container
    const openDetails = () => showTextDetails(point, originalIndex);
    item.addEventListener('click', openDetails, { passive: true });
    item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openDetails();
        }
    });
    
    return item;
}


// Function to show text details
function showTextDetails(point, index, options = {}) {
    const {
        suppressHighlight = false,
        preserveScroll = false,
        focusVisualization = true
    } = options;
    const textList = document.getElementById('text-list');
    const selectedTextView = document.getElementById('selected-text-view');
    const selectedTextContent = document.getElementById('selected-text-content');
    const previousScrollTop = preserveScroll && selectedTextView ? selectedTextView.scrollTop : 0;
    
    // Hide list and show details
    textList.style.display = 'none';
    selectedTextView.style.display = 'block';
    updateExportButtonVisibility();

    // Populate details
    const clusterColor = ensureConsistentColor(point.cluster, point.cluster_color, point.cluster_name);
    // Use getClusterName() to support custom names - it handles -1 (Outlier) and custom names
    const clusterName = getClusterName(point.cluster);
    
    // Build comprehensive metadata display
    let metadataHtml = formatAllMetadata(point);
    lastSelectedTextPoint = point;
    lastSelectedTextIndex = index;
    
    const clusterId = point.cluster;
    const canRename = clusterId !== -1;

    selectedTextContent.innerHTML = `
        <div class="selected-text-header">
            <span class="cluster-indicator" style="background-color: ${clusterColor}"></span>
            <h3>Item ${index + 1} - <span class="${canRename ? 'cluster-name-rename' : ''}"
                ${canRename ? `data-cluster-id="${clusterId}" style="cursor: pointer; border-bottom: 0.2px solid currentColor; padding-bottom: 1px;" title="Click to rename cluster"` : ''}>${escapeHtml(clusterName)}</span></h3>
        </div>
        <div class="selected-text-body">
            <div class="full-text-section">
                <p><strong>Full Text:</strong></p>
                <div class="full-text-content">${escapeHtml(point.text || 'No text available')}</div>
            </div>

            <div class="metadata-section">
                <div class="primary-metadata">
                    ${point.cluster !== -1 ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Cluster Probability:</span>
                        <span class="metadata-value">${(Math.trunc(point.cluster_probability * 10000) / 100).toFixed(2)}%</span>
                    </div>
                    ` : ''}
                    <div class="metadata-item">
                        <span class="metadata-label">Coordinates:</span>
                        <span class="metadata-value">(${point.x.toFixed(3)}, ${point.y.toFixed(3)})</span>
                    </div>
                </div>

                ${metadataHtml}
            </div>
        </div>
    `;

    // Add click handler for cluster name rename in header
    const clusterNameEl = selectedTextContent.querySelector('.cluster-name-rename');
    if (clusterNameEl) {
        clusterNameEl.addEventListener('click', (e) => {
            e.stopPropagation();
            const cid = parseInt(clusterNameEl.dataset.clusterId, 10);
            promptRenameCluster(cid);
        });
    }

    if (selectedTextView) {
        selectedTextView.scrollTop = preserveScroll ? previousScrollTop : 0;
    }

    initMetadataDragAndDrop(selectedTextContent);

    // Highlight point in visualization
    if (!suppressHighlight && window.mainVisualization) {
        window.mainVisualization.highlightPoint(index, { focus: focusVisualization, revealTooltip: true });
    }
}

// Function to format all metadata flexibly
function formatAllMetadata(point) {
    const entries = getSortedMetadataEntries(point);
    
    if (!entries.length) {
        return '<div class="no-metadata">No metadata available</div>';
    }
    
    let metadataItems = '';
    entries.forEach((entry, idx) => {
        metadataItems += `
            <div class="metadata-item metadata-item-reorderable" data-metadata-key="${entry.key}">
                <div class="metadata-item-header">
                    <span class="metadata-label">${entry.label}</span>
                    <span class="metadata-drag-handle" title="Drag to reorder">
                        <i class="fas fa-grip-lines"></i>
                    </span>
                </div>
                <span class="metadata-value">${entry.formattedValue}</span>
            </div>
        `;
        
        if (idx === MAX_METADATA_PREVIEW_ITEMS - 1 && entries.length > MAX_METADATA_PREVIEW_ITEMS) {
            metadataItems += '<div class="metadata-priority-divider" role="presentation"></div>';
        }
    });
    
    return `
        <div class="additional-metadata">
            <div class="metadata-section-title">
                <h4>Metadata</h4>
                <p class="metadata-section-subtitle">Drag metadata to prioritize which fields appear first in the text list.</p>
            </div>
            <div class="metadata-list">
                ${metadataItems}
            </div>
        </div>
    `;
}

function getSortedMetadataEntries(point) {
    if (!point || typeof point !== 'object') {
        return [];
    }
    
    const metadataEntries = [];
    for (const [key, value] of Object.entries(point)) {
        if (!METADATA_EXCLUDED_KEYS.includes(key) && value !== null && value !== undefined && value !== '') {
            metadataEntries.push({ key, value });
        }
    }
    
    if (!metadataEntries.length) {
        return [];
    }
    
    const defaultSortedEntries = metadataEntries.slice().sort((a, b) => defaultMetadataSort(a.key, b.key));
    ensureMetadataOrderForKeys(defaultSortedEntries.map(entry => entry.key));
    
    const orderedEntries = defaultSortedEntries.slice().sort((a, b) => compareMetadataByOrder(a.key, b.key));
    return orderedEntries.map(({ key, value }) => ({
        key,
        label: formatMetadataKey(key),
        formattedValue: formatMetadataValue(value),
        previewValue: formatMetadataPreviewValue(value),
        value
    }));
}

function defaultMetadataSort(keyA, keyB) {
    const aLow = LOW_PRIORITY_METADATA_KEYS.has(keyA);
    const bLow = LOW_PRIORITY_METADATA_KEYS.has(keyB);
    if (aLow !== bLow) {
        return aLow ? 1 : -1;
    }

    const aIsMetadata = keyA.startsWith('metadata_');
    const bIsMetadata = keyB.startsWith('metadata_');
    
    if (aIsMetadata && !bIsMetadata) return -1;
    if (!aIsMetadata && bIsMetadata) return 1;
    return keyA.localeCompare(keyB);
}

function ensureMetadataOrderForKeys(keys) {
    if (!Array.isArray(keys) || !keys.length) return;
    let changed = false;
    
    keys.forEach((key) => {
        if (!metadataSortOrder.includes(key)) {
            metadataSortOrder.push(key);
            changed = true;
        }
    });
    
    if (changed) {
        saveMetadataSortOrder();
    }
}

function compareMetadataByOrder(keyA, keyB) {
    const aLow = LOW_PRIORITY_METADATA_KEYS.has(keyA);
    const bLow = LOW_PRIORITY_METADATA_KEYS.has(keyB);
    if (aLow !== bLow) {
        return aLow ? 1 : -1;
    }

    const indexA = metadataSortOrder.indexOf(keyA);
    const indexB = metadataSortOrder.indexOf(keyB);
    
    if (indexA === -1 && indexB === -1) {
        return defaultMetadataSort(keyA, keyB);
    }
    
    if (indexA === -1) return 1;
    if (indexB === -1) return -1;
    return indexA - indexB;
}

function buildMetadataPreviewItems(point) {
    const entries = getSortedMetadataEntries(point).slice(0, MAX_METADATA_PREVIEW_ITEMS);
    if (!entries.length) return '';
    
    return entries.map((entry) => `
        <div class="metadata-chip" data-metadata-key="${entry.key}">
            <span class="metadata-chip-key">${entry.label}</span>
            <span class="metadata-chip-value">${entry.previewValue}</span>
        </div>
    `).join('');
}

// Function to format metadata keys for display
function formatMetadataKey(key) {
    // Remove metadata_ prefix if present
    let displayKey = key.replace(/^metadata_/, '');
    
    // Remove "Column" prefix if present (case insensitive)
    displayKey = displayKey.replace(/^Column\s+/i, '');
    
    // Convert snake_case and camelCase to Title Case
    displayKey = displayKey
        .replace(/[_-]/g, ' ')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    
    return displayKey;
}

// Function to format metadata values flexibly
function formatMetadataValue(value) {
    if (value === null || value === undefined) {
        return '<em>null</em>';
    }
    
    // Handle different data types
    if (typeof value === 'boolean') {
        return value ? '<span class="bool-true">True</span>' : '<span class="bool-false">False</span>';
    }
    
    if (typeof value === 'number') {
        // Check if it's an integer or float
        if (Number.isInteger(value)) {
            return `<span class="number-int">${value.toLocaleString()}</span>`;
        } else {
            return `<span class="number-float">${value.toFixed(3)}</span>`;
        }
    }
    
    if (typeof value === 'string') {
        // Check if it looks like a date
        const dateRegex = /^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2})?/;
        if (dateRegex.test(value)) {
            try {
                const date = new Date(value);
                if (!isNaN(date.getTime())) {
                    return `<span class="date-value">${date.toLocaleDateString()} ${date.toLocaleTimeString()}</span>`;
                }
            } catch (e) {
                // Fall through to string handling
            }
        }
        
        // Check if it's a very long string (truncate if needed)
        if (value.length > 200) {
            return `<span class="string-long" title="${escapeHtml(value)}">${escapeHtml(value.substring(0, 200))}...</span>`;
        }
        
        return `<span class="string-value">${escapeHtml(value)}</span>`;
    }
    
    // Handle arrays and objects
    if (Array.isArray(value)) {
        const allStrings = value.every(item => typeof item === 'string');
        if (allStrings) {
            const joined = value.join(', ');
            return `<span class="array-string-value">${escapeHtml(joined)}</span>`;
        }
        return `<span class="array-value">[${value.length} items]</span>`;
    }
    
    if (typeof value === 'object') {
        return `<span class="object-value">{${Object.keys(value).length} properties}</span>`;
    }
    
    // Fallback
    return `<span class="unknown-value">${String(value)}</span>`;
}

function formatMetadataPreviewValue(value) {
    let displayValue = '';
    
    if (value === null || value === undefined) {
        displayValue = '—';
    } else if (typeof value === 'boolean') {
        displayValue = value ? 'True' : 'False';
    } else if (typeof value === 'number') {
        displayValue = Number.isInteger(value)
            ? value.toLocaleString()
            : (Math.round(value * 100) / 100).toString();
    } else if (typeof value === 'string') {
        displayValue = value.trim();
    } else if (Array.isArray(value)) {
        const allStrings = value.every(item => typeof item === 'string');
        if (allStrings) {
            displayValue = value.join(', ');
        } else {
            displayValue = `${value.length} item${value.length === 1 ? '' : 's'}`;
        }
    } else if (typeof value === 'object') {
        const keys = Object.keys(value);
        displayValue = `${keys.length} propert${keys.length === 1 ? 'y' : 'ies'}`;
    } else {
        displayValue = String(value);
    }
    
    if (displayValue.length > 60) {
        displayValue = `${displayValue.substring(0, 57)}...`;
    }
    
    return escapeHtml(displayValue);
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function loadMetadataSortOrder() {
    try {
        if (typeof window !== 'undefined' && window.localStorage) {
            const stored = window.localStorage.getItem(METADATA_ORDER_STORAGE_KEY);
            if (stored) {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed)) {
                    return parsed.filter((value) => typeof value === 'string');
                }
            }
        }
    } catch (error) {
        console.warn('Vectoria metadata order: unable to load order from storage', error);
    }
    return [];
}

function saveMetadataSortOrder(order = metadataSortOrder) {
    try {
        if (typeof window !== 'undefined' && window.localStorage) {
            window.localStorage.setItem(METADATA_ORDER_STORAGE_KEY, JSON.stringify(order));
        }
    } catch (error) {
        console.warn('Vectoria metadata order: unable to persist order', error);
    }
}

function refreshTextListMetadataPreview() {
    const container = document.getElementById('text-list');
    const points = window.__currentTextListPoints;
    
    if (!container || !Array.isArray(points) || !points.length) {
        return;
    }
    
    const renderedItems = container.querySelectorAll('.text-item');
    renderedItems.forEach((item) => {
        const listIndex = Number(item.dataset.listIndex);
        if (!Number.isFinite(listIndex)) return;
        const point = points[listIndex];
        if (!point) return;
        
        const previewContent = buildMetadataPreviewItems(point);
        let previewContainer = item.querySelector('.text-item-metadata-preview');
        
        if (!previewContent) {
            if (previewContainer) {
                previewContainer.remove();
            }
            return;
        }
        
        if (!previewContainer) {
            previewContainer = document.createElement('div');
            previewContainer.className = 'text-item-metadata-preview';
            item.appendChild(previewContainer);
        }
        
        previewContainer.innerHTML = previewContent;
    });
}

function handleMetadataOrderChanged() {
    refreshTextListMetadataPreview();
    
    if (lastSelectedTextPoint) {
        showTextDetails(lastSelectedTextPoint, lastSelectedTextIndex || 0, { suppressHighlight: true, preserveScroll: true });
    }
}

function reorderMetadataKey(key, action) {
    if (!key || !action || !Array.isArray(metadataSortOrder) || !metadataSortOrder.length) {
        return;
    }
    
    const currentIndex = metadataSortOrder.indexOf(key);
    if (currentIndex === -1) return;
    
    let targetIndex = currentIndex;
    switch (action) {
        case 'up':
            targetIndex = Math.max(0, currentIndex - 1);
            break;
        case 'down':
            targetIndex = Math.min(metadataSortOrder.length - 1, currentIndex + 1);
            break;
        case 'top':
            targetIndex = 0;
            break;
        case 'bottom':
            targetIndex = metadataSortOrder.length - 1;
            break;
        default:
            return;
    }
    
    if (targetIndex === currentIndex) return;
    
    const [entry] = metadataSortOrder.splice(currentIndex, 1);
    metadataSortOrder.splice(targetIndex, 0, entry);
    saveMetadataSortOrder();
    handleMetadataOrderChanged();
}

function initMetadataDragAndDrop(rootElement) {
    if (!rootElement) return;
    const list = rootElement.querySelector('.metadata-list');
    if (!list) return;

    // Idempotent cleanup via AbortController
    if (_metadataDragAbort) _metadataDragAbort.abort();
    _metadataDragAbort = new AbortController();
    const sig = { signal: _metadataDragAbort.signal };

    let dragged = null;   // the item being dragged
    let startX = 0, startY = 0;
    let hasMoved = false;
    let rafId = 0;
    // Remove any leftover indicator element from previous implementation
    const oldIndicator = list.querySelector('.metadata-drag-indicator');
    if (oldIndicator) oldIndicator.remove();

    // Find the drop target at pointer (x,y).
    // First: direct hit-test (pointer inside a card's rect).
    // Fallback: closest card by center distance.
    function dropTargetAt(x, y) {
        const items = [...list.querySelectorAll('.metadata-item-reorderable:not(.is-pointer-dragging)')];
        if (items.length === 0) return null;

        // Direct hit-test — pointer is inside a card
        for (const item of items) {
            const r = item.getBoundingClientRect();
            if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) {
                return item;
            }
        }

        // Fallback: closest by 2D center distance (for gaps between cards)
        let best = null, bestDist = Infinity;
        for (const item of items) {
            const r = item.getBoundingClientRect();
            const cx = r.left + r.width / 2;
            const cy = r.top + r.height / 2;
            const d = Math.hypot(x - cx, y - cy);
            if (d < bestDist) { bestDist = d; best = item; }
        }
        return best;
    }

    function clearHighlight() {
        list.querySelectorAll('.drag-target')
            .forEach(el => el.classList.remove('drag-target'));
    }

    list.addEventListener('pointerdown', (e) => {
        const handle = e.target.closest('.metadata-drag-handle');
        if (!handle) return;
        const item = handle.closest('.metadata-item-reorderable');
        if (!item) return;

        e.preventDefault();
        dragged = item;
        startX = e.clientX;
        startY = e.clientY;
        hasMoved = false;
        list.setPointerCapture(e.pointerId);
    }, sig);

    list.addEventListener('pointermove', (e) => {
        if (!dragged) return;

        // 5px dead zone (2D)
        if (!hasMoved && Math.hypot(e.clientX - startX, e.clientY - startY) < 5) return;

        if (!hasMoved) {
            hasMoved = true;
            dragged.classList.add('is-pointer-dragging');
            list.classList.add('is-reordering');
        }

        cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
            const target = dropTargetAt(e.clientX, e.clientY);
            if (!target || target === dragged) {
                clearHighlight();
                return;
            }
            clearHighlight();
            target.classList.add('drag-target');
        });
    }, sig);

    const endDrag = (e) => {
        if (!dragged) return;
        cancelAnimationFrame(rafId);
        const wasMoved = hasMoved;
        const draggedItem = dragged;

        draggedItem.classList.remove('is-pointer-dragging');
        list.classList.remove('is-reordering');
        clearHighlight();
        dragged = null;

        if (!wasMoved) return;

        // Determine drop target
        const target = dropTargetAt(e.clientX, e.clientY);
        if (!target || target === draggedItem) return;

        // FLIP: snapshot positions before DOM mutation
        const allChildren = [...list.querySelectorAll('.metadata-item-reorderable, .metadata-priority-divider')];
        const firstRects = new Map();
        for (const child of allChildren) {
            firstRects.set(child, child.getBoundingClientRect());
        }

        // Swap positions: place dragged where target is
        const allItems = [...list.querySelectorAll('.metadata-item-reorderable')];
        const dragIdx = allItems.indexOf(draggedItem);
        const targetIdx = allItems.indexOf(target);
        if (dragIdx < targetIdx) {
            list.insertBefore(draggedItem, target.nextSibling);
        } else {
            list.insertBefore(draggedItem, target);
        }

        // Reposition the priority divider
        const divider = list.querySelector('.metadata-priority-divider');
        if (divider) {
            const allItems = [...list.querySelectorAll('.metadata-item-reorderable')];
            if (allItems.length > MAX_METADATA_PREVIEW_ITEMS) {
                const ref = allItems[MAX_METADATA_PREVIEW_ITEMS];
                list.insertBefore(divider, ref);
            }
        }

        // FLIP: animate from old positions to new
        for (const child of allChildren) {
            const first = firstRects.get(child);
            if (!first) continue;
            const last = child.getBoundingClientRect();
            const dx = first.left - last.left;
            const dy = first.top - last.top;
            if (dx === 0 && dy === 0) continue;
            child.style.transform = `translate(${dx}px, ${dy}px)`;
            child.style.transition = 'none';
            requestAnimationFrame(() => {
                child.style.transition = 'transform 200ms ease';
                child.style.transform = '';
                child.addEventListener('transitionend', function cleanup() {
                    child.style.transition = '';
                    child.removeEventListener('transitionend', cleanup);
                }, { once: true });
            });
        }

        // Update sort order from DOM order
        const newOrder = [...list.querySelectorAll('.metadata-item-reorderable')]
            .map(el => el.dataset.metadataKey)
            .filter(Boolean);
        metadataSortOrder.length = 0;
        metadataSortOrder.push(...newOrder);

        saveMetadataSortOrder();
        refreshTextListMetadataPreview();
    };

    list.addEventListener('pointerup', endDrag, sig);
    list.addEventListener('pointercancel', endDrag, sig);
}

// Duplicate function removed - using global getClusterColor at top of file

// Back to list button
document.addEventListener('DOMContentLoaded', function() {
    const backBtn = document.getElementById('back-to-list-btn');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            document.getElementById('text-list').style.display = 'block';
            document.getElementById('selected-text-view').style.display = 'none';

            // Clear individual point highlight but preserve search results
            if (window.mainVisualization) {
                window.mainVisualization.clearIndividualHighlight();
            }
            updateExportButtonVisibility();
        });
    }
});

// ENHANCED search functionality with fast search integration
let searchTimeout = null;

function initEnhancedSearchFunctionality() {
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-input');
    const clearBtn = document.getElementById('clear-search');
    
    // Initialize the global search interface
    if (!window.globalSearchInterface) {
        window.globalSearchInterface = new SearchInterface();
    }
    
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }
    
    if (searchInput) {
        // Only trigger search on Enter key for semantic search
        // Fast search will be handled by SearchInterface for real-time search
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                performSearch();
            }
        });
        
        // Auto-clear search results when input becomes empty
        searchInput.addEventListener('keyup', (e) => {
            const query = e.target.value.trim();
            if (query === '') {
                // User cleared the input - automatically clear search results
                clearSearch();
            }
        });
        
        // Also handle paste/cut events that might clear the input
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            if (query === '') {
                // Input was cleared via paste/cut - automatically clear search results
                clearSearch();
            }
        });
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', clearSearch);
    }
    
    // Initialize delete data & cache button
    const deleteDataBtn = document.getElementById('delete-data-cache-btn');
    if (deleteDataBtn) {
        deleteDataBtn.addEventListener('click', handleDeleteAllData);
    }
    
}

function debouncedSearch(delay = 300) {
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }
    
    searchTimeout = setTimeout(() => {
        performSearch();
        searchTimeout = null;
    }, delay);
}

function clearSearch() {
    const searchBtn = document.getElementById('search-btn');
    if (searchBtn && searchBtn.disabled) {
        return;
    }

    // Don't clear the search input - user may want to modify and re-search
    // const searchInput = document.getElementById('search-input');
    // if (searchInput) {
    //     searchInput.value = '';
    // }

    const searchResults = document.getElementById('search-results');
    if (searchResults) {
        searchResults.style.display = 'none';
    }

    if (window.mainVisualization) {
        window.mainVisualization.disableSearchHighlightMode();
        window.mainVisualization.clearSearchHighlight();

        // Clear lasso selection and exit lasso mode
        if (window.mainVisualization.lassoMode) {
            window.mainVisualization.toggleLassoMode();
            const lassoBtn = document.getElementById('lasso-select-btn');
            if (lassoBtn) lassoBtn.classList.remove('active');
        } else if (window.mainVisualization.lassoSelectedIndices) {
            window.mainVisualization.clearLassoSelection();
        }
    }

    unlockTextList('search cleared');
    if (currentVisualizationData && currentVisualizationData.points) {
        updateTextList(currentVisualizationData.points, { force: true });
    }
    
    // Hide AI answer card if visible
    const answerCard = document.getElementById('rag-answer-card');
    const answerText = document.getElementById('rag-answer-text');
    const answerMeta = document.getElementById('rag-answer-meta');
    if (answerCard) {
        answerCard.style.display = 'none';
    }
    if (answerText) answerText.textContent = '';
    if (answerMeta) answerMeta.textContent = '';
    
    // Clear global search state
    window.currentSearchResults = null;
    window.currentVisualizationSearchResults = null;

    updateExportButtonVisibility();

    // Clear fast search results if using fast search interface
    if (window.globalSearchInterface) {
        window.globalSearchInterface.clearSearchResults();
    }

    // Update RAG scope text
    updateRAGScopeTextNow();

    showToast('Search cleared', 'info');
}

async function handleDeleteAllData() {
    // Handle delete all data and cache button click
    
    // Confirmation dialog
    if (!confirm('⚠️ DELETE ALL DATA & CACHE\n\nThis will permanently delete:\n• All uploaded documents\n• All embeddings and vector data\n• All visualization cache\n• All search indices\n\nThis action cannot be undone!\n\nAre you sure you want to continue?')) {
        return;
    }
    
    // Second confirmation for safety
    if (!confirm(' FINAL CONFIRMATION\n\nThis will completely reset the system to empty state.\n\nClick OK to DELETE EVERYTHING or Cancel to abort.')) {
        return;
    }
    
    const deleteBtn = document.getElementById('delete-data-cache-btn');
    const originalText = deleteBtn.innerHTML;
    
    try {
        // Show loading state
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
        
        const response = await fetch('/api/delete_all_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            // Clear UI state
            currentVisualizationData = null;
            window.currentVisualizationData = null;
            window.currentSearchResults = null;
            window.currentVisualizationSearchResults = null;
            
            // Clear visualization
            if (window.mainVisualization) {
                window.mainVisualization.clearHighlight();
                window.mainVisualization.data = [];
            }
            
            // Clear text list
            const textList = document.getElementById('text-list');
            if (textList) {
                textList.innerHTML = '<div class="no-data-message">No data available. Upload and process a file to get started.</div>';
            }
            
            // Show success message and reload
            showToast('All data deleted. Reloading page...', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 1500);

        } else {
            throw new Error(result.error || 'Failed to delete data');
        }
        
    } catch (error) {
        console.error('❌ Delete all data failed:', error);
        showToast(`Failed to delete data: ${error.message}`, 'error');
        
    } finally {
        // Restore button state
        deleteBtn.disabled = false;
        deleteBtn.innerHTML = originalText;
    }
}

function performSearch() {
    const searchInput = document.getElementById('search-input');
    const searchTypeSelect = document.getElementById('search-type');
    const resultCountSelect = document.getElementById('result-count');
    const searchBtn = document.getElementById('search-btn');
    
    if (!searchInput) {
        console.error('Search input not found');
        showToast('Search interface not properly loaded', 'error');
        return;
    }
    
    // Check for ongoing search with more specific state tracking
    if (window.searchInProgress) {
        return;
    }
    
    const searchType = searchTypeSelect ? searchTypeSelect.value : 'fast';
    let resultCount = 10;
    if (resultCountSelect && !resultCountSelect.disabled && resultCountSelect.style.display !== 'none') {
        const parsed = parseInt(resultCountSelect.value, 10);
        if (!Number.isNaN(parsed) && parsed > 0) {
            resultCount = parsed;
        }
    }
    if (searchType === 'semantic') {
        const semanticAllowed = [5, 10, 20, 50];
        if (!semanticAllowed.includes(resultCount)) {
            resultCount = 10;
        }
    }
    let ragSearchMode = 'semantic';
    let ragResultCount = Number.isFinite(resultCount) && resultCount > 0 ? resultCount : 5;
    
    if (searchType === 'rag') {
        try {
            const configJson = localStorage.getItem('vectoria_config');
            if (configJson) {
                const config = JSON.parse(configJson);
                const searchConfig = config.search || {};
                const userSelectedCountValid = Number.isFinite(resultCount) && resultCount > 0;
                if ((!userSelectedCountValid || !resultCountSelect) && searchConfig.num_results !== undefined && searchConfig.num_results !== null) {
                    const parsedNum = parseInt(searchConfig.num_results, 10);
                    if (!Number.isNaN(parsedNum) && parsedNum > 0) {
                        ragResultCount = parsedNum;
                    }
                }
            }
        } catch (error) {
            console.warn('Unable to load quick settings config for RAG search:', error);
        }
    }
    
    const query = searchInput.value.trim();
    
    if (!query) {
        showToast('Please enter a search query', 'warning');
        return;
    }
    
    // Route to appropriate search method
    if (searchType === 'fast') {
        // Use fast search interface
        if (window.globalSearchInterface && window.fastSearchReady) {
            window.searchInProgress = true;
            window.globalSearchInterface.performFastSearch(query);
            // Fast search completes quickly, reset state after short delay
            setTimeout(() => {
                window.searchInProgress = false;
            }, 500);
        } else {
            showToast('Fast search is still loading, please wait...', 'warning');
        }
        return;
    }
    
    // Handle semantic/RAG search (existing functionality)
    window.searchInProgress = true;
    lockSearchControls(true);
    
    // Show loading state
    const originalText = searchBtn ? searchBtn.innerHTML : '<i class="fas fa-search"></i> Search';
    if (searchBtn) {
        searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
    }
    
    // After 60s, show "still generating" hint but keep spinner going
    const searchTimeout = setTimeout(() => {
        if (window.searchInProgress) {
            console.warn(' Search timeout - still generating');
            if (searchBtn) {
                searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Still searching...';
            }
            const answerText = document.getElementById('rag-answer-text');
            if (answerText && answerText.innerHTML.includes('Generating answer')) {
                answerText.innerHTML = '<div class="loading-spinner"></div> Still generating...';
            }
            showToast('Still generating — this may take a while for large contexts.', 'info');
        }
    }, 60000); // 60 seconds (RAG can take 30-40s)
    
    // Choose endpoint based on search type
    const endpoint = searchType === 'rag' ? '/query' : '/search';
    
    // Collect metadata filters if available
    const metadataFilters = window.collectMetadataFiltersForSearch ? window.collectMetadataFiltersForSearch() : {};
    const hasActiveFilters = Object.keys(metadataFilters).length > 0;
    
    if (hasActiveFilters) {
    }
    
    // Collect per-search metadata inclusion fields for RAG and semantic search
    let includeMetadata = false;
    let metadataFields = undefined;
    let metadataFieldMode = undefined;
    if (searchType === 'rag' || searchType === 'semantic') {
        const includeBox = document.getElementById('rag-include-metadata');
        const fieldSelect = document.getElementById('rag-metadata-fields');
        includeMetadata = !!(includeBox && includeBox.checked);
        if (includeMetadata && fieldSelect) {
            const selected = Array.from(fieldSelect.selectedOptions).map(o => o.value).filter(Boolean);
            if (selected.length > 0) {
                metadataFields = selected;
                metadataFieldMode = 'custom';
            } else if (Array.isArray(detectedMetadataFields) && detectedMetadataFields.length > 0) {
                metadataFields = detectedMetadataFields.map(field => field.name);
                metadataFieldMode = 'all';
            } else {
                metadataFields = [];
                metadataFieldMode = 'custom';
            }
        }
        
        if (includeMetadata) {
            const logFields = Array.isArray(metadataFields) && metadataFields.length > 0 ? metadataFields : 'all available';
        }
    }

    const basePayload = {
        metadata_filters: metadataFilters,
        include_metadata: includeMetadata,
        metadata_fields: metadataFields,
        metadata_field_mode: metadataFieldMode
    };

    const requestData = searchType === 'rag' ?
        { question: query, num_results: ragResultCount, search_type: ragSearchMode, ...basePayload } :
        { query: query, search_type: searchType, k: resultCount, ...basePayload };

    // Perform search
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Search failed');
            });
        }
        return response.json();
    })
    .then(data => {
        if (searchType === 'rag') {
            // Check if the query was cancelled or stopped - skip result display but keep AI Answer card
            const wasCancelled = data.metadata?.cancelled || data.metadata?.error;
            const wasStopped = data.metadata?.wasStopped;

            if (wasCancelled || (wasStopped && (!data.answer || !data.answer.trim()))) {
                // Don't display results, don't show toast, don't highlight
                // The AI Answer card with "Generation stopped" is already shown by browser-integration.js
                return;
            }

            displayRAGResults(data);

            // For RAG, show source documents in text list and highlight in visualization
            if (data.sources) {
                showSearchResultsInTextList(data.sources, 'rag', query);
                highlightSearchResultsInVisualization(data.sources, query);
            }

            const filterNote = hasActiveFilters ? ` (filtered data)` : '';
            showToast(`RAG response generated with ${data.sources ? data.sources.length : 0} sources${filterNote}`, 'success');
        } else {
            displaySearchResults(data);
            
            // Show search results in text list and highlight in visualization
            if (data.results) {
                showSearchResultsInTextList(data.results, searchType, query);
                highlightSearchResultsInVisualization(data.results, query);
            }
            
            const filterNote = hasActiveFilters ? ` (in filtered data)` : '';
            showToast(`Found ${data.results.length} results${filterNote}`, 'success');
        }
        // Ensure UI reset in success path
        resetSearchUI();
    })
    .catch(error => {
        console.error('Search error:', error);
        
        // More specific error messages
        let errorMessage = 'Search failed';
        if (error.message.includes('Network')) {
            errorMessage = 'Network error - please check your connection';
        } else if (error.message.includes('timeout')) {
            errorMessage = 'Search timed out - please try again';
        } else if (error.message.includes('No processed data')) {
            errorMessage = 'Please upload and process a file first';
        } else {
            errorMessage = error.message;
        }
        
        showToast(errorMessage, 'error');
        // Ensure UI reset in error path
        resetSearchUI();
    })
    .finally(() => {
        // Clear the safety timeout
        clearTimeout(searchTimeout);
        
        // Unlock search controls and restore button (final safeguard)
        try {
            window.searchInProgress = false;
            lockSearchControls(false);
            if (searchBtn) {
                searchBtn.innerHTML = originalText;
                searchBtn.disabled = false;
                searchBtn.classList.remove('loading');
            }
        } catch (cleanupError) {
            console.error('Error during search UI cleanup:', cleanupError);
            // Force reset as fallback
            resetSearchUI();
        }
    });
}

function resetSearchUI() {
    const searchBtn = document.getElementById('search-btn');
    window.searchInProgress = false;
    lockSearchControls(false);
    if (searchBtn) {
        // Force restore to default label in case originalText was corrupted
        searchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
        searchBtn.disabled = false;
        searchBtn.style.opacity = '1';
        searchBtn.style.cursor = 'pointer';
    }
}

// New function to lock/unlock search controls
function lockSearchControls(locked) {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const searchTypeSelect = document.getElementById('search-type');
    const resultCountSelect = document.getElementById('result-count');
    const clearBtn = document.getElementById('clear-search');
    
    // Lock/unlock all search-related controls
    [searchInput, searchBtn, searchTypeSelect, resultCountSelect].forEach(element => {
        if (element) {
            element.disabled = locked;
            if (locked) {
                element.style.opacity = '0.6';
                element.style.cursor = 'not-allowed';
            } else {
                element.style.opacity = '1';
                element.style.cursor = '';
            }
        }
    });
    
    // Clear button should remain enabled during search
    if (clearBtn) {
        clearBtn.disabled = false;
        clearBtn.style.opacity = '1';
        clearBtn.style.cursor = 'pointer';
    }
}

function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const resultsCount = document.getElementById('results-count');
    const resultsArray = Array.isArray(data)
        ? data
        : Array.isArray(data?.results)
            ? data.results
            : [];

    // Detect if this is a lasso selection
    const query = data?.query || '';
    const isLasso = query.toLowerCase().includes('lasso selection');

    // Update results count for compact view
    if (resultsCount) {
        let countText = isLasso
            ? `${resultsArray.length} items selected`
            : `${resultsArray.length} results found`;

        if (!isLasso && !Array.isArray(data) && data?.metadata_filters_applied && Array.isArray(data.active_filters) && data.active_filters.length > 0) {
            countText += ` (${data.active_filters.length} filter${data.active_filters.length > 1 ? 's' : ''} applied)`;
        }

        countText += isLasso ? ' • Lasso' : ' • Search';
        resultsCount.textContent = countText;
    }

    // Show results section
    if (resultsDiv) {
        resultsDiv.style.display = 'block';
    }
}

function highlightQuery(text, query) {
    if (!query || query.length < 2) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
}

// Removed old highlightSearchResults function - now using highlightSearchResultsInVisualization

// Download functionality
function initializeDownloadButtons() {
    // This would be implemented based on the original Vectoria download functionality
}

// Toast notification function
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add to document
    document.body.appendChild(toast);

    // Show and hide
    setTimeout(() => toast.classList.add('show'), 100);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// No global overlay loader (use button spinner only)

function filterTextListBySearch(searchResults) {
    if (!currentVisualizationData || !currentVisualizationData.points) {
        return;
    }
    
    // Get all text items
    const textItems = document.querySelectorAll('.text-item');
    
    // Clear previous search styling
    textItems.forEach(item => {
        item.classList.remove('search-match');
        const scoreSpan = item.querySelector('.search-score');
        if (scoreSpan) {
            scoreSpan.remove();
        }
    });
    
    // Add search result indicators to matching items
    searchResults.slice(0, Math.min(searchResults.length, textItems.length)).forEach((result, index) => {
        if (textItems[index]) {
            textItems[index].classList.add('search-match');
            
            // Add search score indicator
            const header = textItems[index].querySelector('.text-item-header');
            if (header) {
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'search-score';
                scoreSpan.textContent = `${result.score.toFixed(3)}`;
                header.appendChild(scoreSpan);
            }
        }
    });
}

function displayRAGResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const resultsCount = document.getElementById('results-count');
    
    // Update results count (no answer here)
    if (resultsCount) {
        resultsCount.innerHTML = `<div>${data.sources ? data.sources.length : 0} results found • RAG</div>`;
    }

    // Populate dedicated AI answer card
    const answerCard = document.getElementById('rag-answer-card');
    const answerText = document.getElementById('rag-answer-text');
    const answerMeta = document.getElementById('rag-answer-meta');
    if (answerCard && answerText && answerMeta) {
        // Use textContent to avoid HTML injection; CSS preserves newlines
        answerText.textContent = data.answer || 'No answer generated';
        const used = Array.isArray(data.metadata_fields_used) ? data.metadata_fields_used : [];
        if (data.include_metadata && used.length) {
            answerMeta.textContent = `Context metadata: ${used.join(', ')}`;
        } else {
            answerMeta.textContent = '';
        }
        answerCard.style.display = 'block';
        updateExportButtonVisibility();
    }

    // Show results section
    if (resultsDiv) {
        resultsDiv.style.display = 'block';
    }
}

function showSearchResultsInTextList(results, searchType, query, force = false) {
    const textList = document.getElementById('text-list');
    if (!textList) return;

    const isLasso = query && query.toLowerCase().includes('lasso selection');
    if (!force && window.__textListLock === 'lasso' && !isLasso) {
        return;
    }

    if (isLasso) {
        window.__textListLock = 'lasso';
    } else if (force || window.__textListLock === 'lasso') {
        window.__textListLock = null;
    }

    updateTextContentCount(results ? results.length : 0);
    
    // Store current search results globally for consistency
    window.currentSearchResults = {
        results: results,
        searchType: searchType,
        query: query
    };

    updateExportButtonVisibility();

    // Clear current text list
    textList.innerHTML = '';
    window.__currentTextListPoints = Array.isArray(results) ? results : [];

    const activeFilters = window.activeMetadataFilters || {};
    const hasActiveFilters = Object.keys(activeFilters).length > 0;
    const filterCount = Object.keys(activeFilters).length;
    
    // Add search results header
    const header = document.createElement('div');
    header.className = 'search-results-header';
    const isRag = searchType === 'rag';
    const headerGradient = isLasso ? 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)' : 'linear-gradient(135deg, #ffd700 0%, #ff8f00 100%)';
    const textColor = '#000';
    const shadowColor = isLasso ? 'rgba(52, 152, 219, 0.3)' : 'rgba(255, 215, 0, 0.3)';

    const filterBadge = hasActiveFilters ? `<span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 8px;"> ${filterCount} filter${filterCount > 1 ? 's' : ''}</span>` : '';
    const filterNote = hasActiveFilters ? ` (filtered data)` : '';

    const headerTitle = isLasso ? 'Lasso Selection' : (isRag ? 'RAG Search Results' : 'Search Results');
    const headerSubtext = isLasso ? `${results.length} items selected` : `${results.length} matches for "${escapeHtml(query)}"${filterNote}`;

    header.innerHTML = `
        <div style="padding: 12px; background: ${headerGradient}; color: ${textColor}; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 2px 8px ${shadowColor};">
            <h3 style="margin: 0 0 8px 0; font-size: 1.1em; color: ${textColor};">
                ${headerTitle}${filterBadge}
            </h3>
            <div style="font-size: 0.9em; opacity: 0.9; color: ${textColor};">
                ${headerSubtext}
            </div>
        </div>
    `;
    textList.appendChild(header);
    
    // Show search results with enhanced styling
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'text-item search-match priority-result';
        item.dataset.index = index;
        item.dataset.searchResult = 'true';

        const matchingPoint = resolvePointForResult(result) || null;
        if (matchingPoint) {
            result.cluster = matchingPoint.cluster;
            result.cluster_name = matchingPoint.cluster_name || getClusterName(matchingPoint.cluster);
            result.cluster_color = matchingPoint.cluster_color || ensureConsistentColor(matchingPoint.cluster, matchingPoint.cluster_color, matchingPoint.cluster_name);
            result.index = matchingPoint.index;
            result.doc_id = matchingPoint.doc_id;
            result.chunk_id = matchingPoint.chunk_id;
            if (!result.coordinates) {
                result.coordinates = [matchingPoint.x, matchingPoint.y];
            }
        }

        const cluster = matchingPoint ? matchingPoint.cluster : (result.cluster !== undefined ? result.cluster : 0);
        const clusterColor = (matchingPoint?.cluster_color) || result.cluster_color || ensureConsistentColor(cluster, matchingPoint?.cluster_color || result.cluster_color, matchingPoint?.cluster_name || result.cluster_name);
        const clusterName = cluster === -1 ? 'Outlier' : (matchingPoint?.cluster_name || result.cluster_name || getClusterName(cluster));

        // Format the result text with consistent handling
        let displayText = '';
        let displayScore = '';
        let itemTitle = '';

        if (isLasso) {
            // For lasso selections, show original item number and cluster
            displayText = result.text || 'No text available';
            const originalIndex = result.index !== undefined ? result.index + 1 : index + 1;
            itemTitle = `Item ${originalIndex} - ${escapeHtml(clusterName)}`;
            displayScore = ''; // No score for lasso selections
        } else if (searchType === 'rag') {
            displayText = result.content || result.text || 'No content available';
            displayScore = `<span class="rag-source">Source ${index + 1}</span>`;
            itemTitle = `Result ${index + 1} - ${escapeHtml(clusterName)}`;
        } else {
            displayText = result.text || 'No text available';
            const score = result.score || 0;
            displayScore = `<span class="search-score-value">${score.toFixed(3)}</span>`;
            itemTitle = `Result ${index + 1} - ${escapeHtml(clusterName)}`;
        }

        // Enhanced item styling with priority indicator
        const priorityBadge = isLasso ? '' : '<span class="priority-badge">TOP</span>';

        item.innerHTML = `
            <div class="text-item-header">
                <span class="cluster-indicator" style="background-color: ${clusterColor}; box-shadow: 0 0 0 2px rgba(255,255,255,0.8);"></span>
                <span class="item-title">${itemTitle}</span>
                <div class="result-badges">
                    ${displayScore}
                    ${priorityBadge}
                </div>
            </div>
            <div class="text-preview enhanced-preview">
                ${isLasso ? escapeHtml(displayText.substring(0, 200)) : highlightQuery(escapeHtml(displayText.substring(0, 200)), query)}${displayText.length > 200 ? '...' : ''}
            </div>
        `;
        
        // Enhanced click handler with better highlighting
        item.addEventListener('click', () => {
            // Find the actual visualization index for this search result
            let visualizationIndex = index; // fallback to search result index
            if (currentVisualizationData && currentVisualizationData.points) {
                // Try to find matching point by chunk_id or doc_id
                let matchingPointIndex = -1;
                
                // Try exact chunk_id match first
                if (result.chunk_id) {
                    matchingPointIndex = currentVisualizationData.points.findIndex(point => 
                        point.chunk_id === result.chunk_id
                    );
                }
                
                // Try doc_id match if chunk_id failed
                if (matchingPointIndex === -1 && result.doc_id) {
                    matchingPointIndex = currentVisualizationData.points.findIndex(point => 
                        point.doc_id === result.doc_id
                    );
                }
                
                // Use the found index if valid
                if (matchingPointIndex !== -1) {
                    visualizationIndex = matchingPointIndex;
                }
            }
            
            // Get the actual point data from visualization dataset
            let pointData;
            if (currentVisualizationData && currentVisualizationData.points && 
                visualizationIndex < currentVisualizationData.points.length) {
                // Use actual visualization point data
                pointData = {
                    ...currentVisualizationData.points[visualizationIndex],
                    // Add search-specific metadata
                    search_score: result.score || 0,
                    search_type: searchType,
                    search_rank: index + 1,
                    search_query: query || '',
                    is_search_result: true
                };
            } else {
                // Fallback to search result data if visualization data not available
                pointData = {
                    index: visualizationIndex,
                    cluster: cluster,
                    cluster_color: clusterColor,
                    cluster_probability: result.cluster_probability || 1.0,
                    coordinates: result.coordinates || [0, 0],
                    text: displayText,
                    doc_id: result.doc_id,
                    chunk_id: result.chunk_id,
                    score: result.score || 0,
                    search_type: searchType,
                    metadata: result.metadata || {},
                    search_rank: index + 1,
                    search_query: query || '',
                    is_search_result: true,
                    ...result
                };
            }
            
            showTextDetails(pointData, visualizationIndex);
            
            // Enhanced selection highlighting
            document.querySelectorAll('.text-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            
            // Force visualization highlight for this specific result
            if (window.mainVisualization && result.coordinates) {
                window.mainVisualization.highlightPoint(visualizationIndex, { focus: true, revealTooltip: true });
            }
        });
        
        textList.appendChild(item);
    });
    
    // Add footer with search info
    const highlightEnabled = document.getElementById('highlight-results')?.checked !== false;
    const highlightOnText = 'Search results are displayed at the top and highlighted in the visualization';
    const highlightOffText = 'Search results are displayed at the top';
    const footerText = highlightEnabled ? highlightOnText : highlightOffText;
    const footer = document.createElement('div');
    footer.className = 'search-results-footer';
    footer.innerHTML = `
        <div style="padding: 8px 12px; background: rgba(0,0,0,0.05); border-radius: 6px; margin-top: 16px; text-align: center; font-size: 0.85em; color: #666;">
            <i class="fas fa-info-circle"></i> 
            <span class="search-results-footer-message" data-highlight-on="${highlightOnText}" data-highlight-off="${highlightOffText}">${footerText}</span>
        </div>
    `;
    textList.appendChild(footer);
}

function highlightSearchResultsInVisualization(results, query) {
    if (!window.mainVisualization || !results || results.length === 0) {
        console.warn('Cannot highlight search results: missing visualization or results');
        return;
    }

    const highlightEnabled = document.getElementById('highlight-results')?.checked !== false;
    if (!highlightEnabled) {
        window.mainVisualization.disableSearchHighlightMode();
        window.mainVisualization.clearSearchHighlight();
        return;
    }

    const isLasso = query && query.toLowerCase().includes('lasso');
    if (window.__textListLock === 'lasso' && !isLasso) {
        return;
    }
    
    // Enhanced coordinate matching with multiple fallback strategies
    const searchResultsData = results.map((result, index) => {
        let coordinates = [0, 0];
        let coordinateSource = 'default';
        let matchingPoint = resolvePointForResult(result);

        if (matchingPoint) {
            coordinates = [matchingPoint.x, matchingPoint.y];
            coordinateSource = 'matched';
            result.cluster = matchingPoint.cluster;
            result.cluster_name = matchingPoint.cluster_name || getClusterName(matchingPoint.cluster);
            result.cluster_color = matchingPoint.cluster_color || ensureConsistentColor(matchingPoint.cluster, matchingPoint.cluster_color, matchingPoint.cluster_name);
            result.index = matchingPoint.index;
            result.doc_id = matchingPoint.doc_id;
            result.chunk_id = matchingPoint.chunk_id;
            if (!result.coordinates) {
                result.coordinates = coordinates;
            }
        } else if (result.coordinates && Array.isArray(result.coordinates) && result.coordinates.length >= 2) {
            coordinates = [parseFloat(result.coordinates[0]), parseFloat(result.coordinates[1])];
            coordinateSource = 'provided';
        }

        if (matchingPoint) {
            result.cluster = matchingPoint.cluster;
            result.cluster_name = matchingPoint.cluster_name || getClusterName(matchingPoint.cluster);
            result.cluster_color = matchingPoint.cluster_color || ensureConsistentColor(matchingPoint.cluster, matchingPoint.cluster_color, matchingPoint.cluster_name);
            result.index = matchingPoint.index;
            result.doc_id = matchingPoint.doc_id;
            result.chunk_id = matchingPoint.chunk_id;
            if (!result.coordinates) {
                result.coordinates = coordinates;
            }
        }

        return {
            chunk_id: result.chunk_id,
            doc_id: result.doc_id,
            score: result.score || 0,
            rank: index,
            text: result.text || result.content || '',
            coordinates: coordinates,
            coordinateSource: coordinateSource,
            cluster: matchingPoint ? matchingPoint.cluster : (result.cluster !== undefined ? result.cluster : 0),
            cluster_color: matchingPoint ? (matchingPoint.cluster_color || ensureConsistentColor(matchingPoint.cluster, matchingPoint.cluster_color, matchingPoint.cluster_name)) : (result.cluster_color || ensureConsistentColor(result.cluster || 0, null, result.cluster_name)),
            cluster_name: matchingPoint ? (matchingPoint.cluster_name || getClusterName(matchingPoint.cluster)) : (result.cluster_name || getClusterName(result.cluster || 0)),
            isValid: coordinateSource !== 'default' // Flag for valid coordinate matches
        };
    });
    
    // Log matching success rate
    const validMatches = searchResultsData.filter(r => r.isValid).length;
    if (validMatches === 0) {
        console.warn('No search results could be matched to visualization coordinates');
        // Still show the search in progress, but without coordinate highlighting
        window.mainVisualization.enableSearchHighlightMode();
        return;
    }
    
    // Store search results globally for persistence
    window.currentVisualizationSearchResults = searchResultsData;
    
    // Enable enhanced search highlighting mode
    window.mainVisualization.enableSearchHighlightMode();
    
    // Apply highlighting with enhanced error handling
    try {
        window.mainVisualization.highlightSearchResults(searchResultsData);
        
        // Force re-render to ensure highlighting is applied
        setTimeout(() => {
            if (window.mainVisualization && window.mainVisualization.needsRedraw !== false) {
                window.mainVisualization.requestRender();
            }
        }, 100);
        
    } catch (error) {
        console.error('Error applying search result highlighting:', error);
        // Fallback to simple highlighting
        window.mainVisualization.clearSearchHighlight();
    }
}

// Helper function for text similarity calculation
function calculateTextSimilarity(text1, text2) {
    if (!text1 || !text2) return 0;
    
    // Simple similarity based on common words
    const words1 = text1.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const words2 = text2.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    
    if (words1.length === 0 || words2.length === 0) return 0;
    
    const commonWords = words1.filter(word => words2.includes(word));
    const similarity = commonWords.length / Math.min(words1.length, words2.length);
    
    return similarity;
}

// Global access to functions
window.activateTab = activateTab;
window.showTextDetails = showTextDetails;
window.updateTextList = updateTextList;
window.showSearchResultsInTextList = showSearchResultsInTextList;
window.getClusterColor = getClusterColor;
window.getClusterName = getClusterName;
window.getClusterAccent = getClusterAccent;
window.getClusterNames = () => Object.fromEntries(customClusterNames);
window.setClusterNames = (names) => {
    customClusterNames.clear();
    if (names && typeof names === 'object') {
        for (const [id, name] of Object.entries(names)) {
            customClusterNames.set(parseInt(id, 10), name);
        }
    }
    saveCustomClusterNames();
    refreshClusterDisplays();
};
window.performSearch = performSearch;
window.unlockTextList = unlockTextList;

// Enhanced visualization initialization with WebGL support
function loadVisualizationData() {
    fetch('/api/visualization_data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            currentVisualizationData = data;
            window.currentVisualizationData = data; // Make available globally for fast search
            unlockTextList('data refreshed');

            colorCache.clear();
            
            // Log cluster consistency info
            if (data.cluster_info && data.cluster_info.color_consistency) {
            }
            
            // Ensure color consistency from server data
            if (data.cluster_colors) {
                colorCache.clear();
                Object.entries(data.cluster_colors).forEach(([clusterId, color]) => {
                    const numericId = parseInt(clusterId, 10);
                    const topicName = (data.cluster_names && data.cluster_names[clusterId]) || null;
                    colorCache.set(numericId, color);
                    colorManager.registerColor(numericId, topicName, color);
                });
            }
            
            // Initialize enhanced visualization with WebGL support
            initializeEnhancedVisualization(data);
            
            // Update text list with consistent colors
            updateTextList(data.points);
            
            // Load metadata schema from backend if available
            loadMetadataFromProcessedData().catch(console.error);
            
        })
        .catch(error => {
            console.error('Error loading visualization data:', error);
            
            const vizContainer = document.getElementById('main-viz-container');
            const loader = document.getElementById('loader');
            const vizError = document.getElementById('visualization-error');
            
            if (loader) loader.style.display = 'none';
            
            if (vizError) {
                vizError.textContent = `Failed to load visualization: ${error.message}`;
                vizError.style.display = 'block';
            }
        });
}

function initializeEnhancedVisualization(data) {
    // Check if canvas element exists
    const canvas = document.getElementById('viz-canvas');
    if (!canvas) {
        console.warn('Canvas element not found, deferring visualization initialization...');
        // Store data for later initialization when tab becomes visible
        window._pendingVisualizationData = data;
        return;
    }

    try {
        // Initialize WebGL-enhanced canvas visualization
        if (window.mainVisualization) {
            window.mainVisualization.destroy();
        }

        // Try WebGL first, fall back to Canvas 2D
        const useWebGL = window.EnhancedCanvasVisualization && window.WebGLRenderer;

        if (useWebGL) {
            window.mainVisualization = new window.EnhancedCanvasVisualization(
                'viz-canvas',
                'tooltip',
                {
                    useWebGL: true,
                    maxPoints: 2000000,
                    enableOutlines: true
                }
            );
        } else {
            window.mainVisualization = new CanvasVisualization('viz-canvas', 'tooltip');
        }

        // Check if visualization was created successfully
        if (!window.mainVisualization || !window.mainVisualization.canvas) {
            console.warn('Visualization canvas not ready, deferring...');
            window._pendingVisualizationData = data;
            return;
        }

        // Load data with enhanced color consistency
        window.mainVisualization.loadData(data.points);

        // Hide loader and error messages
        const loader = document.getElementById('loader');
        const vizError = document.getElementById('visualization-error');
        const noDataMessage = document.getElementById('no-data-message');

        if (loader) loader.style.display = 'none';
        if (vizError) vizError.style.display = 'none';
        if (noDataMessage) noDataMessage.style.display = 'none';

        // Clear pending data since we've initialized
        window._pendingVisualizationData = null;

        const actualMode = (window.mainVisualization && window.mainVisualization.useWebGL) ? 'WebGL' : 'Canvas 2D';
    } catch (error) {
        console.error('Error initializing visualization:', error);

        // Fallback to basic Canvas 2D if enhanced fails
        try {
            window.mainVisualization = new CanvasVisualization('viz-canvas', 'tooltip');
            if (window.mainVisualization && window.mainVisualization.canvas) {
                window.mainVisualization.loadData(data.points);
                window._pendingVisualizationData = null;
            } else {
                window._pendingVisualizationData = data;
            }
        } catch (fallbackError) {
            console.error('Fallback visualization also failed:', fallbackError);
            window._pendingVisualizationData = data;
        }
    }
}

// =================================================================
// ENHANCED CSS INTEGRATION - Connect new styles with existing logic
// =================================================================

/**
 * Initialize all CSS enhancements when the DOM is ready
 */
function initializeEnhancements() {
    // Initialize fast search integration
    initializeFastSearchIntegration();
    
    // Initialize tooltip enhancements
    initializeTooltipEnhancements();
    
    // Initialize responsive enhancements
    initializeResponsiveEnhancements();
    
}

/**
 * Fast Search Integration
 */
function initializeFastSearchIntegration() {
    // Connect fast search with visualization highlighting
    if (window.globalSearchInterface && window.mainVisualization) {
        const originalHighlightFn = window.globalSearchInterface.highlightFastSearchResults;
        
        window.globalSearchInterface.highlightFastSearchResults = function(results) {
            // Call original function
            if (originalHighlightFn) {
                originalHighlightFn.call(this, results);
            }
            
            // Add CSS class enhancements
            const textItems = document.querySelectorAll('.text-item.search-match');
            textItems.forEach((item, index) => {
                // Add enhanced CSS classes for fast search results
                item.classList.add('fast-match');
                
                // Add performance indicators
                if (index < 5) { // Top 5 results get priority styling
                    item.classList.add('priority-result');
                }
            });
            
            // Update results counter with new styling
            const counter = document.querySelector('.search-results-counter');
            if (counter) {
                counter.classList.add('enhanced-counter');
            }
        };
    }
}

/**
 * Tooltip Enhancement Integration
 */
function initializeTooltipEnhancements() {
    if (window.mainVisualization && window.mainVisualization.showTooltip) {
        const originalShowTooltip = window.mainVisualization.showTooltip;
        
        window.mainVisualization.showTooltip = function(point, clientX, clientY) {
            // Call original tooltip logic
            originalShowTooltip.call(this, point, clientX, clientY);
            
            // Add CSS custom properties for dynamic positioning
            const tooltip = document.getElementById('tooltip');
            if (tooltip) {
                const rect = tooltip.getBoundingClientRect();
                const canvasRect = this.canvas.getBoundingClientRect();
                
                // Calculate optimal position using CSS variables
                let optimalX = clientX - canvasRect.left;
                let optimalY = clientY - canvasRect.top;
                
                // Dynamic repositioning with CSS custom properties
                tooltip.style.setProperty('--tooltip-offset-x', `${optimalX}px`);
                tooltip.style.setProperty('--tooltip-offset-y', `${optimalY}px`);
                
                // Add positioning classes based on viewport constraints
                tooltip.classList.remove('tooltip-top', 'tooltip-bottom', 'tooltip-left', 'tooltip-right');
                
                if (optimalX + rect.width > canvasRect.width) {
                    tooltip.classList.add('tooltip-left');
                }
                if (optimalY - rect.height < 0) {
                    tooltip.classList.add('tooltip-bottom');
                }
            }
        };
    }
}



/**
 * Responsive Enhancement Integration
 */
function initializeResponsiveEnhancements() {
    // Add responsive breakpoint detection
    function updateBreakpoint() {
        const width = window.innerWidth;
        const body = document.body;
        
        body.classList.remove('bp-mobile', 'bp-tablet', 'bp-desktop', 'bp-wide');
        
        if (width < 768) {
            body.classList.add('bp-mobile');
        } else if (width < 1024) {
            body.classList.add('bp-tablet');
        } else if (width < 1440) {
            body.classList.add('bp-desktop');
        } else {
            body.classList.add('bp-wide');
        }
    }
    
    // Update on load and resize
    updateBreakpoint();
    window.addEventListener('resize', updateBreakpoint);
    
    // Add high contrast and reduced motion detection
    if (window.matchMedia) {
        const prefersHighContrast = window.matchMedia('(prefers-contrast: high)');
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        
        function updateAccessibilityClasses() {
            document.body.classList.toggle('high-contrast', prefersHighContrast.matches);
            document.body.classList.toggle('reduced-motion', prefersReducedMotion.matches);
        }
        
        updateAccessibilityClasses();
        
        prefersHighContrast.addListener(updateAccessibilityClasses);
        prefersReducedMotion.addListener(updateAccessibilityClasses);
    }
}

/**
 * Zoom Indicator Integration
 */
function initializeZoomIndicator() {
    if (!window.mainVisualization) return;
    
    const vizContainer = document.querySelector('.visualization-container');
    if (!vizContainer) return;
    
    // Create zoom indicator
    const zoomIndicator = document.createElement('div');
    zoomIndicator.className = 'zoom-indicator';
    zoomIndicator.textContent = '1.0x';
    vizContainer.appendChild(zoomIndicator);
    
    // Update zoom indicator when zoom changes
    const originalHandleWheel = window.mainVisualization.handleWheel;
    if (originalHandleWheel) {
        window.mainVisualization.handleWheel = function(e) {
            originalHandleWheel.call(this, e);
            
            // Update zoom display
            zoomIndicator.textContent = `${this.zoomScale.toFixed(1)}x`;
            zoomIndicator.classList.add('visible');
            
            // Hide after delay
            clearTimeout(this.zoomIndicatorTimeout);
            this.zoomIndicatorTimeout = setTimeout(() => {
                zoomIndicator.classList.remove('visible');
            }, 1500);
        };
    }
}

// Initialize enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for other scripts to load
    setTimeout(() => {
        initializeEnhancements();
        initializeZoomIndicator();
    }, 100);
});

// Also initialize when visualization data is loaded
const originalInitVisualization = initializeVisualization;
if (originalInitVisualization) {
    window.initializeVisualization = function(data) {
        const result = originalInitVisualization.call(this, data);
        
        // Apply enhancements after visualization is ready
        setTimeout(() => {
            initializeEnhancements();
            initializeZoomIndicator();
        }, 200);
        
        return result;
    };
}

// Make enhancement functions globally available for debugging
window.initializeEnhancements = initializeEnhancements;
window.initializeFastSearchIntegration = initializeFastSearchIntegration;
window.initializeTooltipEnhancements = initializeTooltipEnhancements;

// Settings UI initialization function
function initializeSettingsUI() {
    // Initialize settings navigation
    const settingsNavBtns = document.querySelectorAll('.settings-nav-btn');
    const settingsCategories = document.querySelectorAll('.settings-category');
    
    settingsNavBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const category = btn.getAttribute('data-category');
            
            // Update active nav button
            settingsNavBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active settings category
            settingsCategories.forEach(cat => cat.classList.remove('active'));
            const targetCategory = document.querySelector(`.settings-category[data-category="${category}"]`);
            if (targetCategory) {
                targetCategory.classList.add('active');
            }
        });
    });
    
    // Initialize range value displays
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const valueDisplay = range.nextElementSibling;
        if (valueDisplay && valueDisplay.classList.contains('range-value')) {
            // Update display on input
            range.addEventListener('input', () => {
                // Special formatting for percentage-based sliders
                if (range.id === 'hdbscan-sample-ratio') {
                    valueDisplay.textContent = Math.round(parseFloat(range.value) * 100) + '%';
                } else {
                    valueDisplay.textContent = range.value;
                }
            });

            // Initialize display
            if (range.id === 'hdbscan-sample-ratio') {
                valueDisplay.textContent = Math.round(parseFloat(range.value) * 100) + '%';
            } else {
                valueDisplay.textContent = range.value;
            }
        }
    });
    
    // Initialize settings action buttons
    const saveBtn = document.getElementById('save-settings-btn');
    const resetBtn = document.getElementById('reset-settings-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveConfiguration);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetConfiguration);
    }
    
    // Sync advanced panel with persisted RAG defaults/local settings
    applyRAGSettingsToForms();
    
    // Add auto-save functionality - save settings automatically when changed
    setupAutoSaveSettings();
    
}

// Auto-save settings when any input changes (debounced)
let autoSaveTimeout = null;
function setupAutoSaveSettings() {
    const modal = document.getElementById('advanced-settings-modal');
    if (!modal) return;

    // Get all form inputs in the settings modal
    const inputs = modal.querySelectorAll('input, select, textarea');

    // These controls have their own confirmation-modal flow and must NOT be auto-saved
    const modalControlledIds = new Set(['llm-model-id', 'context-window-size']);

    inputs.forEach(input => {
        // Skip the save/reset/load buttons themselves
        if (input.id === 'save-settings-btn' || input.id === 'reset-settings-btn') {
            return;
        }

        // Skip controls that require a confirmation modal (model, context window)
        if (modalControlledIds.has(input.id)) {
            return;
        }

        // Use 'input' event for range sliders and number inputs (fires on each keystroke/drag)
        // Use 'change' event for select, checkbox, and text inputs
        const eventType = (input.type === 'range' || input.type === 'number') ? 'input' : 'change';

        input.addEventListener(eventType, () => {
            // Debounce auto-save to avoid excessive saves
            clearTimeout(autoSaveTimeout);
            autoSaveTimeout = setTimeout(() => {
                autoSaveConfiguration();
            }, 500); // Save 500ms after last change
        });
    });

}

async function autoSaveConfiguration() {
    try {
        const config = collectConfigurationData();

        // Use updateConfig (deep merge) to preserve sections not in the Advanced Settings modal
        // (hyde, ui_preferences, visualization)
        if (window.ConfigManager) {
            window.ConfigManager.updateConfig(config);
            showAutoSaveIndicator();
        } else {
            console.warn('⚠️ ConfigManager not available yet');
        }

    } catch (error) {
        console.error('Error auto-saving configuration:', error);
        // Silently fail for auto-save, don't disturb user with toast
    }
}

function showAutoSaveIndicator() {
    // Show a subtle "Saved" indicator briefly
    const saveBtn = document.getElementById('save-settings-btn');
    if (saveBtn) {
        const originalText = saveBtn.innerHTML;
        saveBtn.innerHTML = '<i class="fas fa-check"></i> Saved';
        saveBtn.style.background = 'var(--success-color)';
        
        setTimeout(() => {
            saveBtn.innerHTML = originalText;
            saveBtn.style.background = '';
        }, 1500);
    }
}

// Configuration management functions
async function saveConfiguration() {
    const saveBtn = document.getElementById('save-settings-btn');
    const originalText = saveBtn.textContent;

    try {
        saveBtn.textContent = 'Saving...';
        saveBtn.disabled = true;

        const config = collectConfigurationData();

        // Use updateConfig (deep merge) to preserve sections not in the Advanced Settings modal
        // (hyde, ui_preferences, visualization)
        if (window.ConfigManager) {
            window.ConfigManager.updateConfig(config);
            showToast('Configuration saved successfully!', 'success');

            // Update current markers
            const llmModelId = document.getElementById('llm-model-id');
            if (llmModelId) llmModelId.setAttribute('data-current-model', llmModelId.value);
        } else {
            throw new Error('Configuration manager not available');
        }

    } catch (error) {
        console.error('Error saving configuration:', error);
        showToast(`Failed to save configuration: ${error.message}`, 'error');
    } finally {
        saveBtn.textContent = originalText;
        saveBtn.disabled = false;
    }
}

function resetConfiguration() {
    const modal = document.getElementById('reset-settings-modal');
    if (!modal) return;

    modal.style.display = 'flex';

    const closeModal = () => {
        modal.style.display = 'none';
        confirmBtn.removeEventListener('click', onConfirm);
        cancelBtn.removeEventListener('click', onCancel);
        cancelXBtn.removeEventListener('click', onCancel);
        modal.removeEventListener('click', onOverlay);
        document.removeEventListener('keydown', onEscape);
    };

    const onConfirm = () => {
        closeModal();
        try {
            if (window.ConfigManager) {
                const defaults = window.ConfigManager.DEFAULT_CONFIG;
                const currentConfig = window.ConfigManager.getConfig();
                const partialReset = {
                    llm: { ...defaults.llm },
                    embeddings: { ...defaults.embeddings },
                    chunking: { ...defaults.chunking },
                    rag_prompts: { ...defaults.rag_prompts },
                    search: { ...defaults.search },
                    clustering: { ...defaults.clustering },
                    visualization: { ...defaults.visualization },
                    ui_preferences: { ...defaults.ui_preferences },
                    hyde: { ...defaults.hyde }
                };
                window.ConfigManager.saveConfig(partialReset);
            }
            window.location.reload();
        } catch (error) {
            console.error('Error resetting configuration:', error);
            showToast(`Failed to reset configuration: ${error.message}`, 'error');
        }
    };

    const onCancel = () => { closeModal(); };
    const onOverlay = (evt) => { if (evt.target === modal) onCancel(); };
    const onEscape = (evt) => { if (evt.key === 'Escape') onCancel(); };

    const confirmBtn = document.getElementById('reset-settings-confirm');
    const cancelBtn = document.getElementById('reset-settings-cancel');
    const cancelXBtn = document.getElementById('reset-settings-cancel-x');

    confirmBtn.addEventListener('click', onConfirm);
    cancelBtn.addEventListener('click', onCancel);
    cancelXBtn.addEventListener('click', onCancel);
    modal.addEventListener('click', onOverlay);
    document.addEventListener('keydown', onEscape);
}

/**
 * Setup bidirectional sync between duplicate controls via ConfigManager observers
 * This ensures when config changes in one location, all UI controls update
 */
function setupConfigSync() {
    if (!window.ConfigManager) {
        console.warn('⚠️ ConfigManager not available, skipping config sync setup');
        return;
    }

    // Observe config changes and sync UI controls
    window.ConfigManager.observeConfig((newConfig) => {
        // Sync search type
        const searchTypeSelect = document.getElementById('search-type');
        if (searchTypeSelect && newConfig.ui_preferences?.search_type) {
            if (searchTypeSelect.value !== newConfig.ui_preferences.search_type) {
                searchTypeSelect.value = newConfig.ui_preferences.search_type;
            }
        }

        // Sync result count
        const resultCountSelect = document.getElementById('result-count');
        if (resultCountSelect && newConfig.ui_preferences?.result_count !== undefined) {
            const newValue = String(newConfig.ui_preferences.result_count);
            if (resultCountSelect.value !== newValue) {
                resultCountSelect.value = newValue;
            }
        }

        // Sync HyDE toggle
        const hydeToggle = document.getElementById('hyde-mode-toggle');
        if (hydeToggle && newConfig.ui_preferences?.hyde_enabled !== undefined) {
            if (hydeToggle.checked !== newConfig.ui_preferences.hyde_enabled) {
                hydeToggle.checked = newConfig.ui_preferences.hyde_enabled;
            }
        }

        // Sync Advanced Settings modal controls
        const advancedModal = document.getElementById('advanced-settings-modal');
        if (advancedModal) {
            // Sync LLM settings
            const tempInput = document.getElementById('temperature');
            if (tempInput && newConfig.llm?.temperature !== undefined) {
                tempInput.value = newConfig.llm.temperature;
            }

            const maxTokensInput = document.getElementById('max-tokens');
            if (maxTokensInput && newConfig.llm?.max_tokens !== undefined) {
                maxTokensInput.value = newConfig.llm.max_tokens;
            }

            // Sync RAG/Search settings in Advanced modal
            const vectorWeightInput = document.getElementById('vector-weight');
            if (vectorWeightInput && newConfig.search?.vector_weight !== undefined) {
                vectorWeightInput.value = newConfig.search.vector_weight * 100;
            }

            const retrievalKInput = document.getElementById('rag-retrieval-k');
            if (retrievalKInput && newConfig.search?.retrieval_k !== undefined) {
                retrievalKInput.value = newConfig.search.retrieval_k;
            }

            const numResultsInput = document.getElementById('rag-num-results');
            if (numResultsInput && newConfig.search?.num_results !== undefined) {
                numResultsInput.value = newConfig.search.num_results;
            }

            const similarityInput = document.getElementById('similarity-threshold');
            if (similarityInput && newConfig.search?.similarity_threshold !== undefined) {
                similarityInput.value = newConfig.search.similarity_threshold;
            }

            // Sync prompts
            const systemPromptInput = document.getElementById('system-prompt');
            if (systemPromptInput && newConfig.rag_prompts?.system_prompt !== undefined) {
                systemPromptInput.value = newConfig.rag_prompts.system_prompt;
            }

            const userTemplateInput = document.getElementById('user-template');
            if (userTemplateInput && newConfig.rag_prompts?.user_template !== undefined) {
                userTemplateInput.value = newConfig.rag_prompts.user_template;
            }

            // Update range displays if they exist
            const ranges = advancedModal.querySelectorAll('input[type="range"]');
            ranges.forEach(range => {
                const valueDisplay = range.nextElementSibling;
                if (valueDisplay && valueDisplay.classList.contains('range-value')) {
                    valueDisplay.textContent = range.value;
                }
            });

        }

        // Sync Quick Settings modal controls (RAG settings)
        const quickModal = document.getElementById('quick-settings-modal');
        if (quickModal) {
            const vectorWeightSlider = document.getElementById('quick-vector-weight');
            if (vectorWeightSlider && newConfig.search?.vector_weight !== undefined) {
                vectorWeightSlider.value = newConfig.search.vector_weight * 100;
            }

            const retrievalKSlider = document.getElementById('quick-retrieval-k');
            if (retrievalKSlider && newConfig.search?.retrieval_k !== undefined) {
                retrievalKSlider.value = newConfig.search.retrieval_k;
            }

            const quickTempSlider = document.getElementById('quick-temperature');
            if (quickTempSlider && newConfig.llm?.temperature !== undefined) {
                quickTempSlider.value = newConfig.llm.temperature;
            }

            const quickMaxTokensSlider = document.getElementById('quick-max-tokens');
            if (quickMaxTokensSlider && newConfig.llm?.max_tokens !== undefined) {
                quickMaxTokensSlider.value = newConfig.llm.max_tokens;
            }

            // Sync prompts in Quick Settings
            const quickSystemPrompt = document.getElementById('quick-system-prompt');
            if (quickSystemPrompt && newConfig.rag_prompts?.system_prompt !== undefined) {
                quickSystemPrompt.value = newConfig.rag_prompts.system_prompt;
            }

            const quickUserTemplate = document.getElementById('quick-user-template');
            if (quickUserTemplate && newConfig.rag_prompts?.user_template !== undefined) {
                quickUserTemplate.value = newConfig.rag_prompts.user_template;
            }

            // Sync HyDE settings
            const hydeTemp = document.getElementById('hyde-temperature');
            if (hydeTemp && newConfig.hyde?.temperature !== undefined) {
                hydeTemp.value = newConfig.hyde.temperature;
            }

            const hydeMaxTokens = document.getElementById('hyde-max-tokens');
            if (hydeMaxTokens && newConfig.hyde?.max_tokens !== undefined) {
                hydeMaxTokens.value = newConfig.hyde.max_tokens;
            }

            const hydePrompt = document.getElementById('hyde-prompt');
            if (hydePrompt && newConfig.hyde?.prompt !== undefined) {
                hydePrompt.value = newConfig.hyde.prompt;
            }

            // Update Quick Settings slider value displays
            const quickRanges = quickModal.querySelectorAll('input[type="range"]');
            quickRanges.forEach(range => {
                const rangeId = range.id;
                let valueDisplay = document.getElementById(rangeId + '-value');
                if (valueDisplay) {
                    if (rangeId === 'quick-vector-weight') {
                        const vectorPercent = parseInt(range.value, 10);
                        const bm25Percent = 100 - vectorPercent;
                        valueDisplay.textContent = `${vectorPercent}% Vector / ${bm25Percent}% BM25`;
                    } else {
                        valueDisplay.textContent = range.value;
                    }
                }
            });

        }
    });

}

function collectConfigurationData() {
    const getValueSafe = (id, defaultValue = '') => {
        const el = document.getElementById(id);
        return el ? el.value : defaultValue;
    };

    const getCheckedSafe = (id, defaultValue = false) => {
        const el = document.getElementById(id);
        return el ? el.checked : defaultValue;
    };

    // Read saved config to preserve values for fields not present in the Advanced Settings modal DOM
    const savedConfig = window.ConfigManager ? window.ConfigManager.getConfig() : {};
    const defaults = window.ConfigManager ? window.ConfigManager.DEFAULT_CONFIG : {};

    // Helper: use DOM element value if it exists, otherwise preserve saved config value, then default
    const getFromDomOrConfig = (id, configPath, defaultValue) => {
        const el = document.getElementById(id);
        if (el) return el.value;
        // Element doesn't exist in DOM - preserve current saved config value
        const parts = configPath.split('.');
        let val = savedConfig;
        for (const p of parts) { val = val?.[p]; }
        return val !== undefined ? val : defaultValue;
    };

    return {
        llm: {
            model_id: savedConfig.llm?.model_id || getValueSafe('llm-model-id', defaults.llm?.model_id),
            temperature: parseFloat(getValueSafe('temperature', String(defaults.llm?.temperature))),
            max_tokens: parseInt(getValueSafe('max-tokens', String(defaults.llm?.max_tokens)), 10),
            top_p: parseFloat(getValueSafe('top-p', String(defaults.llm?.top_p))),
            repeat_penalty: parseFloat(getValueSafe('repeat-penalty', String(defaults.llm?.repeat_penalty))),
            context_window_size: savedConfig.llm?.context_window_size || parseInt(getValueSafe('context-window-size', String(defaults.llm?.context_window_size)), 10)
        },
        embeddings: {
            model_name: 'intfloat/multilingual-e5-small',
            batch_size: getValueSafe('embedding-batch-size') ? parseInt(getValueSafe('embedding-batch-size'), 10) : (savedConfig.embeddings?.batch_size ?? null),
            device: savedConfig.embeddings?.device ?? defaults.embeddings?.device ?? 'auto',
            max_length: parseInt(getValueSafe('embedding-max-length', String(defaults.embeddings?.max_length)), 10),
            tokens_per_batch: getValueSafe('embedding-tokens-per-batch') ? parseInt(getValueSafe('embedding-tokens-per-batch'), 10) : (savedConfig.embeddings?.tokens_per_batch ?? null),
            use_worker: savedConfig.embeddings?.use_worker ?? defaults.embeddings?.use_worker ?? true,
            aggressive_mode: savedConfig.embeddings?.aggressive_mode ?? defaults.embeddings?.aggressive_mode ?? true
        },
        chunking: {
            enabled: getCheckedSafe('chunking-enabled', true),
            chunk_size: parseInt(getValueSafe('chunk-size', String(defaults.chunking?.chunk_size)), 10),
            chunk_overlap: parseInt(getValueSafe('chunk-overlap', String(defaults.chunking?.chunk_overlap)), 10),
            min_chunk_size: parseInt(getValueSafe('min-chunk-size', String(defaults.chunking?.min_chunk_size)), 10)
        },
        rag_prompts: {
            system_prompt: getFromDomOrConfig('system-prompt', 'rag_prompts.system_prompt', defaults.rag_prompts?.system_prompt),
            user_template: getFromDomOrConfig('user-template', 'rag_prompts.user_template', defaults.rag_prompts?.user_template)
        },
        search: {
            num_results: parseInt(getFromDomOrConfig('rag-num-results', 'search.num_results', defaults.search?.num_results), 10),
            retrieval_k: parseInt(getFromDomOrConfig('rag-retrieval-k', 'search.retrieval_k', defaults.search?.retrieval_k), 10),
            vector_weight: document.getElementById('vector-weight')
                ? parseFloat(document.getElementById('vector-weight').value) / 100
                : (savedConfig.search?.vector_weight ?? defaults.search?.vector_weight ?? 0.6),
            similarity_threshold: parseFloat(getFromDomOrConfig('similarity-threshold', 'search.similarity_threshold', defaults.search?.similarity_threshold)),
            retrieval_mode: savedConfig.search?.retrieval_mode ?? defaults.search?.retrieval_mode ?? 'semantic',
            quick_mode: savedConfig.search?.quick_mode ?? defaults.search?.quick_mode ?? 'vector'
        },
        clustering: {
            umap_n_neighbors: parseInt(getValueSafe('umap-n-neighbors', String(defaults.clustering?.umap_n_neighbors)), 10),
            umap_min_dist: parseFloat(getValueSafe('umap-min-dist', String(defaults.clustering?.umap_min_dist))),
            umap_metric: getValueSafe('umap-metric', defaults.clustering?.umap_metric),
            umap_clustering_dimensions: parseInt(getValueSafe('umap-clustering-dimensions', String(defaults.clustering?.umap_clustering_dimensions)), 10),
            umap_sample_size: savedConfig.clustering?.umap_sample_size ?? defaults.clustering?.umap_sample_size ?? 10000,
            hdbscan_min_cluster_size: parseInt(getValueSafe('hdbscan-min-cluster-size', String(defaults.clustering?.hdbscan_min_cluster_size)), 10),
            hdbscan_min_samples: parseInt(getValueSafe('hdbscan-min-samples', String(defaults.clustering?.hdbscan_min_samples)), 10),
            hdbscan_metric: getValueSafe('hdbscan-metric', defaults.clustering?.hdbscan_metric)
        }
    };
}

function populateConfigurationForm(config) {
    // Populate embedding model options from available models
    try {
        const select = document.getElementById('embedding-model-select');
        if (select && config.embeddings && config.embeddings.available_models) {
            const groups = config.embeddings.available_models;
            // Clear old options
            select.innerHTML = '';
            const addOption = (value, label) => {
                const opt = document.createElement('option');
                opt.value = value;
                opt.textContent = label || value;
                select.appendChild(opt);
            };
            // Flatten grouped models to simple options list with friendly labels
            Object.keys(groups).forEach(groupName => {
                const models = groups[groupName];
                Object.keys(models).forEach(key => {
                    const entry = models[key];
                    const name = entry.name || key;
                    const label = entry.description ? `${name} — ${entry.description}` : name;
                    addOption(name, label);
                });
            });
        }
    } catch (e) {
        console.warn('Unable to populate embedding model options:', e);
    }
    // Populate LLM settings
    if (config.llm) {
        const llm = config.llm;
        setElementValue('llm-model-id', llm.model_id || llm.model_name);
        setElementValue('temperature', llm.temperature);
        setElementValue('max-tokens', llm.max_tokens);
        setElementValue('top-p', llm.top_p);
        setElementValue('repeat-penalty', llm.repeat_penalty);
        setElementValue('context-window-size', llm.context_window_size);
    }
    
    // Populate embedding settings
    if (config.embeddings) {
        const emb = config.embeddings;
        setElementValue('embedding-batch-size', emb.batch_size);
        setElementValue('embedding-device', emb.device || 'auto');
        setElementValue('embedding-max-length', emb.max_length);
        setElementValue('embedding-tokens-per-batch', emb.tokens_per_batch);
    }

    // Populate chunking settings
    if (config.chunking) {
        const chunk = config.chunking;
        const chunkingEnabled = document.getElementById('chunking-enabled');
        if (chunkingEnabled) chunkingEnabled.checked = chunk.enabled !== false;
        setElementValue('chunk-size', chunk.chunk_size);
        setElementValue('chunk-overlap', chunk.chunk_overlap);
        setElementValue('min-chunk-size', chunk.min_chunk_size);
    }

    // Populate prompts
    if (config.rag_prompts) {
        const prompts = config.rag_prompts;
        setElementValue('system-prompt', prompts.system_prompt);
        setElementValue('user-template', prompts.user_template);
    }
    
    // Populate search settings
    if (config.search) {
        const search = config.search;
        setElementValue('rag-num-results', search.num_results);
        setElementValue('rag-retrieval-k', search.retrieval_k ?? 60);
        setElementValue('vector-weight', search.vector_weight !== undefined ? search.vector_weight * 100 : 60);
        setElementValue('similarity-threshold', search.similarity_threshold);
        const retrievalSelect = document.getElementById('rag-retrieval-mode');
        if (retrievalSelect) {
            retrievalSelect.value = 'semantic';
        }
    }
    
    // Populate clustering settings
    if (config.clustering) {
        const cluster = config.clustering;
        setElementValue('umap-n-neighbors', cluster.umap_n_neighbors);
        setElementValue('umap-min-dist', cluster.umap_min_dist);
        setElementValue('umap-metric', cluster.umap_metric);
        setElementValue('umap-clustering-dimensions', cluster.umap_clustering_dimensions);
        setElementValue('umap-sample-size', cluster.umap_sample_size);
        setElementValue('hdbscan-min-cluster-size', cluster.hdbscan_min_cluster_size);
        setElementValue('hdbscan-min-samples', cluster.hdbscan_min_samples);
        setElementValue('hdbscan-metric', cluster.hdbscan_metric);
    }
    
    // Update range displays
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const valueDisplay = range.nextElementSibling;
        if (valueDisplay && valueDisplay.classList.contains('range-value')) {
            valueDisplay.textContent = range.value;
        }
    });
}

function setElementValue(id, value) {
    const element = document.getElementById(id);
    if (!element || value === undefined || value === null) return;
    
    if (element.type === 'checkbox') {
        element.checked = value;
    } else {
        element.value = value;
    }
}

function applyRAGSettingsToForms() {
    if (typeof window.loadRAGSettings !== 'function') {
        console.warn('loadRAGSettings not available; skipping RAG form sync');
        return;
    }

    try {
        const defaults = window.RAG_DEFAULTS || {};
        const settings = window.loadRAGSettings() || {};
        const getValue = (key, fallback) => settings[key] !== undefined ? settings[key] : (defaults[key] !== undefined ? defaults[key] : fallback);

        setElementValue('rag-num-results', getValue('numResults', 5));
        setElementValue('rag-retrieval-k', getValue('retrievalK', 60));

        const similarityValue = getValue('similarityThreshold', 0.1);
        const similarityInput = document.getElementById('similarity-threshold');
        if (similarityInput) {
            similarityInput.value = similarityValue;
            const rangeDisplay = similarityInput.nextElementSibling;
            if (rangeDisplay && rangeDisplay.classList.contains('range-value')) {
                rangeDisplay.textContent = similarityValue;
            }
        }

        setElementValue('temperature', getValue('temperature', 0.7));
        setElementValue('max-tokens', getValue('maxTokens', 768));
        setElementValue('system-prompt', getValue('systemPrompt', ''));
        setElementValue('user-template', getValue('userTemplate', ''));
    } catch (error) {
        console.warn('Failed to apply persisted RAG settings to advanced panel:', error);
    }
}

function syncLocalRAGSettingsFromConfig(config) {
    if (typeof window.loadRAGSettings !== 'function' || typeof window.saveRAGSettings !== 'function') {
        return;
    }

    try {
        const current = window.loadRAGSettings() || {};
        const defaults = window.RAG_DEFAULTS || {};
        const next = {
            ...current,
            numResults: config?.search?.num_results ?? current.numResults ?? defaults.numResults ?? 5,
            vectorWeight: current.vectorWeight ?? defaults.vectorWeight ?? 0.6,
            enableBM25: current.enableBM25 ?? defaults.enableBM25 ?? true,
            retrievalK: config?.search?.retrieval_k ?? current.retrievalK ?? defaults.retrievalK ?? 60,
            similarityThreshold: config?.search?.similarity_threshold ?? current.similarityThreshold ?? defaults.similarityThreshold ?? 0.1,
            temperature: config?.llm?.temperature ?? current.temperature ?? defaults.temperature ?? 0.4,
            maxTokens: config?.llm?.max_tokens ?? current.maxTokens ?? defaults.maxTokens ?? 768,
            systemPrompt: config?.rag_prompts?.system_prompt ?? current.systemPrompt ?? defaults.systemPrompt ?? '',
            userTemplate: config?.rag_prompts?.user_template ?? current.userTemplate ?? defaults.userTemplate ?? ''
        };

        window.saveRAGSettings(next);
    } catch (error) {
        console.warn('Failed to sync RAG settings to local storage:', error);
    }
}

// Modal functionality
function initializeModalHandlers() {
    const openButton = document.getElementById('open-advanced-settings');
    const modal = document.getElementById('advanced-settings-modal');
    const closeButton = document.getElementById('close-advanced-settings');
    const modalOverlay = modal;
    
    if (openButton) {
        openButton.addEventListener('click', () => {
            openAdvancedSettingsModal();
        });
    }
    
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            closeAdvancedSettingsModal();
        });
    }
    
    // Close modal with escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal && modal.style.display !== 'none') {
            closeAdvancedSettingsModal();
        }
    });
    
    // Processing summary modal handlers
    const procModal = document.getElementById('processing-summary-modal');
    const procClose = document.getElementById('close-processing-summary');
    const procContinue = document.getElementById('continue-to-explore');
    if (procClose) {
        procClose.addEventListener('click', () => closeProcessingSummaryModal(() => transitionToExplore()));
    }
    if (procContinue) {
        procContinue.addEventListener('click', () => closeProcessingSummaryModal(() => transitionToExplore()));
    }
    if (procModal) {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && procModal.style.display !== 'none') {
                closeProcessingSummaryModal(() => transitionToExplore());
            }
        });
    }

}

function openAdvancedSettingsModal() {
    const modal = document.getElementById('advanced-settings-modal');
    if (modal) {
        // Load current configuration when opening modal
        if (window.ConfigManager) {
            const config = window.ConfigManager.getConfig();
            populateConfigurationForm(config);
        }
        
        // Ensure auto-save is set up (in case it wasn't ready on page load)
        if (!modal.hasAttribute('data-autosave-initialized')) {
            setupAutoSaveSettings();
            modal.setAttribute('data-autosave-initialized', 'true');
        }
        
        // Show modal with animation
        modal.style.display = 'flex';
        document.body.classList.add('modal-open');
        
        // Animate in
        setTimeout(() => {
            modal.classList.add('modal-visible');
        }, 10);
        
    }
}

function closeAdvancedSettingsModal() {
    const modal = document.getElementById('advanced-settings-modal');
    if (modal) {
        // Animate out
        modal.classList.remove('modal-visible');
        
        setTimeout(() => {
            modal.style.display = 'none';
            document.body.classList.remove('modal-open');
        }, 300);
        
    }
}

// Add model download tracking
let modelDownloadStates = new Map();

function startModelDownload(modelType, modelName, modelUrl = null) {
    const downloadId = `${modelType}_${modelName}`;
    modelDownloadStates.set(downloadId, {
        type: modelType,
        name: modelName,
        url: modelUrl,
        status: 'downloading',
        progress: 0,
        startTime: Date.now()
    });
    
    const downloadText = modelUrl ? `${modelName} from ${modelUrl}` : modelName;
    showToast(`Starting download of ${downloadText}...`, 'info');
    
    // Simulate download progress (in real implementation, this would be connected to actual download)
    simulateModelDownload(downloadId);
}

function simulateModelDownload(downloadId) {
    const state = modelDownloadStates.get(downloadId);
    if (!state) return;
    
    const interval = setInterval(() => {
        state.progress += Math.random() * 10;
        
        if (state.progress >= 100) {
            state.progress = 100;
            state.status = 'ready';
            clearInterval(interval);
            
            showToast(`${state.name} model is ready!`, 'success');
            
            // Enable the model in UI
            enableModelInUI(state.type, state.name);
        } else {
            // Update progress in UI if needed
            updateModelDownloadProgress(downloadId, state.progress);
        }
    }, 500);
}

function enableModelInUI(modelType, modelName) {
    // Enable model selection and remove downloading indicator
    const selector = getModelSelector(modelType);
    if (selector) {
        const option = selector.querySelector(`option[value="${modelName}"]`);
        if (option) {
            option.disabled = false;
            option.textContent = option.textContent.replace(' (Downloading...)', '');
        }
    }
}

function updateModelDownloadProgress(downloadId, progress) {
    // Update progress in UI if there's a progress indicator
    const progressElement = document.querySelector(`[data-download-id="${downloadId}"]`);
    if (progressElement) {
        progressElement.style.width = `${progress}%`;
    }
}

function getModelSelector(modelType) {
    switch (modelType) {
        case 'llm':
            return document.getElementById('llm-model-id');
        case 'embedding':
            return null; // Embedding model is fixed in browser version
        default:
            return null;
    }
}

// Enhance settings UI to handle model changes
function enhanceSettingsWithModelDownloads() {
    const llmModelId = document.getElementById('llm-model-id');
    const llmModelUrl = document.getElementById('llm-model-url');
    const llmModelName = document.getElementById('llm-model-name');
    const embeddingSelect = document.getElementById('embedding-model-select');
    
    // Handle LLM model changes
    if (llmModelId) {
        llmModelId.addEventListener('change', (e) => {
            const newModel = e.target.value;
            const currentUrl = e.target.getAttribute('data-current-url');
            const newUrl = llmModelUrl ? llmModelUrl.value || '' : '';
            const modelName = llmModelName ? llmModelName.value || newModel : newModel;
            
            if (newUrl && newUrl !== currentUrl) {
                if (confirm(`Download and switch to ${modelName} from this URL? This may take a few minutes.`)) {
                    startModelDownload('llm', modelName, newUrl);
                    e.target.setAttribute('data-current-url', newUrl);
                } else {
                    // Revert to previous URL
                    e.target.value = currentUrl || '';
                }
            }
        });
    }
    
    // Handle embedding model changes
    if (embeddingSelect) {
        embeddingSelect.addEventListener('change', (e) => {
            const newModel = e.target.value;
            const currentModel = e.target.getAttribute('data-current-model');
            
            if (newModel !== currentModel) {
                if (confirm(`Download and switch to ${newModel} embedding model? This may take a few minutes.`)) {
                    startModelDownload('embedding', newModel);
                    e.target.setAttribute('data-current-model', newModel);
                } else {
                    // Revert selection
                    e.target.value = currentModel;
                }
            }
        });
    }

    // Handle Context Window Size changes - save via ConfigManager
    const maxContextInput = document.getElementById('max-context');
    if (maxContextInput) {
        maxContextInput.addEventListener('change', (e) => {
            const contextSize = parseInt(e.target.value, 10);
            if (contextSize >= 1024 && contextSize <= 32768) {
                if (window.ConfigManager) {
                    window.ConfigManager.updateConfig({ llm: { context_window_size: contextSize } });
                }
                // Update current size display
                const currentSizeSpan = document.getElementById('current-context-size');
                if (currentSizeSpan) {
                    currentSizeSpan.textContent = contextSize;
                }

                // Show warning about reload
                showToast('Context window size saved. Reload page to apply changes.', 'warning');
            }
        });

        // Load current value from ConfigManager
        try {
            if (window.ConfigManager) {
                const config = window.ConfigManager.getConfig();
                const storedSize = config.llm?.context_window_size;
                if (storedSize) {
                    maxContextInput.value = storedSize;
                    const currentSizeSpan = document.getElementById('current-context-size');
                    if (currentSizeSpan) {
                        currentSizeSpan.textContent = storedSize;
                    }
                }
            }
        } catch (error) {
            console.warn('Failed to load context window from config:', error);
        }
    }

    // Handle LLM Model ID changes - show confirmation modal before saving/reloading
    const llmModelSelect = document.getElementById('llm-model-id');
    let _previousModelValue = llmModelSelect ? llmModelSelect.value : '';
    if (llmModelSelect) {
        llmModelSelect.addEventListener('change', async (e) => {
            const modelId = e.target.value;
            const constraints = typeof getModelConstraints === 'function' ? getModelConstraints(modelId) : {};
            const modelName = constraints.description || modelId;
            // Use real download size captured from previous downloads, fall back to estimate
            let modelSize = constraints.estimatedSize || '~2.0 GB';
            let hasRealSize = false;
            if (window.__webllmRealDownloadSizes && window.__webllmRealDownloadSizes[modelId]) {
                modelSize = window.__webllmRealDownloadSizes[modelId];
                hasRealSize = true;
            }

            // Show confirmation modal
            const modal = document.getElementById('model-change-modal');
            const nameEl = document.getElementById('model-change-name');
            const sizeEl = document.getElementById('model-change-size');
            const sizeLabelEl = document.getElementById('model-change-size-label');
            if (modal && nameEl && sizeEl) {
                nameEl.textContent = modelName;
                sizeEl.textContent = modelSize;
                if (sizeLabelEl) sizeLabelEl.textContent = hasRealSize ? 'Download size:' : 'Estimated download:';
                modal.style.display = 'flex';

                const closeModal = () => {
                    modal.style.display = 'none';
                    // Remove listeners
                    confirmBtn.removeEventListener('click', onConfirm);
                    cancelBtn.removeEventListener('click', onCancel);
                    cancelXBtn.removeEventListener('click', onCancel);
                    modal.removeEventListener('click', onOverlay);
                    document.removeEventListener('keydown', onEscape);
                };

                const onConfirm = () => {
                    closeModal();
                    try {
                        if (window.ConfigManager) {
                            window.ConfigManager.updateConfig({ llm: { model_id: modelId } });
                        }
                        updateSlidersForModel(modelId);
                        _previousModelValue = modelId;
                        window.location.reload();
                    } catch (error) {
                        console.error('Failed to save LLM model:', error);
                    }
                };

                const onCancel = () => {
                    closeModal();
                    llmModelSelect.value = _previousModelValue;
                };

                const onOverlay = (evt) => {
                    if (evt.target === modal) onCancel();
                };

                const onEscape = (evt) => {
                    if (evt.key === 'Escape') onCancel();
                };

                const confirmBtn = document.getElementById('model-change-confirm');
                const cancelBtn = document.getElementById('model-change-cancel');
                const cancelXBtn = document.getElementById('model-change-cancel-x');

                confirmBtn.addEventListener('click', onConfirm);
                cancelBtn.addEventListener('click', onCancel);
                cancelXBtn.addEventListener('click', onCancel);
                modal.addEventListener('click', onOverlay);
                document.addEventListener('keydown', onEscape);
            }
        });

        // Load current value from unified config
        try {
            if (window.ConfigManager) {
                const config = window.ConfigManager.getConfig();
                const storedModel = config.llm?.model_id;
                if (storedModel) {
                    llmModelSelect.value = storedModel;
                    _previousModelValue = storedModel;
                    // Update sliders for current model
                    updateSlidersForModel(storedModel);
                }
            }
        } catch (error) {
            console.warn('Failed to load LLM model from config:', error);
        }
    }

    // Handle Context Window Size changes - auto-reload
    const contextWindowSelect = document.getElementById('context-window-size');
    let _previousContextValue = contextWindowSelect ? contextWindowSelect.value : '';
    if (contextWindowSelect) {
        contextWindowSelect.addEventListener('change', async (e) => {
            const contextSize = parseInt(e.target.value, 10);

            // Show confirmation modal
            const modal = document.getElementById('context-window-change-modal');
            const sizeEl = document.getElementById('context-window-new-size');
            if (modal && sizeEl) {
                sizeEl.textContent = contextSize.toLocaleString();
                modal.style.display = 'flex';

                const closeModal = () => {
                    modal.style.display = 'none';
                    confirmBtn.removeEventListener('click', onConfirm);
                    cancelBtn.removeEventListener('click', onCancel);
                    cancelXBtn.removeEventListener('click', onCancel);
                    modal.removeEventListener('click', onOverlay);
                    document.removeEventListener('keydown', onEscape);
                };

                const onConfirm = () => {
                    closeModal();
                    try {
                        if (window.ConfigManager) {
                            window.ConfigManager.updateConfig({ llm: { context_window_size: contextSize } });
                        }
                        _previousContextValue = String(contextSize);
                        window.location.reload();
                    } catch (error) {
                        console.error('Failed to save context window size:', error);
                    }
                };

                const onCancel = () => {
                    closeModal();
                    contextWindowSelect.value = _previousContextValue;
                };

                const onOverlay = (evt) => {
                    if (evt.target === modal) onCancel();
                };

                const onEscape = (evt) => {
                    if (evt.key === 'Escape') onCancel();
                };

                const confirmBtn = document.getElementById('context-window-confirm');
                const cancelBtn = document.getElementById('context-window-cancel');
                const cancelXBtn = document.getElementById('context-window-cancel-x');

                confirmBtn.addEventListener('click', onConfirm);
                cancelBtn.addEventListener('click', onCancel);
                cancelXBtn.addEventListener('click', onCancel);
                modal.addEventListener('click', onOverlay);
                document.addEventListener('keydown', onEscape);
            }
        });

        // Load current value from unified config
        try {
            if (window.ConfigManager) {
                const config = window.ConfigManager.getConfig();
                const storedContextSize = config.llm?.context_window_size;
                if (storedContextSize) {
                    contextWindowSelect.value = storedContextSize;
                    _previousContextValue = String(storedContextSize);
                }
            }
        } catch (error) {
            console.warn('Failed to load context window size from config:', error);
        }
    }

    // Handle RAG number of results changes - save to localStorage immediately
    const ragNumResultsInput = document.getElementById('rag-num-results');
    if (ragNumResultsInput) {
        ragNumResultsInput.addEventListener('change', (e) => {
            const numResults = parseInt(e.target.value, 10);
            if (numResults >= 1 && numResults <= 20) {
                if (window.ConfigManager) {
                    window.ConfigManager.updateConfig({ search: { num_results: numResults } });
                }
                showToast(`RAG will now use ${numResults} results for context.`, 'success');
            }
        });

        // Load current value
        try {
            if (window.ConfigManager) {
                const config = window.ConfigManager.getConfig();
                const storedNumResults = config.search?.num_results;
                if (storedNumResults) {
                    ragNumResultsInput.value = storedNumResults;
                }
            }
        } catch (error) {
            console.warn('Failed to load RAG num results from config:', error);
        }
    }

    // Handle Check Storage button
    const checkStorageBtn = document.getElementById('check-storage-btn');
    if (checkStorageBtn) {
        checkStorageBtn.addEventListener('click', async () => {
            try {
                if ('storage' in navigator && 'estimate' in navigator.storage) {
                    const estimate = await navigator.storage.estimate();
                    const usage = estimate.usage || 0;
                    const quota = estimate.quota || 0;
                    const available = quota - usage;
                    const percentUsed = quota > 0 ? (usage / quota) * 100 : 0;

                    // Format sizes intelligently
                    const formatSize = (bytes) => {
                        if (bytes >= 1024 * 1024 * 1024) {
                            return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB';
                        } else if (bytes >= 1024 * 1024) {
                            return (bytes / 1024 / 1024).toFixed(1) + ' MB';
                        } else if (bytes >= 1024) {
                            return (bytes / 1024).toFixed(0) + ' KB';
                        }
                        return bytes + ' B';
                    };

                    const storageInfo = document.getElementById('storage-info');
                    const storageDetails = document.getElementById('storage-details');
                    const storageBarFill = document.getElementById('storage-bar-fill');
                    const storageUsedLabel = document.getElementById('storage-used-label');
                    const storageAvailableLabel = document.getElementById('storage-available-label');

                    if (storageInfo && storageDetails) {
                        // Update progress bar
                        if (storageBarFill) {
                            storageBarFill.style.width = `${Math.min(percentUsed, 100)}%`;
                            storageBarFill.classList.toggle('warning', percentUsed > 80);
                        }

                        // Update labels
                        if (storageUsedLabel) {
                            storageUsedLabel.textContent = `${formatSize(usage)} used (${percentUsed.toFixed(1)}%)`;
                        }
                        if (storageAvailableLabel) {
                            storageAvailableLabel.textContent = `${formatSize(available)} available`;
                        }

                        // Build detailed breakdown
                        storageDetails.innerHTML = `
                            <div class="storage-item">
                                <span class="storage-item-label">Used</span>
                                <span class="storage-item-value highlight">${formatSize(usage)}</span>
                            </div>
                            <div class="storage-item">
                                <span class="storage-item-label">Available</span>
                                <span class="storage-item-value">${formatSize(available)}</span>
                            </div>
                            <div class="storage-item">
                                <span class="storage-item-label">Browser Limit</span>
                                <span class="storage-item-value">${formatSize(quota)}</span>
                            </div>
                        `;

                        storageInfo.style.display = 'block';
                    }

                } else {
                    showToast('Storage API not supported in this browser', 'warning');
                }
            } catch (error) {
                console.error('❌ Failed to check storage:', error);
                showToast('Failed to check storage usage', 'error');
            }
        });
    }

    // Handle Reset and clear all data button
    const clearCacheBtn = document.getElementById('clear-model-cache-btn');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', () => {
            const modal = document.getElementById('reset-all-data-modal');
            if (!modal) return;

            modal.style.display = 'flex';

            const closeModal = () => {
                modal.style.display = 'none';
                confirmBtn.removeEventListener('click', onConfirm);
                cancelBtn.removeEventListener('click', onCancel);
                cancelXBtn.removeEventListener('click', onCancel);
                modal.removeEventListener('click', onOverlay);
                document.removeEventListener('keydown', onEscape);
            };

            const onConfirm = async () => {
                closeModal();
                try {
                    clearCacheBtn.disabled = true;
                    clearCacheBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resetting...';

                    // 1. Clear all Cache API caches
                    if ('caches' in window) {
                        const cacheNames = await caches.keys();
                        for (const cacheName of cacheNames) {
                            await caches.delete(cacheName);
                        }
                    }

                    // 2. Clear all IndexedDB databases
                    const dbNames = [
                        'webllm', 'webllm-cache', 'mlc-llm',
                        'vectoria-embeddings', 'vectoria-indexes', 'vectoria-data',
                        'localforage', 'transformers-cache'
                    ];
                    for (const dbName of dbNames) {
                        try {
                            await indexedDB.deleteDatabase(dbName);
                        } catch (err) {
                            // Database might not exist, ignore
                        }
                    }

                    // 3. Clear localStorage
                    try {
                        const keysToRemove = [];
                        for (let i = 0; i < localStorage.length; i++) {
                            const key = localStorage.key(i);
                            if (key && key.startsWith('vectoria')) {
                                keysToRemove.push(key);
                            }
                        }
                        keysToRemove.forEach(key => localStorage.removeItem(key));
                    } catch (e) {
                        console.warn('Unable to clear localStorage:', e);
                    }
                    // 4. Call API to clear server-side data (if any)
                    try {
                        await fetch('/api/delete_all_data', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' }
                        });
                    } catch (err) {
                        // API might not exist or fail, continue anyway
                    }

                    showToast('All data cleared. Reloading...', 'success');
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);

                } catch (error) {
                    console.error('❌ Failed to reset:', error);
                    showToast(`Failed to reset: ${error.message}`, 'error');
                    clearCacheBtn.disabled = false;
                    clearCacheBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Reset and clear all data';
                }
            };

            const onCancel = () => { closeModal(); };
            const onOverlay = (evt) => { if (evt.target === modal) onCancel(); };
            const onEscape = (evt) => { if (evt.key === 'Escape') onCancel(); };

            const confirmBtn = document.getElementById('reset-all-data-confirm');
            const cancelBtn = document.getElementById('reset-all-data-cancel');
            const cancelXBtn = document.getElementById('reset-all-data-cancel-x');

            confirmBtn.addEventListener('click', onConfirm);
            cancelBtn.addEventListener('click', onCancel);
            cancelXBtn.addEventListener('click', onCancel);
            modal.addEventListener('click', onOverlay);
            document.addEventListener('keydown', onEscape);
        });
    }
}

// Make settings functions globally available
window.initializeSettingsUI = initializeSettingsUI;
window.saveConfiguration = saveConfiguration;
window.resetConfiguration = resetConfiguration;
window.initializeModalHandlers = initializeModalHandlers;
window.openAdvancedSettingsModal = openAdvancedSettingsModal;
window.closeAdvancedSettingsModal = closeAdvancedSettingsModal;
window.applyRAGSettingsToForms = applyRAGSettingsToForms;

// Setting restrictions functionality
let isFileProcessed = false;
let restrictedSettings = [];

function initializeSettingRestrictions() {
    // Track file processing state
    const processBtn = document.getElementById('process-csv-btn');
    if (processBtn) {
        processBtn.addEventListener('click', () => {
            handleFileProcessingStart();
        });
    }
    
    // Define which settings should be restricted after file processing
    restrictedSettings = [
        'llm-model-id',
        'embedding-batch-size',
        'embedding-max-length',
        'embedding-tokens-per-batch'
    ];
    
    // Define clustering settings that should be restricted after data is processed
    clusteringRestrictedSettings = [
        'umap-n-neighbors',
        'umap-min-dist',
        'umap-metric',
        'umap-clustering-dimensions',
        'hdbscan-min-cluster-size',
        'hdbscan-min-samples',
        'hdbscan-metric',
        'hdbscan-sample-ratio',
        'hdbscan-epsilon',
        'hdbscan-method'
    ];
    
}

function handleFileProcessingStart() {
    // Mark that a file has been processed
    isFileProcessed = true;
    
    // Apply restrictions to model-related settings
    applySettingRestrictions();
    
    // Add visual indicators
    addRestrictionIndicators();
}

function applySettingRestrictions() {
    // Restrict model-related settings
    restrictedSettings.forEach(settingId => {
        const element = document.getElementById(settingId);
        if (element) {
            element.disabled = true;
            element.classList.add('setting-restricted');
            
            // Add tooltip explaining restriction
            addRestrictionTooltip(element, 'This setting cannot be changed after processing a file. Upload a new file to modify.');
        }
    });
    
    // Restrict clustering parameters
    clusteringRestrictedSettings.forEach(settingId => {
        const element = document.getElementById(settingId);
        if (element) {
            element.disabled = true;
            element.classList.add('setting-clustering-restricted');
            
            // Add tooltip explaining clustering restriction
            addRestrictionTooltip(element, 'Clustering parameters are locked after data processing. Reprocess data to modify these settings.');
        }
    });
    
    // Show warning message in modal
    showSettingRestrictionWarning();
}

function addRestrictionIndicators() {
    // Add indicators for model restrictions
    const restrictedElements = document.querySelectorAll('.setting-restricted');
    restrictedElements.forEach(element => {
        const parent = element.parentElement;
        if (parent && !parent.querySelector('.restriction-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'restriction-indicator';
            indicator.innerHTML = '<i class="fas fa-lock"></i> Setting locked after file processing';
            indicator.style.cssText = `
                color: var(--warning-color);
                font-size: 0.8em;
                margin-top: 4px;
                display: flex;
                align-items: center;
                gap: 4px;
            `;
            parent.appendChild(indicator);
        }
    });
    
    // Add indicators for clustering restrictions
    const clusteringRestrictedElements = document.querySelectorAll('.setting-clustering-restricted');
    clusteringRestrictedElements.forEach(element => {
        const parent = element.parentElement;
        if (parent && !parent.querySelector('.clustering-restriction-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'clustering-restriction-indicator';
            indicator.innerHTML = '<i class="fas fa-chart-bar"></i> Clustering locked after data processing';
            indicator.style.cssText = `
                color: var(--info-color);
                font-size: 0.8em;
                margin-top: 4px;
                display: flex;
                align-items: center;
                gap: 4px;
            `;
            parent.appendChild(indicator);
        }
    });
}

function addRestrictionTooltip(element, message) {
    element.title = message;
    element.style.cursor = 'not-allowed';
}

function showSettingRestrictionWarning() {
    const modal = document.getElementById('advanced-settings-modal');
    if (modal && modal.style.display !== 'none') {
        const modalBody = modal.querySelector('.modal-body');
        if (modalBody && !modalBody.querySelector('.setting-restriction-warning')) {
            const warning = document.createElement('div');
            warning.className = 'setting-restriction-warning';
            warning.innerHTML = `
                <div style="background: rgba(255, 152, 0, 0.1); border: 1px solid var(--warning-color); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: var(--warning-color); font-weight: 600; margin-bottom: 8px;">
                        <i class="fas fa-exclamation-triangle"></i>
                        Settings Restricted
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9em; line-height: 1.4;">
                        Some model settings are locked after processing a file to ensure consistency. 
                        You can still modify LLM parameters (temperature, prompts, etc.) and RAG search settings.
                        To change model settings, upload a new file.
                    </div>
                </div>
            `;
            modalBody.insertBefore(warning, modalBody.firstChild);
        }
    }
}

function removeSettingRestrictions() {
    isFileProcessed = false;
    
    // Remove model restrictions from elements
    restrictedSettings.forEach(settingId => {
        const element = document.getElementById(settingId);
        if (element) {
            element.disabled = false;
            element.classList.remove('setting-restricted');
            element.style.cursor = '';
            element.removeAttribute('title');
        }
    });
    
    // Remove clustering restrictions from elements
    clusteringRestrictedSettings.forEach(settingId => {
        const element = document.getElementById(settingId);
        if (element) {
            element.disabled = false;
            element.classList.remove('setting-clustering-restricted');
            element.style.cursor = '';
            element.removeAttribute('title');
        }
    });
    
    // Remove all restriction indicators
    const indicators = document.querySelectorAll('.restriction-indicator, .clustering-restriction-indicator');
    indicators.forEach(indicator => indicator.remove());
    
    // Remove warning from modal
    const warning = document.querySelector('.setting-restriction-warning');
    if (warning) {
        warning.remove();
    }
    
    showToast('All settings unlocked for new file', 'info');
}

// Function to check if settings are currently restricted
function areSettingsRestricted() {
    return isFileProcessed;
}

// Function to get list of restricted settings
function getRestrictedSettings() {
    return restrictedSettings.filter(settingId => {
        const element = document.getElementById(settingId);
        return element && element.disabled;
    });
}

// Enhanced settings UI to handle restrictions
function enhanceSettingsWithRestrictions() {
    // Override the settings initialization to handle restrictions
    const originalInitializeSettings = window.initializeSettingsUI;
    window.initializeSettingsUI = function() {
        originalInitializeSettings();
        
        // Apply restrictions if file has been processed
        if (isFileProcessed) {
            setTimeout(() => {
                applySettingRestrictions();
                addRestrictionIndicators();
            }, 100);
        }
        
        // Add file upload listener to remove restrictions
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    removeSettingRestrictions();
                }
            });
        }
    };
}

// Track processing states for better UX
let processingStates = {
    uploading: false,
    processing: false,
    embedding: false,
    clustering: false
};

function updateProcessingState(state, isActive) {
    processingStates[state] = isActive;
    
    // Update UI based on processing state
    if (isActive) {
        showProcessingIndicator(state);
    } else {
        hideProcessingIndicator(state);
    }
    
    // Check if any processing is active
    const isAnyProcessing = Object.values(processingStates).some(active => active);
    
    if (isAnyProcessing) {
        // Disable all restricted settings during processing
        applyTemporaryRestrictions();
    } else {
        // Remove temporary restrictions but keep permanent ones
        removeTemporaryRestrictions();
    }
}

function applyTemporaryRestrictions() {
    const allSettings = [
        ...restrictedSettings
        // Note: LLM URL and name are already included in restrictedSettings
    ];
    
    allSettings.forEach(settingId => {
        const element = document.getElementById(settingId);
        if (element) {
            element.disabled = true;
            element.classList.add('setting-processing');
        }
    });
}

function removeTemporaryRestrictions() {
    // Remove temporary processing restrictions
    const processingElements = document.querySelectorAll('.setting-processing');
    processingElements.forEach(element => {
        element.classList.remove('setting-processing');
        
        // Only re-enable if not permanently restricted
        if (!element.classList.contains('setting-restricted')) {
            element.disabled = false;
        }
    });
}

function showProcessingIndicator(state) {
    const stateMessages = {
        uploading: 'Uploading file...',
        processing: 'Processing data...',
        embedding: 'Generating embeddings...',
        clustering: 'Creating clusters...'
    };
    
    showToast(stateMessages[state] || 'Processing...', 'info');
}

function hideProcessingIndicator(state) {
    const stateMessages = {
        uploading: 'File uploaded successfully',
        processing: 'Data processed successfully',
        embedding: 'Embeddings generated successfully',
        clustering: 'Clusters created successfully'
    };
    
    showToast(stateMessages[state] || 'Processing completed', 'success');
}

// Enhanced model change detection
function handleModelChange(modelType, newModel, oldModel) {
    if (areSettingsRestricted() && restrictedSettings.includes(`${modelType}-model-select`)) {
        // Revert the change and show warning
        const select = document.getElementById(`${modelType}-model-select`);
        if (select) {
            select.value = oldModel;
        }
        
        showToast('Cannot change model settings after processing a file. Upload a new file to modify.', 'error');
        return false;
    }
    
    // Proceed with model change
    if (confirm(`Download and switch to ${newModel} model? This may take a few minutes.`)) {
        startModelDownload(modelType, newModel);
        return true;
    }
    
    return false;
}

// Initialize restriction enhancements
enhanceSettingsWithRestrictions();

// Make restriction functions globally available
window.initializeSettingRestrictions = initializeSettingRestrictions;
window.handleFileProcessingStart = handleFileProcessingStart;
window.removeSettingRestrictions = removeSettingRestrictions;
window.areSettingsRestricted = areSettingsRestricted;
window.updateProcessingState = updateProcessingState;
window.handleModelChange = handleModelChange;

// Metadata system functionality
let detectedMetadataFields = [];
let activeMetadataFilters = {};
let metadataFilterUI = null;

function initializeMetadataSystem() {
    // Listen for file upload to detect metadata
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
    
    // Create metadata filter UI container
    createMetadataFilterContainer();
    
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        analyzeFileForMetadata(file);
    }
}

function analyzeFileForMetadata(file) {
    // Reset previous metadata
    detectedMetadataFields = [];
    activeMetadataFilters = {};
    
    if (file.name.endsWith('.csv')) {
        analyzeCSVMetadata(file);
    } else if (file.name.endsWith('.json') || file.name.endsWith('.jsonl')) {
        analyzeJSONMetadata(file);
    } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
        analyzeExcelMetadata(file);
    } else if (file.name.endsWith('.txt')) {
        analyzeTXTMetadata(file);
    } else {
    }
}

function analyzeCSVMetadata(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const lines = text.split('\n');
        if (lines.length > 0) {
            const headers = lines[0].split(',').map(header => header.trim().replace(/['"]/g, ''));
            
            // Sample some rows to understand data types
            const sampleRows = lines.slice(1, Math.min(11, lines.length))
                .map(line => line.split(',').map(cell => cell.trim().replace(/['"]/g, '')));
            
            detectedMetadataFields = headers.map((header, index) => {
                const sampleValues = sampleRows.map(row => row[index]).filter(val => val && val.length > 0);
                const dataType = inferDataType(sampleValues);
                const uniqueValues = [...new Set(sampleValues)].slice(0, 50); // Limit unique values
                
                return {
                    name: header,
                    displayName: formatFieldName(header),
                    type: dataType,
                    uniqueValues: uniqueValues,
                    isTextColumn: header.toLowerCase().includes('text') || 
                                  header.toLowerCase().includes('content') ||
                                  header.toLowerCase().includes('description') ||
                                  header.toLowerCase().includes('body')
                };
            });
            
            updateMetadataUI();
        }
    };
    reader.readAsText(file);
}

function analyzeJSONMetadata(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const text = e.target.result;
            let jsonData;
            
            if (file.name.endsWith('.jsonl')) {
                // Handle JSON Lines format
                const lines = text.split('\n').filter(line => line.trim());
                jsonData = lines.slice(0, 10).map(line => JSON.parse(line)); // Sample first 10 lines
            } else {
                jsonData = JSON.parse(text);
                if (!Array.isArray(jsonData)) {
                    jsonData = [jsonData]; // Convert single object to array
                }
            }
            
            if (jsonData.length > 0) {
                const allFields = new Set();
                const fieldSamples = {};
                
                // Collect all possible fields and their sample values
                jsonData.forEach(item => {
                    Object.keys(item).forEach(key => {
                        allFields.add(key);
                        if (!fieldSamples[key]) fieldSamples[key] = [];
                        if (fieldSamples[key].length < 10) {
                            fieldSamples[key].push(item[key]);
                        }
                    });
                });
                
                detectedMetadataFields = Array.from(allFields).map(fieldName => {
                    const sampleValues = fieldSamples[fieldName].filter(val => val != null);
                    const dataType = inferDataType(sampleValues);
                    const uniqueValues = [...new Set(sampleValues.map(String))].slice(0, 50);
                    
                    return {
                        name: fieldName,
                        displayName: formatFieldName(fieldName),
                        type: dataType,
                        uniqueValues: uniqueValues,
                        isTextColumn: fieldName.toLowerCase().includes('text') || 
                                      fieldName.toLowerCase().includes('content') ||
                                      fieldName.toLowerCase().includes('description') ||
                                      fieldName.toLowerCase().includes('body')
                    };
                });
                
                updateMetadataUI();
            }
        } catch (error) {
            console.error('Error parsing JSON file for metadata:', error);
        }
    };
    reader.readAsText(file);
}

function analyzeExcelMetadata(file) {
    // For Excel files, we'd need a library like xlsx
    // For now, show a message that Excel metadata detection is not yet implemented
    showToast('Excel metadata detection coming soon', 'info');
}

function analyzeTXTMetadata(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);

        detectedMetadataFields = [
            {
                name: 'text',
                displayName: 'Text',
                type: 'text',
                uniqueValues: lines.slice(0, 50),
                isTextColumn: true
            },
            {
                name: 'line_number',
                displayName: 'Line Number',
                type: 'number',
                uniqueValues: [],
                isTextColumn: false
            }
        ];

        updateMetadataUI();
    };
    reader.readAsText(file);
}

function inferDataType(sampleValues) {
    if (sampleValues.length === 0) return 'text';
    
    const stringValues = sampleValues.map(String);
    
    // Check if all values are numbers
    const numericValues = stringValues.filter(val => !isNaN(val) && val.trim() !== '');
    if (numericValues.length === stringValues.length) {
        return stringValues.some(val => val.includes('.')) ? 'number' : 'integer';
    }
    
    // Check if all values are dates
    const dateValues = stringValues.filter(val => !isNaN(Date.parse(val)));
    if (dateValues.length === stringValues.length && dateValues.length > 0) {
        return 'date';
    }
    
    // Check if values are boolean-like
    const booleanLike = stringValues.filter(val => 
        ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n'].includes(val.toLowerCase())
    );
    if (booleanLike.length === stringValues.length) {
        return 'boolean';
    }
    
    // Check if it's categorical (limited unique values)
    const uniqueValues = new Set(stringValues);
    if (uniqueValues.size <= Math.max(5, stringValues.length * 0.1)) {
        return 'category';
    }
    
    return 'text';
}

function formatFieldName(fieldName) {
    return fieldName
        .replace(/^Column\s+/i, '')  // Remove "Column" prefix if present
        .replace(/[_-]/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/\b\w/g, l => l.toUpperCase())
        .trim();
}

function createMetadataFilterContainer() {
    // Use the inline metadata filters under the visualization
    metadataFilterUI = document.getElementById('metadata-filters-section');
    if (!metadataFilterUI) {
        console.warn('⚠️ Inline metadata filters section not found (may be on hidden tab)');
        // Try again later when tab becomes visible
        return;
    }
    metadataFilterUI.style.display = 'block';

    // Wire actions - use direct event listeners without cloning
    // This ensures they work even if buttons are on hidden tabs
    attachFilterButtonHandlers();
    
}

// Separate function to attach button handlers - can be called multiple times safely
function attachFilterButtonHandlers() {
    const applyBtn = document.getElementById('apply-metadata-filters');
    const clearBtn = document.getElementById('clear-metadata-filters');
    
    if (applyBtn) {
        // Remove old listeners by cloning
        const newApplyBtn = applyBtn.cloneNode(true);
        applyBtn.parentNode.replaceChild(newApplyBtn, applyBtn);
        
        // Add new listener with capture phase to ensure it runs
        newApplyBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            try {
                applyMetadataFilters();
            } catch (err) {
                console.error('Error in applyMetadataFilters:', err);
            }
        }, true);
        
        // Also add in bubble phase as backup
        newApplyBtn.addEventListener('click', function(e) {
        }, false);
        
    } else {
        console.warn('⚠️ Apply Filters button (#apply-metadata-filters) not found in DOM');
    }
    
    if (clearBtn) {
        // Remove old listeners by cloning
        const newClearBtn = clearBtn.cloneNode(true);
        clearBtn.parentNode.replaceChild(newClearBtn, clearBtn);
        
        // Add new listener with capture phase
        newClearBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            try {
                clearMetadataFilters();
            } catch (err) {
                console.error('Error in clearMetadataFilters:', err);
            }
        }, true);
        
        // Also add in bubble phase as backup
        newClearBtn.addEventListener('click', function(e) {
        }, false);
        
    } else {
        console.warn('⚠️ Clear All button (#clear-metadata-filters) not found in DOM');
    }
    
}

async function loadMetadataSchemaFromBackend() {
    try {
        const response = await fetch('/metadata_schema');
        
        if (response.ok) {
            const data = await response.json();
            if (data.metadata_fields && Object.keys(data.metadata_fields).length > 0) {
                return data.metadata_fields;
            } else {
            }
        } else {
            const errorData = await response.json().catch(() => ({}));
        }
    } catch (error) {
    }
    return null;
}

function extractMetadataFromVisualizationData() {
    // Use the same data that's already working in the text display
    if (!currentVisualizationData || !currentVisualizationData.points) {
        return false;
    }
    
    const excludedKeys = ['index', 'x', 'y', 'cluster', 'text', 'cluster_probability', 'cluster_color', 'cluster_name', 'doc_id', 'chunk_id', 'cluster_keyword_scores', 'cluster_keywords', 'cluster_keywords_viz', 'color', 'filename', 'original_filename', 'processing_type', 'selected_column', 'num_chunks', 'metadata'];
    const columnData = {};
    
    // Extract all unique columns and their unique values from ALL points
    currentVisualizationData.points.forEach(point => {
        Object.entries(point).forEach(([key, value]) => {
            // Skip excluded keys
            if (excludedKeys.includes(key)) return;
            
            // Skip null, undefined, or empty string values
            if (value === null || value === undefined || value === '') return;
            
            // Initialize column if not seen before
            if (!columnData[key]) {
                columnData[key] = {
                    name: key,
                    displayName: formatMetadataKey(key),
                    uniqueValues: new Set(),
                    type: null
                };
            }
            
            // Add unique value
            columnData[key].uniqueValues.add(String(value));
            
            // Determine type on first non-null value
            if (!columnData[key].type) {
                columnData[key].type = inferTypeFromValue(value);
            }
        });
    });
    
    // Convert to final format
    detectedMetadataFields = Object.values(columnData)
        .filter(column => column.uniqueValues.size > 0) // Only include columns with values
        .map(column => {
            const uniqueValues = Array.from(column.uniqueValues).sort();
            
            // Calculate min/max for numeric fields
            let minValue = null;
            let maxValue = null;
            if (column.type === 'number') {
                const numericValues = uniqueValues.map(Number).filter(n => !isNaN(n));
                if (numericValues.length > 0) {
                    minValue = Math.min(...numericValues);
                    maxValue = Math.max(...numericValues);
                }
            }
            
            return {
                name: column.name,
                displayName: column.displayName,
                type: column.type,
                uniqueValues: uniqueValues,
                minValue: minValue,
                maxValue: maxValue
            };
        })
        .sort((a, b) => a.displayName.localeCompare(b.displayName)); // Sort by display name
    
    detectedMetadataFields.forEach(field => {
    });

    if (window.browserML?.pipeline?.currentDataset) {
        window.browserML.pipeline.currentDataset.metadataSchema = detectedMetadataFields;
    }
    
    updateMetadataUI();
    
    // Also update RAG metadata controls
    if (window.populateRAGMetadataFields) {
        try {
            window.populateRAGMetadataFields();
        } catch (error) {
            console.error('❌ Error populating RAG metadata fields:', error);
        }
    } else {
        console.warn('⚠️ window.populateRAGMetadataFields not available - will be populated when switching to RAG mode');
    }
    
    return true;
}

function inferTypeFromValue(value) {
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number') return 'number';
    
    // Try to parse as number
    const numValue = Number(value);
    if (!isNaN(numValue) && String(numValue) === String(value)) {
        return 'number';
    }
    
    // Try to detect dates
    if (typeof value === 'string') {
        const datePatterns = [
            /^\d{4}-\d{2}-\d{2}$/,           // YYYY-MM-DD
            /^\d{4}-\d{2}-\d{2}T/,          // ISO datetime
            /^\d{2}\/\d{2}\/\d{4}$/,        // MM/DD/YYYY
            /^\d{2}-\d{2}-\d{4}$/           // MM-DD-YYYY
        ];
        
        if (datePatterns.some(pattern => pattern.test(value))) {
            return 'date';
        }
        
        // Short strings are likely categories
        if (value.length < 50 && !value.includes('\n') && !value.includes('\t')) {
            return 'category';
        }
    }
    
    return 'text';
}

async function loadMetadataFromProcessedData() {
    // First try to extract from visualization data (this is the working source)
    if (extractMetadataFromVisualizationData()) {
        return true;
    }
    
    // Fallback to backend schema if visualization data isn't available
    const backendSchema = await loadMetadataSchemaFromBackend();
    if (backendSchema) {
        // Convert backend schema to our format and filter out invalid fields
        detectedMetadataFields = Object.values(backendSchema)
            .filter(field => {
                // Filter out fields with no valid data
                const hasValidValues = field.unique_values && field.unique_values.length > 0;
                const hasValidRange = field.type === 'number' && (field.min_value !== null || field.max_value !== null);
                const isValidField = field.name && field.type && (hasValidValues || hasValidRange || field.type === 'text');
                
                if (!isValidField) {
                }
                
                return isValidField;
            })
            .map(field => ({
                name: field.name,
                type: field.type,
                uniqueValues: (field.unique_values || []).filter(value => 
                    value !== null && 
                    value !== undefined && 
                    value !== '' && 
                    String(value).toLowerCase() !== 'undefined'
                ),
                minValue: field.min_value,
                maxValue: field.max_value
            }));
        
        if (window.browserML?.pipeline?.currentDataset) {
            window.browserML.pipeline.currentDataset.metadataSchema = detectedMetadataFields;
        }
        updateMetadataUI();
        return true;
    }
    return false;
}

function updateMetadataUI() {
    // Try to get the filter UI if we don't have it yet
    if (!metadataFilterUI) {
        metadataFilterUI = document.getElementById('metadata-filters-section');
    }
    
    if (!metadataFilterUI) {
        console.warn('⚠️ Cannot update metadata UI - metadata-filters-section element not found');
        return;
    }
    
    if (detectedMetadataFields.length === 0) {
        return;
    }
    
    // Show inline filters by default
    metadataFilterUI.style.display = 'block';
    
    const filtersGrid = document.getElementById('metadata-filters-grid');
    const filterActions = metadataFilterUI.querySelector('.metadata-filter-actions');
    
    if (detectedMetadataFields.length === 0) {
        filtersGrid.innerHTML = '<p class="no-metadata-message">No metadata fields detected in this file.</p>';
        filterActions.style.display = 'none';
        return;
    }
    
    // Filter out text columns from metadata filters (they're used for content, not filtering)
    const filterableFields = detectedMetadataFields.filter(field => !field.isTextColumn);
    
    if (filterableFields.length === 0) {
        filtersGrid.innerHTML = '<p class="no-metadata-message">No filterable metadata fields detected. All fields appear to be text content.</p>';
        filterActions.style.display = 'none';
        return;
    }
    
    // Create filter UI for each field
    let filtersHTML = '';
    filterableFields.forEach(field => {
        filtersHTML += createFilterUI(field);
    });
    
    filtersGrid.innerHTML = filtersHTML;
    filterActions.style.display = 'flex';
    
    // Add event listeners to filter controls
    addFilterEventListeners();
    
    showToast(`Detected ${filterableFields.length} filterable metadata fields`, 'success');
}

function createFilterUI(field) {
    const filterId = `filter-${field.name.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const displayName = field.displayName || formatMetadataKey(field.name);
    
    switch (field.type) {
        case 'category':
            return `
                <div class="metadata-filter-item" data-field="${field.name}" data-type="category">
                    <label for="${filterId}">${displayName}</label>
                    <select id="${filterId}" class="metadata-filter" multiple>
                        <option value="">All ${displayName}</option>
                        ${field.uniqueValues.map(value => 
                            `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`
                        ).join('')}
                    </select>
                    <small class="filter-help">${field.uniqueValues.length} categories</small>
                </div>
            `;
        
        case 'integer':
        case 'number':
            const minVal = field.minValue !== null ? field.minValue : Math.min(...field.uniqueValues.map(Number));
            const maxVal = field.maxValue !== null ? field.maxValue : Math.max(...field.uniqueValues.map(Number));
            return `
                <div class="metadata-filter-item" data-field="${field.name}" data-type="number">
                    <label for="${filterId}-min">${displayName}</label>
                    <div class="range-filter">
                        <input type="number" id="${filterId}-min" class="metadata-filter" 
                               data-range="min" placeholder="Min" min="${minVal}" max="${maxVal}">
                        <span>to</span>
                        <input type="number" id="${filterId}-max" class="metadata-filter"
                               data-range="max" placeholder="Max" min="${minVal}" max="${maxVal}">
                    </div>
                    <small class="filter-help">Range: ${minVal} - ${maxVal}</small>
                </div>
            `;
        
        case 'date':
            return `
                <div class="metadata-filter-item" data-field="${field.name}" data-type="date">
                    <label for="${filterId}-from">${displayName}</label>
                    <div class="date-filter">
                        <input type="date" id="${filterId}-from" class="metadata-filter" data-range="from">
                        <span>to</span>
                        <input type="date" id="${filterId}-to" class="metadata-filter" data-range="to">
                    </div>
                    <small class="filter-help">Date range filter</small>
                </div>
            `;
        
        case 'boolean':
            return `
                <div class="metadata-filter-item" data-field="${field.name}" data-type="boolean">
                    <label for="${filterId}">${displayName}</label>
                    <select id="${filterId}" class="metadata-filter">
                        <option value="">All</option>
                        <option value="true">True/Yes</option>
                        <option value="false">False/No</option>
                    </select>
                    <small class="filter-help">Boolean filter</small>
                </div>
            `;
        
        default:
            // For text fields with limited unique values, treat as category with checkboxes
            if (field.uniqueValues.length <= 20) {
                const checkboxes = field.uniqueValues.map((value, idx) => {
                    const checkboxId = `${filterId}-${idx}`;
                    // For cluster_label field, use custom cluster names if available
                    let displayValue = escapeHtml(value);
                    if (field.name === 'cluster_label') {
                        // Parse cluster ID from "Cluster X" format
                        const clusterMatch = String(value).match(/^Cluster\s+(\d+)$/i);
                        if (clusterMatch) {
                            const clusterId = parseInt(clusterMatch[1], 10);
                            displayValue = escapeHtml(getClusterName(clusterId));
                        } else if (String(value).toLowerCase() === 'outlier') {
                            displayValue = 'Outlier';
                        }
                    }
                    return `
                        <label class="checkbox-option" for="${checkboxId}">
                            <input type="checkbox"
                                   id="${checkboxId}"
                                   class="metadata-filter-checkbox"
                                   data-field="${field.name}"
                                   value="${escapeHtml(value)}">
                            <span class="checkbox-custom"></span>
                            <span class="checkbox-label-text">${displayValue}</span>
                        </label>
                    `;
                }).join('');

                return `
                    <div class="metadata-filter-item metadata-filter-checkboxes" data-field="${field.name}" data-type="category">
                        <div class="filter-header">
                            <label class="filter-title">
                                ${displayName}
                                <span class="multi-select-badge">Multi-select</span>
                            </label>
                            <button class="select-all-btn" data-field="${field.name}" type="button">
                                <i class="fas fa-check-double"></i> All
                            </button>
                            <button class="clear-all-btn" data-field="${field.name}" type="button">
                                <i class="fas fa-times"></i> Clear
                            </button>
                        </div>
                        <div class="checkbox-group" id="${filterId}">
                            ${checkboxes}
                        </div>
                        <small class="filter-help">
                            <i class="fas fa-mouse-pointer"></i>
                            Click to select • Double-click to apply • ${field.uniqueValues.length} options
                        </small>
                    </div>
                `;
            } else {
                // Create datalist with suggestions (limit to 100 for performance)
                const datalistId = `${filterId}-suggestions`;
                const suggestions = field.uniqueValues.slice(0, 100);
                const datalistOptions = suggestions.map(v => `<option value="${escapeHtml(String(v))}">`).join('');

                return `
                    <div class="metadata-filter-item" data-field="${field.name}" data-type="text">
                        <label for="${filterId}">${displayName}</label>
                        <input type="text" id="${filterId}" class="metadata-filter"
                               placeholder="Search ${displayName.toLowerCase()}..."
                               list="${datalistId}" autocomplete="off">
                        <datalist id="${datalistId}">${datalistOptions}</datalist>
                        <small class="filter-help">Text search in ${field.uniqueValues.length} values</small>
                    </div>
                `;
            }
    }
}

function addFilterEventListeners() {
    const filterElements = document.querySelectorAll('.metadata-filter');
    filterElements.forEach(element => {
        // Update count on change/input
        element.addEventListener('change', updateFilterCount);
        element.addEventListener('input', updateFilterCount);

        // Auto-apply for range inputs (number/date) with debounce
        if (element.type === 'number' || element.type === 'date') {
            let rangeDebounceTimer;
            element.addEventListener('input', (e) => {
                clearTimeout(rangeDebounceTimer);
                rangeDebounceTimer = setTimeout(() => {
                    if (typeof applyMetadataFilters === 'function') {
                        applyMetadataFilters();
                    }
                }, 800); // Auto-apply 800ms after user stops typing
            });
        }
    });

    // Add checkbox event listeners
    const checkboxes = document.querySelectorAll('.metadata-filter-checkbox');
    checkboxes.forEach(checkbox => {
        // Update count on change
        checkbox.addEventListener('change', updateFilterCount);

        // Double-click to apply filters instantly
        checkbox.addEventListener('dblclick', (e) => {
            e.preventDefault();

            // Visual feedback
            const label = checkbox.closest('.checkbox-option');
            if (label) {
                label.style.transition = 'all 0.2s ease';
                label.style.transform = 'scale(1.05)';
                label.style.background = 'rgba(102, 126, 234, 0.2)';

                setTimeout(() => {
                    label.style.transform = 'scale(1)';
                    label.style.background = '';
                }, 200);
            }

            // Apply filters
            if (typeof applyMetadataFilters === 'function') {
                applyMetadataFilters();
            }
        });
    });

    // Add Select All / Clear All button handlers
    const selectAllBtns = document.querySelectorAll('.select-all-btn');
    selectAllBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const fieldName = btn.dataset.field;
            const checkboxes = document.querySelectorAll(`.metadata-filter-checkbox[data-field="${fieldName}"]`);
            checkboxes.forEach(cb => cb.checked = true);
            updateFilterCount();
        });
    });

    const clearAllBtns = document.querySelectorAll('.clear-all-btn');
    clearAllBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const fieldName = btn.dataset.field;
            const checkboxes = document.querySelectorAll(`.metadata-filter-checkbox[data-field="${fieldName}"]`);
            checkboxes.forEach(cb => cb.checked = false);
            updateFilterCount();
        });
    });

    // Double-click to apply filters instantly
    filterElements.forEach(element => {
        // Double-click to apply filters instantly
        element.addEventListener('dblclick', (e) => {
            e.preventDefault();
            // Show visual feedback
            element.style.transition = 'all 0.2s ease';
            element.style.transform = 'scale(1.02)';
            element.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.3)';
            
            setTimeout(() => {
                element.style.transform = 'scale(1)';
                element.style.boxShadow = '';
            }, 200);
            
            // Apply filters
            if (typeof applyMetadataFilters === 'function') {
                applyMetadataFilters();
            }
        });
        
    });

    // Also add Enter key support for text inputs
    filterElements.forEach(element => {
        if (element.tagName === 'INPUT' && element.type === 'text') {
            element.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (typeof applyMetadataFilters === 'function') {
                        applyMetadataFilters();
                    }
                }
            });
        }
    });
}

function updateFilterCount() {
    const activeFilters = collectActiveFilters();
    const count = Object.keys(activeFilters).length;
    
    const filterCountElement = document.getElementById('active-filter-count');
    if (filterCountElement) {
        if (count === 0) {
            filterCountElement.textContent = '';
            filterCountElement.className = 'filter-count';
            filterCountElement.style.display = 'none';
        } else {
            filterCountElement.textContent = `${count}`;
            filterCountElement.className = 'filter-count active';
            filterCountElement.style.display = 'inline-block';
        }
    }

    // Also update persistent status bar near the search input
    updateFilterStatusBar();
}

// Persistent filter status bar (always accessible near search)
function setupFilterStatusBar() {
    const clearBtn = document.getElementById('clear-active-filters');
    if (clearBtn) {
        // Remove any existing handlers to avoid duplicates
        const newBtn = clearBtn.cloneNode(true);
        clearBtn.parentNode.replaceChild(newBtn, clearBtn);
        
        newBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            
            try {
                // Clear all filters - this function handles everything
                clearMetadataFilters();
            } catch (e) {
                console.error('❌ Failed to clear filters:', e);
                if (typeof showToast === 'function') {
                    showToast('Error clearing filters: ' + e.message, 'error');
                }
            }
        });
    } else {
        console.warn('⚠️ Clear Filters button not found in DOM');
    }
    // Initial paint
    updateFilterStatusBar();
}

function updateFilterStatusBar() {
    const bar = document.getElementById('filter-status-bar');
    const text = document.getElementById('filter-status-text');
    const countEl = document.getElementById('filter-status-count');
    const listEl = document.getElementById('filter-status-list');
    if (!bar || !text || !countEl) return;

    // Prefer the active tracked filters; fall back to current UI collection
    const filters = (window.getActiveMetadataFilters && window.getActiveMetadataFilters()) ||
                    (window.collectMetadataFiltersForSearch && window.collectMetadataFiltersForSearch()) || {};
    const count = Object.keys(filters).length;

    if (count > 0) {
        bar.style.display = 'flex';
        countEl.textContent = String(count);
        countEl.style.display = 'inline-block';
        text.textContent = 'Filters active';
        if (listEl) {
            listEl.innerHTML = renderActiveFilterChips(filters);
            attachFilterChipHandlers();
        }
    } else {
        // Keep bar visible but update the display to show no filters
        bar.style.display = 'none';
        countEl.style.display = 'none';
        if (listEl) listEl.innerHTML = '';
    }
}

function collectActiveFilters() {
    const filters = {};
    const filterElements = document.querySelectorAll('.metadata-filter');
    
    filterElements.forEach(element => {
        const field = element.dataset.field;
        const type = element.dataset.type;
        const value = element.value;
        
        if (value && value.trim()) {
            if (!filters[field]) {
                filters[field] = { type: type, conditions: [] };
            }
            
            if (type === 'range') {
                const rangeType = element.dataset.range;
                if (!filters[field].range) filters[field].range = {};
                filters[field].range[rangeType] = value;
            } else if (type === 'date') {
                const rangeType = element.dataset.range;
                if (!filters[field].range) filters[field].range = {};
                filters[field].range[rangeType] = value;
            } else if (type === 'category') {
                const selectedValues = Array.from(element.selectedOptions).map(option => option.value);
                if (selectedValues.length > 0) {
                    filters[field].conditions = selectedValues;
                }
            } else {
                filters[field].conditions = [value];
            }
        }
    });
    
    // Clean up empty filters
    Object.keys(filters).forEach(field => {
        const filter = filters[field];
        if (filter.conditions && filter.conditions.length === 0) {
            if (!filter.range || Object.keys(filter.range).length === 0) {
                delete filters[field];
            }
        }
    });
    
    return filters;
}

window.collectMetadataFiltersForSearch = function() {
    const filters = {};
    const filterElements = document.querySelectorAll('.metadata-filter-item input, .metadata-filter-item select');

    filterElements.forEach(element => {
        const filterItem = element.closest('.metadata-filter-item');
        const field = filterItem.dataset.field;
        const type = filterItem.dataset.type;
        
        if (!field || !type) return;
        
        let value = null;
        
        if (type === 'number') {
            // Handle range inputs
            const minInput = filterItem.querySelector('input[data-range="min"]');
            const maxInput = filterItem.querySelector('input[data-range="max"]');
            
            if (minInput && maxInput && (minInput.value || maxInput.value)) {
                value = {};
                if (minInput.value) value.min = parseFloat(minInput.value);
                if (maxInput.value) value.max = parseFloat(maxInput.value);
            }
        } else if (type === 'date') {
            // Handle date range inputs
            const minInput = filterItem.querySelector('input[data-range="from"]');
            const maxInput = filterItem.querySelector('input[data-range="to"]');
            
            if (minInput && maxInput && (minInput.value || maxInput.value)) {
                value = {};
                if (minInput.value) value.min = minInput.value;
                if (maxInput.value) value.max = maxInput.value;
            }
        } else if (type === 'boolean') {
            // Handle select dropdown for boolean
            const select = filterItem.querySelector('select');
            if (select && select.value && select.value !== '') {
                value = select.value.toLowerCase();
            }
        } else if (type === 'category') {
            // Handle checkbox-based multi-select
            const checkboxes = filterItem.querySelectorAll('input[type="checkbox"]:checked');
            if (checkboxes.length > 0) {
                // Collect all checked values
                value = Array.from(checkboxes)
                    .map(checkbox => checkbox.value)
                    .filter(val => val !== '');
            } else {
                const multiSelect = filterItem.querySelector('select[multiple]');
                if (multiSelect) {
                    const selected = Array.from(multiSelect.selectedOptions)
                        .map(option => option.value)
                        .filter(val => val !== '');
                    if (selected.length > 0) {
                        value = selected;
                    }
                }
            }
        } else if (type === 'text') {
            // Handle text input
            const textInput = filterItem.querySelector('input[type="text"]');
            if (textInput && textInput.value.trim()) {
                value = textInput.value.trim();
            }
        }
        
        if (value !== null) {
            if (Array.isArray(value) && value.length === 0) {
                value = null;
            }
        }

        if (value !== null) {
            filters[field] = {
                type: type,
                value: value
            };
        }
    });

    return filters;
};

// Render active filter chips (as HTML string)
function renderActiveFilterChips(filters) {
    const parts = [];
    const formatRange = (v, isDate = false) => {
        if (!v || (v.min === undefined && v.max === undefined)) return '';
        const min = v.min !== undefined ? v.min : '';
        const max = v.max !== undefined ? v.max : '';
        if (min !== '' && max !== '') return `${min} – ${max}`;
        if (min !== '') return `≥ ${min}`;
        if (max !== '') return `≤ ${max}`;
        return '';
    };
    Object.entries(filters).forEach(([field, cfg]) => {
        const type = cfg.type;
        const value = cfg.value;
        const prettyKey = formatMetadataKey(field);
        if (type === 'category') {
            if (Array.isArray(value)) {
                value.forEach(val => {
                    parts.push(
                        `<span class="filter-chip" data-field="${escapeHtml(field)}" data-type="category" data-value="${escapeHtml(val)}">` +
                        `<span class="chip-key">${escapeHtml(prettyKey)}:</span> <span class="chip-value">${escapeHtml(val)}</span>` +
                        `<button class="chip-remove" title="Remove">×</button>` +
                        `</span>`
                    );
                });
            } else if (value !== undefined && value !== null && value !== '') {
                parts.push(
                    `<span class="filter-chip" data-field="${escapeHtml(field)}" data-type="category" data-value="${escapeHtml(String(value))}">` +
                    `<span class="chip-key">${escapeHtml(prettyKey)}:</span> <span class="chip-value">${escapeHtml(String(value))}</span>` +
                    `<button class="chip-remove" title="Remove">×</button>` +
                    `</span>`
                );
            }
        } else if (type === 'number' || type === 'date') {
            const rangeText = formatRange(value, type === 'date');
            if (rangeText) {
                parts.push(
                    `<span class="filter-chip" data-field="${escapeHtml(field)}" data-type="${type}" data-value="">` +
                    `<span class="chip-key">${escapeHtml(prettyKey)}:</span> <span class="chip-value">${escapeHtml(rangeText)}</span>` +
                    `<button class="chip-remove" title="Remove">×</button>` +
                    `</span>`
                );
            }
        } else if (type === 'boolean') {
            const label = (value === true || value === 'true') ? 'True' : 'False';
            parts.push(
                `<span class="filter-chip" data-field="${escapeHtml(field)}" data-type="boolean" data-value="${value}">` +
                `<span class="chip-key">${escapeHtml(prettyKey)}:</span> <span class="chip-value">${label}</span>` +
                `<button class="chip-remove" title="Remove">×</button>` +
                `</span>`
            );
        } else if (type === 'text') {
            const label = String(value);
            if (label) {
                parts.push(
                    `<span class="filter-chip" data-field="${escapeHtml(field)}" data-type="text" data-value="">` +
                    `<span class="chip-key">${escapeHtml(prettyKey)}:</span> <span class="chip-value">"${escapeHtml(label)}"</span>` +
                    `<button class="chip-remove" title="Remove">×</button>` +
                    `</span>`
                );
            }
        }
    });
    return parts.join('');
}

function attachFilterChipHandlers() {
    const listEl = document.getElementById('filter-status-list');
    if (!listEl) return;

    // Remove any existing listeners to prevent duplicates
    const oldListEl = listEl.cloneNode(true);
    listEl.parentNode.replaceChild(oldListEl, listEl);

    // Use event delegation on the list container
    oldListEl.addEventListener('click', (e) => {
        const removeBtn = e.target.closest('.chip-remove');
        if (!removeBtn) return;

        e.preventDefault();
        e.stopPropagation();

        const chip = removeBtn.closest('.filter-chip');
        if (!chip) return;

        const field = chip.getAttribute('data-field');
        const type = chip.getAttribute('data-type');
        const value = chip.getAttribute('data-value');

        removeSingleFilter(field, type, value);
    });
}

function removeSingleFilter(field, type, value) {
    const filterItem = document.querySelector(`.metadata-filter-item[data-field="${CSS.escape(field)}"]`);
    if (!filterItem) {
        console.warn(`⚠️ Filter item not found for field: ${field}`);
        return;
    }

    if (type === 'number') {
        const minInput = filterItem.querySelector('input[data-range="min"]');
        const maxInput = filterItem.querySelector('input[data-range="max"]');
        if (minInput) {
            minInput.value = '';
        }
        if (maxInput) {
            maxInput.value = '';
        }
    } else if (type === 'date') {
        const fromInput = filterItem.querySelector('input[data-range="from"]');
        const toInput = filterItem.querySelector('input[data-range="to"]');
        if (fromInput) {
            fromInput.value = '';
        }
        if (toInput) {
            toInput.value = '';
        }
    } else if (type === 'boolean') {
        const select = filterItem.querySelector('select');
        if (select) {
            select.value = '';
        }
    } else if (type === 'category') {
        // Handle both select-based and checkbox-based category filters
        const selectElement = filterItem.querySelector('select[multiple]');
        const hasCheckboxes = filterItem.querySelectorAll('input[type="checkbox"]').length > 0;

        if (selectElement) {
            // Handle multi-select dropdown
            if (value) {
                const options = Array.from(selectElement.options);
                const option = options.find(opt => opt.value === value);
                if (option) {
                    option.selected = false;
                } else {
                    console.warn(`  ⚠️ Option not found for value: "${value}"`);
                }
            } else {
                // Clear all selections
                Array.from(selectElement.options).forEach(opt => opt.selected = false);
            }
        } else if (hasCheckboxes) {
            // Handle checkbox-based filters
            if (value) {
                // Clear specific value
                // Find checkbox by value
                const allCheckboxes = filterItem.querySelectorAll('input[type="checkbox"]');
                const checkbox = Array.from(allCheckboxes).find(cb => {
                    return cb.value === value;
                });

                if (checkbox) {
                    checkbox.checked = false;
                } else {
                    console.warn(`  ⚠️ Checkbox not found for value: "${value}"`);
                }
            } else {
                // Clear all checkboxes in this category
                const checkboxes = filterItem.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = false);
            }
        } else {
            console.warn(`  ⚠️ No select or checkboxes found in category filter`);
        }
    } else if (type === 'text') {
        const input = filterItem.querySelector('input[type="text"]');
        if (input) {
            input.value = '';
        }
    }

    // Re-apply filters
    updateFilterCount();
    if (typeof applyMetadataFilters === 'function') {
        applyMetadataFilters();
    } else {
        console.warn('⚠️ applyMetadataFilters function not found');
    }
}

// Function to filter visualization points based on metadata filters (for local filtering)
function filterVisualizationPointsByMetadata(points, metadataFilters) {
    if (!metadataFilters || Object.keys(metadataFilters).length === 0) {
        return points;
    }
    
    // Debug: Show what filters are actually being applied
    Object.entries(metadataFilters).forEach(([fieldName, filterConfig]) => {
    });
    
    const normalizeValue = (value) => String(value ?? '').trim().toLowerCase();
    const toValueArray = (rawValue) => {
        if (rawValue === null || rawValue === undefined) return [];
        if (Array.isArray(rawValue)) {
            return rawValue.map(normalizeValue).filter(val => val.length > 0);
        }
        const normalized = normalizeValue(rawValue);
        return normalized ? [normalized] : [];
    };
    const toBoolean = (value) => {
        const normalized = normalizeValue(value);
        if (!normalized) return null;
        if (['true', 'yes', '1', 'y'].includes(normalized)) return true;
        if (['false', 'no', '0', 'n'].includes(normalized)) return false;
        return null;
    };

    return points.filter((point, index) => {
        // Debug first few points to see what's happening
        const isDebugPoint = index < 3;
        
        // Check if point matches all filters
        for (const [fieldName, filterConfig] of Object.entries(metadataFilters)) {
            const filterType = filterConfig.type;
            const filterValue = filterConfig.value;
            
            // Get the actual value from the point (same as text display)
            let actualValue = point[fieldName];
            if (actualValue === undefined && point.metadata) {
                actualValue = point.metadata[fieldName];
            }
            if (actualValue === undefined && point.data) {
                actualValue = point.data[fieldName];
            }
            
            if (isDebugPoint) {
            }
            
            // Skip if field doesn't exist on this point
            if (actualValue === null || actualValue === undefined || actualValue === '' ||
                (Array.isArray(actualValue) && actualValue.length === 0)) {
                if (isDebugPoint) {
                }
                return false;
            }
            
            // Apply type-specific filtering
            if (filterType === 'category') {
                // Handle both single values and arrays of values
                const selectedValues = toValueArray(filterValue);
                if (selectedValues.length === 0) continue;
                const candidateValues = toValueArray(actualValue);
                const matches = candidateValues.some(val => selectedValues.includes(val));
                if (!matches) {
                    if (isDebugPoint) {
                    }
                    return false;
                }
            } else if (filterType === 'number') {
                const actualNum = Number(actualValue);
                if (!Number.isFinite(actualNum)) return false;
                
                if (filterValue.min !== undefined) {
                    const minVal = Number(filterValue.min);
                    if (!Number.isFinite(minVal) || actualNum < minVal) {
                        return false;
                    }
                }
                if (filterValue.max !== undefined) {
                    const maxVal = Number(filterValue.max);
                    if (!Number.isFinite(maxVal) || actualNum > maxVal) {
                        return false;
                    }
                }
            } else if (filterType === 'boolean') {
                const expectedBool = toBoolean(filterValue);
                if (expectedBool === null) continue;
                const actualBool = toBoolean(actualValue);
                if (actualBool === null || actualBool !== expectedBool) {
                    return false;
                }
            } else if (filterType === 'text') {
                const searchText = normalizeValue(filterValue);
                const actualText = normalizeValue(actualValue);
                if (!actualText.includes(searchText)) {
                    return false;
                }
            } else if (filterType === 'date') {
                try {
                    const actualDate = new Date(actualValue);
                    if (filterValue.min) {
                        const minDate = new Date(filterValue.min);
                        if (actualDate < minDate) return false;
                    }
                    if (filterValue.max) {
                        const maxDate = new Date(filterValue.max);
                        if (actualDate > maxDate) return false;
                    }
                } catch (error) {
                    return false;
                }
            }
        }
        
        return true;
    });
}

function showFilterNotification(message, isCleared = false) {
    const banner = document.getElementById('filter-notification-banner');
    const textEl = document.getElementById('filter-notification-text');
    
    if (!banner || !textEl) return;
    
    // Update text and style
    textEl.textContent = message;
    banner.classList.remove('cleared', 'show');
    
    if (isCleared) {
        banner.classList.add('cleared');
    }
    
    // Show banner
    setTimeout(() => banner.classList.add('show'), 10);
    
    // Hide after 2.5 seconds
    setTimeout(() => {
        banner.classList.remove('show');
    }, 2500);
}

// Update RAG scope text based on current filter/lasso state
function updateRAGScopeTextNow() {
    const scopeText = document.getElementById('rag-scope-text');
    if (!scopeText) return;

    const lassoIndices = window.mainVisualization?.lassoSelectedIndices;
    const filteredIndices = window.mainVisualization?.metadataFilteredIndices;
    const totalCount = currentVisualizationData?.points?.length || 0;

    if (lassoIndices && lassoIndices.size > 0) {
        const count = lassoIndices.size;
        scopeText.textContent = `Analyzing ${count.toLocaleString()} document${count === 1 ? '' : 's'} from lasso selection`;
    } else if (filteredIndices && filteredIndices.size > 0) {
        const count = filteredIndices.size;
        scopeText.textContent = `Analyzing ${count.toLocaleString()} document${count === 1 ? '' : 's'} from filtered view`;
    } else {
        scopeText.textContent = `Analyzing ${totalCount.toLocaleString()} document${totalCount === 1 ? '' : 's'} from the entire dataset`;
    }
}
window.updateRAGScopeTextNow = updateRAGScopeTextNow;

async function applyMetadataFilters() {
    const filters = window.collectMetadataFiltersForSearch();
    activeMetadataFilters = filters;

    updateExportButtonVisibility();

    // Count active filters
    const filterCount = Object.keys(filters).length;

    // INSTANT UPDATE: Apply visual filter preview immediately for responsiveness
    applyInstantFilterPreview(filters);

    // Update RAG scope text immediately after filter preview is applied
    updateRAGScopeTextNow();

    // Update the persistent status bar immediately
    updateFilterStatusBar();

    // Show notification when filters are applied (clearing message handled elsewhere)
    if (filterCount > 0) {
        showFilterNotification(`Filters applied: ${filterCount} active`, false);
    }

    /* DISABLED: Backend sync not needed - frontend filtering works perfectly
    try {
        // BACKEND CONFIRMATION: Get authoritative filtered data from backend (optional)
        const backendStartTime = performance.now();
        const response = await fetch('/apply_metadata_filters', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filters: filters
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const backendTime = performance.now() - backendStartTime;
        // AUTHORITATIVE UPDATE: If backend provides filtered content, sync it. Otherwise keep preview state.
        const syncStartTime = performance.now();
        
        // If backend returns a filtered list, use it to refine UI; otherwise, keep instant preview
        if (data.text_content_list && Array.isArray(data.text_content_list)) {
            updateTextList(data.text_content_list);
            
            // Update visualization highlight from backend doc_ids
            const backendDocIds = new Set(data.text_content_list.map(item => item.doc_id));
            const matchingPoints = (currentVisualizationData.points || []).filter(point => 
                backendDocIds.has(point.doc_id) || backendDocIds.has(point.id)
            );
            const visibleIndices = new Set(matchingPoints.map(point => point.index));
            if (window.mainVisualization && window.mainVisualization.enableMetadataFilterMode) {
                window.mainVisualization.enableMetadataFilterMode(visibleIndices);
            }
        } else {
            // No authoritative list from backend; retain instant preview state
        }
        
        const syncTime = performance.now() - syncStartTime;
        // If there are active search results, re-apply them with metadata filters
        if (window.currentSearchResults || window.currentVisualizationSearchResults) {
            reapplySearchWithMetadataFilters();
        }

        // Update the persistent status bar
        updateFilterStatusBar();

    } catch (error) {
        console.error('❌ Error applying metadata filters:', error);
        showToast('Error applying filters. Please try again.', 'error');

        // ROLLBACK: If backend fails, revert to unfiltered state
        applyInstantFilterPreview({});
    }
    */ // End of disabled backend sync code

    // Update UI
    updateFilterCount();
    sendFiltersToBackend(filters);

    return filters;
}

// Helper function to debug data structure
function debugPointStructure() {
    if (currentVisualizationData && currentVisualizationData.points && currentVisualizationData.points.length > 0) {
        const firstPoint = currentVisualizationData.points[0];
        // Look for sentiment field specifically
        const sentimentLocations = [];
        if (firstPoint.sentiment !== undefined) sentimentLocations.push('point.sentiment');
        if (firstPoint.metadata && firstPoint.metadata.sentiment !== undefined) sentimentLocations.push('point.metadata.sentiment');
        if (firstPoint.data && firstPoint.data.sentiment !== undefined) sentimentLocations.push('point.data.sentiment');
        
    }
}

// INSTANT FILTERING: Apply visual preview immediately for responsiveness  
function applyInstantFilterPreview(filters) {
    if (!window.mainVisualization || !currentVisualizationData || !currentVisualizationData.points) {
        return;
    }
    
    const normalizeValue = (value) => String(value ?? '').trim().toLowerCase();
    const toValueArray = (rawValue) => {
        if (rawValue === null || rawValue === undefined) return [];
        if (Array.isArray(rawValue)) {
            return rawValue.map(normalizeValue).filter(val => val.length > 0);
        }
        const normalized = normalizeValue(rawValue);
        return normalized ? [normalized] : [];
    };
    const toBoolean = (value) => {
        const normalized = normalizeValue(value);
        if (!normalized) return null;
        if (['true', 'yes', '1', 'y'].includes(normalized)) return true;
        if (['false', 'no', '0', 'n'].includes(normalized)) return false;
        return null;
    };

    // Debug the data structure on first run
    if (Object.keys(filters).length > 0) {
        debugPointStructure();
    }
    
    const hasFilters = filters && Object.keys(filters).length > 0;
    
    if (hasFilters) {
        // Apply instant client-side filtering for immediate visual feedback
        const filteredPoints = currentVisualizationData.points.filter(point => {
            // Check if point matches all filters
            for (const [fieldName, filterConfig] of Object.entries(filters)) {
                if (!filterConfig || !filterConfig.value) continue;
                
                const filterType = filterConfig.type;
                const filterValue = filterConfig.value;
                
                // Look for field in multiple locations: direct property, metadata, or nested
                let actualValue = point[fieldName];
                if (actualValue === undefined && point.metadata) {
                    actualValue = point.metadata[fieldName];
                }
                if (actualValue === undefined && point.data) {
                    actualValue = point.data[fieldName];
                }
                
                // Debug: Log field lookup for troubleshooting (use a counter since filteredPoints is being built)
                if (currentVisualizationData.points.indexOf(point) < 3) {
                }
                
                // Quick filtering logic (same as DataManager but optimized for speed)
                if (filterType === 'category') {
                    const selectedValues = toValueArray(filterValue);
                    if (selectedValues.length === 0) continue;
                    const candidateValues = toValueArray(actualValue);
                    const matches = candidateValues.some(value => selectedValues.includes(value));
                    if (!matches) return false;
                } else if (filterType === 'text') {
                    const searchText = normalizeValue(filterValue);
                    const actualText = normalizeValue(actualValue);
                    if (!actualText.includes(searchText)) return false;
                } else if (filterType === 'number') {
                    const actualNum = Number(actualValue);
                    if (!Number.isFinite(actualNum)) return false;
                    const minVal = filterValue.min !== undefined ? Number(filterValue.min) : -Infinity;
                    const maxVal = filterValue.max !== undefined ? Number(filterValue.max) : Infinity;
                    if ((filterValue.min !== undefined && !Number.isFinite(minVal)) ||
                        (filterValue.max !== undefined && !Number.isFinite(maxVal))) {
                        return false;
                    }
                    if (actualNum < minVal || actualNum > maxVal) return false;
                } else if (filterType === 'boolean') {
                    const expectedBool = toBoolean(filterValue);
                    if (expectedBool === null) continue;
                    const actualBool = toBoolean(actualValue);
                    if (actualBool === null) return false;
                    if (actualBool !== expectedBool) return false;
                }
            }
            return true;
        });
        
        // Create instant visual filtering effect
        const visibleIndices = new Set(filteredPoints.map(point => point.index));

        // Apply instant visual filtering
        if (window.mainVisualization && window.mainVisualization.enableMetadataFilterMode) {
            window.mainVisualization.enableMetadataFilterMode(visibleIndices);
        }
        
        // Update text list instantly with filtered points (preserve original indices)
        updateTextList(filteredPoints);
        
    } else {
        // Clear all filters
        if (window.mainVisualization.disableMetadataFilterMode) {
            window.mainVisualization.disableMetadataFilterMode();
        }
        
        // Show all data
        if (currentVisualizationData.points) {
            updateTextList(currentVisualizationData.points);
        }
    }
}

// BACKEND SYNC: Synchronize with authoritative backend data 
async function syncVisualizationWithBackend() {
    try {
        const response = await fetch('/visualization_data?refresh=true', {
            method: 'GET',
            headers: {'Content-Type': 'application/json'}
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            return;
        }
        
        // Update current data with authoritative backend data
        currentVisualizationData = data;
        
        // Re-render visualization with authoritative data (this may correct any preview inaccuracies)
        if (window.mainVisualization && data.points) {
            window.mainVisualization.loadData(data.points);
        }
        
    } catch (error) {
        console.error('❌ Error syncing with backend:', error);
    }
}

// DEPRECATED: Replace with instant preview approach
// New function to refresh visualization data from backend with current filters
async function refreshVisualizationWithFilters() {
    try {
        const response = await fetch('/visualization_data?refresh=true', {
            method: 'GET',
            headers: {'Content-Type': 'application/json'}
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            return;
        }
        
        // Update the current visualization data with backend-filtered data
        currentVisualizationData = data;
        
        // Re-render the visualization with the updated data
        if (window.mainVisualization && data.points) {
            window.mainVisualization.loadData(data.points);
        }
        
    } catch (error) {
        console.error('❌ Error refreshing visualization data:', error);
    }
}

function updateVisualizationWithMetadataFilters(filters) {
    if (!window.mainVisualization || !currentVisualizationData) return;
    
    const hasFilters = filters && Object.keys(filters).length > 0;
    
    if (hasFilters) {
        // Backend has already filtered the data - just show which points match
        // The filtered data is already in currentVisualizationData.points from refreshVisualizationWithFilters()
        const visibleIndices = new Set(currentVisualizationData.points.map(point => point.index));
        
        // Use the existing search highlighting system but for metadata filtering
        window.mainVisualization.enableMetadataFilterMode(visibleIndices);
    } else {
        // Disable metadata filter mode
        window.mainVisualization.disableMetadataFilterMode();
    }
}

function reapplySearchWithMetadataFilters() {
    // If there are active search results, we need to perform a new search
    // with the current metadata filters to get updated results
    const searchInput = document.getElementById('search-input');
    const searchTypeSelect = document.getElementById('search-type');
    
    if (searchInput && searchInput.value.trim()) {
        // Trigger a new search which will include the metadata filters
        performSearch();
    }
}

function clearMetadataFilters() {
    // Clear all filter inputs
    const filterElements = document.querySelectorAll('.metadata-filter-item input, .metadata-filter-item select');
    filterElements.forEach(element => {
        if (element.tagName === 'SELECT') {
            // For multi-select, clear all selections
            if (element.multiple) {
                Array.from(element.options).forEach(option => option.selected = false);
            } else {
                element.selectedIndex = 0;
            }
        } else if (element.type === 'checkbox') {
            element.checked = false;
        } else if (element.type === 'radio') {
            element.checked = false;
        } else {
            element.value = '';
        }
    });
    
    activeMetadataFilters = {};
    // Ensure any in-flight preview state is reset immediately
    if (typeof applyInstantFilterPreview === 'function') {
        applyInstantFilterPreview({});
    }

    // Reset visualization to show all points
    if (currentVisualizationData && currentVisualizationData.points) {
        unlockTextList('filters cleared');
        updateTextList(currentVisualizationData.points, { force: true });
        // Clear metadata filtering from visualization
        updateVisualizationWithMetadataFilters({});
    }
    
    // Clear any visualization highlights
    if (window.mainVisualization) {
        window.mainVisualization.clearHighlight();
    }
    
    // Send empty filters to backend
    sendFiltersToBackend({});
    
    // Update UI
    updateFilterCount();
    updateFilterStatusBar();
    
    // Show notification
    showFilterNotification('All filters cleared', true);

    // Also show toast for better visibility
    if (typeof showToast === 'function') {
        showToast('All filters cleared', 'success');
    }

    // Update RAG scope text immediately
    updateRAGScopeTextNow();

    updateExportButtonVisibility();
}

// Expose globally for console testing
window.clearAllFilters = clearMetadataFilters;
window.applyFilters = applyMetadataFilters;

// Expose function to reinitialize button handlers if needed
window.reinitFilterButtons = function() {
    if (typeof attachFilterButtonHandlers === 'function') {
        attachFilterButtonHandlers();
    }
    if (typeof setupFilterStatusBar === 'function') {
        setupFilterStatusBar();
    }
};

// Expose attachFilterButtonHandlers globally for debugging
window.attachFilterButtonHandlers = attachFilterButtonHandlers;

// Debug function - expose globally
window.debugVectoriaState = function() {
};

function sendFiltersToBackend(filters) {
    // Send the filters to the backend
    fetch('/api/set-metadata-filters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filters: filters })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
        } else {
            console.error('Failed to apply metadata filters on backend:', data.error);
        }
    })
    .catch(error => {
        console.error('Error sending metadata filters to backend:', error);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make metadata functions globally available
window.initializeMetadataSystem = initializeMetadataSystem;
window.analyzeFileForMetadata = analyzeFileForMetadata;
window.applyMetadataFilters = applyMetadataFilters;
window.clearMetadataFilters = clearMetadataFilters;

// Debug functions for metadata system
window.debugMetadata = function() {
    if (currentVisualizationData && currentVisualizationData.points) {
        const excludedKeys = ['index', 'x', 'y', 'cluster', 'text', 'cluster_probability'];
        const metadataKeys = Object.keys(currentVisualizationData.points[0] || {})
            .filter(key => !excludedKeys.includes(key));
        // Show sample values for each metadata key
        metadataKeys.forEach(key => {
            const sampleValues = currentVisualizationData.points
                .slice(0, 5)
                .map(point => point[key])
                .filter(value => value !== null && value !== undefined && value !== '');
        });
    }
};

window.testMetadataFiltering = function() {
    if (currentVisualizationData && currentVisualizationData.points) {
        // Show sample point structure
        if (currentVisualizationData.points.length > 0) {
            const samplePoint = currentVisualizationData.points[0];
            // Show metadata fields (excluding system fields)
            const excludedKeys = ['index', 'x', 'y', 'cluster', 'text', 'cluster_probability', 'cluster_color', 'cluster_name', 'doc_id', 'chunk_id', 'cluster_keyword_scores', 'cluster_keywords', 'cluster_keywords_viz', 'color', 'filename', 'original_filename', 'processing_type', 'selected_column', 'num_chunks'];
            const metadataKeys = Object.keys(samplePoint).filter(key => !excludedKeys.includes(key));
            // Show sample values
            metadataKeys.slice(0, 5).forEach(key => {
            });
        }
        
        // Try to extract metadata again
        const success = extractMetadataFromVisualizationData();
        return {
            totalPoints: currentVisualizationData.points.length,
            detectedFields: detectedMetadataFields.length,
            fields: detectedMetadataFields.map(f => ({name: f.name, type: f.type, uniqueCount: f.uniqueValues.length}))
        };
    } else {
        return null;
    }
};

window.testMetadataIntegration = function() {
    // Test 1: Metadata extraction
    const extractionResult = window.testMetadataFiltering();
    if (!extractionResult) {
        return false;
    }
    // Test 2: UI generation
    const filtersGrid = document.getElementById('metadata-filters-grid');
    if (filtersGrid && filtersGrid.children.length > 0) {
    } else {
    }
    
    // Test 3: Filter collection
    const filters = window.collectMetadataFiltersForSearch ? window.collectMetadataFiltersForSearch() : {};
    // Test 4: Visualization connection
    if (window.mainVisualization && typeof window.mainVisualization.enableMetadataFilterMode === 'function') {
    } else {
    }
    
    // Test 5: Search integration
    const searchFunctions = {
        performSearch: typeof window.performSearch === 'function',
        fastSearch: window.globalSearchInterface && typeof window.globalSearchInterface.matchesMetadataFilters === 'function',
        collectFilters: typeof window.collectMetadataFiltersForSearch === 'function'
    };
    
    const searchIntegrationOk = Object.values(searchFunctions).every(fn => fn);
    if (searchIntegrationOk) {
    } else {
    }
    
    return {
        extraction: !!extractionResult,
        uiGeneration: filtersGrid && filtersGrid.children.length > 0,
        filterCollection: typeof window.collectMetadataFiltersForSearch === 'function',
        visualization: window.mainVisualization && typeof window.mainVisualization.enableMetadataFilterMode === 'function',
        search: searchIntegrationOk,
        overall: extractionResult && searchIntegrationOk
    };
};
window.getActiveMetadataFilters = () => activeMetadataFilters;
window.getDetectedMetadataFields = () => detectedMetadataFields;

// Debug function to check filter collection
window.debugFilterCollection = function() {
    const filterElements = document.querySelectorAll('.metadata-filter-item');
    filterElements.forEach((filterItem, index) => {
        const field = filterItem.dataset.field;
        const type = filterItem.dataset.type;
        const label = filterItem.querySelector('label')?.textContent;
        
        if (type === 'category') {
            const select = filterItem.querySelector('select');
            if (select) {
                const selectedOptions = Array.from(select.selectedOptions);
                selectedOptions.forEach(option => {
                });
                
                // Show if first option is empty (should be "All")
                if (select.options.length > 0) {
                    const firstOption = select.options[0];
                }
            }
        }
    });
    
    // Test the collection function
    const collectedFilters = window.collectMetadataFiltersForSearch();
};

// Debug function to diagnose metadata filtering issues
window.debugMetadataFilteringIssues = function() {
    // Step 1: Check if visualization data is available
    if (!currentVisualizationData || !currentVisualizationData.points) {
        return { issue: 'no_visualization_data' };
    }
    // Step 2: Check metadata extraction
    const samplePoint = currentVisualizationData.points[0];
    const excludedKeys = ['index', 'x', 'y', 'cluster', 'text', 'cluster_probability', 'cluster_color', 'cluster_name', 'doc_id', 'chunk_id', 'cluster_keyword_scores', 'cluster_keywords', 'cluster_keywords_viz', 'color', 'filename', 'original_filename', 'processing_type', 'selected_column', 'num_chunks'];
    const metadataKeys = Object.keys(samplePoint).filter(key => !excludedKeys.includes(key));
    if (metadataKeys.length === 0) {
        return { issue: 'no_metadata_fields' };
    }
    
    // Step 3: Check if metadata was extracted
    if (detectedMetadataFields.length === 0) {
        extractMetadataFromVisualizationData();
    }
    
    // Step 4: Check UI elements
    const toggleContainer = document.getElementById('metadata-filters-toggle');
    const filtersSection = document.getElementById('metadata-filters-section');
    const filtersGrid = document.getElementById('metadata-filters-grid');
    
    if (filtersGrid) {
    }
    
    // Step 5: Inline filters are always visible
    // Step 6: Test filter collection
    if (typeof window.collectMetadataFiltersForSearch === 'function') {
        const filters = window.collectMetadataFiltersForSearch();
    } else {
    }
    
    return {
        visualizationData: !!currentVisualizationData,
        metadataFields: metadataKeys.length,
        detectedFields: detectedMetadataFields.length,
        uiElements: !!(toggleContainer && filtersSection && filtersGrid),
        toggleVisible: toggleContainer && toggleContainer.style.display !== 'none'
    };
};

// Define RAG metadata population function GLOBALLY (outside DOM-dependent init)
async function populateRAGMetadataFields() {
    const chipsContainer = document.getElementById('rag-metadata-chips');
    const contextContainer = document.getElementById('rag-metadata-context');
    const fieldSelectEl = document.getElementById('rag-metadata-fields');
    const includeCheckbox = document.getElementById('rag-include-metadata');
    if (!chipsContainer || !fieldSelectEl || !includeCheckbox) {
        console.warn('⚠️ RAG metadata chip elements not found');
        return;
    }

    let fields = [];

    // First try detectedMetadataFields (most reliable)
    if (detectedMetadataFields && detectedMetadataFields.length) {
        fields = detectedMetadataFields.map(f => f.name || f.displayName);
    }
    // Try backend schema second
    else {
        const schema = await loadMetadataSchemaFromBackend();
        if (schema && schema.fields) {
            fields = Object.keys(schema.fields);
        } else if (schema && typeof schema === 'object') {
            fields = Object.keys(schema);
        }
    }

    // Fallback: inspect currentVisualizationData
    if ((!fields || fields.length === 0) && window.currentVisualizationData && window.currentVisualizationData.points && window.currentVisualizationData.points.length > 0) {
        const excluded = new Set(['index','x','y','cluster','text','cluster_probability','cluster_color','cluster_name','doc_id','chunk_id','cluster_keyword_scores','cluster_keywords','cluster_keywords_viz','color','filename','original_filename','processing_type','selected_column','num_chunks','metadata']);
        const sample = window.currentVisualizationData.points[0] || {};
        fields = Object.keys(sample).filter(k => !excluded.has(k));
    }

    if (fields && fields.length > 0) {
        fields.sort();
        // Restore saved selection from localStorage
        let saved = [];
        try {
            const raw = localStorage.getItem('vectoria_rag_metadata_fields');
            if (raw) saved = JSON.parse(raw);
        } catch(e) {}
        const savedSet = new Set(saved);

        chipsContainer.innerHTML = '';
        fieldSelectEl.innerHTML = '';
        fields.forEach(name => {
            // Create chip button
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'rag-context-chip';
            chip.dataset.field = name;
            chip.textContent = name;
            if (savedSet.has(name)) chip.classList.add('active');
            chip.addEventListener('click', () => {
                chip.classList.toggle('active');
                syncChipsToHiddenElements();
            });
            chipsContainer.appendChild(chip);

            // Create backing option
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (savedSet.has(name)) opt.selected = true;
            fieldSelectEl.appendChild(opt);
        });

        // Sync initial state
        syncChipsToHiddenElements();

        if (contextContainer) contextContainer.style.display = 'flex';
    } else {
        chipsContainer.innerHTML = '';
        if (contextContainer) contextContainer.style.display = 'none';
        includeCheckbox.checked = false;
    }

    function syncChipsToHiddenElements() {
        const activeFields = [];
        chipsContainer.querySelectorAll('.rag-context-chip.active').forEach(c => {
            activeFields.push(c.dataset.field);
        });
        // Sync hidden checkbox
        includeCheckbox.checked = activeFields.length > 0;
        // Sync hidden select
        Array.from(fieldSelectEl.options).forEach(opt => {
            opt.selected = activeFields.includes(opt.value);
        });
        // Persist to localStorage
        try { localStorage.setItem('vectoria_rag_metadata_fields', JSON.stringify(activeFields)); } catch (_) {}
    }
}

// Expose globally
window.populateRAGMetadataFields = populateRAGMetadataFields;
// Initialize RAG metadata field selection UI
function initializeRAGMetadataControls() {
    const searchType = document.getElementById('search-type');
    const controls = document.getElementById('rag-metadata-controls');
    const fieldSelect = document.getElementById('rag-metadata-fields');
    const includeBox = document.getElementById('rag-include-metadata');
    
    // If elements are missing, we still expose the function but can't set up handlers
    if (!searchType || !controls || !fieldSelect || !includeBox) {
        console.warn('⚠️ Missing RAG metadata control elements - handlers not set up yet');
        return;
    }

    const handleIncludeToggle = () => {
        fieldSelect.disabled = !includeBox.checked;
        if (!includeBox.checked) {
            fieldSelect.selectedIndex = -1;
        }
    };

    if (!includeBox.dataset.listenerAttached) {
        includeBox.addEventListener('change', handleIncludeToggle);
        includeBox.dataset.listenerAttached = 'true';
    }

    // Show/hide on search type changes
    function updateVisibility() {
        const searchMode = searchType.value;
        // Show metadata controls only for RAG (Semantic Search RAG)
        const showControls = searchMode === 'rag';
        controls.hidden = !showControls;
        if (showControls) {
            controls.style.display = 'flex';
            // Always try to populate when switching to RAG mode
            populateRAGMetadataFields();
            handleIncludeToggle();
        } else {
            controls.style.display = 'none';
        }
    }

    searchType.addEventListener('change', updateVisibility);
    updateVisibility();
}

// ============================================================================
// Quick Settings Panel for Explore Tab
// ============================================================================

// Quick settings modal initialization moved to fast-search.js
// See initializeQuickSettingsModal() in fast-search.js

// ============================================================================
// Export/Import Handlers
// ============================================================================

/**
 * Show/hide the "Export data" button.
 * Visible only when showing the full unfiltered dataset (no search, lasso, filter, or RAG active).
 */
function updateExportButtonVisibility() {
    const btn = document.getElementById('export-selection-btn');
    if (!btn) return;

    const lassoActive = window.__textListLock === 'lasso';
    const searchActive = window.currentSearchResults && Array.isArray(window.currentSearchResults.results) && window.currentSearchResults.results.length > 0;
    const filtersActive = typeof activeMetadataFilters !== 'undefined' && activeMetadataFilters && Object.keys(activeMetadataFilters).length > 0;
    const ragCard = document.getElementById('rag-answer-card');
    const ragActive = ragCard && ragCard.style.display !== 'none';
    const detailView = document.getElementById('selected-text-view');
    const detailActive = detailView && detailView.style.display !== 'none';
    const hasData = (window.browserML && window.browserML.pipeline) ||
        (Array.isArray(window.currentDataset) && window.currentDataset.length > 0) ||
        (Array.isArray(window.canvasData) && window.canvasData.length > 0) ||
        (window.currentVisualizationData && Array.isArray(window.currentVisualizationData.points) && window.currentVisualizationData.points.length > 0);

    if (hasData && !lassoActive && !searchActive && !filtersActive && !ragActive && !detailActive) {
        btn.style.display = '';
        btn.disabled = false;
    } else {
        btn.style.display = 'none';
        btn.disabled = true;
    }
}

/**
 * Initialize export/import buttons
 */
function initializeExportImportHandlers() {
    // Export Button (full dataset only)
    const exportSelectionBtn = document.getElementById('export-selection-btn');
    if (exportSelectionBtn) {
        exportSelectionBtn.addEventListener('click', async () => {
            // Prefer pipeline's full dataset, fall back to visualization points
            const pipeline = window.browserML?.pipeline;
            if (pipeline && pipeline.currentDataset && typeof exportFullDataset === 'function') {
                try {
                    await exportFullDataset();
                } catch (error) {
                    console.error('Export failed:', error);
                }
                return;
            }

            // Fallback: export from visualization data
            const allDocs = getFilteredDocuments();
            if (!allDocs || allDocs.length === 0) {
                showToast('No documents to export', 'warning');
                return;
            }

            if (typeof exportSelection !== 'function') {
                console.error('❌ exportSelection function not found!');
                showToast('Export function not available. Please refresh the page.', 'error');
                return;
            }

            await exportSelection(allDocs, 'full', {});
        });
    }

    // Export Full Dataset Button
    const exportDatasetBtn = document.getElementById('export-dataset-btn');
    if (exportDatasetBtn) {
        exportDatasetBtn.addEventListener('click', async () => {
            if (!window.browserML || !window.browserML.pipeline) {
                showToast('No processed dataset available', 'warning');
                return;
            }

            if (typeof exportFullDataset !== 'function') {
                console.error('❌ exportFullDataset function not found!');
                showToast('Export function not available. Please refresh the page.', 'error');
                return;
            }

            try {
                await exportFullDataset();
            } catch (error) {
                console.error('Export failed:', error);
            }
        });
    }

    // Import Dataset Button
    const importDatasetBtn = document.getElementById('import-dataset-btn');
    const importDatasetFile = document.getElementById('import-dataset-file');

    if (importDatasetBtn && importDatasetFile) {
        importDatasetBtn.addEventListener('click', () => {
            importDatasetFile.click();
        });

        importDatasetFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            if (!window.browserML || !window.browserML.pipeline) {
                showToast('Browser ML not initialized', 'error');
                e.target.value = '';
                return;
            }

            if (typeof importDataset !== 'function') {
                console.error('❌ importDataset function not found!');
                showToast('Import function not available. Please refresh the page.', 'error');
                e.target.value = '';
                return;
            }

            try {
                const result = await importDataset(file);

                if (result.success) {
                    // Update visualization with loaded data
                    await updateVisualizationWithLoadedData(result.data);

                    // Rebuild metadata filters and RAG controls from loaded snapshot
                    if (window.loadMetadataFromProcessedData) {
                        try {
                            await window.loadMetadataFromProcessedData();
                        } catch (err) {
                            console.error('Metadata load failed after import:', err);
                        }
                    }

                    if (window.populateRAGMetadataFields) {
                        try {
                            await window.populateRAGMetadataFields();
                        } catch (err) {
                            console.error('RAG metadata population failed after import:', err);
                        }
                    }

                    // Switch to explore tab
                    activateTab('explore-tab');

                }
            } catch (error) {
                console.error('Import failed:', error);
                showToast(`Import failed: ${error.message}`, 'error');
            }

            // Reset file input
            e.target.value = '';
        });
    }

    // HyDE Viewer Button
    const hydeViewerBtn = document.getElementById('hyde-viewer-btn');
    if (hydeViewerBtn) {
        hydeViewerBtn.addEventListener('click', () => {
            const hydeModal = document.getElementById('hyde-modal');
            const hydeContent = document.getElementById('hyde-answer-content');

            if (window.lastHyDEAnswer) {
                hydeContent.textContent = window.lastHyDEAnswer;
                hydeModal.style.display = 'flex';
            } else {
                showToast('HyDE answer not available', 'warning');
            }
        });
    }

}

/**
 * Get currently filtered/displayed documents
 */
function getFilteredDocuments() {
    if (Array.isArray(window.currentDataset) && window.currentDataset.length > 0) {
        return window.currentDataset;
    }

    if (Array.isArray(window.canvasData) && window.canvasData.length > 0) {
        return window.canvasData;
    }

    if (window.currentVisualizationData && Array.isArray(window.currentVisualizationData.points) && window.currentVisualizationData.points.length > 0) {
        return window.currentVisualizationData.points;
    }

    return [];
}

/**
 * Update visualization with loaded dataset
 */
async function updateVisualizationWithLoadedData(vizData) {
    // Update metadata schema
    if (vizData.metadataSchema) {
        window.metadataSchema = vizData.metadataSchema;
    }

    // Clear any stale visualization data so loadVisualizationData() goes through
    // the fetch interception path which builds proper points via
    // getBrowserVisualizationData() (with cluster_color, cluster_name, etc.)
    currentVisualizationData = null;
    window.currentVisualizationData = null;

    // Populate filters
    if (typeof populateFilterPanel === 'function') {
        populateFilterPanel();
    }

    updateExportButtonVisibility();
}

function spawnLogoBurst(logo, intensity) {
    const burst = document.createElement('span');
    burst.className = 'logo-burst';

    const size = 14 + intensity * 18;
    const width = Math.max(logo.clientWidth, 40);
    const height = Math.max(logo.clientHeight, 24);
    const x = 10 + Math.random() * Math.max(width - size - 10, 10);
    const y = height * (0.3 + Math.random() * 0.4);
    const rotation = (Math.random() * 36 - 18);
    const endRotation = rotation * (1.2 + intensity * 0.6);

    burst.style.setProperty('--logo-burst-size', `${size.toFixed(1)}px`);
    burst.style.setProperty('--logo-burst-x', `${x.toFixed(1)}px`);
    burst.style.setProperty('--logo-burst-y', `${y.toFixed(1)}px`);
    burst.style.setProperty('--logo-burst-rotate', `${rotation.toFixed(1)}deg`);
    burst.style.setProperty('--logo-burst-rotate-end', `${endRotation.toFixed(1)}deg`);
    burst.style.setProperty('--logo-burst-opacity', `${(0.35 + intensity * 0.5).toFixed(2)}`);

    logo.appendChild(burst);
    burst.addEventListener('animationend', () => burst.remove());
}

function initLogoEasterEgg() {
    const logo = document.querySelector('header .logo');
    if (!logo) return;

    if (logo.dataset.logoEasterEggBound === 'true') return;
    logo.dataset.logoEasterEggBound = 'true';

    if (!logo.hasAttribute('role')) {
        logo.setAttribute('role', 'button');
    }
    if (!logo.hasAttribute('tabindex')) {
        logo.tabIndex = 0;
    }

    let streak = 0;
    let lastClick = 0;
    let resetTimer = null;

    logo.addEventListener('click', () => {
        const now = performance.now();
        if (now - lastClick < 550) {
            streak += 1;
        } else {
            streak = 1;
        }
        lastClick = now;

        const level = Math.min(streak, 12);
        const intensity = level / 12;
        const scale = 1 + intensity * 0.6;
        const x = (Math.random() - 0.5) * (6 + intensity * 16);
        const y = -10 - intensity * 20;
        const rotate = (Math.random() - 0.5) * (6 + intensity * 14);
        const shadowY = Math.round(4 + intensity * 10);
        const shadowBlur = Math.round(12 + intensity * 26);
        const shadowAlpha = (0.18 + intensity * 0.18).toFixed(2);

        logo.style.setProperty('--logo-pop-scale', scale.toFixed(3));
        logo.style.setProperty('--logo-pop-x', `${x.toFixed(1)}px`);
        logo.style.setProperty('--logo-pop-y', `${y.toFixed(1)}px`);
        logo.style.setProperty('--logo-pop-x-back', `${(-x * 0.45).toFixed(1)}px`);
        logo.style.setProperty('--logo-pop-y-back', `${(-y * 0.2).toFixed(1)}px`);
        logo.style.setProperty('--logo-pop-scale-back', `${(1 + scale) / 2}`);
        logo.style.setProperty('--logo-pop-rotate', `${rotate.toFixed(1)}deg`);
        logo.style.setProperty('--logo-pop-shadow', `0 ${shadowY}px ${shadowBlur}px rgba(0, 0, 0, ${shadowAlpha})`);

        logo.classList.remove('logo-pop');
        void logo.offsetWidth;
        logo.classList.add('logo-pop');

        if (level >= 4) {
            spawnLogoBurst(logo, intensity);
        }

        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => {
            streak = 0;
        }, 700);
    });

    logo.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            logo.click();
        }
    });
}

function scheduleLogoEasterEgg() {
    const start = () => {
        try {
            initLogoEasterEgg();
        } catch (error) {
            console.warn('Logo easter egg failed to initialize:', error);
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start, { once: true });
    } else {
        start();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeExportImportHandlers();
    updateModelSetupModalModelNames();
});
scheduleLogoEasterEgg();

// ============================================================================
// End of vectoria.js - Log completion
// ============================================================================
