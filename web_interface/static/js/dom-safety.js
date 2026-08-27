(function initializeVectoriaDOMSafety(root) {
    const HTML_ENTITIES = Object.freeze({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    });
    const SAFE_COLOR_FALLBACK = /^(?:#[\da-f]{3,4}|#[\da-f]{6}|#[\da-f]{8}|[a-z]+|rgba?\(\s*[\d.%]+\s*,\s*[\d.%]+\s*,\s*[\d.%]+(?:\s*,\s*[\d.%]+)?\s*\)|hsla?\(\s*[-\d.]+(?:deg)?\s*,\s*[\d.]+%\s*,\s*[\d.]+%(?:\s*,\s*[\d.%]+)?\s*\))$/i;

    function escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, character => HTML_ENTITIES[character]);
    }

    function safeColor(value, fallback = '#9CA3AF') {
        const candidate = typeof value === 'string' ? value.trim() : '';
        if (!candidate || candidate.length > 80 || /["'<>;{}]/.test(candidate)) return fallback;
        if (typeof root.CSS?.supports === 'function') {
            return root.CSS.supports('color', candidate) ? candidate : fallback;
        }
        return SAFE_COLOR_FALLBACK.test(candidate) ? candidate : fallback;
    }

    root.VectoriaDOM = Object.freeze({ escapeHTML, safeColor });
})(typeof window !== 'undefined' ? window : globalThis);
