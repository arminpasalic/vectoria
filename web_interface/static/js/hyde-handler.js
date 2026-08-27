/**
 * HyDE (Hypothetical Document Embeddings) Handler
 * Manages HyDE mode toggle and workflow
 */

// Global HyDE state
window.hydeMode = {
    enabled: false,
    currentQuery: null,
    generatedText: null,
    resolveCallback: null,
    rejectCallback: null,
    resultMode: 'legacy',
    previousFocus: null,
    inertElements: []
};

/**
 * Initialize HyDE mode handlers
 */
function initializeHyDEHandlers() {
    const toggleBtn = document.getElementById('toggle-hyde-mode');
    const indicator = document.getElementById('hyde-mode-indicator');
    const modal = document.getElementById('hyde-review-modal');
    const closeBtn = document.getElementById('close-hyde-modal');
    const cancelBtn = document.getElementById('hyde-cancel');
    const withoutBtn = document.getElementById('hyde-without');
    const searchBtn = document.getElementById('hyde-search');
    const originalQueryDiv = document.getElementById('hyde-original-query');
    const generatedTextarea = document.getElementById('hyde-generated-text');

    if (!toggleBtn || !modal) {
        console.warn('⚠️ HyDE elements not found');
        return;
    }

    // Toggle HyDE mode
    toggleBtn.addEventListener('click', () => {
        window.hydeMode.enabled = !window.hydeMode.enabled;

        if (window.hydeMode.enabled) {
            toggleBtn.classList.add('active');
            toggleBtn.style.background = 'var(--button-primary-bg)';
            toggleBtn.style.color = 'white';
            if (indicator) indicator.style.display = 'block';

            if (typeof showToast === 'function') {
                showToast('HyDE Mode enabled: Queries will generate hypothetical answers first', 'info');
            }
        } else {
            toggleBtn.classList.remove('active');
            toggleBtn.style.background = '';
            toggleBtn.style.color = '';
            if (indicator) indicator.style.display = 'none';

            if (typeof showToast === 'function') {
                showToast('HyDE Mode disabled', 'info');
            }
        }

        // Sync with new RAG UI toggle
        const hydeModeToggle = document.getElementById('hyde-mode-toggle');
        if (hydeModeToggle) {
            hydeModeToggle.checked = window.hydeMode.enabled;
        }

    });

    if (modal.dataset.hydeInitialized === 'true') return;
    modal.dataset.hydeInitialized = 'true';

    const restoreBackground = () => {
        for (const item of window.hydeMode.inertElements || []) item.element.inert = item.wasInert;
        window.hydeMode.inertElements = [];
        const focusTarget = window.hydeMode.previousFocus;
        window.hydeMode.previousFocus = null;
        if (focusTarget && typeof focusTarget.focus === 'function') focusTarget.focus();
    };

    const settleModal = (action = 'cancel') => {
        modal.style.display = 'none';
        const detailed = window.hydeMode.resultMode === 'detailed';
        const resolve = window.hydeMode.resolveCallback;
        const reject = window.hydeMode.rejectCallback;
        window.hydeMode.resolveCallback = null;
        window.hydeMode.rejectCallback = null;
        restoreBackground();
        if (detailed && resolve) {
            resolve({ action });
        } else if (reject) {
            reject(new Error('HyDE cancelled by user'));
        }
    };

    if (closeBtn) closeBtn.addEventListener('click', () => settleModal('cancel'));
    if (cancelBtn) cancelBtn.addEventListener('click', () => settleModal('cancel'));

    if (withoutBtn) {
        withoutBtn.addEventListener('click', () => {
            modal.style.display = 'none';
            const resolve = window.hydeMode.resolveCallback;
            window.hydeMode.resolveCallback = null;
            window.hydeMode.rejectCallback = null;
            restoreBackground();
            resolve?.({ action: 'without_hyde' });
        });
    }

    // Search with HyDE text
    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            const editedText = generatedTextarea.value.trim();

            if (!editedText) {
                if (typeof showToast === 'function') {
                    showToast('Cannot search with empty text', 'error');
                }
                return;
            }

            modal.style.display = 'none';
            const resolve = window.hydeMode.resolveCallback;
            const detailed = window.hydeMode.resultMode === 'detailed';
            window.hydeMode.resolveCallback = null;
            window.hydeMode.rejectCallback = null;
            restoreBackground();
            resolve?.(detailed ? { action: 'use', text: editedText } : editedText);
        });
    }

    modal.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
            event.preventDefault();
            settleModal('cancel');
            return;
        }
        if (event.key !== 'Tab') return;
        const focusable = [...modal.querySelectorAll('button:not([disabled]):not([hidden]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])')]
            .filter(element => element.offsetParent !== null);
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    });

}

/**
 * Show HyDE review modal
 * @param {string} originalQuery - User's original question
 * @param {string} generatedText - LLM-generated hypothetical answer
 * @returns {Promise<string>} Edited/approved text to use for search
 */
function showHyDEReviewModal(originalQuery, generatedText, options = {}) {
    return new Promise((resolve, reject) => {
        const modal = document.getElementById('hyde-review-modal');
        const originalQueryDiv = document.getElementById('hyde-original-query');
        const generatedTextarea = document.getElementById('hyde-generated-text');

        if (!modal || !originalQueryDiv || !generatedTextarea) {
            reject(new Error('HyDE modal elements not found'));
            return;
        }

        // Store callbacks
        window.hydeMode.currentQuery = originalQuery;
        window.hydeMode.generatedText = generatedText;
        window.hydeMode.resolveCallback = resolve;
        window.hydeMode.rejectCallback = reject;
        window.hydeMode.resultMode = options?.detailed === true ? 'detailed' : 'legacy';
        window.hydeMode.previousFocus = document.activeElement;
        window.hydeMode.inertElements = [document.querySelector('header'), document.querySelector('main')]
            .filter(Boolean)
            .map(element => ({ element, wasInert: element.inert }));
        window.hydeMode.inertElements.forEach(item => { item.element.inert = true; });

        // Populate modal
        originalQueryDiv.textContent = originalQuery;
        generatedTextarea.value = generatedText;
        const withoutBtn = document.getElementById('hyde-without');
        if (withoutBtn) withoutBtn.hidden = options?.detailed !== true;

        // Show modal
        window.clearVisualizationTransientState?.();
        modal.style.display = 'flex';
        requestAnimationFrame(() => generatedTextarea.focus());
    });
}

function cancelHyDEReview() {
    const modal = document.getElementById('hyde-review-modal');
    if (!modal || modal.style.display === 'none') return false;
    document.getElementById('hyde-cancel')?.click();
    return true;
}

/**
 * Process query with HyDE if enabled
 * @param {string} question - User question
 * @param {Function} generateHyDEFunc - Function to generate HyDE text
 * @returns {Promise<Object>} { useHyDE: boolean, text: string (original or HyDE) }
 */
async function processHyDEQuery(question, generateHyDEFunc) {
    if (!window.hydeMode.enabled) {
        return { useHyDE: false, text: question };
    }

    try {
        // Generate hypothetical answer
        const hydeText = await generateHyDEFunc(question);

        // Show review modal
        const approvedText = await showHyDEReviewModal(question, hydeText);

        return { useHyDE: true, text: approvedText };
    } catch (error) {
        console.error('❌ HyDE process failed:', error);

        if (typeof showToast === 'function') {
            showToast('HyDE cancelled, using original query', 'info');
        }

        // Fallback to original query
        return { useHyDE: false, text: question };
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeHyDEHandlers);
} else {
    initializeHyDEHandlers();
}

// Export for use in other modules
window.initializeHyDEHandlers = initializeHyDEHandlers;
window.showHyDEReviewModal = showHyDEReviewModal;
window.processHyDEQuery = processHyDEQuery;
window.cancelHyDEReview = cancelHyDEReview;
