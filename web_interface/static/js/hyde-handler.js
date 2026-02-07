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
    rejectCallback: null
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

    // Close modal handlers
    const closeModal = () => {
        modal.style.display = 'none';
        if (window.hydeMode.rejectCallback) {
            window.hydeMode.rejectCallback(new Error('HyDE cancelled by user'));
            window.hydeMode.rejectCallback = null;
        }
    };

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeModal);

    // Force users to click buttons - no backdrop click or Escape to close
    // This ensures users make an explicit choice to Cancel or Search

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

            if (window.hydeMode.resolveCallback) {
                window.hydeMode.resolveCallback(editedText);
                window.hydeMode.resolveCallback = null;
                window.hydeMode.rejectCallback = null;
            }
        });
    }

}

/**
 * Show HyDE review modal
 * @param {string} originalQuery - User's original question
 * @param {string} generatedText - LLM-generated hypothetical answer
 * @returns {Promise<string>} Edited/approved text to use for search
 */
function showHyDEReviewModal(originalQuery, generatedText) {
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

        // Populate modal
        originalQueryDiv.textContent = originalQuery;
        generatedTextarea.value = generatedText;

        // Show modal
        modal.style.display = 'flex';
    });
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
