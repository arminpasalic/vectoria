/**
 * Search Interface Enhancements - Simplified Version
 * Works in conjunction with fast-search.js for improved search UX
 * Focuses only on UI animations and visual feedback
 */

class SearchEnhancement {
    constructor() {
        this.init();
    }

    init() {
        this.setupSearchAnimations();
        this.setupKeyboardShortcuts();
    }

    setupSearchAnimations() {
        const searchInput = document.getElementById('search-input');
        if (!searchInput) return;

        // Add focus/blur animations
        searchInput.addEventListener('focus', (e) => {
            e.target.parentElement.classList.add('focused');
            this.addSearchGlow(e.target);
        });

        searchInput.addEventListener('blur', (e) => {
            e.target.parentElement.classList.remove('focused');
            this.removeSearchGlow(e.target);
        });

        // Add typing animation
        searchInput.addEventListener('input', (e) => {
            this.addTypingEffect(e.target);
        });
    }

    addSearchGlow(input) {
        input.style.boxShadow = '0 0 0 4px rgba(255, 215, 0, 0.2), 0 8px 32px rgba(255, 215, 0, 0.15)';
        input.style.transform = 'translateY(-2px) scale(1.01)';
    }

    removeSearchGlow(input) {
        setTimeout(() => {
            if (document.activeElement !== input) {
                input.style.boxShadow = '';
                input.style.transform = '';
            }
        }, 150);
    }

    addTypingEffect(input) {
        input.classList.add('typing');
        clearTimeout(input.typingTimeout);
        input.typingTimeout = setTimeout(() => {
            input.classList.remove('typing');
        }, 500);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const input = document.getElementById('search-input');
                if (input) {
                    input.focus();
                    input.select();
                    this.showToast('Search focused', 'info');
                }
            }
            
            // Escape to clear search when focused
            if (e.key === 'Escape' && document.activeElement && document.activeElement.id === 'search-input') {
                if (window.globalSearchInterface) {
                    window.globalSearchInterface.clearSearchResults();
                }
            }
        });
    }

    showToast(message, type = 'info') {
        if (window.showToast) {
            window.showToast(message, type);
        } else {
        }
    }

    // Utility method for highlighting search terms
    highlightSearchTerm(text, term) {
        if (!term || term.trim() === '') return text;
        
        const regex = new RegExp(`(${this.escapeRegExp(term)})`, 'gi');
        return text.replace(regex, '<mark class="fast-highlight">$1</mark>');
    }

    escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Update results counter in UI
    updateResultsCounter(count, searchTime = null) {
        const counter = document.querySelector('.search-results-counter');
        if (counter) {
            let text = `${count} result${count !== 1 ? 's' : ''}`;
            if (searchTime !== null) {
                text += ` (${searchTime.toFixed(1)}ms)`;
            }
            counter.textContent = text;
        }
    }

    // Show search loading state
    showSearchLoading() {
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.classList.add('loading');
            searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
        }
    }

    // Hide search loading state
    hideSearchLoading() {
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.classList.remove('loading');
            searchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
        }
    }

    // Add smooth transitions for result appearance
    animateResultsIn(resultsContainer) {
        if (!resultsContainer) return;
        
        const results = resultsContainer.querySelectorAll('.text-item');
        results.forEach((result, index) => {
            result.style.opacity = '0';
            result.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                result.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                result.style.opacity = '1';
                result.style.transform = 'translateY(0)';
            }, index * 50); // Stagger the animations
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.searchEnhancement = new SearchEnhancement();
});

// Make available globally
window.SearchEnhancement = SearchEnhancement;
