/**
 * Browser Capabilities Detection
 * Checks WebGPU, WebAssembly, and other requirements for Vectoria
 * Shows warnings for unsupported browsers
 */

const BrowserCapabilities = {
    // Capability results cache
    _cache: null,

    /**
     * Check all browser capabilities
     * @returns {Object} Capability check results
     */
    async checkAll() {
        if (this._cache) return this._cache;

        const results = {
            webgpu: await this.checkWebGPU(),
            webassembly: this.checkWebAssembly(),
            indexedDB: this.checkIndexedDB(),
            serviceWorker: this.checkServiceWorker(),
            sharedArrayBuffer: this.checkSharedArrayBuffer(),
            isMobile: this.checkMobile(),
            memory: await this.checkMemory(),
            browser: this.detectBrowser(),
            timestamp: Date.now()
        };

        // Determine overall compatibility
        results.isFullySupported = results.webgpu.supported &&
                                    results.webassembly.supported &&
                                    results.indexedDB.supported &&
                                    !results.isMobile.isMobile;

        results.canRunEmbeddings = results.webassembly.supported && results.indexedDB.supported;
        results.canRunLLM = results.webgpu.supported && results.webassembly.supported;

        this._cache = results;
        return results;
    },

    /**
     * Check WebGPU support (required for LLM inference)
     */
    async checkWebGPU() {
        const result = {
            supported: false,
            adapter: null,
            device: null,
            reason: null
        };

        if (!navigator.gpu) {
            result.reason = 'WebGPU API not available in this browser';
            return result;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                result.reason = 'No WebGPU adapter found (GPU may not be supported)';
                return result;
            }

            result.adapter = {
                name: adapter.name || 'Unknown',
                vendor: adapter.vendor || 'Unknown',
                isFallbackAdapter: adapter.isFallbackAdapter || false
            };

            // Try to get device to confirm full support
            try {
                const device = await adapter.requestDevice();
                result.device = true;
                result.supported = true;
                device.destroy(); // Clean up
            } catch (e) {
                result.reason = `WebGPU device creation failed: ${e.message}`;
            }
        } catch (e) {
            result.reason = `WebGPU check failed: ${e.message}`;
        }

        return result;
    },

    /**
     * Check WebAssembly support (required for embeddings and clustering)
     */
    checkWebAssembly() {
        const result = {
            supported: false,
            simd: false,
            threads: false,
            reason: null
        };

        if (typeof WebAssembly === 'undefined') {
            result.reason = 'WebAssembly not supported';
            return result;
        }

        result.supported = true;

        // Check SIMD support
        try {
            // Simple SIMD detection
            result.simd = WebAssembly.validate(new Uint8Array([
                0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
            ]));
        } catch (e) {
            result.simd = false;
        }

        // Check threads support
        try {
            result.threads = typeof SharedArrayBuffer !== 'undefined';
        } catch (e) {
            result.threads = false;
        }

        return result;
    },

    /**
     * Check IndexedDB support (required for storage)
     */
    checkIndexedDB() {
        return {
            supported: typeof indexedDB !== 'undefined',
            reason: typeof indexedDB === 'undefined' ? 'IndexedDB not available' : null
        };
    },

    /**
     * Check Service Worker support
     */
    checkServiceWorker() {
        return {
            supported: 'serviceWorker' in navigator
        };
    },

    /**
     * Check SharedArrayBuffer (needed for multi-threaded WASM)
     */
    checkSharedArrayBuffer() {
        const result = {
            supported: false,
            reason: null
        };

        try {
            new SharedArrayBuffer(1);
            result.supported = true;
        } catch (e) {
            result.reason = 'SharedArrayBuffer not available (requires COOP/COEP headers)';
        }

        return result;
    },

    /**
     * Check if running on mobile device
     */
    checkMobile() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;

        // Check for mobile user agents
        const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile|tablet/i;
        const isMobileUA = mobileRegex.test(userAgent.toLowerCase());

        // Check for touch capability and small screen
        const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        const isSmallScreen = window.innerWidth < 768;

        // Check for mobile-specific features
        const isMobileDevice = /Mobi|Android/i.test(userAgent);

        return {
            isMobile: isMobileUA || (hasTouch && isSmallScreen) || isMobileDevice,
            isTablet: /ipad|tablet/i.test(userAgent.toLowerCase()),
            hasTouch,
            screenWidth: window.innerWidth,
            userAgent: userAgent.substring(0, 100) // Truncate for display
        };
    },

    /**
     * Check available memory
     */
    async checkMemory() {
        const result = {
            deviceMemory: null,
            jsHeapLimit: null,
            recommended: false
        };

        // Device memory API (Chrome only)
        if (navigator.deviceMemory) {
            result.deviceMemory = navigator.deviceMemory;
            result.recommended = navigator.deviceMemory >= 4;
        }

        // Performance memory API (Chrome only)
        if (performance.memory) {
            result.jsHeapLimit = Math.round(performance.memory.jsHeapSizeLimit / (1024 * 1024));
        }

        return result;
    },

    /**
     * Detect browser type and version
     */
    detectBrowser() {
        const ua = navigator.userAgent;
        let browser = 'Unknown';
        let version = null;

        if (ua.includes('Firefox/')) {
            browser = 'Firefox';
            version = ua.match(/Firefox\/(\d+)/)?.[1];
        } else if (ua.includes('Edg/')) {
            browser = 'Edge';
            version = ua.match(/Edg\/(\d+)/)?.[1];
        } else if (ua.includes('Chrome/')) {
            browser = 'Chrome';
            version = ua.match(/Chrome\/(\d+)/)?.[1];
        } else if (ua.includes('Safari/') && !ua.includes('Chrome')) {
            browser = 'Safari';
            version = ua.match(/Version\/(\d+)/)?.[1];
        }

        return {
            name: browser,
            version: version ? parseInt(version) : null,
            isModern: this._isModernBrowser(browser, version)
        };
    },

    _isModernBrowser(browser, version) {
        if (!version) return false;
        const v = parseInt(version);

        switch (browser) {
            case 'Chrome':
            case 'Edge':
                return v >= 113; // WebGPU stable
            case 'Firefox':
                return v >= 121; // WebGPU behind flag
            case 'Safari':
                return v >= 17; // WebGPU partial
            default:
                return false;
        }
    },

    /**
     * Get recommended actions based on capabilities
     */
    getRecommendations(capabilities) {
        const recommendations = [];

        if (!capabilities.webgpu.supported) {
            recommendations.push({
                type: 'warning',
                title: 'WebGPU Not Available',
                message: 'AI-powered answers (RAG) require WebGPU. You can still use embeddings and search.',
                action: 'Try Chrome 113+ or Edge 113+ for full features',
                feature: 'llm'
            });
        }

        if (!capabilities.webassembly.supported) {
            recommendations.push({
                type: 'error',
                title: 'WebAssembly Required',
                message: 'Your browser does not support WebAssembly which is required for all ML features.',
                action: 'Please use a modern browser (Chrome, Firefox, Safari, or Edge)',
                feature: 'core'
            });
        }

        if (capabilities.isMobile.isMobile) {
            recommendations.push({
                type: 'warning',
                title: 'Mobile Device Detected',
                message: 'Vectoria is optimized for desktop browsers. Mobile performance may be limited.',
                action: 'For best experience, use a desktop or laptop computer',
                feature: 'performance'
            });
        }

        if (capabilities.memory.deviceMemory && capabilities.memory.deviceMemory < 4) {
            recommendations.push({
                type: 'warning',
                title: 'Limited Memory',
                message: `Your device has ${capabilities.memory.deviceMemory}GB RAM. 8GB+ is recommended.`,
                action: 'Processing large datasets may be slow or fail',
                feature: 'performance'
            });
        }

        if (!capabilities.browser.isModern) {
            recommendations.push({
                type: 'info',
                title: 'Browser Update Recommended',
                message: `${capabilities.browser.name} ${capabilities.browser.version || ''} may have limited features.`,
                action: 'Update to the latest version for best compatibility',
                feature: 'compatibility'
            });
        }

        return recommendations;
    }
};

/**
 * Show capability warning banner
 */
function showCapabilityWarning(capabilities) {
    const recommendations = BrowserCapabilities.getRecommendations(capabilities);

    if (recommendations.length === 0) return;

    // Create banner container
    const banner = document.createElement('div');
    banner.id = 'capability-warning-banner';
    banner.className = 'capability-banner';

    // Determine severity
    const hasError = recommendations.some(r => r.type === 'error');
    const hasWarning = recommendations.some(r => r.type === 'warning');
    banner.classList.add(hasError ? 'capability-error' : hasWarning ? 'capability-warning' : 'capability-info');

    // Build content
    const mainIssue = recommendations[0];
    const otherCount = recommendations.length - 1;

    banner.innerHTML = `
        <div class="capability-banner-content">
            <div class="capability-icon">
                ${hasError ? '<i class="fas fa-exclamation-circle"></i>' :
                  hasWarning ? '<i class="fas fa-exclamation-triangle"></i>' :
                  '<i class="fas fa-info-circle"></i>'}
            </div>
            <div class="capability-message">
                <strong>${mainIssue.title}</strong>
                <span>${mainIssue.message}</span>
                ${otherCount > 0 ? `<span class="capability-more">+${otherCount} more issue${otherCount > 1 ? 's' : ''}</span>` : ''}
            </div>
            <div class="capability-actions">
                <button class="btn btn-sm capability-details-btn" onclick="showCapabilityDetails()">
                    Details
                </button>
                <button class="btn btn-sm btn-icon capability-dismiss" onclick="dismissCapabilityWarning()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;

    // Insert after header
    const header = document.querySelector('header');
    if (header && header.nextSibling) {
        header.parentNode.insertBefore(banner, header.nextSibling);
    } else {
        document.body.insertBefore(banner, document.body.firstChild);
    }

    // Store recommendations for details modal
    window._capabilityRecommendations = recommendations;
    window._browserCapabilities = capabilities;
}

/**
 * Show detailed capability information
 */
function showCapabilityDetails() {
    const capabilities = window._browserCapabilities;
    const recommendations = window._capabilityRecommendations || [];

    const modal = document.createElement('div');
    modal.id = 'capability-modal';
    modal.className = 'modal-overlay ml-modal-overlay';
    modal.innerHTML = `
        <div class="modal-content ml-modal-content capability-modal-content">
            <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="margin: 0;"><i class="fas fa-microchip"></i> Browser Capabilities</h2>
                <button class="btn btn-icon" onclick="closeCapabilityModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>

            <div class="capability-grid">
                ${buildCapabilityRow('WebGPU (LLM)', capabilities.webgpu.supported,
                    capabilities.webgpu.supported ?
                        (capabilities.webgpu.adapter?.name || 'Available') :
                        capabilities.webgpu.reason)}

                ${buildCapabilityRow('WebAssembly', capabilities.webassembly.supported,
                    capabilities.webassembly.supported ?
                        `SIMD: ${capabilities.webassembly.simd ? 'Yes' : 'No'}, Threads: ${capabilities.webassembly.threads ? 'Yes' : 'No'}` :
                        capabilities.webassembly.reason)}

                ${buildCapabilityRow('IndexedDB', capabilities.indexedDB.supported,
                    capabilities.indexedDB.supported ? 'Available for storage' : 'Not available')}

                ${buildCapabilityRow('Device Type', !capabilities.isMobile.isMobile,
                    capabilities.isMobile.isMobile ?
                        `Mobile device (${capabilities.isMobile.screenWidth}px width)` :
                        'Desktop browser')}

                ${capabilities.memory.deviceMemory ?
                    buildCapabilityRow('Memory', capabilities.memory.deviceMemory >= 4,
                        `${capabilities.memory.deviceMemory}GB RAM${capabilities.memory.deviceMemory < 8 ? ' (8GB+ recommended)' : ''}`) : ''}

                ${buildCapabilityRow('Browser', capabilities.browser.isModern,
                    `${capabilities.browser.name} ${capabilities.browser.version || ''}`)}
            </div>

            ${recommendations.length > 0 ? `
                <div class="capability-recommendations">
                    <h3 style="margin: 1.5rem 0 1rem; font-size: 1rem;">Recommendations</h3>
                    ${recommendations.map(r => `
                        <div class="recommendation-item recommendation-${r.type}">
                            <strong>${r.title}</strong>
                            <p style="margin: 0.25rem 0;">${r.message}</p>
                            <small style="color: var(--text-muted);">${r.action}</small>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            <div class="capability-summary" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-subtle); border-radius: 8px;">
                <strong>Feature Support:</strong>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap;">
                    <span class="feature-badge ${capabilities.canRunEmbeddings ? 'supported' : 'unsupported'}">
                        ${capabilities.canRunEmbeddings ? '<i class="fas fa-check"></i>' : '<i class="fas fa-times"></i>'}
                        Embeddings & Search
                    </span>
                    <span class="feature-badge ${capabilities.canRunLLM ? 'supported' : 'unsupported'}">
                        ${capabilities.canRunLLM ? '<i class="fas fa-check"></i>' : '<i class="fas fa-times"></i>'}
                        AI Answers (RAG)
                    </span>
                    <span class="feature-badge ${capabilities.indexedDB.supported ? 'supported' : 'unsupported'}">
                        ${capabilities.indexedDB.supported ? '<i class="fas fa-check"></i>' : '<i class="fas fa-times"></i>'}
                        Data Storage
                    </span>
                </div>
            </div>

            <div style="margin-top: 1.5rem; text-align: right;">
                <button class="btn btn-primary" onclick="closeCapabilityModal()">Got it</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

function buildCapabilityRow(label, supported, detail) {
    return `
        <div class="capability-row">
            <span class="capability-label">${label}</span>
            <span class="capability-status ${supported ? 'status-ok' : 'status-warning'}">
                ${supported ? '<i class="fas fa-check-circle"></i>' : '<i class="fas fa-exclamation-circle"></i>'}
            </span>
            <span class="capability-detail">${detail || ''}</span>
        </div>
    `;
}

function closeCapabilityModal() {
    const modal = document.getElementById('capability-modal');
    if (modal) modal.remove();
}

function dismissCapabilityWarning() {
    const banner = document.getElementById('capability-warning-banner');
    if (banner) {
        banner.style.animation = 'slideUp 0.3s ease-out reverse';
        setTimeout(() => banner.remove(), 300);
    }
    // Store dismissal in session
    sessionStorage.setItem('capabilityWarningDismissed', 'true');
}

/**
 * Initialize capability check on page load
 */
async function initCapabilityCheck() {
    // Skip if already dismissed this session
    if (sessionStorage.getItem('capabilityWarningDismissed') === 'true') {
        return;
    }

    const capabilities = await BrowserCapabilities.checkAll();

    // Store globally for other modules
    window.browserCapabilities = capabilities;

    // Show warning if needed
    if (!capabilities.isFullySupported) {
        showCapabilityWarning(capabilities);
    }

    // Log summary
    if (capabilities.isFullySupported) {
    } else {
        console.warn('Some browser capabilities are limited:',
            BrowserCapabilities.getRecommendations(capabilities).map(r => r.title).join(', '));
    }

    return capabilities;
}

// Export for use in other modules
window.BrowserCapabilities = BrowserCapabilities;
window.initCapabilityCheck = initCapabilityCheck;
window.showCapabilityDetails = showCapabilityDetails;
window.closeCapabilityModal = closeCapabilityModal;
window.dismissCapabilityWarning = dismissCapabilityWarning;
