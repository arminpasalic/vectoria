// Canvas Visualization for RAG-Vectoria Integration
// Cleaned version with all contour code removed

class CanvasVisualization {
    constructor(canvasId, tooltipId) {
        this.canvasId = canvasId;
        this.tooltipId = tooltipId;
        this.canvas = document.getElementById(canvasId);
        this.tooltip = document.getElementById(tooltipId);
        this.tooltipDefaultParent = document.body;
        this.ctx = null;
        this.data = [];

        // Interaction state
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.zoomScale = 1;

        // Lasso selection state
        this.lassoMode = false;
        this.lassoPath = [];
        this.lassoSelectedIndices = null;
        this.isDrawingLasso = false;

        // Enhanced zoom settings - UNLIMITED ZOOM IN
        this.minZoom = 0.1; // Reasonable minimum to prevent getting lost
        this.maxZoom = Infinity; // UNLIMITED zoom in

        // Highlighting
        this.highlightedPoint = null;
        this.hoveredPoint = null;
        this.highlightedDocs = null;
        this.searchResults = null;
        this.searchResultsMap = null; // Pre-computed coordinate lookup for O(1) access
        this.metadataFilteredIndices = null; // For metadata filtering

        // Performance-optimized highlighting settings
        this.highlightConfig = {
            normalOpacity: 1.0,
            dimmedOpacity: 0.4,
            highlightGlow: false,
            glowRadius: 4,
            glowIntensity: 0.5,
            highlightBorder: true,
            borderWidth: 1.5
        };
        this.baseZoomScale = 1;

        // Performance optimizations
        this.renderRequestId = null;
        this.lastRenderTime = 0;
        this.renderCooldown = 16; // 60fps cap for better performance
        this.isRendering = false;
        this.rafThrottle = false;

        // Canvas controls
        this.showLabels = false;
        this.isFullscreen = false;
        this.controlsContainer = null;
        this.skipBackgroundForScreenshot = false;
        this.filteredClusterIds = null;

        // Performance caches
        this.colorCache = new Map();
        this.lightenedColorCache = new Map();
        this.clusterColorCache = new Map();
        const sharedColorManager = window.VectoriaColorManager || null;
        this.colorManager = sharedColorManager;
        this.clusterKeywords = new Map();
        this.clusterKeywordsViz = new Map();

        if (this.colorManager) {
            this.clusterBasePalette = [...this.colorManager.basePalette];
            this.accentColors = { ...this.colorManager.accentColors };
            this.goldenAngle = this.colorManager.goldenAngle;
            this.outlierColor = this.colorManager.outlierColor;
            this.outlierDimColor = this.colorManager.outlierDimColor;
            this.dimmedNeutralColor = this.colorManager.dimmedNeutralColor;
        } else {
            const fallbackPalette = [
                '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
                '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
                '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
                '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038',
                '#EEC3FF'
            ];
            this.clusterBasePalette = fallbackPalette.slice(0, fallbackPalette.length - 3);
            this.accentColors = {
                hover: '#EEC3FF',
                focus: '#0D2A63',
                search: '#AF0038'
            };
            this.goldenAngle = 137.508; // Degrees for palette rotations
            this.outlierColor = '#9CA3AF';
            this.outlierDimColor = 'rgba(148, 155, 170, 0.65)';
            this.dimmedNeutralColor = 'rgba(82, 86, 94, 0.75)';
        }
        this.hoverTransition = {
            value: 0,
            from: 0,
            target: 0,
            start: 0,
            duration: 160 // ms
        };
        this.selectionPulse = null;
        this.selectionPulseConfig = {
            duration: 900,
            repeats: 3
        };
        this.deferredTooltipTimeout = null;

        // WebGL layer for performance (inspired by LoglineExplorer)
        this.glCanvas = null;
        this.gl = null;
        this.useWebGL = false;

        // Cached cluster centroids (calculated once, reused)
        this.clusterCentroidsCache = null;
        this.clusterCountsCache = null;

        // Label rendering optimization
        this.lastLabelRenderTime = 0;
        this.labelRenderThrottle = 100; // Only render labels every 100ms during interactions
        this.shouldRenderLabels = true;

        // Display properties
        this.dpr = window.devicePixelRatio || 1;

        // Cached gradient for background (recreate only on resize)
        this.cachedGradient = null;

        // Spatial index for fast hover picking (built on demand)
        this.spatialIndex = null;
        this.gridConfig = null;
        this.minIndexGridPoints = 2000; // build grid only for larger datasets

        // Smooth zoom animation state
        this.isZoomAnimating = false;
        this.zoomAnimStart = 0;
        // Slightly longer animation for smoother feel
        this.zoomAnimDuration = 150; // ms - faster, more responsive
        this.startZoomScale = 1;
        this.startOffsetX = 0;
        this.startOffsetY = 0;
        this.targetZoomScale = 1;
        this.targetOffsetX = 0;
        this.targetOffsetY = 0;
        this.zoomAnimationId = null;

        // Wheel event throttling for smoother zoom
        this.pendingWheelDelta = 0;
        this.wheelAnimationFrame = null;
        this.lastWheelTime = 0;
        this.lastWheelMouseX = 0;
        this.lastWheelMouseY = 0;

        // Contour layer removed per request

        if (!this.canvas) {
            console.error(`Canvas element with id "${this.canvasId}" not found.`);
            return;
        }

        if (this.tooltip) {
            if (this.tooltip.parentElement && this.tooltip.parentElement !== this.tooltipDefaultParent) {
                this.tooltipDefaultParent.appendChild(this.tooltip);
            }
            this.tooltip.style.position = 'fixed';
            this.tooltip.style.pointerEvents = 'none';
        }

        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        this.initWebGL();
        this.setupCanvasControls();
        this.setupEventListeners();
    }

    applyFullscreenLayout(isFullscreen) {
        const canvasContainer = this.canvas ? this.canvas.parentElement : null;
        const visualizationWrapper = canvasContainer ? canvasContainer.parentElement : null;

        if (!canvasContainer || !visualizationWrapper) return;

        const toggleClass = (element, className) => {
            if (!element) return;
            if (isFullscreen) {
                element.classList.add(className);
            } else {
                element.classList.remove(className);
            }
        };

        toggleClass(visualizationWrapper, 'canvas-fullscreen-active');
        toggleClass(canvasContainer, 'canvas-fullscreen-active');
        toggleClass(this.canvas, 'canvas-fullscreen-active');
        if (this.glCanvas) {
            toggleClass(this.glCanvas, 'canvas-fullscreen-active');
        }
    }

    setupCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.canvas.width = width * this.dpr;
        this.canvas.height = height * this.dpr;
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
        // Avoid cumulative scaling on repeated resizes
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

        // Initial offset to center the view
        this.offsetX = width / 2;
        this.offsetY = height / 2;

        // No contour canvas
    }

    initWebGL() {
        // Create background WebGL canvas for high-performance point rendering
        try {
            const container = this.canvas.parentElement;
            if (!container) return;

            this.glCanvas = document.createElement('canvas');
            this.glCanvas.style.position = 'absolute';
            this.glCanvas.style.top = '0';
            this.glCanvas.style.left = '0';
            this.glCanvas.style.zIndex = '0';
            this.glCanvas.style.pointerEvents = 'none';

            // Ensure container is positioned
            if (window.getComputedStyle(container).position === 'static') {
                container.style.position = 'relative';
            }

            container.insertBefore(this.glCanvas, this.canvas);

            // Match canvas size
            const rect = container.getBoundingClientRect();
            this.glCanvas.width = rect.width * this.dpr;
            this.glCanvas.height = rect.height * this.dpr;
            this.glCanvas.style.width = `${rect.width}px`;
            this.glCanvas.style.height = `${rect.height}px`;

            // Get WebGL context
            this.gl = this.glCanvas.getContext('webgl', {
                antialias: false,
                alpha: true,
                preserveDrawingBuffer: false
            });

            if (this.gl) {
                this.useWebGL = true;
                this.gl.enable(this.gl.BLEND);
                this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
                this.gl.clearColor(0, 0, 0, 0);
            } else {
                this.glCanvas.remove();
                this.glCanvas = null;
            }
        } catch (e) {
            console.warn('WebGL init failed:', e);
            this.useWebGL = false;
            if (this.glCanvas) {
                this.glCanvas.remove();
                this.glCanvas = null;
            }
        }

        // Ensure 2D canvas is on top
        this.canvas.style.position = 'relative';
        this.canvas.style.zIndex = '1';
    }

    setupCanvasControls() {
        // Setup vertical control buttons from HTML
        const zoomInBtn = document.getElementById('zoom-in-btn');
        const zoomOutBtn = document.getElementById('zoom-out-btn');
        const resetViewBtn = document.getElementById('reset-view-btn');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        const toggleLabelsBtn = document.getElementById('toggle-labels-btn');
        const lassoSelectBtn = document.getElementById('lasso-select-btn');
        const screenshotBtn = document.getElementById('screenshot-btn');

        if (zoomInBtn) {
            zoomInBtn.addEventListener('click', () => this.zoomIn());
        }

        if (zoomOutBtn) {
            zoomOutBtn.addEventListener('click', () => this.zoomOut());
        }

        if (resetViewBtn) {
            resetViewBtn.addEventListener('click', () => this.resetView());
        }

        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }

        if (toggleLabelsBtn) {
            toggleLabelsBtn.addEventListener('click', () => {
                this.toggleLabels();
                toggleLabelsBtn.classList.toggle('active', this.showLabels);
            });
            // Set initial state
            toggleLabelsBtn.classList.toggle('active', this.showLabels);
        }

        if (lassoSelectBtn) {
            lassoSelectBtn.addEventListener('click', () => {
                this.toggleLassoMode();
                lassoSelectBtn.classList.toggle('active', this.lassoMode);
            });
        }

        if (screenshotBtn) {
            screenshotBtn.addEventListener('click', () => this.takeScreenshot());
        }

        // Setup hover labels for all control buttons
        this.setupHoverLabels();

        // Keep container reference for compatibility
        this.controlsContainer = this.canvas.parentElement;
    }

    setupHoverLabels() {
        // Define labels for each button
        const buttonLabels = {
            'zoom-in-btn': 'Zoom in',
            'zoom-out-btn': 'Zoom out',
            'reset-view-btn': 'Reset view',
            'fullscreen-btn': 'Enter fullscreen',
            'toggle-labels-btn': 'Toggle labels',
            'lasso-select-btn': 'Lasso selection',
            'screenshot-btn': 'Take screenshot'
        };

        // Setup delayed hover effect (0.5 seconds)
        Object.entries(buttonLabels).forEach(([id, label]) => {
            const btn = document.getElementById(id);
            if (!btn) return;

            // Set data attribute for CSS
            btn.setAttribute('data-label', label);

            let hoverTimeout = null;

            btn.addEventListener('mouseenter', () => {
                // Clear any existing timeout
                if (hoverTimeout) clearTimeout(hoverTimeout);

                // Show label after 0.5 seconds
                hoverTimeout = setTimeout(() => {
                    btn.classList.add('show-label');
                }, 500);
            });

            btn.addEventListener('mouseleave', () => {
                // Clear timeout and hide label
                if (hoverTimeout) clearTimeout(hoverTimeout);
                btn.classList.remove('show-label');
            });
        });
    }

    createControlButton(icon, title, onClick) {
        const button = document.createElement('button');
        button.innerHTML = icon;
        button.title = title;
        button.className = 'viz-control-btn-custom'; // Use CSS class for styling

        button.addEventListener('click', onClick);
        this.controlsContainer.appendChild(button);
        return button;
    }

    resetView(options = {}) {
        const { animate = true } = options;
        this.centerView({ preferClusterCenter: true, animate });
    }

    toggleFullscreen() {
        if (!this.canvas) return;

        // Get the wrapper that contains both canvas and controls
        const canvasContainer = this.canvas.parentElement;
        const visualizationWrapper = canvasContainer ? canvasContainer.parentElement : null;

        if (!this.isFullscreen) {
            // Enter fullscreen - use wrapper to include controls
            const fullscreenTarget = visualizationWrapper || canvasContainer;
            if (fullscreenTarget.requestFullscreen) {
                fullscreenTarget.requestFullscreen();
            } else if (fullscreenTarget.webkitRequestFullscreen) {
                fullscreenTarget.webkitRequestFullscreen();
            } else if (fullscreenTarget.mozRequestFullScreen) {
                fullscreenTarget.mozRequestFullScreen();
            }
        } else {
            // Exit fullscreen
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                document.mozCancelFullScreen();
            }
        }
    }

    toggleLabels() {
        this.showLabels = !this.showLabels;
        this.requestRender();
    }

    toggleLassoMode() {
        this.lassoMode = !this.lassoMode;

        if (this.lassoMode) {
            this.canvas.style.cursor = 'crosshair';
        } else {
            this.canvas.style.cursor = 'grab';
            this.clearLassoSelection();
        }
    }

    clearLassoSelection() {
        this.lassoPath = [];
        this.lassoSelectedIndices = null;
        this.isDrawingLasso = false;

        // Clear highlighting and restore normal view
        if (typeof window.highlightSearchResultsInVisualization === 'function') {
            window.highlightSearchResultsInVisualization([], '');
        }

        // Restore full text list with all data points
        if (typeof window.updateTextList === 'function' && this.data && this.data.length > 0) {
            if (!window.__textListLock || window.__textListLock !== 'lasso') {
                window.updateTextList(this.data, { force: true });
            }
        }

        // Update RAG scope text
        if (typeof window.updateRAGScopeTextNow === 'function') {
            window.updateRAGScopeTextNow();
        }

        this.requestRender();
    }

    takeScreenshot() {
        const scale = 2; // render at higher pixel density
        const originalCanvas = this.canvas;
        const originalCtx = this.ctx;
        const originalDpr = this.dpr || 1;
        const originalSkipBackground = this.skipBackgroundForScreenshot || false;

        const cssWidth = originalCanvas.width / originalDpr;
        const cssHeight = originalCanvas.height / originalDpr;
        const targetDpr = originalDpr * scale;

        const screenshotCanvas = document.createElement('canvas');
        screenshotCanvas.width = Math.max(1, Math.round(cssWidth * targetDpr));
        screenshotCanvas.height = Math.max(1, Math.round(cssHeight * targetDpr));
        const screenshotCtx = screenshotCanvas.getContext('2d', { alpha: true });

        if (!screenshotCtx) {
            console.error('❌ Failed to access screenshot context');
            if (typeof window.showToast === 'function') {
                window.showToast('Screenshot failed: unable to access drawing context', 'error');
            }
            return;
        }

        let renderError = null;

        try {
            screenshotCtx.imageSmoothingEnabled = true;
            screenshotCtx.imageSmoothingQuality = 'high';

            this.canvas = screenshotCanvas;
            this.ctx = screenshotCtx;
            this.dpr = targetDpr;
            this.skipBackgroundForScreenshot = true;

            this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            this.render();
        } catch (error) {
            renderError = error;
        } finally {
            this.canvas = originalCanvas;
            this.ctx = originalCtx;
            this.dpr = originalDpr;
            this.skipBackgroundForScreenshot = originalSkipBackground;

            if (this.ctx && typeof this.ctx.setTransform === 'function') {
                this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            }

            this.requestRender();
        }

        if (renderError) {
            console.error('❌ Screenshot failed during rendering:', renderError);
            if (typeof window.showToast === 'function') {
                window.showToast('Screenshot failed: ' + renderError.message, 'error');
            }
            return;
        }

        screenshotCanvas.toBlob((blob) => {
            if (!blob) {
                console.error('❌ Failed to create screenshot blob');
                if (typeof window.showToast === 'function') {
                    window.showToast('Failed to create screenshot', 'error');
                }
                return;
            }

            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            link.download = `vectoria-visualization-${timestamp}.png`;
            link.href = url;

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            setTimeout(() => URL.revokeObjectURL(url), 100);

            if (typeof window.showToast === 'function') {
                window.showToast('Screenshot saved!', 'success');
            }
        }, 'image/png');
    }

    completeLassoSelection() {
        if (!this.lassoPath || this.lassoPath.length < 3) {
            return;
        }

        const selectedIndices = new Set();

        // Test each point to see if it's inside the lasso polygon
        for (let i = 0; i < this.data.length; i++) {
            const point = this.data[i];

            // Convert world coordinates to screen coordinates
            const screenX = point.x * this.zoomScale + this.offsetX;
            const screenY = point.y * this.zoomScale + this.offsetY;

            if (this.pointInPolygon(screenX, screenY, this.lassoPath)) {
                selectedIndices.add(i);
            }
        }

        this.lassoSelectedIndices = selectedIndices;

        // Update visualization to highlight selected points
        const selectedPoints = Array.from(selectedIndices).map(i => this.data[i]);

        // For WebGL renderer, use the specialized method
        if (typeof this.highlightLassoSelection === 'function') {
            this.highlightLassoSelection(selectedIndices);
        }

        // Also update the global highlighting
        if (typeof window.highlightSearchResultsInVisualization === 'function') {
            window.highlightSearchResultsInVisualization(selectedPoints, 'lasso selection');
        }

        // Filter text content list to show only selected items
        if (typeof window.showSearchResultsInTextList === 'function') {
            const selectedDocs = Array.from(selectedIndices).map((i, rank) => ({
                text: this.data[i].text || '',
                metadata: this.data[i].metadata || {},
                score: 1.0, // All lasso selections have equal priority
                index: i,
                coordinates: [this.data[i].x, this.data[i].y],
                doc_id: this.data[i].doc_id,
                chunk_id: this.data[i].chunk_id,
                cluster: this.data[i].cluster,
                cluster_color: this.data[i].cluster_color || ensureConsistentColor(this.data[i].cluster, this.data[i].cluster_color, this.data[i].cluster_name),
                cluster_name: this.data[i].cluster === -1 ? 'Outlier' : (this.data[i].cluster_name || getClusterName(this.data[i].cluster)),
                rank: rank
            }));
            window.showSearchResultsInTextList(selectedDocs, 'semantic', `Lasso Selection (${selectedDocs.length} items)`);
        }

        // Also update the results count display
        if (typeof window.displaySearchResults === 'function') {
            window.displaySearchResults({
                results: Array.from(selectedIndices).map(i => this.data[i]),
                search_type: 'lasso',
                query: `Lasso Selection (${selectedIndices.size} items)`
            });
        }

        // Update RAG scope text
        if (typeof window.updateRAGScopeTextNow === 'function') {
            window.updateRAGScopeTextNow();
        }

        // Clear the lasso path after selection
        this.lassoPath = [];
        this.requestRender();
    }

    pointInPolygon(x, y, polygon) {
        // Ray casting algorithm for point-in-polygon test
        let inside = false;

        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];

            const intersect = ((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

            if (intersect) {
                inside = !inside;
            }
        }

        return inside;
    }

    zoomIn() {
        const centerX = (this.canvas.width / this.dpr) / 2;
        const centerY = (this.canvas.height / this.dpr) / 2;
        const worldX = (centerX - this.offsetX) / this.zoomScale;
        const worldY = (centerY - this.offsetY) / this.zoomScale;
        const newZoomScale = Math.min(this.maxZoom, this.zoomScale * 1.2);
        const targetOffsetX = centerX - worldX * newZoomScale;
        const targetOffsetY = centerY - worldY * newZoomScale;
        this.startSmoothZoom(newZoomScale, targetOffsetX, targetOffsetY);
    }

    zoomOut() {
        const centerX = (this.canvas.width / this.dpr) / 2;
        const centerY = (this.canvas.height / this.dpr) / 2;
        const worldX = (centerX - this.offsetX) / this.zoomScale;
        const worldY = (centerY - this.offsetY) / this.zoomScale;
        const newZoomScale = Math.max(this.minZoom, this.zoomScale / 1.2);
        const targetOffsetX = centerX - worldX * newZoomScale;
        const targetOffsetY = centerY - worldY * newZoomScale;
        this.startSmoothZoom(newZoomScale, targetOffsetX, targetOffsetY);
    }

    updateFullscreenState() {
        const canvasContainer = this.canvas.parentElement;
        const visualizationWrapper = canvasContainer.parentElement;
        const wasFullscreen = this.isFullscreen;

        // Check if either the wrapper or container is fullscreen
        this.isFullscreen = document.fullscreenElement === visualizationWrapper ||
            document.fullscreenElement === canvasContainer ||
            document.webkitFullscreenElement === visualizationWrapper ||
            document.webkitFullscreenElement === canvasContainer ||
            document.mozFullScreenElement === visualizationWrapper ||
            document.mozFullScreenElement === canvasContainer;

        // Update button sizes for fullscreen
        if (this.controlsContainer) {
            const buttons = this.controlsContainer.querySelectorAll('button');
            buttons.forEach(button => {
                if (this.isFullscreen) {
                    button.style.width = '36px';
                    button.style.height = '36px';
                    button.style.fontSize = '14px';
                } else {
                    button.style.width = '28px';
                    button.style.height = '28px';
                    button.style.fontSize = '12px';
                }
            });
        }

        const fullscreenElement = document.fullscreenElement ||
            document.webkitFullscreenElement ||
            document.mozFullScreenElement ||
            null;

        if (this.tooltip) {
            if (this.isFullscreen && fullscreenElement && this.tooltip.parentElement !== fullscreenElement) {
                fullscreenElement.appendChild(this.tooltip);
            } else if (!this.isFullscreen && this.tooltip.parentElement !== this.tooltipDefaultParent) {
                this.tooltipDefaultParent.appendChild(this.tooltip);
            }
        }

        this.applyFullscreenLayout(this.isFullscreen);

        // Update fullscreen button icon and tooltip
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        if (fullscreenBtn) {
            const icon = fullscreenBtn.querySelector('i');
            if (icon) {
                if (this.isFullscreen) {
                    icon.className = 'fas fa-compress';
                    fullscreenBtn.setAttribute('data-label', 'Exit fullscreen');
                } else {
                    icon.className = 'fas fa-expand';
                    fullscreenBtn.setAttribute('data-label', 'Enter fullscreen');
                }
            }
        }

        const scheduleResizeAndReset = () => {
            this.resizeCanvas();
            this.resetView({ animate: false });
            this.requestRender();
        };

        // Force render to ensure canvas is visible
        const forceRender = () => {
            this.requestRender();
            this.render();
        };

        // Reset view after resize when entering or exiting fullscreen
        if (this.isFullscreen !== wasFullscreen) {
            if (this.isFullscreen) {
                // Entering fullscreen - need multiple frames for browser to update dimensions
                requestAnimationFrame(() => {
                    requestAnimationFrame(scheduleResizeAndReset);
                });
                setTimeout(scheduleResizeAndReset, 150);
                setTimeout(forceRender, 200);
            } else {
                // Exiting fullscreen - need extra time for layout to settle
                requestAnimationFrame(() => {
                    requestAnimationFrame(scheduleResizeAndReset);
                });
                setTimeout(scheduleResizeAndReset, 100);
                setTimeout(scheduleResizeAndReset, 200);
                setTimeout(forceRender, 300);
            }
        }
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));

        // Critical: Attach wheel event with passive:false to enable preventDefault
        const wheelHandler = (e) => this.handleWheel(e);
        this.canvas.addEventListener('wheel', wheelHandler, { passive: false, capture: true });

        // Also attach to parent container for better coverage
        const container = this.canvas.parentElement;
        if (container) {
            container.addEventListener('wheel', wheelHandler, { passive: false, capture: true });
            // Set explicit style to ensure it captures pointer events
            container.style.touchAction = 'none';
        }

        this.canvas.addEventListener('click', (e) => this.handleClick(e));

        // Debounced resize handler
        this.resizeTimeout = null;
        window.addEventListener('resize', () => {
            clearTimeout(this.resizeTimeout);
            this.resizeTimeout = setTimeout(() => {
                this.resizeCanvas();
                this.requestRender();
            }, 150);
        });

        // Fullscreen change listeners
        document.addEventListener('fullscreenchange', () => this.updateFullscreenState());
        document.addEventListener('webkitfullscreenchange', () => this.updateFullscreenState());
        document.addEventListener('mozfullscreenchange', () => this.updateFullscreenState());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Escape key cancels lasso mode
            if (e.key === 'Escape' && this.lassoMode) {
                e.preventDefault();
                this.toggleLassoMode();
                const lassoBtn = document.getElementById('lasso-select-btn');
                if (lassoBtn) lassoBtn.classList.remove('active');
            }
        });
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;

        // Allow layout to dictate display size when not in fullscreen
        if (!this.isFullscreen) {
            this.canvas.style.width = '';
            this.canvas.style.height = '';
            if (this.glCanvas) {
                this.glCanvas.style.width = '100%';
                this.glCanvas.style.height = '100%';
            }
        }

        const rect = container.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        // Store current view state for coordinate preservation
        const oldWidth = this.canvas.width / this.dpr;
        const oldHeight = this.canvas.height / this.dpr;
        const currentZoom = this.zoomScale;
        const currentOffsetX = this.offsetX;
        const currentOffsetY = this.offsetY;

        const cssWidth = rect.width;
        const cssHeight = rect.height;

        this.canvas.width = cssWidth * this.dpr;
        this.canvas.height = cssHeight * this.dpr;
        if (this.isFullscreen) {
            this.canvas.style.width = `${cssWidth}px`;
            this.canvas.style.height = `${cssHeight}px`;
        }
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

        // Resize WebGL canvas to match
        if (this.glCanvas) {
            this.glCanvas.width = cssWidth * this.dpr;
            this.glCanvas.height = cssHeight * this.dpr;
            if (this.isFullscreen) {
                this.glCanvas.style.width = `${cssWidth}px`;
                this.glCanvas.style.height = `${cssHeight}px`;
            } else {
                this.glCanvas.style.width = '100%';
                this.glCanvas.style.height = '100%';
            }
            if (this.gl) {
                this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
            }
        }

        // Invalidate cached gradient on resize
        this.cachedGradient = null;

        // Preserve view center when resizing (coordinate-aware)
        if (oldWidth > 0 && oldHeight > 0) {
            const worldCenterX = (oldWidth / 2 - currentOffsetX) / currentZoom;
            const worldCenterY = (oldHeight / 2 - currentOffsetY) / currentZoom;

            this.offsetX = cssWidth / 2 - worldCenterX * currentZoom;
            this.offsetY = cssHeight / 2 - worldCenterY * currentZoom;
            this.zoomScale = currentZoom;
        }
    }

    handleMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.lastMouseX = e.clientX - rect.left;
        this.lastMouseY = e.clientY - rect.top;

        // Lasso mode takes priority
        if (this.lassoMode) {
            this.isDrawingLasso = true;
            this.lassoPath = [[this.lastMouseX, this.lastMouseY]];
            this.canvas.style.cursor = 'crosshair';
            return;
        }

        const clickedPoint = this.findPointUnderMouse(this.lastMouseX, this.lastMouseY);

        if (!clickedPoint) {
            this.isDragging = true;
            this.canvas.style.cursor = 'grabbing';
        }
    }

    setHoveredPoint(point) {
        const previousIndex = this.hoveredPoint ? this.hoveredPoint.index : null;
        const nextIndex = point ? point.index : null;

        if (previousIndex === nextIndex) {
            return;
        }

        if (point) {
            this.hoverTransition.from = 0;
            this.hoverTransition.value = 0;
            this.hoverTransition.target = 1;
            this.hoverTransition.start = performance.now();
        } else {
            this.hoverTransition.from = this.hoverTransition.value;
            this.hoverTransition.target = 0;
            this.hoverTransition.start = performance.now();
        }

        this.hoveredPoint = point;
        this.requestRender();
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Handle lasso drawing
        if (this.isDrawingLasso) {
            // Add point to path with slight throttling for performance
            const lastPoint = this.lassoPath[this.lassoPath.length - 1];
            const distance = lastPoint ? Math.hypot(mouseX - lastPoint[0], mouseY - lastPoint[1]) : Infinity;

            // Only add point if moved at least 3 pixels (reduces path complexity)
            if (distance > 3) {
                this.lassoPath.push([mouseX, mouseY]);
                // Render immediately for responsive visual feedback
                this.performRender();
            }
            return;
        }

        if (this.isDragging) {
            // Instant drag response - no throttling
            this.offsetX += mouseX - this.lastMouseX;
            this.offsetY += mouseY - this.lastMouseY;
            this.lastMouseX = mouseX;
            this.lastMouseY = mouseY;

            // Skip labels during drag
            this.shouldRenderLabels = false;

            // Direct render without requestRender delay
            this.performRender();
        } else {
            // Update cursor for lasso mode
            if (this.lassoMode) {
                this.canvas.style.cursor = 'crosshair';
                return;
            }

            const hoveredPoint = this.findPointUnderMouse(mouseX, mouseY);
            const previousIndex = this.hoveredPoint ? this.hoveredPoint.index : null;
            const nextIndex = hoveredPoint ? hoveredPoint.index : null;

            if (previousIndex !== nextIndex) {
                this.setHoveredPoint(hoveredPoint);
                if (!hoveredPoint) {
                    this.hideTooltip();
                    this.canvas.style.cursor = 'grab';
                }
            }

            if (hoveredPoint) {
                this.showTooltip(hoveredPoint, e.clientX, e.clientY);
                this.canvas.style.cursor = 'pointer';
            } else if (!this.hoveredPoint) {
                this.hideTooltip();
                this.canvas.style.cursor = 'grab';
            }
        }
    }

    handleMouseUp() {
        // Complete lasso selection
        if (this.isDrawingLasso) {
            this.isDrawingLasso = false;
            if (this.lassoPath.length > 2) {
                this.completeLassoSelection();
            } else {
                this.lassoPath = [];
                this.requestRender();
            }
            this.canvas.style.cursor = 'crosshair';
            return;
        }

        if (this.isDragging) {
            this.isDragging = false;
            this.canvas.style.cursor = this.hoveredPoint ? 'pointer' : 'grab';

            // Re-enable labels immediately after drag ends
            this.shouldRenderLabels = true;
            this.requestRender();
        }
    }

    handleMouseLeave() {
        if (this.isDragging) {
            this.isDragging = false;

            // Re-enable labels when leaving canvas during drag
            this.shouldRenderLabels = true;
        }
        if (this.hoveredPoint) {
            this.setHoveredPoint(null);
            this.hideTooltip();
            this.canvas.style.cursor = 'grab';
        }
    }

    handleWheel(e) {
        // Prevent default scrolling behavior
        e.preventDefault();
        e.stopPropagation();

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Normalize delta across devices (pixels, lines, pages)
        let deltaY = e.deltaY;
        if (e.deltaMode === 1) { // DOM_DELTA_LINE
            deltaY *= 33;
        } else if (e.deltaMode === 2) { // DOM_DELTA_PAGE
            deltaY *= 120;
        }

        // Store mouse position for this zoom operation
        this.lastWheelMouseX = mouseX;
        this.lastWheelMouseY = mouseY;
        this.lastWheelTime = performance.now();

        // Accumulate wheel delta
        this.pendingWheelDelta += deltaY;

        // Mark as zooming for ultra-fast rendering mode
        this.isZoomAnimating = true;
        this.shouldRenderLabels = false;

        // Schedule zoom processing if not already scheduled
        if (!this.wheelAnimationFrame) {
            this.wheelAnimationFrame = requestAnimationFrame(() => {
                this.processAccumulatedWheel();
            });
        }

        // Re-enable labels after zooming stops (debounced)
        clearTimeout(this.labelDebounceTimer);
        this.labelDebounceTimer = setTimeout(() => {
            this.isZoomAnimating = false;
            this.shouldRenderLabels = true;
            this.render(); // Final render with labels
        }, 150);

        return false;
    }

    processAccumulatedWheel() {
        this.wheelAnimationFrame = null;

        if (this.pendingWheelDelta === 0) return;

        const mouseX = this.lastWheelMouseX;
        const mouseY = this.lastWheelMouseY;
        const deltaY = this.pendingWheelDelta;

        // Reset accumulated delta
        this.pendingWheelDelta = 0;

        // Calculate zoom factor from accumulated delta
        const zoomFactor = Math.pow(1.1, -deltaY / 50);
        let newZoomScale = this.zoomScale * zoomFactor;

        // Clamp to min/max zoom
        newZoomScale = Math.max(this.minZoom, Math.min(newZoomScale, this.maxZoom));

        // Calculate world coordinates at mouse position
        const worldX = (mouseX - this.offsetX) / this.zoomScale;
        const worldY = (mouseY - this.offsetY) / this.zoomScale;

        // Calculate new offset to keep mouse position fixed
        this.offsetX = mouseX - worldX * newZoomScale;
        this.offsetY = mouseY - worldY * newZoomScale;
        this.zoomScale = newZoomScale;

        // Render immediately
        this.render();
    }

    handleClick(e) {
        // Disable click handling during lasso mode
        if (this.lassoMode || this.isDrawingLasso) {
            return;
        }

        if (this.isDragging) return;

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const clickedPoint = this.findPointUnderMouse(mouseX, mouseY);

        if (clickedPoint) {
            this.highlightedPoint = clickedPoint.index;
            this.requestRender();

            if (window.showTextDetails) {
                window.showTextDetails(clickedPoint, clickedPoint.index, { focusVisualization: false });
            }
        } else {
            this.highlightedPoint = null;
            this.requestRender();
        }
    }

    findPointUnderMouse(mouseX, mouseY) {
        if (!this.data || this.data.length === 0) return null;

        const worldX = (mouseX - this.offsetX) / this.zoomScale;
        const worldY = (mouseY - this.offsetY) / this.zoomScale;
        const threshold = 8 / this.zoomScale;

        // Use spatial grid if available for faster lookups
        if (this.spatialIndex && this.gridConfig) {
            const { cellW, cellH } = this.gridConfig;
            const nx = Math.max(1, Math.ceil(threshold / cellW));
            const ny = Math.max(1, Math.ceil(threshold / cellH));
            const { ci, cj } = this.worldToCell(worldX, worldY);

            let closestPoint = null;
            let minDistance = threshold;

            for (let di = -nx; di <= nx; di++) {
                for (let dj = -ny; dj <= ny; dj++) {
                    const key = this.cellKey(ci + di, cj + dj);
                    const bucket = this.spatialIndex.get(key);
                    if (!bucket) continue;
                    for (let k = 0; k < bucket.length; k++) {
                        const idx = bucket[k];
                        const p = this.data[idx];
                        const dx = p.x - worldX;
                        const dy = p.y - worldY;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < minDistance) {
                            minDistance = dist;
                            closestPoint = p;
                        }
                    }
                }
            }
            return closestPoint;
        }

        // Fallback to linear scan
        let closestPoint = null;
        let minDistance = threshold;
        for (let i = this.data.length - 1; i >= 0; i--) {
            const point = this.data[i];
            const dx = point.x - worldX;
            const dy = point.y - worldY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < minDistance) {
                minDistance = distance;
                closestPoint = point;
            }
        }
        return closestPoint;
    }

    showTooltip(point, clientX, clientY) {
        if (!this.tooltip) return;

        // Ensure tooltip is visible (especially important in fullscreen)
        this.tooltip.style.display = 'block';
        this.tooltip.style.position = 'fixed';
        this.tooltip.style.zIndex = '10001';

        const clusterColor = point.cluster_color || this.getClusterColor(point.cluster, point.cluster_name);
        const clusterName = this.getClusterName(point.cluster);

        // Adjust text length based on viewport size
        const maxTextLength = window.innerWidth < 768 ? 120 : 200;

        const darkMode = document.documentElement && document.documentElement.classList.contains('dark');
        const titleColor = darkMode ? '#F5F6FA' : '#1a1a1a';
        const textColor = darkMode ? '#E3E7F0' : '#4a4a4a';
        const detailColor = darkMode ? '#B5BED1' : '#666';
        const keywordColor = darkMode ? '#9AA6C2' : '#888';
        const barBackground = darkMode ? 'rgba(255,255,255,0.15)' : '#e0e0e0';

        let content = `<div class="tooltip-content" style="border-left: 3px solid ${clusterColor}; padding-left: 14px; padding-right: 4px; font-family: 'Neue Montreal', 'Helvetica Neue', Arial, sans-serif; color: ${textColor};">`;
        content += `<div class="tooltip-title" style="font-size: 15px; font-weight: 700; margin-bottom: 6px; color: ${titleColor}; letter-spacing: -0.01em;">Item ${point.index + 1}</div>`;
        content += `<div class="tooltip-cluster" style="display: inline-block; background: ${clusterColor}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">${clusterName}</div>`;

        if (point.text) {
            const text = point.text.length > maxTextLength ? point.text.substring(0, maxTextLength) + '...' : point.text;
            content += `<div class="tooltip-text" style="margin-top: 10px; font-size: 13px; line-height: 1.5; color: ${textColor}; word-wrap: break-word; overflow-wrap: break-word; font-weight: 400;">${text}</div>`;
        }

        // Show CTFIDF keywords if available
        const keywords = this.clusterKeywords && this.clusterKeywords.get(point.cluster);
        if (keywords && keywords.length > 0) {
            const topKeywords = keywords.slice(0, 10);
            const keywordText = topKeywords.join(', ');
            content += `<div class="tooltip-keywords" style="margin-top: 10px; font-size: 11px; color: ${keywordColor}; font-style: italic; font-weight: 400;">Top keywords: ${keywordText}</div>`;
        }

        // Only show confidence for non-outlier points (cluster !== -1)
        if (typeof point.cluster_probability === 'number' && point.cluster !== -1) {
            const percent = Math.trunc(point.cluster_probability * 10000) / 100;
            content += `<div class="tooltip-confidence" style="margin-top: 10px; font-size: 12px; color: ${detailColor}; font-weight: 400;">
                <span style="display: inline-block; width: 100px; height: 6px; background: ${barBackground}; border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 6px;">
                    <span style="display: block; width: ${percent}%; height: 100%; background: ${clusterColor}; border-radius: 3px;"></span>
                </span>
                <span style="font-weight: 600; color: ${titleColor};">${percent.toFixed(1)}%</span> confidence
            </div>`;
        }

        content += `</div>`;

        this.tooltip.innerHTML = content;
        this.tooltip.style.display = 'block';

        requestAnimationFrame(() => {
            this.positionTooltipResponsively(clientX, clientY);
        });
    }

    positionTooltipResponsively(clientX, clientY) {
        if (!this.tooltip) return;

        const tooltipRect = this.tooltip.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();
        const viewport = {
            width: window.innerWidth,
            height: window.innerHeight
        };

        // Generous margins to prevent cutoff
        const edgeMargin = 20;
        const cursorOffset = 15;

        // Calculate available space in each direction
        const spaceRight = Math.min(viewport.width, canvasRect.right) - clientX;
        const spaceLeft = clientX - Math.max(0, canvasRect.left);
        const spaceBelow = Math.min(viewport.height, canvasRect.bottom) - clientY;
        const spaceAbove = clientY - Math.max(0, canvasRect.top);

        let left, top;
        let position = '';

        // Smart horizontal positioning - choose side with more space
        if (spaceRight >= tooltipRect.width + cursorOffset + edgeMargin) {
            // Enough space on right
            left = clientX + cursorOffset;
            position += 'right';
        } else if (spaceLeft >= tooltipRect.width + cursorOffset + edgeMargin) {
            // Better to show on left
            left = clientX - tooltipRect.width - cursorOffset;
            position += 'left';
        } else if (spaceRight > spaceLeft) {
            // Prefer right but with clamping
            left = clientX + cursorOffset;
            position += 'right-clamped';
        } else {
            // Prefer left but with clamping
            left = clientX - tooltipRect.width - cursorOffset;
            position += 'left-clamped';
        }

        // Smart vertical positioning - choose side with more space
        if (spaceBelow >= tooltipRect.height + cursorOffset + edgeMargin) {
            // Enough space below
            top = clientY + cursorOffset;
            position += '-below';
        } else if (spaceAbove >= tooltipRect.height + cursorOffset + edgeMargin) {
            // Better to show above
            top = clientY - tooltipRect.height - cursorOffset;
            position += '-above';
        } else if (spaceBelow > spaceAbove) {
            // Prefer below but with clamping
            top = clientY + cursorOffset;
            position += '-below-clamped';
        } else {
            // Prefer above but with clamping
            top = clientY - tooltipRect.height - cursorOffset;
            position += '-above-clamped';
        }

        // Clamp to canvas bounds first (prefer staying in canvas)
        const canvasLeft = Math.max(0, canvasRect.left);
        const canvasRight = Math.min(viewport.width, canvasRect.right);
        const canvasTop = Math.max(0, canvasRect.top);
        const canvasBottom = Math.min(viewport.height, canvasRect.bottom);

        left = Math.max(canvasLeft + edgeMargin, Math.min(left, canvasRight - tooltipRect.width - edgeMargin));
        top = Math.max(canvasTop + edgeMargin, Math.min(top, canvasBottom - tooltipRect.height - edgeMargin));

        // Final viewport clamp as safety net
        left = Math.max(edgeMargin, Math.min(left, viewport.width - tooltipRect.width - edgeMargin));
        top = Math.max(edgeMargin, Math.min(top, viewport.height - tooltipRect.height - edgeMargin));

        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
        this.tooltip.className = `tooltip visible tooltip-${position}`;
    }

    hideTooltip() {
        if (this.tooltip) {
            this.tooltip.classList.remove('visible');
            this.tooltip.style.display = 'none';
        }
    }

    requestRender() {
        if (this.rafThrottle) return;

        this.rafThrottle = true;
        requestAnimationFrame(() => {
            this.performRender();
            this.rafThrottle = false;

            // Schedule next frame if animation is active (checked after render completes)
            if (this._needsAnimationFrame) {
                this._needsAnimationFrame = false;
                this.requestRender();
            }
        });
    }

    performRender() {
        const now = performance.now();
        // Skip cooldown check if animation is active or forced render requested
        const hasActiveAnimation = this.selectionPulse || this.hoverAnimation;
        const forceRender = this._forceRender;
        this._forceRender = false;
        if (!hasActiveAnimation && !forceRender && now - this.lastRenderTime < this.renderCooldown) {
            return;
        }

        this.isRendering = true;
        this.render();
        this.lastRenderTime = now;
        this.isRendering = false;
    }

    loadData(points) {
        this.data = points || [];
        this.clusterColorCache.clear();
        this.lightenedColorCache.clear();

        // Extract cluster topic labels for better naming
        this.clusterTopicLabels = new Map();
        this.clusterKeywords = new Map();
        this.clusterKeywordsViz = new Map();

        // Calculate cluster centroids ONCE during data load
        this.clusterCentroidsCache = new Map();
        this.clusterCountsCache = new Map();

        this.data.forEach((point, index) => {
            if (point.cluster_color) {
                point.color = point.cluster_color;
            } else if (point.cluster !== undefined) {
                point.color = this.getClusterColor(point.cluster, point.cluster_name);
            }
            if (this.colorManager && typeof this.colorManager.registerColor === 'function') {
                this.colorManager.registerColor(point.cluster, point.cluster_name, point.color);
            }
            point.index = index;

            // Collect CTFIDF cluster names and keywords
            if (point.cluster !== undefined && point.cluster_name && !this.clusterTopicLabels.has(point.cluster)) {
                this.clusterTopicLabels.set(point.cluster, point.cluster_name);
            }
            // Also store keywords if available
            if (point.cluster !== undefined && point.cluster_keywords && point.cluster_keywords.length > 0) {
                this.clusterKeywords.set(point.cluster, point.cluster_keywords);
            }
            if (point.cluster !== undefined && point.cluster_keywords_viz && point.cluster_keywords_viz.length > 0) {
                this.clusterKeywordsViz.set(point.cluster, point.cluster_keywords_viz);
            }

            // Pre-calculate cluster centroids
            if (typeof point.x === 'number' && typeof point.y === 'number' && point.cluster !== undefined) {
                const clusterId = point.cluster;
                if (!this.clusterCentroidsCache.has(clusterId)) {
                    this.clusterCentroidsCache.set(clusterId, { x: 0, y: 0 });
                    this.clusterCountsCache.set(clusterId, 0);
                }
                const centroid = this.clusterCentroidsCache.get(clusterId);
                centroid.x += point.x;
                centroid.y += point.y;
                this.clusterCountsCache.set(clusterId, this.clusterCountsCache.get(clusterId) + 1);
            }
        });

        // Finalize centroids (divide by count)
        this.clusterCentroidsCache.forEach((centroid, clusterId) => {
            const count = this.clusterCountsCache.get(clusterId);
            if (count > 0) {
                centroid.x /= count;
                centroid.y /= count;
            }
        });

        if (this.canvas) {
            this.centerView();
        }

        // Build spatial index for larger datasets
        if (this.data.length >= this.minIndexGridPoints) {
            this.buildSpatialIndex();
        } else {
            this.spatialIndex = null;
            this.gridConfig = null;
        }

        this.updateFilteredClusterVisibility();
    }

    updateHoverAnimation() {
        if (!this.hoverTransition) return;

        const { value, target, start, duration, from } = this.hoverTransition;
        if (value === target) {
            return;
        }

        const now = performance.now();
        const elapsed = Math.max(0, now - start);
        const animDuration = Math.max(80, duration || 160);
        const progress = Math.min(1, elapsed / animDuration);

        let eased;
        if (target > from) {
            // Ease-out for hover-in
            eased = 1 - Math.pow(1 - progress, 3);
        } else {
            // Ease-in for hover-out
            eased = Math.pow(progress, 2);
        }

        this.hoverTransition.value = from + (target - from) * eased;

        if (progress < 1) {
            this.requestRender();
        } else {
            this.hoverTransition.value = target;
            this.hoverTransition.from = target;
        }
    }

    // Smooth zoom animation helper
    startSmoothZoom(newScale, targetOffsetX, targetOffsetY) {
        if (this.zoomAnimationId) {
            cancelAnimationFrame(this.zoomAnimationId);
            this.zoomAnimationId = null;
        }
        this.isZoomAnimating = true;
        this.zoomAnimStart = performance.now();
        this.startZoomScale = this.zoomScale;
        this.startOffsetX = this.offsetX;
        this.startOffsetY = this.offsetY;
        this.targetZoomScale = newScale;
        this.targetOffsetX = targetOffsetX;
        this.targetOffsetY = targetOffsetY;

        const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

        const step = (now) => {
            const t = Math.min(1, (now - this.zoomAnimStart) / this.zoomAnimDuration);
            const k = easeOutCubic(t);
            this.zoomScale = this.startZoomScale + (this.targetZoomScale - this.startZoomScale) * k;
            this.offsetX = this.startOffsetX + (this.targetOffsetX - this.startOffsetX) * k;
            this.offsetY = this.startOffsetY + (this.targetOffsetY - this.startOffsetY) * k;
            this.requestRender();
            if (t < 1) {
                this.zoomAnimationId = requestAnimationFrame(step);
            } else {
                this.isZoomAnimating = false;
                this.zoomAnimationId = null;
            }
        };

        this.zoomAnimationId = requestAnimationFrame(step);
    }

    // Spatial grid builders
    buildSpatialIndex() {
        if (!this.data || this.data.length === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < this.data.length; i++) {
            const p = this.data[i];
            if (typeof p.x === 'number' && typeof p.y === 'number') {
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
        }
        if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) return;

        const cols = 128;
        const rows = 128;
        const cellW = (maxX - minX || 1) / cols;
        const cellH = (maxY - minY || 1) / rows;
        this.gridConfig = { minX, minY, cols, rows, cellW, cellH };
        this.spatialIndex = new Map();

        for (let idx = 0; idx < this.data.length; idx++) {
            const p = this.data[idx];
            if (typeof p.x !== 'number' || typeof p.y !== 'number') continue;
            const { ci, cj } = this.worldToCell(p.x, p.y);
            const key = this.cellKey(ci, cj);
            let bucket = this.spatialIndex.get(key);
            if (!bucket) {
                bucket = [];
                this.spatialIndex.set(key, bucket);
            }
            bucket.push(idx);
        }
    }

    worldToCell(x, y) {
        const { minX, minY, cellW, cellH, cols, rows } = this.gridConfig;
        let ci = Math.floor((x - minX) / cellW);
        let cj = Math.floor((y - minY) / cellH);
        ci = Math.max(0, Math.min(cols - 1, ci));
        cj = Math.max(0, Math.min(rows - 1, cj));
        return { ci, cj };
    }

    cellKey(ci, cj) {
        return `${ci},${cj}`;
    }

    centerView(options = {}) {
        if (!this.canvas) return;

        const {
            preferClusterCenter = false,
            animate = false
        } = options;

        const bounds = this.computeDataBounds();
        if (!bounds) {
            this.zoomScale = 1;
            const viewWidth = (this.canvas.width || 0) / (this.dpr || 1);
            const viewHeight = (this.canvas.height || 0) / (this.dpr || 1);
            this.offsetX = viewWidth / 2;
            this.offsetY = viewHeight / 2;
            this.requestRender();
            return;
        }

        let { centerX, centerY, paddedRangeX, paddedRangeY } = bounds;
        if (preferClusterCenter) {
            const clusterCenter = this.computeClusterCenter();
            if (clusterCenter) {
                centerX = clusterCenter.x;
                centerY = clusterCenter.y;
            }
        }

        const viewWidth = Math.max(1, (this.canvas.width || 0) / (this.dpr || 1));
        const viewHeight = Math.max(1, (this.canvas.height || 0) / (this.dpr || 1));

        const zoomX = viewWidth / paddedRangeX;
        const zoomY = viewHeight / paddedRangeY;
        const targetZoom = Math.min(zoomX, zoomY);
        const targetOffsetX = viewWidth / 2 - centerX * targetZoom;
        const targetOffsetY = viewHeight / 2 - centerY * targetZoom;

        this.baseZoomScale = targetZoom;

        if (animate && this.zoomScale) {
            this.startSmoothZoom(targetZoom, targetOffsetX, targetOffsetY);
        } else {
            this.zoomScale = targetZoom;
            this.offsetX = targetOffsetX;
            this.offsetY = targetOffsetY;
            this.requestRender();
        }
    }

    computeDataBounds() {
        if (!this.data || this.data.length === 0) {
            return null;
        }

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (let i = 0; i < this.data.length; i++) {
            const point = this.data[i];
            if (typeof point.x === 'number' && typeof point.y === 'number') {
                if (point.x < minX) minX = point.x;
                if (point.x > maxX) maxX = point.x;
                if (point.y < minY) minY = point.y;
                if (point.y > maxY) maxY = point.y;
            }
        }

        if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
            return null;
        }

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const paddedRangeX = rangeX === 0 ? 100 : Math.max(60, rangeX * 1.2);
        const paddedRangeY = rangeY === 0 ? 100 : Math.max(60, rangeY * 1.2);
        const centerX = minX + rangeX / 2;
        const centerY = minY + rangeY / 2;

        return {
            minX,
            maxX,
            minY,
            maxY,
            rangeX,
            rangeY,
            paddedRangeX,
            paddedRangeY,
            centerX,
            centerY
        };
    }

    computeClusterCenter() {
        if (!this.clusterCentroidsCache || this.clusterCentroidsCache.size === 0) {
            return null;
        }

        let weightedX = 0;
        let weightedY = 0;
        let totalWeight = 0;

        this.clusterCentroidsCache.forEach((centroid, clusterId) => {
            if (!centroid || typeof centroid.x !== 'number' || typeof centroid.y !== 'number') {
                return;
            }
            const weight = (this.clusterCountsCache && this.clusterCountsCache.get(clusterId)) || 1;
            weightedX += centroid.x * weight;
            weightedY += centroid.y * weight;
            totalWeight += weight;
        });

        if (totalWeight === 0) {
            return null;
        }

        return {
            x: weightedX / totalWeight,
            y: weightedY / totalWeight
        };
    }

    worldToScreen(x, y) {
        return {
            x: x * this.zoomScale + this.offsetX,
            y: y * this.zoomScale + this.offsetY
        };
    }

    focusOnPoint(point, options = {}) {
        if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') return;
        if (!this.canvas) return;

        const viewWidth = Math.max(1, (this.canvas.width || 1) / this.dpr);
        const viewHeight = Math.max(1, (this.canvas.height || 1) / this.dpr);
        const {
            zoom = null,
            animate = true,
            minZoom = null,
            maxZoom = Math.min(this.maxZoom, this.baseZoomScale * 4)
        } = options;

        const currentZoom = this.zoomScale || 1;
        const baseline = this.baseZoomScale || currentZoom || 1;

        let targetZoom;
        if (typeof zoom === 'number' && isFinite(zoom)) {
            targetZoom = zoom;
        } else {
            const minFocusedZoom = minZoom != null ? minZoom : baseline * 1.12;
            if (currentZoom < minFocusedZoom) {
                targetZoom = minFocusedZoom;
            } else {
                targetZoom = currentZoom * 1.06;
            }
        }

        const noZoomOutThreshold = Math.max(currentZoom * 0.98, baseline * 0.9);
        targetZoom = Math.max(targetZoom, noZoomOutThreshold);

        const desiredZoom = Math.min(maxZoom, Math.max(targetZoom, baseline * 0.9));

        const targetOffsetX = viewWidth / 2 - point.x * desiredZoom;
        const targetOffsetY = viewHeight / 2 - point.y * desiredZoom;

        if (animate && (Math.abs(desiredZoom - currentZoom) > 0.01 ||
            Math.abs(targetOffsetX - this.offsetX) > 1 || Math.abs(targetOffsetY - this.offsetY) > 1)) {
            this.startSmoothZoom(desiredZoom, targetOffsetX, targetOffsetY);
        } else {
            this.zoomScale = desiredZoom;
            this.offsetX = targetOffsetX;
            this.offsetY = targetOffsetY;
            this.requestRender();
        }
    }

    startSelectionPulse(index) {
        const now = performance.now();
        this.selectionPulse = {
            index,
            start: now,
            duration: this.selectionPulseConfig.duration,
            repeats: this.selectionPulseConfig.repeats,
            progress: 0
        };
        this.requestRender();
    }

    updateSelectionPulse(now = performance.now()) {
        if (!this.selectionPulse) {
            return false;
        }

        const { start, duration, repeats, index } = this.selectionPulse;
        if (this.highlightedPoint !== index) {
            this.selectionPulse = null;
            return false;
        }

        const elapsed = now - start;
        const totalDuration = duration * repeats;
        if (elapsed > totalDuration) {
            this.selectionPulse = null;
            return false;
        }

        this.selectionPulse.progress = (elapsed % duration) / duration;
        return true; // Animation still active
    }

    cancelDeferredTooltip() {
        if (this.deferredTooltipTimeout) {
            clearTimeout(this.deferredTooltipTimeout);
            this.deferredTooltipTimeout = null;
        }
    }

    showTooltipAtPoint(point) {
        if (!this.tooltip || !point) return;
        const { x, y } = this.worldToScreen(point.x, point.y);
        const rect = this.canvas.getBoundingClientRect();
        this.showTooltip(point, rect.left + x, rect.top + y);
    }

    scheduleTooltip(point, delay = 220) {
        if (!this.tooltip) return;
        this.cancelDeferredTooltip();
        this.deferredTooltipTimeout = setTimeout(() => {
            this.deferredTooltipTimeout = null;
            this.showTooltipAtPoint(point);
        }, Math.max(0, delay));
    }

    highlightPoint(index, options = {}) {
        if (!this.data || typeof index !== 'number' || index < 0 || index >= this.data.length) {
            return;
        }

        const point = this.data[index];
        const {
            focus = false,
            zoom = null,
            animate = true,
            revealTooltip = false,
            tooltipDelay = 240,
            pulse = true
        } = options;

        this.highlightedPoint = index;

        if (focus && point) {
            this.focusOnPoint(point, { zoom, animate });
        }

        if (pulse !== false) {
            this.startSelectionPulse(index);
        }

        if (revealTooltip && point) {
            this.scheduleTooltip(point, tooltipDelay);
        } else if (!this.hoveredPoint) {
            this.cancelDeferredTooltip();
        }

        this.requestRender();
    }

    clearHighlight() {
        this.highlightedPoint = null;
        this.highlightedDocs = null;
        this.searchResults = null;
        this.searchResultsMap = null;
        this.metadataFilteredIndices = null;
        this.filteredClusterIds = null;
        this.selectionPulse = null;
        this.cancelDeferredTooltip();
        this.hideTooltip();
        this.requestRender();
    }

    clearIndividualHighlight() {
        // Clear only individual point/document highlights, preserve search results
        this.highlightedPoint = null;
        this.highlightedDocs = null;
        // Keep this.searchResults intact
        // Keep this.metadataFilteredIndices intact
        this.selectionPulse = null;
        this.cancelDeferredTooltip();
        this.hideTooltip();
        // Force immediate render to clear visual state, bypassing cooldown
        this._forceRender = true;
        this.requestRender();
    }

    highlightDocuments(docIds) {
        this.highlightedDocs = new Set(docIds);
        this.requestRender();
    }

    highlightSearchResults(searchResults) {
        this.searchResults = searchResults;

        // Pre-compute coordinate lookup map for O(1) access during rendering
        this.searchResultsMap = new Map();
        if (searchResults && searchResults.length > 0) {
            searchResults.forEach(result => {
                if (result.coordinates && result.coordinates.length >= 2) {
                    const key = `${result.coordinates[0]},${result.coordinates[1]}`;
                    this.searchResultsMap.set(key, result);
                }
            });
        }

        // Update dimming baseline
        this.highlightConfig.dimmedOpacity = searchResults && searchResults.length > 0 ? 0.4 : 0.4;

        this.requestRender();
    }

    clearSearchHighlight() {
        this.highlightedDocs = null;
        this.searchResults = null;
        this.searchResultsMap = null;
        this.requestRender();
    }

    enableSearchHighlightMode() {
        this.highlightConfig.dimmedOpacity = 0.4;
        this.requestRender();
    }

    disableSearchHighlightMode() {
        this.highlightConfig.dimmedOpacity = 0.4;
        this.requestRender();
    }

    updateFilteredClusterVisibility() {
        if (!this.metadataFilteredIndices) {
            this.filteredClusterIds = null;
            return;
        }

        const clusterSet = new Set();
        const addClusterForIndex = (index) => {
            if (index === null || index === undefined) return;
            const numericIndex = typeof index === 'number' ? index : parseInt(index, 10);
            if (!Number.isFinite(numericIndex) || numericIndex < 0 || numericIndex >= this.data.length) {
                return;
            }
            const point = this.data[numericIndex];
            if (!point) {
                return;
            }
            let clusterValue = point.cluster;
            if (clusterValue === undefined && point.cluster_id !== undefined) {
                clusterValue = point.cluster_id;
            }
            if (clusterValue === undefined && point.metadata && point.metadata.cluster !== undefined) {
                clusterValue = point.metadata.cluster;
            }
            if (clusterValue === undefined || clusterValue === null) {
                return;
            }
            clusterSet.add(String(clusterValue));
        };

        const source = this.metadataFilteredIndices;
        if (source && typeof source.forEach === 'function') {
            source.forEach(addClusterForIndex);
        } else if (Array.isArray(source)) {
            source.forEach(addClusterForIndex);
        } else if (source && typeof source === 'object') {
            Object.values(source).forEach(addClusterForIndex);
        }

        this.filteredClusterIds = clusterSet;
    }

    enableMetadataFilterMode(filteredIndices) {
        this.metadataFilteredIndices = filteredIndices;
        this.updateFilteredClusterVisibility();
        this.highlightConfig.dimmedOpacity = 0.4;
        this.requestRender();
    }

    disableMetadataFilterMode() {
        this.metadataFilteredIndices = null;
        this.filteredClusterIds = null;
        this.highlightConfig.dimmedOpacity = this.highlightConfig.normalOpacity;
        this.requestRender();
    }

    shouldDimPoint(index) {
        if (!Array.isArray(this.data) || index < 0 || index >= this.data.length) {
            return false;
        }

        const point = this.data[index];
        if (!point) {
            return false;
        }

        const pointIndex = typeof point.index === 'number' ? point.index : index;
        const hasSelection = typeof this.highlightedPoint === 'number';
        const isHighlighted = hasSelection && this.highlightedPoint === pointIndex;
        const isHovered = this.hoveredPoint && this.hoveredPoint.index === pointIndex;

        const hasMetadataFilters = this.metadataFilteredIndices !== null;
        let isMetadataFiltered = true;
        if (hasMetadataFilters) {
            if (this.metadataFilteredIndices && typeof this.metadataFilteredIndices.has === 'function') {
                isMetadataFiltered = this.metadataFilteredIndices.has(pointIndex);
            } else if (Array.isArray(this.metadataFilteredIndices)) {
                isMetadataFiltered = this.metadataFilteredIndices.includes(pointIndex);
            }
        }

        const docId = point.doc_id;
        const isSearchHighlighted = !!(this.highlightedDocs && docId != null &&
            typeof this.highlightedDocs.has === 'function' && this.highlightedDocs.has(docId));

        let isSearchResult = false;
        if (this.searchResultsMap && typeof this.searchResultsMap.has === 'function' &&
            typeof point.x === 'number' && typeof point.y === 'number') {
            const coordKey = `${point.x},${point.y}`;
            isSearchResult = this.searchResultsMap.has(coordKey);
        }

        const hasSearchResults = Array.isArray(this.searchResults) && this.searchResults.length > 0;

        let shouldDim = false;

        if (hasSearchResults && !isSearchResult) {
            shouldDim = true;
        }

        if (hasMetadataFilters && !isMetadataFiltered) {
            shouldDim = true;
        }

        if (hasSelection && !isHighlighted && !isHovered && !isSearchResult && !isSearchHighlighted) {
            shouldDim = true;
        }

        return shouldDim;
    }

    isPointHighlighted(index) {
        if (!Array.isArray(this.data) || index < 0 || index >= this.data.length) {
            return false;
        }

        const point = this.data[index];
        if (!point) {
            return false;
        }

        const pointIndex = typeof point.index === 'number' ? point.index : index;

        if (typeof this.highlightedPoint === 'number' && this.highlightedPoint === pointIndex) {
            return true;
        }

        if (this.lassoSelectedIndices && typeof this.lassoSelectedIndices.has === 'function' &&
            this.lassoSelectedIndices.has(pointIndex)) {
            return true;
        }

        if (this.highlightedDocs && typeof this.highlightedDocs.has === 'function') {
            const docId = point.doc_id;
            if (docId != null && this.highlightedDocs.has(docId)) {
                return true;
            }
        }

        if (this.searchResultsMap && typeof this.searchResultsMap.has === 'function' &&
            typeof point.x === 'number' && typeof point.y === 'number') {
            const coordKey = `${point.x},${point.y}`;
            if (this.searchResultsMap.has(coordKey)) {
                return true;
            }
        }

        return false;
    }

    // Contour visibility removed

    render() {
        if (!this.ctx) return;

        this.updateHoverAnimation();
        const pulseActive = this.updateSelectionPulse();

        // Flag that we need another frame for animation
        this._needsAnimationFrame = pulseActive;

        const viewWidth = this.canvas.width / this.dpr;
        const viewHeight = this.canvas.height / this.dpr;

        // Clear entire canvas
        this.ctx.save();
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Skip background for transparent screenshots
        if (!this.skipBackgroundForScreenshot) {
            // Draw professional gradient background
            const darkMode = document.documentElement && document.documentElement.classList.contains('dark');
            if (!this.cachedGradient || this.cachedGradientDark !== darkMode) {
                const gradient = this.ctx.createRadialGradient(
                    this.canvas.width * 0.5, this.canvas.height * 0.32, 0,
                    this.canvas.width * 0.5, this.canvas.height * 0.52, this.canvas.width * 0.85
                );
                if (darkMode) {
                    gradient.addColorStop(0, '#0f1422');
                    gradient.addColorStop(0.45, '#080b14');
                    gradient.addColorStop(1, '#05070d');
                } else {
                    gradient.addColorStop(0, '#fafbfc');
                    gradient.addColorStop(0.55, '#f8f9fa');
                    gradient.addColorStop(1, '#f1f3f5');
                }
                this.cachedGradient = gradient;
                this.cachedGradientDark = darkMode;
            }
            this.ctx.fillStyle = this.cachedGradient;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Grid is now drawn by WebGL renderer (see webgl-renderer.js)
            // Canvas 2D fallback no longer draws perspective grid
        }

        this.ctx.restore();

        if (!this.data || this.data.length === 0) return;

        // Draw points with batch optimization
        if (this.data.length > 500) {
            this.drawPointsBatched();
        } else {
            this.data.forEach(point => {
                this.drawPoint(point);
            });
        }

        // Draw cluster labels
        if (this.showLabels) {
            this.drawClusterLabels();
        }

        // Draw lasso path if active
        if (this.lassoPath.length > 0) {
            this.drawLassoPath();
        }
    }

    drawGrid(viewWidth, viewHeight) {
        const gridSize = 70;
        if (gridSize < 12) return;

        this.ctx.save();

        const isDarkMode = document.documentElement && document.documentElement.classList.contains('dark');
        const baseColor = isDarkMode ? 'rgba(150, 170, 215, ' : 'rgba(80, 100, 140, ';
        const majorOpacity = isDarkMode ? 0.16 : 0.1;
        const minorOpacity = isDarkMode ? 0.09 : 0.06;

        const shiftX = 0;
        const shiftY = 0;
        const horizonY = viewHeight * 0.5;
        const vpX = viewWidth * 0.5;
        const vpY = horizonY;
        const maxDepth = Math.max(1, Math.max(vpY, viewHeight - vpY));

        this.ctx.lineWidth = 0.6;

        // Converging vertical lines toward a vanishing point
        const lineCount = Math.ceil(viewWidth / gridSize) + 4;
        for (let i = -lineCount; i <= lineCount; i++) {
            const xBottom = vpX + i * gridSize + shiftX;
            if (xBottom < -gridSize || xBottom > viewWidth + gridSize) continue;
            const opacity = i % 2 === 0 ? majorOpacity : minorOpacity;
            this.ctx.strokeStyle = `${baseColor}${opacity})`;
            this.ctx.beginPath();
            this.ctx.moveTo(xBottom, 0);
            this.ctx.lineTo(vpX, vpY);
            this.ctx.lineTo(xBottom, viewHeight);
            this.ctx.stroke();
        }

        // Receding horizontal lines with perspective spacing (both directions)
        const rowCount = Math.ceil(maxDepth / gridSize) + 4;
        for (let i = -rowCount; i <= rowCount; i++) {
            if (i === 0) continue;
            const depth = i * gridSize + shiftY;
            const t = Math.min(Math.abs(depth) / maxDepth, 1.25);
            const eased = Math.pow(t, 1.25);
            const y = vpY + Math.sign(depth) * eased * maxDepth;
            if (y < -gridSize || y > viewHeight + gridSize) continue;
            const xLeft = vpX + (0 - vpX) * eased;
            const xRight = vpX + (viewWidth - vpX) * eased;
            const opacity = i % 2 === 0 ? minorOpacity : minorOpacity * 0.7;
            this.ctx.strokeStyle = `${baseColor}${opacity})`;
            this.ctx.beginPath();
            this.ctx.moveTo(xLeft, y);
            this.ctx.lineTo(xRight, y);
            this.ctx.stroke();
        }

        this.ctx.restore();
    }

    drawPoint(point) {
        if (typeof point.x !== 'number' || typeof point.y !== 'number') return;

        const screenX = point.x * this.zoomScale + this.offsetX;
        const screenY = point.y * this.zoomScale + this.offsetY;

        const viewWidth = this.canvas.width / this.dpr;
        const viewHeight = this.canvas.height / this.dpr;
        const margin = Math.max(50, 10 / this.zoomScale);
        if (screenX < -margin || screenX > viewWidth + margin || screenY < -margin || screenY > viewHeight + margin) {
            return;
        }

        const hasSelection = typeof this.highlightedPoint === 'number';
        const isHighlighted = hasSelection && this.highlightedPoint === point.index;
        const isHovered = this.hoveredPoint && this.hoveredPoint.index === point.index;
        const isSearchHighlighted = this.highlightedDocs && this.highlightedDocs.has(point.doc_id);

        let searchResult = null;
        let isSearchResult = false;
        if (!this.isDragging && this.searchResultsMap) {
            const coordKey = `${point.x},${point.y}`;
            searchResult = this.searchResultsMap.get(coordKey);
            isSearchResult = !!searchResult;
        }

        let isMetadataFiltered = true;
        if (this.metadataFilteredIndices) {
            if (typeof this.metadataFilteredIndices.has === 'function') {
                isMetadataFiltered = this.metadataFilteredIndices.has(point.index);
            } else if (Array.isArray(this.metadataFilteredIndices)) {
                isMetadataFiltered = this.metadataFilteredIndices.includes(point.index);
            } else if (typeof this.metadataFilteredIndices === 'object') {
                isMetadataFiltered = Object.values(this.metadataFilteredIndices).includes(point.index);
            }
        }
        const hasSearchResults = this.searchResults && this.searchResults.length > 0;
        const hasMetadataFilters = this.metadataFilteredIndices !== null;

        const isOutlierCluster = point.cluster === -1;
        const baseColor = point.color || point.cluster_color || this.getClusterColor(point.cluster, point.cluster_name);
        let color = isOutlierCluster ? this.outlierColor : baseColor;
        let radius = 4;
        let opacity = 1.0;
        const hoverIntensity = isHovered ? this.hoverTransition.value : 0;

        if (isOutlierCluster && !isHighlighted && hoverIntensity === 0 && !isSearchResult && !isSearchHighlighted) {
            // Default outlier/noise alpha is 40%
            opacity = 0.40;
            // Keep original color - no desaturation
        }

        let shouldDim = (hasSearchResults && !isSearchResult) ||
            (hasMetadataFilters && !isMetadataFiltered);
        if (hasSelection && !isHighlighted && !isHovered && !isSearchResult && !isSearchHighlighted) {
            shouldDim = true;
        }

        // If outlier is in the metadata filter, restore full opacity
        if (isOutlierCluster && isMetadataFiltered && hasMetadataFilters && !shouldDim) {
            opacity = 1.0;
        }

        if (shouldDim) {
            // Desaturate AND reduce opacity for better dimming effect
            const desaturatedColor = this.desaturateColor(color, 0.75); // 75% desaturation
            color = desaturatedColor;
            opacity = Math.min(opacity, 0.06);
        }

        let accentColor = null;
        const darkMode = document.documentElement && document.documentElement.classList.contains('dark');

        if (isHighlighted) {
            accentColor = this.getAccentColor('focus');
            color = this.brightenColor(color, 0.18);
            radius = Math.max(6, radius * 1.5);
            opacity = 1;
        } else if (isSearchResult || isSearchHighlighted) {
            accentColor = this.getAccentColor('search');
            const rankBonus = isSearchResult ? Math.max(0, (10 - (searchResult.rank || 0)) / 10) : 0.35;
            color = this.brightenColor(color, 0.14 + rankBonus * 0.08);
            radius = 4.6 + rankBonus * 2.1;
            opacity = 1;
        } else if (hoverIntensity > 0) {
            accentColor = this.getAccentColor('hover');
            color = this.brightenColor(color, 0.27 * hoverIntensity);
            radius = Math.max(5, radius * (1 + 0.25 * hoverIntensity));
            opacity = Math.min(1, opacity + 0.12 * hoverIntensity);
        }

        if (shouldDim) {
            accentColor = null;
        }

        this.ctx.save();
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 0;

        if (isHighlighted) {
            this.ctx.shadowColor = this.applyAlphaToColor(this.getAccentColor('focus'), 0.35);
            this.ctx.shadowBlur = Math.max(12, radius * 2.1);
        } else if (!shouldDim && (isHovered || isSearchResult || isSearchHighlighted)) {
            const accentForShadow = accentColor || this.getAccentColor('hover');
            this.ctx.shadowColor = this.applyAlphaToColor(accentForShadow, 0.24);
            this.ctx.shadowBlur = 8;
            this.ctx.shadowOffsetY = 1;
        } else if (!shouldDim) {
            this.ctx.shadowColor = 'rgba(0, 0, 0, 0.08)';
            this.ctx.shadowBlur = 3;
            this.ctx.shadowOffsetY = 1;
        } else {
            this.ctx.shadowColor = 'transparent';
            this.ctx.shadowBlur = 0;
        }

        this.ctx.globalAlpha = opacity;
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(screenX, screenY, radius, 0, Math.PI * 2);
        this.ctx.fill();

        if (!shouldDim) {
            this.ctx.globalAlpha = Math.min(1, opacity + 0.05);
            this.ctx.fillStyle = this.lightenColor(color, 0.08);
            this.ctx.beginPath();
            this.ctx.arc(screenX, screenY, radius * 0.55, 0, Math.PI * 2);
            this.ctx.fill();
        }

        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;
        this.ctx.shadowOffsetY = 0;
        this.ctx.globalAlpha = opacity;

        if (accentColor && !shouldDim) {
            const accentAlpha = isHighlighted
                ? 0.9
                : (isSearchResult || isSearchHighlighted)
                    ? 0.76
                    : Math.min(0.8, 0.35 + hoverIntensity * 0.45);
            this.ctx.lineWidth = 1;
            this.ctx.strokeStyle = this.applyAlphaToColor(accentColor, accentAlpha);
            this.ctx.beginPath();
            this.ctx.arc(screenX, screenY, radius + 0.6, 0, Math.PI * 2);
            this.ctx.stroke();
        } else if (isOutlierCluster && (isHovered || isHighlighted || isSearchResult || isSearchHighlighted)) {
            this.ctx.lineWidth = 1;
            this.ctx.strokeStyle = this.applyAlphaToColor(this.getAccentColor('hover'), 0.45);
            this.ctx.beginPath();
            this.ctx.arc(screenX, screenY, radius + 0.5, 0, Math.PI * 2);
            this.ctx.stroke();
        }

        if (isHighlighted && this.selectionPulse && this.selectionPulse.index === point.index) {
            const pulseProgress = this.selectionPulse.progress || 0;
            const eased = 1 - Math.pow(1 - pulseProgress, 3);
            const pulseRadius = radius + 6 + eased * 9;
            const pulseAlpha = Math.max(0.12, 0.48 - pulseProgress * 0.4);
            this.ctx.save();
            this.ctx.globalAlpha = pulseAlpha;
            this.ctx.lineWidth = 2;
            const pulseColor = accentColor || this.getAccentColor('focus');
            this.ctx.strokeStyle = this.applyAlphaToColor(pulseColor, Math.min(0.85, pulseAlpha + 0.35));
            this.ctx.beginPath();
            this.ctx.arc(screenX, screenY, pulseRadius, 0, Math.PI * 2);
            this.ctx.stroke();
            this.ctx.restore();
        }

        this.ctx.restore();
    }

    getAccentColor(type) {
        if (this.accentColors && this.accentColors[type]) {
            return this.accentColors[type];
        }
        const fallback = {
            hover: '#EEC3FF',
            focus: '#0D2A63',
            search: '#AF0038'
        };
        return fallback[type] || fallback.focus;
    }

    applyAlphaToColor(color, alpha = 1) {
        const rgba = this.hexToRgba(color);
        const clampedAlpha = this.clamp(alpha, 0, 1);
        return `rgba(${Math.round(rgba.r)}, ${Math.round(rgba.g)}, ${Math.round(rgba.b)}, ${clampedAlpha})`;
    }


    lightenColor(color, amount) {
        const cacheKey = `${color}_${amount}`;
        if (this.lightenedColorCache.has(cacheKey)) {
            return this.lightenedColorCache.get(cacheKey);
        }

        const rgba = this.hexToRgba(color);
        const delta = amount * 255;
        const result = `rgba(${this.clampColor(rgba.r + delta)}, ${this.clampColor(rgba.g + delta)}, ${this.clampColor(rgba.b + delta)}, ${rgba.a})`;

        this.lightenedColorCache.set(cacheKey, result);
        return result;
    }

    brightenColor(color, amount) {
        // Increase luminance by specified amount (0.0 to 1.0)
        const cacheKey = `bright_${color}_${amount}`;
        if (this.lightenedColorCache.has(cacheKey)) {
            return this.lightenedColorCache.get(cacheKey);
        }

        const rgba = this.hexToRgba(color);
        const hsl = this.rgbToHsl(rgba.r, rgba.g, rgba.b);

        // Increase lightness by 20-30% for contextual focus
        const newL = Math.min(95, hsl.l + (amount * 100));
        const newRgb = this.hslToRgb(hsl.h, hsl.s, newL);

        const result = `rgba(${Math.round(newRgb.r)}, ${Math.round(newRgb.g)}, ${Math.round(newRgb.b)}, ${rgba.a})`;
        this.lightenedColorCache.set(cacheKey, result);
        return result;
    }

    desaturateColor(color, amount) {
        // Desaturate color by specified amount (0.0 to 1.0) for dimming
        const cacheKey = `desat_${color}_${amount}`;
        if (this.lightenedColorCache.has(cacheKey)) {
            return this.lightenedColorCache.get(cacheKey);
        }

        const rgba = this.hexToRgba(color);
        const hsl = this.rgbToHsl(rgba.r, rgba.g, rgba.b);

        // Reduce saturation while maintaining hue for spatial legibility
        const newS = hsl.s * (1 - amount);
        const newRgb = this.hslToRgb(hsl.h, newS, hsl.l);

        const result = `rgba(${Math.round(newRgb.r)}, ${Math.round(newRgb.g)}, ${Math.round(newRgb.b)}, ${rgba.a})`;
        this.lightenedColorCache.set(cacheKey, result);
        return result;
    }

    rgbToHsl(r, g, b) {
        r /= 255;
        g /= 255;
        b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;

        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
                case g: h = ((b - r) / d + 2) / 6; break;
                case b: h = ((r - g) / d + 4) / 6; break;
            }
        }

        return { h: h * 360, s: s * 100, l: l * 100 };
    }

    hslToRgb(h, s, l) {
        h /= 360;
        s /= 100;
        l /= 100;
        let r, g, b;

        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1 / 6) return p + (q - p) * 6 * t;
                if (t < 1 / 2) return q;
                if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }

        return { r: r * 255, g: g * 255, b: b * 255 };
    }

    hexToRgba(hex, alpha = null) {
        const cacheKey = `${hex}_${alpha}`;
        if (this.colorCache.has(cacheKey)) {
            return this.colorCache.get(cacheKey);
        }

        let rgba;
        if (typeof hex === 'string' && hex.trim().startsWith('#')) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex.trim());
            if (result) {
                rgba = {
                    r: parseInt(result[1], 16),
                    g: parseInt(result[2], 16),
                    b: parseInt(result[3], 16),
                    a: alpha != null ? alpha : 1
                };
            }
        } else if (typeof hex === 'string' && hex.trim().startsWith('rgba')) {
            const match = /^rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/.exec(hex.trim());
            if (match) {
                rgba = {
                    r: parseFloat(match[1]),
                    g: parseFloat(match[2]),
                    b: parseFloat(match[3]),
                    a: alpha != null ? alpha : parseFloat(match[4])
                };
            }
        } else if (typeof hex === 'string' && hex.trim().startsWith('rgb')) {
            const match = /^rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/.exec(hex.trim());
            if (match) {
                rgba = {
                    r: parseFloat(match[1]),
                    g: parseFloat(match[2]),
                    b: parseFloat(match[3]),
                    a: alpha != null ? alpha : 1
                };
            }
        }

        if (!rgba) {
            rgba = { r: 52, g: 152, b: 219, a: alpha != null ? alpha : 1 };
        }

        this.colorCache.set(cacheKey, rgba);
        return rgba;
    }

    hexToHsl(hex) {
        const { r, g, b } = this.hexToRgba(hex);
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

    adjustPaletteColor(baseHex, variantIndex) {
        const baseHsl = this.hexToHsl(baseHex);
        const hueOffset = (variantIndex * this.goldenAngle) % 360;
        const hue = (baseHsl.h + hueOffset) % 360;
        const lightnessShift = ((variantIndex % 3) - 1) * 15; // -15%, 0%, +15%
        const saturation = baseHsl.s; // No clamping - preserve original vibrancy
        const lightness = this.clamp(baseHsl.l + lightnessShift, 35, 75);
        return this.hslToHex(hue, saturation, lightness);
    }

    clampColor(value) {
        return Math.max(0, Math.min(255, Math.round(value)));
    }

    createPointGradient(x, y, radius, color) {
        const gradient = this.ctx.createRadialGradient(x, y, Math.max(0.1, radius * 0.25), x, y, radius);
        gradient.addColorStop(0, this.lightenColor(color, 0.12));
        gradient.addColorStop(0.6, this.lightenColor(color, 0.05));
        gradient.addColorStop(1, this.lightenColor(color, -0.08));
        return gradient;
    }

    drawClusterLabels() {
        if (!this.showLabels || !this.data || this.data.length === 0 || this.zoomScale < 0.4) {
            return; // Skip labels when disabled, zoomed out too far, or no data
        }

        // Skip label rendering during rapid interactions for performance
        if (this.isDragging || this.isZoomAnimating) {
            return;
        }

        // CRITICAL: Skip labels during rapid zooming (debounced flag)
        if (!this.shouldRenderLabels) {
            return;
        }

        // Use pre-calculated centroids from cache (MUCH faster!)
        if (!this.clusterCentroidsCache || !this.clusterCountsCache) {
            return; // Cache not ready
        }

        const filteredClusters = this.filteredClusterIds;
        const filterActive = filteredClusters && typeof filteredClusters.has === 'function';
        if (filterActive && filteredClusters.size === 0) {
            return;
        }

        // Draw labels at cluster centroids
        this.ctx.save();
        const fontSize = Math.max(11, 13 * Math.min(1, this.zoomScale));
        this.ctx.font = `600 ${fontSize}px 'Neue Montreal', 'Helvetica Neue', Arial, sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        // Batch operations for performance
        const viewWidth = this.canvas.width / this.dpr;
        const viewHeight = this.canvas.height / this.dpr;
        const padding = 8;
        const visibleLabels = [];

        // First pass: collect visible labels and measure once
        this.clusterCentroidsCache.forEach((centroid, clusterId) => {
            const count = this.clusterCountsCache.get(clusterId);
            if (count === 0 || clusterId === -1) return;

            if (filterActive) {
                const clusterKey = String(clusterId);
                if (!filteredClusters.has(clusterKey)) {
                    return;
                }
            }

            const screenX = centroid.x * this.zoomScale + this.offsetX;
            const screenY = centroid.y * this.zoomScale + this.offsetY;

            // Skip if outside view (with margin)
            if (screenX < -100 || screenX > viewWidth + 100 || screenY < -50 || screenY > viewHeight + 50) {
                return;
            }

            const topicLabel = this.clusterTopicLabels && this.clusterTopicLabels.get(clusterId);
            // Use getClusterName to support custom cluster names
            const clusterLabel = this.getClusterName(clusterId);
            const normalizedTopic = typeof topicLabel === 'string' ? topicLabel.trim() : '';
            // Only show topic separately if it differs from the cluster label
            const hasDistinctTopic = normalizedTopic &&
                normalizedTopic.toLowerCase() !== clusterLabel.toLowerCase() &&
                !clusterLabel.toLowerCase().includes(normalizedTopic.toLowerCase());
            const baseLabel = hasDistinctTopic
                ? `${clusterLabel} • ${normalizedTopic}`
                : clusterLabel;
            const keywordsSource = (this.clusterKeywordsViz && this.clusterKeywordsViz.get(clusterId))
                || (this.clusterKeywords && this.clusterKeywords.get(clusterId));
            const keywordPreview = Array.isArray(keywordsSource) && keywordsSource.length > 0
                ? keywordsSource.slice(0, 3).join(', ')
                : null;
            const labelText = keywordPreview ? `${baseLabel}: ${keywordPreview}` : baseLabel;
            const clusterColor = this.getClusterColor(clusterId, topicLabel);

            visibleLabels.push({ clusterId, labelText, screenX, screenY, clusterColor });
        });

        // Early exit if no visible labels
        if (visibleLabels.length === 0) {
            this.ctx.restore();
            return;
        }

        // Second pass: batch render backgrounds with professional styling
        visibleLabels.forEach(label => {
            const textWidth = this.ctx.measureText(label.labelText).width;
            const textHeight = fontSize * 1.3;
            label.textWidth = textWidth;
            label.textHeight = textHeight;

            const x = label.screenX - textWidth / 2 - padding;
            const y = label.screenY - textHeight / 2 - padding / 2;
            const width = textWidth + padding * 2;
            const height = textHeight + padding;
            const radius = 6;

            // Draw subtle shadow for depth
            this.ctx.save();
            const darkMode = document.documentElement && document.documentElement.classList.contains('dark');
            this.ctx.shadowColor = darkMode ? 'rgba(0, 0, 0, 0.55)' : 'rgba(0, 0, 0, 0.12)';
            this.ctx.shadowBlur = 8;
            this.ctx.shadowOffsetX = 0;
            this.ctx.shadowOffsetY = 2;

            // Draw rounded rectangle background
            this.ctx.fillStyle = darkMode ? 'rgba(8, 10, 16, 0.92)' : 'rgba(255, 255, 255, 0.95)';
            this.ctx.beginPath();
            this.ctx.roundRect(x, y, width, height, radius);
            this.ctx.fill();

            this.ctx.restore();

            // Draw subtle border
            this.ctx.strokeStyle = this.applyAlphaToColor(label.clusterColor, 0.9);
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.roundRect(x, y, width, height, radius);
            this.ctx.stroke();
        });

        // Third pass: batch render text with enhanced clarity
        this.ctx.textBaseline = 'middle';
        this.ctx.textAlign = 'center';

        visibleLabels.forEach(label => {
            const x = label.screenX;
            const y = label.screenY;

            // Subtle text shadow for depth and readability
            const darkMode = document.documentElement && document.documentElement.classList.contains('dark');
            this.ctx.shadowColor = darkMode ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.08)';
            this.ctx.shadowBlur = darkMode ? 2 : 1;
            this.ctx.shadowOffsetX = 0;
            this.ctx.shadowOffsetY = darkMode ? 1 : 0.5;

            const outlineColor = darkMode ? 'rgba(0, 0, 0, 0.75)' : 'rgba(255, 255, 255, 0.9)';
            this.ctx.strokeStyle = outlineColor;
            this.ctx.lineWidth = darkMode ? 3.5 : 3.5;
            this.ctx.lineJoin = 'round';
            this.ctx.miterLimit = 2;
            this.ctx.strokeText(label.labelText, x, y);

            this.ctx.shadowBlur = 0;
            this.ctx.fillStyle = darkMode ? '#F5F6FA' : '#1a1a1a';
            this.ctx.fillText(label.labelText, x, y);
        });

        this.ctx.restore();
    }

    drawLassoPath() {
        if (!this.lassoPath || this.lassoPath.length < 2) {
            return;
        }

        this.ctx.save();

        // Draw the lasso path with enhanced visual feedback
        this.ctx.beginPath();
        this.ctx.moveTo(this.lassoPath[0][0], this.lassoPath[0][1]);

        for (let i = 1; i < this.lassoPath.length; i++) {
            this.ctx.lineTo(this.lassoPath[i][0], this.lassoPath[i][1]);
        }

        // Close the path if we have enough points
        if (this.lassoPath.length > 2) {
            this.ctx.closePath();
        }

        // Fill the lasso area with semi-transparent blue
        this.ctx.fillStyle = 'rgba(52, 152, 219, 0.12)';
        this.ctx.fill();

        // Draw an outer glow effect for better visibility
        this.ctx.shadowColor = 'rgba(41, 128, 185, 0.6)';
        this.ctx.shadowBlur = 8;
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 0;

        // Draw the lasso border with animated dashes
        this.ctx.strokeStyle = 'rgba(41, 128, 185, 0.9)';
        this.ctx.lineWidth = 2.5;
        this.ctx.setLineDash([8, 4]);
        this.ctx.lineDashOffset = -(Date.now() / 30) % 12; // Animated dashes
        this.ctx.stroke();

        // Draw start point indicator
        this.ctx.shadowBlur = 0;
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = 'rgba(41, 128, 185, 0.95)';
        this.ctx.beginPath();
        this.ctx.arc(this.lassoPath[0][0], this.lassoPath[0][1], 4, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        this.ctx.lineWidth = 1.5;
        this.ctx.stroke();

        this.ctx.restore();
    }

    drawPointsBatched() {
        if (!this.data || this.data.length === 0) {
            return;
        }
        for (let i = 0; i < this.data.length; i++) {
            this.drawPoint(this.data[i]);
        }
    }


    getClusterColor(clusterId, clusterName = null) {
        if (clusterId === -1) {
            return this.outlierColor;
        }

        if (this.colorManager && typeof this.colorManager.getColor === 'function') {
            return this.colorManager.getColor(clusterId, clusterName);
        }

        const key = clusterName ? `${clusterId}_${clusterName}` : clusterId;
        if (this.clusterColorCache.has(key)) {
            return this.clusterColorCache.get(key);
        }

        const baseIndex = ((clusterId % this.clusterBasePalette.length) + this.clusterBasePalette.length) % this.clusterBasePalette.length;
        const variantIndex = Math.floor(clusterId / this.clusterBasePalette.length);
        const baseColor = this.clusterBasePalette[baseIndex];
        const color = this.adjustPaletteColor(baseColor, variantIndex);

        this.clusterColorCache.set(key, color);
        return color;
    }

    clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    hslToHex(h, s, l) {
        const sNorm = this.clamp(s, 0, 100) / 100;
        const lNorm = this.clamp(l, 0, 100) / 100;

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

    getClusterName(clusterId) {
        if (clusterId === -1) {
            return 'Outlier';
        }

        // Check for custom name from user renaming (global customClusterNames map)
        if (typeof customClusterNames !== 'undefined' && customClusterNames.has(clusterId)) {
            return customClusterNames.get(clusterId);
        }

        // Use CTFIDF cluster name if available, otherwise use generic cluster name
        const clusterName = this.clusterTopicLabels && this.clusterTopicLabels.get(clusterId);
        if (clusterName) {
            return clusterName;
        }

        return `Cluster ${clusterId}`;
    }

    updateClusterLabels() {
        // Force a re-render to update cluster labels after renaming
        this.scheduleRender();
    }

    destroy() {
        if (this.renderRequestId) {
            clearTimeout(this.renderRequestId);
            this.renderRequestId = null;
        }
        if (this.resizeTimeout) {
            clearTimeout(this.resizeTimeout);
            this.resizeTimeout = null;
        }
        this.cancelDeferredTooltip();
        this.selectionPulse = null;

        this.data = [];
        this.highlightedPoint = null;
        this.hoveredPoint = null;
        this.highlightedDocs = null;
        this.searchResults = null;
        this.searchResultsMap = null;

        // Clean up controls
        if (this.controlsContainer) {
            this.controlsContainer.remove();
        }

        if (this.canvas) {
            const newCanvas = this.canvas.cloneNode();
            this.canvas.parentNode.replaceChild(newCanvas, this.canvas);
        }
    }
}

// Make it available globally
window.CanvasVisualization = CanvasVisualization;
