import {
    WORKSPACE_SPLIT, clampSplitRatio, createPaneState, parseStoredSplitRatio,
    resolveWorkspaceMode, serializeSplitRatio, transitionPane
} from './workspace-state.js';

const state = {
    mode: resolveWorkspaceMode(window.innerWidth),
    pane: createPaneState('map'),
    splitRatio: WORKSPACE_SPLIT.defaultRatio,
    filterTrigger: null,
    resizeFrame: 0,
    lastCanvasBox: { width: 0, height: 0, dpr: 0 }
};

const byId = (id) => document.getElementById(id);
const elements = {};

function currentScroller(pane = state.pane.activePane) {
    if (pane === 'chat') return byId('chat-messages');
    if (pane === 'documents') {
        const detail = byId('selected-text-view');
        return detail && detail.style.display !== 'none' ? detail : byId('text-list');
    }
    if (pane === 'filters') return document.querySelector('.filter-panel-body');
    return null;
}

function currentScrollKey(pane = state.pane.activePane) {
    if (pane === 'documents' && byId('selected-text-view')?.style.display !== 'none') return 'documentDetail';
    return pane;
}

function rememberCurrentScroll() {
    const scroller = currentScroller();
    if (scroller) state.pane.scrollPositions[currentScrollKey()] = scroller.scrollTop;
}

function restoreScroll(pane) {
    requestAnimationFrame(() => {
        const scroller = currentScroller(pane);
        if (scroller) scroller.scrollTop = state.pane.scrollPositions[currentScrollKey(pane)] || 0;
    });
}

function emitPaneChange() {
    document.dispatchEvent(new CustomEvent('vectoria:workspace-pane-changed', {
        detail: { pane: state.pane.activePane, mode: state.mode }
    }));
}

function refreshLayout() {
    if (!document.body.classList.contains('explore-workbench-active')) return;
    requestAnimationFrame(() => {
        const viz = window.mainVisualization;
        if (viz?.resizeCanvas) {
            viz.resizeCanvas({ preserveView: true });
            viz.requestRender?.();
        }
    });
}

function showPane(pane, options = {}) {
    if (!['map', 'documents', 'chat', 'filters'].includes(pane)) return false;
    rememberCurrentScroll();
    state.pane = transitionPane(state.pane, pane);
    elements.explore?.setAttribute('data-workspace-pane', pane);
    elements.paneTabs.forEach((tab) => {
        const active = tab.dataset.workspacePane === pane;
        tab.classList.toggle('active', active);
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
    });

    if (!options.skipWorkspaceTab && pane === 'documents') byId('workspace-documents-tab')?.click();
    if (!options.skipWorkspaceTab && pane === 'chat') byId('workspace-chat-tab')?.click();
    if (pane === 'filters' && state.mode !== 'narrow') openFilters();
    restoreScroll(pane);
    refreshLayout();
    emitPaneChange();

    if (pane === 'chat' && options.focus !== false) {
        requestAnimationFrame(() => byId('chat-input')?.focus({ preventScroll: true }));
    }
    return true;
}

function showSearchResults() {
    closeFloatingSurfaces({ restoreFocus: false });
    showPane('documents', { focus: false });
    const list = byId('text-list');
    if (list) list.scrollTop = 0;
}

function openFilters(trigger = elements.filterRail) {
    state.filterTrigger = trigger || document.activeElement;
    elements.explore?.classList.add('filters-overlay-open');
    elements.filters?.classList.add('open');
    elements.filterRail?.setAttribute('aria-expanded', 'true');
    if (state.mode === 'narrow') showPane('filters', { focus: false });
    requestAnimationFrame(() => elements.filters?.querySelector('.card input, .card select, .card button')?.focus({ preventScroll: true }));
}

function closeFilters({ restoreFocus = true } = {}) {
    const wasOpen = elements.filters?.classList.contains('open');
    elements.explore?.classList.remove('filters-overlay-open');
    elements.filters?.classList.remove('open');
    elements.filterRail?.setAttribute('aria-expanded', 'false');
    if (restoreFocus && wasOpen) state.filterTrigger?.focus?.({ preventScroll: true });
}

function focusFirstControl(surface) {
    requestAnimationFrame(() => surface?.querySelector(
        'button:not([disabled]), select:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )?.focus({ preventScroll: true }));
}

function closeSurface(surface, trigger, { restoreFocus = true } = {}) {
    if (!surface) return false;
    const isOptions = surface === elements.optionsPanel;
    const wasOpen = isOptions ? surface.classList.contains('open') : !surface.hidden;
    if (!wasOpen) return false;
    if (isOptions) surface.classList.remove('open');
    else surface.hidden = true;
    trigger?.setAttribute('aria-expanded', 'false');
    if (restoreFocus) trigger?.focus?.({ preventScroll: true });
    return true;
}

function closeFloatingSurfaces({ restoreFocus = false, except = null } = {}) {
    if (except !== elements.datasetMenu) closeSurface(elements.datasetMenu, elements.datasetButton, { restoreFocus });
    if (except !== elements.vizMenu) closeSurface(elements.vizMenu, elements.vizButton, { restoreFocus });
    if (except !== elements.optionsPanel) closeSurface(elements.optionsPanel, elements.optionsButton, { restoreFocus });
}

function toggleSurface(trigger, surface, { preserveOptions = false } = {}) {
    if (!trigger || !surface) return;
    const isOptions = surface === elements.optionsPanel;
    const willOpen = isOptions ? !surface.classList.contains('open') : surface.hidden;
    if (!willOpen) {
        closeSurface(surface, trigger);
        return;
    }
    if (preserveOptions) {
        closeSurface(elements.vizMenu, elements.vizButton, { restoreFocus: false });
    } else {
        closeFloatingSurfaces({ except: surface });
    }
    if (isOptions) surface.classList.add('open');
    else surface.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
    focusFirstControl(surface);
}

function closeTopMenu() {
    if (closeSurface(elements.datasetMenu, elements.datasetButton)) return true;
    if (closeSurface(elements.vizMenu, elements.vizButton)) return true;
    if (closeSurface(elements.optionsPanel, elements.optionsButton)) return true;
    return false;
}

function handleEscape() {
    if (closeTopMenu()) return true;
    if (elements.filters?.classList.contains('open')) {
        closeFilters();
        return true;
    }
    return false;
}

function handleSearchShortcut(event) {
    if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== 'k') return;
    const input = byId('search-input');
    if (!input || input.disabled) return;
    event.preventDefault();
    input.focus({ preventScroll: true });
    input.select();
}

function activateRelativeTab(event, tabs) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const current = tabs.indexOf(event.currentTarget);
    if (current < 0) return;
    event.preventDefault();
    let next = current;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else next = (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next]?.focus({ preventScroll: true });
    tabs[next]?.click();
}

function handleMenuKeyboard(event, menu) {
    const items = [...menu.querySelectorAll('[role="menuitem"]:not([disabled])')];
    if (!items.length) return;
    const current = items.indexOf(document.activeElement);
    let next = current;
    if (event.key === 'ArrowDown') next = (current + 1 + items.length) % items.length;
    else if (event.key === 'ArrowUp') next = (current - 1 + items.length) % items.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = items.length - 1;
    else if (['Enter', ' '].includes(event.key) && current >= 0) {
        event.preventDefault();
        items[current].click();
        return;
    } else return;
    event.preventDefault();
    items[next]?.focus({ preventScroll: true });
}

function applySplitRatio(ratio, persist = false) {
    const available = elements.grid?.getBoundingClientRect().width || window.innerWidth;
    state.splitRatio = clampSplitRatio(ratio, available);
    elements.grid?.style.setProperty('--workspace-split-ratio', state.splitRatio);
    elements.grid?.style.setProperty('--workspace-panel-width', `${Math.round(state.splitRatio * available)}px`);
    elements.splitter?.setAttribute('aria-valuenow', String(Math.round(state.splitRatio * 100)));
    if (persist) {
        try { localStorage.setItem(WORKSPACE_SPLIT.key, serializeSplitRatio(state.splitRatio, available)); } catch (_) {}
    }
    refreshLayout();
}

function setupSplitter() {
    let dragging = false;
    const updateFromPointer = (event) => {
        const rect = elements.grid.getBoundingClientRect();
        applySplitRatio((rect.right - event.clientX) / rect.width);
    };
    elements.splitter?.addEventListener('pointerdown', (event) => {
        if (state.mode === 'narrow') return;
        dragging = true;
        elements.splitter.setPointerCapture(event.pointerId);
        document.body.classList.add('workspace-resizing');
        event.preventDefault();
    });
    elements.splitter?.addEventListener('pointermove', (event) => {
        if (!dragging) return;
        updateFromPointer(event);
        event.preventDefault();
    });
    const finish = () => {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove('workspace-resizing');
        applySplitRatio(state.splitRatio, true);
    };
    elements.splitter?.addEventListener('pointerup', finish);
    elements.splitter?.addEventListener('pointercancel', finish);
    elements.splitter?.addEventListener('dblclick', () => applySplitRatio(WORKSPACE_SPLIT.defaultRatio, true));
    elements.splitter?.addEventListener('keydown', (event) => {
        if (event.key === 'Home') {
            event.preventDefault();
            applySplitRatio(WORKSPACE_SPLIT.defaultRatio, true);
            return;
        }
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const width = elements.grid.getBoundingClientRect().width || 1;
        const pixels = event.shiftKey ? 48 : 16;
        applySplitRatio(state.splitRatio + (event.key === 'ArrowLeft' ? pixels : -pixels) / width, true);
    });
}

function updateMode() {
    const nextMode = resolveWorkspaceMode(window.innerWidth);
    state.mode = nextMode;
    elements.explore?.setAttribute('data-workspace-mode', nextMode);
    if (nextMode === 'narrow') {
        elements.optionsPanel?.setAttribute('role', 'dialog');
        elements.optionsPanel?.setAttribute('aria-label', 'Explore options');
        const activeWorkspaceTab = byId('workspace-chat-tab')?.getAttribute('aria-selected') === 'true' ? 'chat' : 'documents';
        state.pane = transitionPane(state.pane, activeWorkspaceTab);
        elements.explore?.setAttribute('data-workspace-pane', activeWorkspaceTab);
        elements.paneTabs.forEach((tab) => {
            const active = tab.dataset.workspacePane === activeWorkspaceTab;
            tab.classList.toggle('active', active);
            tab.setAttribute('aria-selected', String(active));
            tab.tabIndex = active ? 0 : -1;
        });
    } else {
        closeSurface(elements.optionsPanel, elements.optionsButton, { restoreFocus: false });
        elements.optionsPanel?.removeAttribute('role');
        elements.optionsPanel?.removeAttribute('aria-label');
        elements.explore?.setAttribute('data-workspace-pane', state.pane.activePane);
    }
    const width = elements.grid?.getBoundingClientRect().width || window.innerWidth;
    applySplitRatio(state.splitRatio, false);
    emitPaneChange();
}

function setupCanvasObserver() {
    if (!('ResizeObserver' in window) || !elements.canvasContainer) return;
    const observer = new ResizeObserver((entries) => {
        const rect = entries[entries.length - 1]?.contentRect;
        if (!rect || rect.width <= 0 || rect.height <= 0) return;
        const dpr = window.devicePixelRatio || 1;
        if (rect.width === state.lastCanvasBox.width && rect.height === state.lastCanvasBox.height && dpr === state.lastCanvasBox.dpr) return;
        state.lastCanvasBox = { width: rect.width, height: rect.height, dpr };
        cancelAnimationFrame(state.resizeFrame);
        state.resizeFrame = requestAnimationFrame(refreshLayout);
    });
    observer.observe(elements.canvasContainer);
}

function initialize() {
    Object.assign(elements, {
        explore: byId('explore-tab'), grid: document.querySelector('.main-viz-text-container'),
        filters: byId('workspace-filters'), filterRail: byId('workspace-filter-rail'),
        splitter: byId('workspace-splitter'), canvasContainer: byId('main-viz-container'),
        optionsButton: byId('explore-options-btn'), optionsPanel: byId('explore-options-panel'),
        datasetButton: byId('dataset-actions-btn'), datasetMenu: byId('dataset-actions-menu'),
        vizButton: byId('viz-more-btn'), vizMenu: byId('viz-more-menu'),
        paneTabs: [...document.querySelectorAll('[data-workspace-pane]')]
    });
    if (!elements.explore || !elements.grid) return;

    const available = elements.grid.getBoundingClientRect().width || window.innerWidth;
    try { state.splitRatio = parseStoredSplitRatio(localStorage.getItem(WORKSPACE_SPLIT.key), available); } catch (_) {}
    applySplitRatio(state.splitRatio);
    elements.paneTabs.forEach((tab) => {
        tab.addEventListener('click', () => showPane(tab.dataset.workspacePane));
        tab.addEventListener('keydown', (event) => activateRelativeTab(event, elements.paneTabs));
    });
    elements.filterRail?.addEventListener('click', (event) => openFilters(event.currentTarget));
    byId('workspace-filter-close')?.addEventListener('click', () => closeFilters());
    elements.optionsButton?.addEventListener('click', () => toggleSurface(elements.optionsButton, elements.optionsPanel));
    elements.datasetButton?.addEventListener('click', () => toggleSurface(elements.datasetButton, elements.datasetMenu, { preserveOptions: state.mode === 'narrow' }));
    elements.vizButton?.addEventListener('click', () => toggleSurface(elements.vizButton, elements.vizMenu));
    elements.vizMenu?.addEventListener('keydown', (event) => handleMenuKeyboard(event, elements.vizMenu));
    elements.vizMenu?.addEventListener('click', (event) => {
        const proxy = event.target.closest('[data-control-target]');
        if (!proxy) return;
        byId(proxy.dataset.controlTarget)?.click();
        closeTopMenu();
    });
    document.addEventListener('pointerdown', (event) => {
        const target = event.target;
        if (!elements.datasetMenu?.hidden && !target.closest('.dataset-actions')) {
            closeSurface(elements.datasetMenu, elements.datasetButton);
        }
        if (!elements.vizMenu?.hidden && !target.closest('.viz-more-menu, #viz-more-btn')) {
            closeSurface(elements.vizMenu, elements.vizButton);
        }
        if (elements.optionsPanel?.classList.contains('open') && !target.closest('#explore-options-panel, #explore-options-btn')) {
            closeSurface(elements.optionsPanel, elements.optionsButton);
        }
        // The filter drawer is a persistent workspace surface, not a popover.
        // Keep it open while users inspect the map or documents; its close
        // button and Escape key remain the deliberate dismissal paths.
    });
    document.addEventListener('keydown', handleSearchShortcut);
    window.addEventListener('resize', updateMode, { passive: true });
    setupSplitter();
    setupCanvasObserver();
    updateMode();
}

window.VectoriaWorkspace = { showPane, showSearchResults, openFilters, closeFilters, refreshLayout, handleEscape };
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
else initialize();
