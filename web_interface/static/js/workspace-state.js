export const WORKSPACE_BREAKPOINTS = Object.freeze({ narrow: 960, wide: 1440 });
export const WORKSPACE_SPLIT = Object.freeze({
    key: 'vectoria_workspace_split_v1',
    defaultRatio: 0.38,
    minimumRatio: 0.22,
    maximumRatio: 0.55,
    rightMinimum: 360,
    rightMaximum: 620,
    canvasMinimum: 480
});

const PANE_NAMES = new Set(['map', 'documents', 'chat', 'filters']);

export function resolveWorkspaceMode(width) {
    const viewportWidth = Number.isFinite(Number(width)) ? Number(width) : 0;
    if (viewportWidth < WORKSPACE_BREAKPOINTS.narrow) return 'narrow';
    if (viewportWidth < WORKSPACE_BREAKPOINTS.wide) return 'laptop';
    return 'wide';
}

export function clampSplitRatio(value, availableWidth = 0) {
    const width = Math.max(0, Number(availableWidth) || 0);
    const parsed = Number(value);
    let minimum = WORKSPACE_SPLIT.minimumRatio;
    let maximum = WORKSPACE_SPLIT.maximumRatio;

    if (width > 0) {
        minimum = Math.max(minimum, WORKSPACE_SPLIT.rightMinimum / width);
        maximum = Math.min(maximum, WORKSPACE_SPLIT.rightMaximum / width);
        maximum = Math.min(maximum, 1 - (WORKSPACE_SPLIT.canvasMinimum / width));
    }

    if (maximum < minimum) return WORKSPACE_SPLIT.defaultRatio;
    const fallback = WORKSPACE_SPLIT.defaultRatio;
    return Math.min(maximum, Math.max(minimum, Number.isFinite(parsed) ? parsed : fallback));
}

export function parseStoredSplitRatio(serialized, availableWidth = 0) {
    if (typeof serialized !== 'string' || !serialized.trim()) {
        return clampSplitRatio(WORKSPACE_SPLIT.defaultRatio, availableWidth);
    }
    try {
        const parsed = JSON.parse(serialized);
        const value = typeof parsed === 'number' ? parsed : parsed?.ratio;
        return clampSplitRatio(value, availableWidth);
    } catch (_) {
        return clampSplitRatio(WORKSPACE_SPLIT.defaultRatio, availableWidth);
    }
}

export function serializeSplitRatio(value, availableWidth = 0) {
    return JSON.stringify({ version: 1, ratio: clampSplitRatio(value, availableWidth) });
}

export function createPaneState(activePane = 'map') {
    return {
        activePane: PANE_NAMES.has(activePane) ? activePane : 'map',
        previousPane: null,
        scrollPositions: { map: 0, documents: 0, documentDetail: 0, chat: 0, filters: 0 }
    };
}

export function transitionPane(state, nextPane, scrollTop) {
    if (!PANE_NAMES.has(nextPane)) return state;
    const current = state || createPaneState();
    const positions = { ...current.scrollPositions };
    if (Number.isFinite(Number(scrollTop))) positions[current.activePane] = Math.max(0, Number(scrollTop));
    return { ...current, previousPane: current.activePane, activePane: nextPane, scrollPositions: positions };
}

export function rememberPaneScroll(state, pane, scrollTop) {
    if (!PANE_NAMES.has(pane) && pane !== 'documentDetail') return state;
    return {
        ...state,
        scrollPositions: { ...state.scrollPositions, [pane]: Math.max(0, Number(scrollTop) || 0) }
    };
}

export function createBatchWindow(total = 0, batchSize = 100) {
    const safeTotal = Math.max(0, Number(total) || 0);
    const safeSize = Math.max(1, Number(batchSize) || 100);
    return { total: safeTotal, batchSize: safeSize, displayed: Math.min(safeTotal, safeSize) };
}

export function advanceBatchWindow(state) {
    return { ...state, displayed: Math.min(state.total, state.displayed + state.batchSize) };
}

export function resetBatchWindow(state, total = state?.total || 0) {
    return createBatchWindow(total, state?.batchSize || 100);
}
