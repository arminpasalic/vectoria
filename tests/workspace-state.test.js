import test from 'node:test';
import assert from 'node:assert/strict';
import {
    advanceBatchWindow, clampSplitRatio, createBatchWindow, createPaneState,
    parseStoredSplitRatio, rememberPaneScroll, resetBatchWindow, resolveWorkspaceMode,
    serializeSplitRatio, transitionPane
} from '../web_interface/static/js/workspace-state.js';

test('workspace modes use the exact 960 and 1440 boundaries', () => {
    assert.equal(resolveWorkspaceMode(959), 'narrow');
    assert.equal(resolveWorkspaceMode(960), 'laptop');
    assert.equal(resolveWorkspaceMode(1439), 'laptop');
    assert.equal(resolveWorkspaceMode(1440), 'wide');
});

test('split ratios clamp, persist, and recover from corrupt storage', () => {
    assert.equal(clampSplitRatio(-1, 1600), 0.225);
    assert.equal(clampSplitRatio(0.9, 1600), 0.3875);
    const encoded = serializeSplitRatio(0.34, 1600);
    assert.equal(parseStoredSplitRatio(encoded, 1600), 0.34);
    assert.equal(parseStoredSplitRatio('{nope', 1600), 0.38);
});

test('pane transitions retain independent scroll positions', () => {
    let state = createPaneState('documents');
    state = transitionPane(state, 'chat', 88);
    state = rememberPaneScroll(state, 'chat', 144);
    assert.equal(state.previousPane, 'documents');
    assert.equal(state.scrollPositions.documents, 88);
    assert.equal(state.scrollPositions.chat, 144);
});

test('batch windows initialize, advance, and reset', () => {
    let state = createBatchWindow(250, 100);
    assert.equal(state.displayed, 100);
    state = advanceBatchWindow(state);
    assert.equal(state.displayed, 200);
    state = advanceBatchWindow(state);
    assert.equal(state.displayed, 250);
    assert.equal(resetBatchWindow(state, 12).displayed, 12);
});
