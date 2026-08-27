import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const [safetySource, searchSource, visualizationSource, vectoriaSource, capabilitiesSource] = await Promise.all([
    readFile(new URL('../web_interface/static/js/dom-safety.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/fast-search.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/viz.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/browser-capabilities.js', import.meta.url), 'utf8')
]);

function browserContext() {
    const context = {
        console,
        performance: { now: () => 0 },
        requestAnimationFrame: () => 1,
        document: {
            addEventListener() {},
            documentElement: { classList: { contains: () => false } }
        },
        CSS: { supports: property => property === 'color' }
    };
    context.window = context;
    vm.createContext(context);
    vm.runInContext(safetySource, context);
    return context;
}

test('fast-search highlighting preserves marks while escaping document text', () => {
    const context = browserContext();
    vm.runInContext(searchSource, context);
    const search = new context.FastSearch([{ text: '<img src=x onerror=alert(1)> Alpha & beta', cluster: 0, x: 0, y: 0 }]);
    const [result] = search.search('alpha').results;

    assert.match(result.matchedText, /<mark class="fast-highlight">Alpha<\/mark>/);
    assert.match(result.matchedText, /&lt;img/);
    assert.doesNotMatch(result.matchedText, /<img[^>]*onerror=/i);
});

test('visualization tooltip escapes dataset text, labels, and keywords and validates colors', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);
    const tooltip = { style: {}, innerHTML: '' };
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        tooltip,
        clusterKeywords: new Map([[1, ['<script>alert(3)</script>']]]),
        getClusterColor: () => '#3B82F6',
        getClusterName: () => '<img src=x onerror=alert(2)>'
    });

    visualization.showTooltip({
        index: 0,
        cluster: 1,
        cluster_color: 'red; background:url(javascript:alert(1))',
        text: '<svg onload=alert(1)>Text',
        cluster_probability: 2
    }, 0, 0);

    assert.match(tooltip.innerHTML, /#9CA3AF/);
    assert.match(tooltip.innerHTML, /&lt;svg onload=alert\(1\)&gt;Text/);
    assert.match(tooltip.innerHTML, /&lt;img src=x onerror=alert\(2\)&gt;/);
    assert.match(tooltip.innerHTML, /width: 100%/);
    assert.doesNotMatch(tooltip.innerHTML, /<script|<svg|<img/i);
});

test('tooltip metadata follows the text-list priority order and escapes keys and values', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);

    const seen = [];
    context.window.getMetadataPreviewEntries = (point, limit) => {
        seen.push({ point, limit });
        return [
            { key: 'source', label: 'Source', previewValue: '&lt;img src=x onerror=alert(4)&gt;' },
            { key: 'author', label: '<script>alert(5)</script>', previewValue: 'Ada' }
        ];
    };

    const tooltip = { style: {}, innerHTML: '' };
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        tooltip,
        clusterKeywords: new Map(),
        getClusterColor: () => '#3B82F6',
        getClusterName: () => 'Cluster A'
    });

    visualization.showTooltip({ index: 0, cluster: 1, source: 'x', author: 'y' }, 0, 0);

    // The point is handed through untouched so ordering stays owned by the text list.
    assert.equal(seen.length, 1);
    assert.equal(seen[0].point.source, 'x');
    assert.ok(seen[0].limit > 0);

    // Priority order from the helper is preserved in the rendered markup.
    assert.ok(tooltip.innerHTML.indexOf('Source') < tooltip.innerHTML.indexOf('alert(5)'));
    assert.match(tooltip.innerHTML, /&lt;img src=x onerror=alert\(4\)&gt;/);
    assert.doesNotMatch(tooltip.innerHTML, /<script|<img/i);
});

test('the Appearance opt-out suppresses tooltip metadata without touching the rest', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);

    let helperCalls = 0;
    context.window.getMetadataPreviewEntries = () => {
        helperCalls += 1;
        return [{ key: 'author', label: 'Author', previewValue: 'Ada' }];
    };

    const tooltip = { style: {}, innerHTML: '' };
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        tooltip,
        clusterKeywords: new Map(),
        getClusterColor: () => '#3B82F6',
        getClusterName: () => 'Cluster A'
    });
    const point = { index: 0, cluster: 1, text: 'Body', author: 'Ada' };

    const setPreference = value => {
        context.window.ConfigManager = { getConfig: () => ({ ui_preferences: { hover_metadata: value } }) };
    };

    setPreference(false);
    visualization.showTooltip(point, 0, 0);
    assert.doesNotMatch(tooltip.innerHTML, /tooltip-metadata/);
    assert.equal(helperCalls, 0, 'disabled preference should not query metadata at all');
    assert.match(tooltip.innerHTML, /Body/, 'the rest of the tooltip still renders');

    setPreference(true);
    visualization.showTooltip(point, 0, 0);
    assert.match(tooltip.innerHTML, /tooltip-metadata/);
    assert.equal(helperCalls, 1);

    // Unset preference and absent ConfigManager both default to showing metadata.
    context.window.ConfigManager = { getConfig: () => ({ ui_preferences: {} }) };
    visualization.showTooltip(point, 0, 0);
    assert.match(tooltip.innerHTML, /tooltip-metadata/);

    delete context.window.ConfigManager;
    visualization.showTooltip(point, 0, 0);
    assert.match(tooltip.innerHTML, /tooltip-metadata/);
});

test('tooltip omits the metadata block when no fields are available', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);

    const tooltip = { style: {}, innerHTML: '' };
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        tooltip,
        clusterKeywords: new Map(),
        getClusterColor: () => '#3B82F6',
        getClusterName: () => 'Cluster A'
    });

    // No helper on window at all: the tooltip still renders, minus metadata.
    visualization.showTooltip({ index: 0, cluster: 1, text: 'Body' }, 0, 0);
    assert.doesNotMatch(tooltip.innerHTML, /tooltip-metadata/);
    assert.match(tooltip.innerHTML, /Body/);

    context.window.getMetadataPreviewEntries = () => [];
    visualization.showTooltip({ index: 0, cluster: 1, text: 'Body' }, 0, 0);
    assert.doesNotMatch(tooltip.innerHTML, /tooltip-metadata/);
});

test('hidden outliers are excluded from both visualization hit-test paths', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);
    const visualization = Object.create(context.CanvasVisualization.prototype);
    const outlier = { index: 0, x: 0, y: 0, cluster: -1 };
    const visible = { index: 1, x: 20, y: 20, cluster: 2 };
    Object.assign(visualization, {
        data: [outlier, visible],
        outliersHidden: true,
        offsetX: 0,
        offsetY: 0,
        zoomScale: 1,
        spatialIndex: null,
        gridConfig: null
    });
    assert.equal(visualization.findPointUnderMouse(0, 0), null);
    assert.equal(visualization.findPointUnderMouse(20, 20), visible);

    Object.assign(visualization, {
        spatialIndex: new Map([['0,0', [0]], ['1,1', [1]]]),
        gridConfig: { cellW: 20, cellH: 20 },
        worldToCell: (x, y) => ({ ci: Math.floor(x / 20), cj: Math.floor(y / 20) }),
        cellKey: (i, j) => `${i},${j}`
    });
    assert.equal(visualization.findPointUnderMouse(0, 0), null);
    assert.equal(visualization.findPointUnderMouse(20, 20), visible);
});

test('chat source previews cannot revive a hidden outlier', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        data: [{ index: 0, x: 0, y: 0, cluster: -1 }],
        outliersHidden: true,
        chatPreviewPoint: null,
        selectionPulse: null,
        cancelDeferredTooltip() {},
        hideTooltip() {},
        requestRender() {},
        startSelectionPulse() { throw new Error('hidden outlier must not pulse'); }
    });
    assert.equal(visualization.previewChatPoint(0), false);
    assert.equal(visualization.chatPreviewPoint, null);
});

test('chat source preview temporarily owns point focus and dims its peers', () => {
    const context = browserContext();
    vm.runInContext(visualizationSource, context);
    const visualization = Object.create(context.CanvasVisualization.prototype);
    Object.assign(visualization, {
        data: [
            { index: 0, x: 0, y: 0, cluster: 1, doc_id: 'a' },
            { index: 1, x: 10, y: 10, cluster: 2, doc_id: 'b' }
        ],
        highlightedPoint: 1,
        chatPreviewPoint: 0,
        hoveredPoint: null,
        highlightedDocs: null,
        searchResults: null,
        searchResultsMap: null,
        metadataFilteredIndices: null,
        lassoSelectedIndices: new Set()
    });

    assert.equal(visualization.shouldDimPoint(0), false);
    assert.equal(visualization.shouldDimPoint(1), true);
    assert.equal(visualization.isPointHighlighted(0), true);

    visualization.chatPreviewPoint = null;
    assert.equal(visualization.shouldDimPoint(0), true);
    assert.equal(visualization.shouldDimPoint(1), false);
});

test('document metadata templates escape both keys and fallback values', () => {
    assert.match(vectoriaSource, /data-metadata-key="\$\{escapeHtml\(entry\.key\)\}"/);
    assert.match(vectoriaSource, /metadata-label">\$\{escapeHtml\(entry\.label\)\}/);
    assert.match(vectoriaSource, /unknown-value">\$\{escapeHtml\(String\(value\)\)\}/);
    assert.match(vectoriaSource, /safeClusterColor\(providedColor\)/);
    assert.match(vectoriaSource, /function metadataFilterId\(fieldName\)/);
    assert.match(vectoriaSource, /data-field="\$\{escapedFieldName\}"/);
    assert.match(vectoriaSource, /aria-label="\$\{removeLabel\}"/);
});

test('browser capability details escape adapter and recommendation text', () => {
    assert.match(capabilitiesSource, /capability-detail">\$\{escapeCapabilityHTML\(detail \|\| ''\)\}/);
    assert.match(capabilitiesSource, /<strong>\$\{escapeCapabilityHTML\(r\.title\)\}<\/strong>/);
    assert.match(capabilitiesSource, /aria-label="Close browser capabilities"/);
    assert.match(capabilitiesSource, /aria-label="Dismiss browser capability warning"/);
});

test('the unified search controller exclusively owns submit and clear-input events', () => {
    assert.match(searchSource, /searchBtn\.addEventListener\('click', \(\) => this\.performSearch\(\)\)/);
    assert.match(searchSource, /typeof window\.clearSearch === 'function'/);
    assert.doesNotMatch(vectoriaSource, /searchBtn\.addEventListener\('click', performSearch\)/);
    assert.doesNotMatch(vectoriaSource, /function debouncedSearch/);
    assert.match(vectoriaSource, /window\.clearSearch = clearSearch/);
});

test('search result previews use the FastSearch highlighter owned by the controller', () => {
    assert.match(searchSource, /this\.fastSearch\.highlightMatches\(previewText, matchedTerms\)/);
    assert.doesNotMatch(searchSource, /this\.highlightMatches\(previewText, matchedTerms\)/);
});

test('settings restrictions follow successful dataset lifecycle events', () => {
    const restrictionInitializer = vectoriaSource.slice(
        vectoriaSource.indexOf('function initializeSettingRestrictions()'),
        vectoriaSource.indexOf('function handleFileProcessingStart()')
    );
    assert.match(vectoriaSource, /const clusteringRestrictedSettings = \[/);
    assert.match(restrictionInitializer, /document\.addEventListener\('vectoria:dataset-changed'/);
    assert.match(restrictionInitializer, /if \(event\.detail\?\.datasetId\) handleFileProcessingStart\(\)/);
    assert.doesNotMatch(restrictionInitializer, /process-csv-btn/);
});
