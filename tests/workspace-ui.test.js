import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = path => readFile(new URL(path, import.meta.url), 'utf8');
const [index, mainCss, exploreCss, controller, state, vectoria, chat, fastSearch, viz, config, serviceWorker] = await Promise.all([
    read('../web_interface/index.html'), read('../web_interface/static/css/main.css'),
    read('../web_interface/static/css/explore-workbench.css'), read('../web_interface/static/js/workspace-controller.js'),
    read('../web_interface/static/js/workspace-state.js'), read('../web_interface/static/js/vectoria.js'),
    read('../web_interface/static/js/chat-interface.js'), read('../web_interface/static/js/fast-search.js'),
    read('../web_interface/static/js/viz.js'),
    read('../web_interface/static/js/config-manager.js'), read('../web_interface/sw.js')
]);

test('Explore exposes panes, live status, labelled popovers, and an accessible splitter', () => {
    for (const pane of ['map', 'documents', 'chat', 'filters']) assert.match(index, new RegExp(`data-workspace-pane="${pane}"`));
    assert.match(index, /id="workspace-filter-rail"[^>]*aria-controls="workspace-filters"[^>]*aria-expanded="false"/);
    assert.match(index, /id="workspace-splitter"[^>]*role="separator"[^>]*aria-orientation="vertical"[^>]*aria-controls="workspace-evidence-panel"[^>]*aria-valuemin="22"[^>]*aria-valuenow="38"[^>]*aria-valuemax="55"/);
    assert.match(index, /id="workspace-live-status"[^>]*role="status"[^>]*aria-live="polite"/);
    assert.match(index, /id="dataset-actions-btn"[^>]*aria-haspopup="dialog"[^>]*aria-controls="dataset-actions-menu"/);
    assert.match(index, /id="dataset-actions-menu"[^>]*role="dialog"[^>]*aria-label="Dataset actions"/);
    assert.doesNotMatch(index, /id="dataset-actions-menu"[^>]*role="menu"/);
    const datasetMenu = index.slice(index.indexOf('id="dataset-actions-menu"'), index.indexOf('</div>', index.indexOf('id="cluster-label-toolbar"')));
    assert.match(datasetMenu, /id="export-selection-btn"[^>]*title="Export full dataset"/);
    assert.equal((index.match(/id="export-selection-btn"/g) || []).length, 1);
    assert.doesNotMatch(index, /id="documents-workspace-footer"/);
    assert.match(index, /id="viz-more-menu"[^>]*role="menu"/);
});

test('Explore design authority is isolated, loaded last, stamped, and precached', () => {
    const mainLink = index.indexOf('static/css/main.css?v=');
    const browserLink = index.indexOf('static/css/browser-ml.css?v=');
    const exploreLink = index.indexOf('static/css/explore-workbench.css?v=');
    assert.ok(mainLink >= 0 && browserLink > mainLink && exploreLink > browserLink);
    assert.doesNotMatch(mainCss, /Viewport-bound Explore workbench|Explore toolbar and filter drawer refinements|Final workbench sizing authority|explore-workbench-active/);
    assert.match(exploreCss, /^\/\* Explore-only viewport workbench/);
    assert.doesNotMatch(exploreCss, /(^|\n)\s*:root\s*\{|(^|\n)\s*html\.(?:light|dark)\s*\{|(^|\n)\s*\.btn\s*\{/);
    assert.ok(serviceWorker.includes("v('/static/css/explore-workbench.css')"));
});

test('workbench removes old scroll, staged-filter, and destructive interaction contracts', () => {
    assert.doesNotMatch(mainCss + exploreCss, /--text-panel-height/);
    assert.doesNotMatch(chat, /scrollIntoView|bringIntoView/);
    assert.doesNotMatch(vectoria, /Auto-apply 800ms|Click to select • Double-click to apply/);
    assert.doesNotMatch(index, />Apply Filters<|>Reset Changes|Unapplied changes/);
    assert.doesNotMatch(state, /createFilterStage|stageFilters|hasUnappliedFilters|resetStagedFilters/);
    assert.doesNotMatch(vectoria, /updateFilterDirtyState|has-unapplied-changes/);
    assert.doesNotMatch(vectoria, /Cannot highlight search results: missing visualization or results/);
});

test('the active workspace controller owns the search keyboard shortcut', () => {
    assert.match(controller, /event\.ctrlKey \|\| event\.metaKey/);
    assert.match(controller, /event\.key\.toLowerCase\(\) !== 'k'/);
    assert.match(controller, /byId\('search-input'\)[\s\S]*?input\.focus\(\{ preventScroll: true \}\)[\s\S]*?input\.select\(\)/);
});

test('filters are a closed 288px desktop drawer and categories use additive double-click toggles', () => {
    assert.match(exploreCss, /#workspace-filters\s*>\s*\.card\s*\{[\s\S]*?display:\s*none/);
    assert.match(exploreCss, /#workspace-filters\.open\s*>\s*\.card\s*\{[\s\S]*?width:\s*min\(288px/);
    assert.match(controller, /elements\.filters\?\.classList\.contains\('open'\)[\s\S]*?closeFilters\(\)/);
    assert.match(vectoria, /function scheduleMetadataFilterApply\(immediate = false\)/);
    assert.match(vectoria, /metadataFilterApplyTimer = setTimeout\(apply, 250\)/);
    assert.match(vectoria, /option\?\.addEventListener\('dblclick'[\s\S]*?toggleCategoryCheckbox\(checkbox\)/);
    assert.match(vectoria, /checkbox\.addEventListener\('keydown'[\s\S]*?event\.key !== 'Enter'[\s\S]*?event\.key !== ' '/);
    assert.match(index, /Double-click<\/strong> a value to add or remove it/);
    assert.match(mainCss, /\.checkbox-group\s*\{[\s\S]*?flex-flow:\s*row wrap[\s\S]*?background:\s*transparent/);
    assert.match(mainCss, /\.checkbox-option:has\(input\[type="checkbox"\]:checked\)/);
});

test('the filter drawer stays open during workspace interaction and closes deliberately', () => {
    const outsidePointerHandler = controller.slice(
        controller.indexOf("document.addEventListener('pointerdown'"),
        controller.indexOf("window.addEventListener('resize'")
    );
    assert.doesNotMatch(outsidePointerHandler, /closeFilters\(/);
    assert.match(controller, /workspace-filter-close'\)\?\.addEventListener\('click', \(\) => closeFilters\(\)\)/);
    assert.match(controller, /function handleEscape\(\)[\s\S]*?filters\?\.classList\.contains\('open'\)[\s\S]*?closeFilters\(\)/);
});

test('toolbar has fixed desktop controls and equal mobile icon targets without wrapping', () => {
    assert.match(exploreCss, /grid-template-columns:\s*minmax\(380px, 1fr\) auto 132px/);
    assert.match(exploreCss, /#search-btn,[\s\S]*?#dataset-actions-btn[\s\S]*?width:\s*132px/);
    assert.match(exploreCss, /#search-type\s*\{[\s\S]*?width:\s*180px/);
    assert.match(exploreCss, /#result-count\s*\{[\s\S]*?width:\s*150px/);
    assert.match(exploreCss, /@media \(max-width:\s*959px\)[\s\S]*?#search-btn,[\s\S]*?\.explore-options-btn[\s\S]*?width:\s*40px;[\s\S]*?height:\s*40px/);
    assert.match(index, /id="explore-options-btn"[^>]*aria-label="Explore options"[^>]*aria-controls="explore-options-panel"/);
    assert.doesNotMatch(index, /id="explore-options-btn"[\s\S]{0,180}fa-ellipsis/);
});

test('Highlight lives only in Search & answers settings and is persisted with an enabled default', () => {
    assert.equal((index.match(/id="highlight-results"/g) || []).length, 1);
    const settingsStart = index.indexOf('id="settings-panel-rag"');
    assert.ok(index.indexOf('id="highlight-results"') > settingsStart);
    assert.match(config, /ui_preferences:\s*\{[\s\S]*?highlight_results:\s*true/);
    assert.match(fastSearch, /highlight_results:\s*enabled/);
    assert.match(fastSearch, /highlightToggle\.checked = config\.ui_preferences\?\.highlight_results \?\? true/);
    assert.match(vectoria, /newConfig\.ui_preferences\?\.highlight_results/);
});

test('every submitted search opens Documents without touching the Chat draft', () => {
    const submit = fastSearch.slice(fastSearch.indexOf('performSearch() {'), fastSearch.indexOf('performFastSearch(query)'));
    assert.match(submit, /window\.VectoriaWorkspace\?\.showSearchResults\?\.\(\)/);
    assert.doesNotMatch(submit, /chat-input|conversation|VectoriaChat/);
    assert.match(controller, /function showSearchResults\(\)[\s\S]*?showPane\('documents', \{ focus: false \}\)[\s\S]*?list\.scrollTop = 0/);
});

test('document list batches, delegates, exposes a fallback, and marks batch work busy', () => {
    assert.match(vectoria, /const TEXT_LIST_BATCH_SIZE = 100/);
    assert.match(vectoria, /new IntersectionObserver/);
    assert.match(vectoria, /container\.addEventListener\('click', activateRow\)/);
    assert.match(vectoria, /container\.addEventListener\('keydown'/);
    assert.match(vectoria, /container\.setAttribute\('aria-busy', 'true'\)[\s\S]*?container\.setAttribute\('aria-busy', 'false'\)/);
    assert.match(vectoria, /pagination\.hidden = !hasMore \|\| Boolean\(textListObserver\)/);
    assert.match(index, />Show 100 more documents</);
    const renderer = vectoria.slice(vectoria.indexOf('function renderTextListItem'), vectoria.indexOf('function showTextDetails'));
    assert.doesNotMatch(renderer, /addEventListener/);
});

test('canvas More is narrow-only and keyboard contracts are implemented', () => {
    assert.match(exploreCss, /@media \(min-width:\s*960px\)[\s\S]*?#viz-more-btn,[\s\S]*?#viz-more-menu[\s\S]*?display:\s*none !important/);
    assert.match(exploreCss, /@media \(max-width:\s*959px\)[\s\S]*?#viz-more-btn\s*\{[\s\S]*?display:\s*inline-flex/);
    assert.match(controller, /\['ArrowLeft', 'ArrowRight', 'Home', 'End'\]/);
    assert.match(controller, /event\.key === 'ArrowDown'/);
    assert.match(controller, /items\[next\]\?\.focus\(\{ preventScroll: true \}\)/);
    assert.match(controller, /focusFirstControl\(surface\)/);
});

test('Escape consumes chat, workspace surfaces, then lasso exactly in that order', () => {
    const handler = vectoria.slice(vectoria.indexOf("if (event.key !== 'Escape'"), vectoria.indexOf('// Lightweight scroll performance'));
    const chatAt = handler.indexOf('VectoriaChat?.closeTopSurface');
    const workspaceAt = handler.indexOf('VectoriaWorkspace?.handleEscape');
    const canvasAt = handler.indexOf('mainVisualization?.cancelActiveTool');
    assert.ok(chatAt >= 0 && workspaceAt > chatAt && canvasAt > workspaceAt);
    assert.match(chat, /function closeTopSurface\(\)[\s\S]*?return true;[\s\S]*?return false;/);
    assert.match(controller, /function handleEscape\(\)[\s\S]*?if \(closeTopMenu\(\)\) return true;[\s\S]*?closeFilters\(\)/);
    assert.match(viz, /cancelActiveTool\(\)\s*\{[\s\S]*?if \(!this\.lassoMode\) return false;[\s\S]*?return true;/);
    assert.doesNotMatch(chat, /addEventListener\('keydown',[\s\S]{0,180}Escape/);
    assert.doesNotMatch(viz, /Escape key cancels lasso mode/);
});

test('workspace controller keeps split persistence, resize observation, and touch canvas compatibility', () => {
    assert.match(controller, /vectoria_workspace_split_v1|WORKSPACE_SPLIT\.key/);
    assert.match(controller, /new ResizeObserver/);
    assert.match(controller, /vectoria:workspace-pane-changed/);
    assert.match(controller, /window\.VectoriaWorkspace = \{ showPane, showSearchResults, openFilters, closeFilters, refreshLayout, handleEscape \}/);
    assert.match(viz, /addEventListener\('pointerdown'/);
    assert.match(viz, /activePointers\.size === 2/);
    assert.match(viz, /window\.devicePixelRatio/);
});

test('footer stays global and workspace modules remain stamped and precached', () => {
    assert.ok(index.indexOf('<footer>') > index.indexOf('</main>'));
    assert.match(exploreCss, /body\.explore-workbench-active\s*>\s*footer\s*\{[\s\S]*?display:\s*none/);
    assert.match(index, /static\/js\/workspace-controller\.js\?v=/);
    for (const asset of ['workspace-state.js', 'workspace-controller.js']) {
        assert.ok(serviceWorker.includes(`v('/static/js/${asset}')`));
    }
});
