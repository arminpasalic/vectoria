import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const index = await readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8');
const css = await readFile(new URL('../web_interface/static/css/main.css', import.meta.url), 'utf8');
const vectoria = await readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8');
const fastSearch = await readFile(new URL('../web_interface/static/js/fast-search.js', import.meta.url), 'utf8');
const chatInterface = await readFile(new URL('../web_interface/static/js/chat-interface.js', import.meta.url), 'utf8');

test('search modes use concise labels with keyword first and selected by default', () => {
    assert.match(index, /<option value="fast" selected>Keyword<\/option>[\s\S]*?<option value="semantic">Semantic<\/option>[\s\S]*?<option value="hybrid">Hybrid<\/option>/);
    assert.match(fastSearch, /fastOption\.textContent = 'Keyword'[\s\S]*?semanticOption\.textContent = 'Semantic'[\s\S]*?hybridOption\.textContent = 'Hybrid'/);
    assert.match(fastSearch, /const defaultSearchType = [\s\S]*?\? savedSearchType[\s\S]*?: 'fast'/);
});

test('settings use six accessible categories while preserving internal keys', () => {
    assert.match(index, /class="settings-nav" role="tablist" aria-label="Settings categories"/);
    const categories = [
        ['storage', 'General'],
        ['models', 'Models'],
        ['processing', 'Data preparation'],
        ['rag', 'Search &amp; answers'],
        ['explore', 'Projection &amp; clustering'],
        ['mcp', 'MCP integration']
    ];
    for (const [key, label] of categories) {
        assert.match(index, new RegExp(`id="settings-tab-${key}"[\\s\\S]*?role="tab"[\\s\\S]*?aria-controls="settings-panel-${key}"[\\s\\S]*?data-category="${key}"[\\s\\S]*?${label}`));
        assert.match(index, new RegExp(`id="settings-panel-${key}"[^>]*role="tabpanel"[^>]*aria-labelledby="settings-tab-${key}"[^>]*data-category="${key}"`));
    }
    assert.match(vectoria, /ArrowDown'[\s\S]*?ArrowRight'[\s\S]*?ArrowUp'[\s\S]*?ArrowLeft'/);
    assert.match(vectoria, /button\.setAttribute\('aria-selected', String\(isSelected\)\)/);
    assert.match(vectoria, /panel\.hidden = !isSelected/);
});

test('settings have a persistent app-header entry and restore invoking focus', () => {
    assert.match(index, /id="header-settings-btn"[^>]*aria-label="Open settings"/);
    assert.doesNotMatch(index, /id="header-theme-toggle"/);
    assert.match(vectoria, /headerSettingsButton\.addEventListener\('click'[\s\S]*?openAdvancedSettingsModal\('storage'\)/);
    assert.match(vectoria, /advancedSettingsInvoker = activeElement/);
    assert.match(vectoria, /advancedSettingsInvoker\?\.isConnected[\s\S]*?advancedSettingsInvoker\.focus\(\)/);
});

test('theme switching is a full-width settings action labelled with the destination theme', () => {
    assert.match(index, /id="theme-toggle"[^>]*class="theme-toggle-setting-btn"[^>]*aria-label="Switch to dark mode"/);
    assert.match(index, /id="theme-toggle-label">Dark mode/);
    assert.match(css, /\.theme-toggle-setting-btn\s*\{[\s\S]*?width:\s*100%[\s\S]*?min-height:\s*46px/);
    assert.match(vectoria, /const targetTheme = normalized === 'dark' \? 'light' : 'dark'/);
    assert.match(vectoria, /label\.textContent = actionText/);
    assert.match(vectoria, /btn\.classList\.toggle\('is-dark', normalized === 'dark'\)/);
});

test('settings controls share aligned icons, inputs, and right-aligned toggles', () => {
    assert.match(css, /#advanced-settings-modal \.settings-nav-icon > i\s*\{[\s\S]*?place-items:\s*center[\s\S]*?width:\s*16px[\s\S]*?height:\s*16px/);
    assert.match(css, /#advanced-settings-modal input\[type="number"\]\s*\{[\s\S]*?width:\s*100%[\s\S]*?background:\s*var\(--control-bg\)/);
    assert.match(css, /#advanced-settings-modal \.rag-settings-toggle\s*\{[\s\S]*?grid-template-columns:\s*minmax\(0, 1fr\) 44px/);
    assert.match(css, /#advanced-settings-modal \.rag-settings-toggle \.rag-toggle-slider\s*\{[\s\S]*?grid-column:\s*2/);
    assert.match(index, /class="rag-toggle-title"><i class="fas fa-tags"[^>]*><\/i><span>Metadata on hover<\/span>/);
    assert.doesNotMatch(index, /Page reload required/);
});

test('General reset actions align and About names the MCP boundary', () => {
    assert.match(css, /\.settings-reset-option \.btn\s*\{[\s\S]*?width:\s*150px[\s\S]*?justify-content:\s*center/);
    assert.match(index, /In AI client mode via MCP, only retrieved excerpts or cluster examples are shared with your selected provider\./);
});

test('legacy quick-settings shells and duplicate action models are removed', () => {
    assert.doesNotMatch(index, /id="quick-settings-modal"/);
    assert.doesNotMatch(index, /id="apply-quick-settings"/);
    assert.doesNotMatch(index, /id="legacy-generation-controls"/);
    assert.doesNotMatch(index, /id="advanced-storage-settings-host"/);
    assert.doesNotMatch(index, /id="save-settings-btn"/);
    assert.doesNotMatch(vectoria, /appendChild\(storageSection\)|appendChild\(ragBody\)/);
    assert.doesNotMatch(fastSearch, /document\.getElementById\('quick-settings-modal'\)/);
});

test('all element ids are unique and existing settings controls remain addressable', () => {
    const ids = [...index.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    const duplicates = [...new Set(ids.filter((id, position) => ids.indexOf(id) !== position))];
    assert.deepEqual(duplicates, []);

    for (const id of [
        'llm-model-id', 'context-window-size',
        'embedding-batch-size', 'embedding-max-length', 'embedding-tokens-per-batch',
        'chunking-enabled', 'chunking-strategy', 'chunk-size', 'chunk-overlap', 'min-chunk-size',
        'semantic-threshold', 'semantic-similarity-window', 'semantic-filter-window',
        'semantic-filter-polyorder', 'semantic-filter-tolerance', 'semantic-skip-window',
        'code-language', 'table-mode', 'table-rows-per-chunk',
        'umap-n-neighbors', 'umap-min-dist', 'umap-metric', 'umap-clustering-dimensions',
        'hdbscan-min-cluster-size', 'hdbscan-min-samples', 'hdbscan-metric',
        'hyde-mode-toggle', 'quick-retrieval-k', 'quick-vector-weight', 'quick-temperature',
        'quick-max-tokens', 'quick-top-p', 'quick-repeat-penalty', 'quick-system-prompt',
        'quick-user-template', 'hyde-temperature', 'hyde-max-tokens', 'hyde-prompt',
        'generation-mode-local', 'generation-mode-external', 'mcp-bridge-enabled', 'mcp-bridge-enabled-connected',
        'theme-toggle', 'check-storage-btn', 'reset-settings-btn', 'clear-model-cache-btn'
    ]) assert.ok(ids.includes(id), `missing preserved control #${id}`);
});

test('configuration collection reads the active Search & answers controls', () => {
    for (const id of [
        'quick-temperature', 'quick-max-tokens', 'quick-top-p', 'quick-repeat-penalty',
        'quick-system-prompt', 'quick-user-template', 'quick-retrieval-k', 'quick-vector-weight'
    ]) assert.match(vectoria, new RegExp(`['"]${id}['"]`));
    assert.doesNotMatch(vectoria, /getValueSafe\('(temperature|max-tokens|top-p|repeat-penalty)'/);
    assert.doesNotMatch(vectoria, /getFromDomOrConfig\('(system-prompt|user-template|rag-retrieval-k)'/);
});

test('desktop and mobile settings navigation use the canonical persistent shell', () => {
    assert.match(css, /\.settings-categories\s*\{[\s\S]*?grid-template-columns:\s*190px minmax\(0, 1fr\)/);
    assert.match(css, /\.settings-content\s*\{[\s\S]*?overflow-y:\s*auto/);
    assert.match(css, /@media \(max-width:\s*768px\)[\s\S]*?\.settings-nav\s*\{[\s\S]*?flex-direction:\s*row[\s\S]*?overflow-x:\s*auto/);
    assert.match(css, /@media \(max-width:\s*768px\)[\s\S]*?\.settings-nav-btn \.settings-nav-label\s*\{\s*display:\s*inline-flex/);
});

test('chat setup groups and reveals only applicable dependent fields', () => {
    assert.match(index, /id="chat-retrieval-settings-title">Retrieval/);
    assert.match(index, /id="chat-memory-settings-title">Conversation context/);
    assert.match(index, /Applies to future questions until changed/);
    assert.match(index, /id="chat-metadata-fields-group"[^>]*hidden/);
    assert.match(chatInterface, /metadataFieldsGroup\.hidden = metadataMode !== 'selected'/);
    assert.match(chatInterface, /memoryTurnsGroup\.hidden = elements\.memoryMode\?\.value === 'none'/);
    assert.match(chatInterface, /window\.addEventListener\('resize', syncSettingsPanelPortal/);
});

test('model onboarding and settings confirmations share structured modal regions', () => {
    assert.match(index, /id="model-setup-settings-btn"[^>]*>Change the language model in Settings → Models/);
    assert.doesNotMatch(index, />1\. Embeddings model</);
    assert.doesNotMatch(index, />2\. Large Language Model</);
    for (const id of ['model-change-modal', 'context-window-change-modal', 'reset-settings-modal', 'reset-all-data-modal']) {
        assert.match(index, new RegExp(`id="${id}"[^>]*role="dialog"[^>]*aria-modal="true"`));
    }
    assert.equal((index.match(/class="settings-confirmation-actions"/g) || []).length, 4);
});
