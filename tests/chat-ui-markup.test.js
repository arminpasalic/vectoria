import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const index = await readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8');
const css = await readFile(new URL('../web_interface/static/css/main.css', import.meta.url), 'utf8');
const chatInterface = await readFile(new URL('../web_interface/static/js/chat-interface.js', import.meta.url), 'utf8');
const fastSearch = await readFile(new URL('../web_interface/static/js/fast-search.js', import.meta.url), 'utf8');
const vectoria = await readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8');
const serviceWorker = await readFile(new URL('../web_interface/sw.js', import.meta.url), 'utf8');

test('Ask workspace exposes accessible tabs, controls, settings, and exports', () => {
    assert.match(index, /class="workspace-view-tabs" role="tablist"/);
    assert.match(index, /id="workspace-documents-tab"[^>]*role="tab"[^>]*aria-controls="documents-workspace"/);
    assert.match(index, /id="workspace-chat-tab"[^>]*role="tab"[^>]*aria-controls="chat-workspace"/);
    assert.match(index, /id="chat-workspace"[^>]*role="tabpanel"[^>]*aria-labelledby="workspace-chat-tab"/);
    for (const id of [
        'chat-hyde-toggle',
        'chat-source-count',
        'chat-similarity-threshold',
        'chat-metadata-mode',
        'chat-metadata-fields',
        'chat-memory-mode',
        'chat-memory-turns',
        'chat-open-rag-settings',
        'chat-open-model-settings',
        'chat-trim-memory',
        'chat-restore-memory'
    ]) assert.match(index, new RegExp(`id="${id}"`));
    assert.match(index, /data-chat-export="json"/);
    assert.match(index, /data-chat-export="csv"/);
    assert.match(index, /data-chat-export="markdown"/);
    assert.match(index, /id="chat-live-status"[^>]*role="status"[^>]*aria-live="polite"/);
    assert.doesNotMatch(index, /id="chat-mode-select"/);
    assert.doesNotMatch(index, /id="quick-conversation-system-prompt"/);
    assert.doesNotMatch(fastSearch, /quick-conversation-system-prompt/);
    assert.doesNotMatch(index, /searches the captured document scope for every substantive answer/i);
    assert.match(index, /AI-generated answers can be inaccurate\. Verify important claims in the cited sources\./);
    assert.match(index, /does not filter or rerank search/i);
});

test('HyDE review is editable, modal, and exposes exactly the three workflow actions', () => {
    assert.match(index, /id="hyde-review-modal"[^>]*role="dialog"[^>]*aria-modal="true"/);
    assert.match(index, /id="hyde-generated-text"/);
    assert.match(index, /id="hyde-cancel"[^>]*>[\s\S]*?Cancel turn/);
    assert.match(index, /id="hyde-without"[^>]*>[\s\S]*?Search without HyDE/);
    assert.match(index, /id="hyde-search"[^>]*>\s*<i[^>]*><\/i> Search with this draft/);
    assert.match(index, /may contain invented details/i);
});

test('documents-only turns retain source behavior and transient visualization previews', () => {
    assert.doesNotMatch(chatInterface, /async function searchDocumentsForTurn\(/);
    assert.match(chatInterface, /if \(assistantMessage\.route\.resolved === 'documents'\)[\s\S]*?extractCitations/);
    assert.match(chatInterface, /previewChatPoint/);
    assert.match(chatInterface, /clearChatPreview/);
    assert.match(chatInterface, /pulseChatDocuments/);
    assert.match(chatInterface, /`Rank \$\{rank\}`/);
    assert.doesNotMatch(chatInterface, /return `Score \$\{/);
    assert.match(chatInterface, /cited · \$\{sources\.length\} context sources/);
    assert.match(chatInterface, /pointercancel/);
    assert.match(chatInterface, /pointerout/);
    assert.match(chatInterface, /sourcePreviewOwner/);
    assert.match(chatInterface, /!owner\.contains\(event\.target\)/);
    assert.match(chatInterface, /chat-unavailable-citation/);
    assert.match(chatInterface, /source that was not available/);
    assert.match(chatInterface, /clearSourcePreview\(\);[\s\S]*?content\.replaceChildren/);
    assert.match(chatInterface, /const requestId = \+\+state\.loadRequestId;[\s\S]*?if \(requestId !== state\.loadRequestId\) return;/);
});

test('chat typography remains compact without falling below readable sizes', () => {
    assert.match(css, /\.chat-message-content\s*\{[\s\S]*?font-size:\s*14px[\s\S]*?line-height:\s*1\.55/);
    assert.match(css, /#chat-input\s*\{[\s\S]*?font-size:\s*14px/);
    assert.match(css, /\.chat-stop-btn\[hidden\]\s*\{[\s\S]*?display:\s*none\s*!important/);
});

test('MCP handoff and responsive sticky composer are present', () => {
    assert.match(index, /id="chat-external-note"/);
    assert.match(index, /Website Ask is paused in AI-client mode/);
    assert.match(index, /id="chat-enable-local-btn"/);
    assert.match(css, /@media\s*\(max-width:\s*900px\)[\s\S]*?\.chat-composer-shell\s*\{[\s\S]*?position:\s*sticky/);
});

test('chat composer receives focus on chat entry and after an assistant turn', () => {
    assert.match(chatInterface, /function focusChatInput\(\)/);
    assert.match(chatInterface, /state\.activeView !== 'chat'[\s\S]*?elements\.input\.disabled[\s\S]*?elements\.chatWorkspace\?\.hidden/);
    assert.match(chatInterface, /elements\.input\.focus\(\{ preventScroll: true \}\)/);
    assert.match(chatInterface, /function showChat[\s\S]*?focusChatInput\(\);/);
    assert.match(chatInterface, /finally \{[\s\S]*?renderConversation\(\);[\s\S]*?focusChatInput\(\);/);
    assert.doesNotMatch(chatInterface, /scrollIntoView/);
});

test('local chat explains model setup and first-run dismissal keeps the app open', () => {
    assert.match(index, /id="chat-model-note"/);
    assert.match(index, /id="chat-open-model-setup"/);
    assert.match(index, /Explore without models/);
    assert.match(index, /modelSetupLaterBtn\.addEventListener\('click', dismissModelSetup\)/);
    assert.match(index, /modelSetupCloseBtn\.addEventListener\('click', dismissModelSetup\)/);
    assert.doesNotMatch(index, /You can close this tab now/);
});

test('icon-only modal close buttons have accessible labels', () => {
    const closeButtons = [...index.matchAll(/<button\b[^>]*class="[^"]*modal-close[^"]*"[^>]*>/g)].map(match => match[0]);
    assert.ok(closeButtons.length >= 8);
    for (const button of closeButtons) assert.match(button, /aria-label="[^"]+"/);
});

test('Ask setup and export popovers soften and disable the investigation behind them', () => {
    assert.match(index, /id="chat-settings-close"[^>]*aria-label="Close Ask settings"/);
    assert.match(css, /\.chat-popover-open\s+\.workspace-panel-header\s*\{[\s\S]*?z-index:\s*20/);
    assert.match(css, /\.chat-popover-open\s+\.chat-workspace\s*\{[\s\S]*?filter:\s*blur\(3px\)[\s\S]*?opacity:\s*0\.38[\s\S]*?pointer-events:\s*none/);
    assert.match(css, /\.chat-settings-panel\.chat-popover-portal/);
    assert.match(chatInterface, /workspaceCard\?\.classList\.toggle\('chat-popover-open', popoverOpen\)/);
    assert.match(chatInterface, /elements\.chatWorkspace\.inert = popoverOpen/);
    assert.match(chatInterface, /document\.body\.append\(elements\.settingsPanel\)/);
});

test('all processed-data lock indicators use the warning color', () => {
    assert.match(css, /\.restriction-indicator,\s*\n\.clustering-restriction-indicator\s*\{[\s\S]*?color:\s*var\(--warning-color\)/);
    assert.doesNotMatch(vectoria, /clustering-restriction-indicator[\s\S]{0,400}var\(--info-color\)/);
});

test('clear filters hover remains theme-aware and readable', () => {
    assert.match(css, /\.clear-filters-btn:hover\s*\{[\s\S]*?background:\s*var\(--control-bg-hover\)/);
    assert.match(css, /\.clear-filters-btn:hover\s*\{[\s\S]*?color:\s*var\(--text-primary\)/);
    assert.doesNotMatch(css, /\.clear-filters-btn:hover\s*\{[^}]*rgba\(255,\s*255,\s*255,\s*0\.9\)/);
});

test('service worker updates bypass the HTTP cache and precache every chat module', () => {
    assert.match(index, /register\('\/sw\.js\?v=\d{4}-\d{2}-\d{2}-[a-f0-9]+', \{ updateViaCache: 'none' \}\)/);
    assert.doesNotMatch(index, /(?:src|href)="static\/(?:css|js)\/[^"?]+"/);
    assert.match(serviceWorker, /v\('\/static\/js\/chat-interface\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/chat-context\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/chat-export\.js'\)/);
    assert.doesNotMatch(serviceWorker, /claim-validation|legacy-grounding/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/retrieval-ranking\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/reranker\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/reranker-worker\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/generation-mode-controller\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/browser-ml\/chat-store\.js'\)/);
    assert.match(serviceWorker, /v\('\/static\/js\/dom-safety\.js'\)/);
    assert.match(serviceWorker, /v\('\/'\)/);
    assert.match(serviceWorker, /v\('\/index\.html'\)/);
    assert.match(serviceWorker, /request\.mode === 'navigate'[\s\S]*?fetch\(request, \{ cache: 'no-store' \}\)[\s\S]*?caches\.match\(v\('\/index\.html'\)\)/);
});
