/**
 * Local Chat with Documents UI.
 *
 * The website conversation is intentionally local-only. MCP clients keep
 * using the existing one-shot tools and render their conversation themselves.
 */

import { pipeline } from './browser-ml/index.js';
import { createChatMessage, createEmptyConversation } from './browser-ml/chat-store.js';
import {
    FALLBACK_SUGGESTIONS,
    buildSuggestionPrompt,
    parseSuggestionResponse,
    sampleDocumentsForSuggestions
} from './browser-ml/suggested-questions.js';
import {
    applyChatContextCutoff,
    captureChatScope,
    createChatOptionsSnapshot,
    extractCitations,
    normalizeLocalAIError,
    restoreChatScope,
    routeChatTurn,
    tokenizeCitations
} from './browser-ml/chat-context.js';

const state = {
    initialized: false,
    activeView: 'documents',
    conversation: null,
    datasetId: null,
    isGenerating: false,
    chatScrollTop: 0,
    renderFrame: null,
    loadRequestId: 0,
    suggestionRequestId: 0,
    sourcePreviewOwner: null,
    expandedSourceMessageIds: new Set()
};

const elements = {};
let settingsPanelHome = null;
let settingsPanelBreakpoint = null;

function cacheElements() {
    Object.assign(elements, {
        documentsTab: document.getElementById('workspace-documents-tab'),
        chatTab: document.getElementById('workspace-chat-tab'),
        documentsWorkspace: document.getElementById('documents-workspace'),
        chatWorkspace: document.getElementById('chat-workspace'),
        workspaceCard: document.getElementById('chat-workspace')?.closest('.fullheight-card'),
        chatToolbar: document.getElementById('chat-toolbar'),
        messages: document.getElementById('chat-messages'),
        empty: document.getElementById('chat-empty-state'),
        suggestions: document.getElementById('chat-suggestions'),
        input: document.getElementById('chat-input'),
        send: document.getElementById('chat-send-btn'),
        stop: document.getElementById('chat-stop-btn'),
        scope: document.getElementById('chat-scope-select'),
        live: document.getElementById('chat-live-status'),
        externalNote: document.getElementById('chat-external-note'),
        modelNote: document.getElementById('chat-model-note'),
        openModelSetup: document.getElementById('chat-open-model-setup'),
        localNote: document.getElementById('chat-local-note'),
        newChat: document.getElementById('chat-new-btn'),
        settings: document.getElementById('chat-settings-btn'),
        settingsClose: document.getElementById('chat-settings-close'),
        settingsPanel: document.getElementById('chat-settings-panel'),
        hydeToggle: document.getElementById('chat-hyde-toggle'),
        metadataMode: document.getElementById('chat-metadata-mode'),
        sourceCount: document.getElementById('chat-source-count'),
        similarityThreshold: document.getElementById('chat-similarity-threshold'),
        metadataFields: document.getElementById('chat-metadata-fields'),
        metadataFieldsGroup: document.getElementById('chat-metadata-fields-group'),
        memoryMode: document.getElementById('chat-memory-mode'),
        memoryTurns: document.getElementById('chat-memory-turns'),
        memoryTurnsGroup: document.getElementById('chat-memory-turns-group'),
        configSummary: document.getElementById('chat-config-summary'),
        exportChat: document.getElementById('chat-export-btn'),
        exportMenu: document.getElementById('chat-export-menu'),
        trimMemory: document.getElementById('chat-trim-memory'),
        memoryNote: document.getElementById('chat-memory-note'),
        restoreMemory: document.getElementById('chat-restore-memory'),
        enableLocal: document.getElementById('chat-enable-local-btn'),
        openRagSettings: document.getElementById('chat-open-rag-settings'),
        openModelSettings: document.getElementById('chat-open-model-settings'),
        backToList: document.getElementById('back-to-list-btn')
    });
    settingsPanelHome = elements.settingsPanel?.parentElement || null;
}

function isExternalMode() {
    return window.__vectoriaGenerationMode === 'external';
}

function currentDatasetId() {
    return pipeline.currentDatasetId || pipeline.currentDataset?.id || null;
}

function getConfig() {
    return window.ConfigManager?.getConfig?.() || {};
}

function getChatOptionsSnapshot() {
    return createChatOptionsSnapshot(getConfig());
}

function contextHistory(messages) {
    return applyChatContextCutoff(messages, state.conversation?.contextCutoffAt);
}

function showToast(message, type = 'info') {
    if (typeof window.showToast === 'function') window.showToast(message, type);
}

function setLiveStatus(message = '') {
    if (elements.live) elements.live.textContent = message;
}

function setActiveTab(tab, active) {
    if (!tab) return;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active ? 'true' : 'false');
    tab.tabIndex = active ? 0 : -1;
}

function focusChatInput() {
    requestAnimationFrame(() => {
        if (state.activeView !== 'chat'
            || !elements.input
            || elements.input.disabled
            || elements.chatWorkspace?.hidden
            || elements.chatWorkspace?.inert
            || (elements.settingsPanel && !elements.settingsPanel.hidden)
            || (elements.exportMenu && !elements.exportMenu.hidden)) return;
        elements.input.focus({ preventScroll: true });
    });
}

function showDocuments() {
    if (!elements.documentsWorkspace) return;
    if (state.activeView === 'chat' && elements.messages) state.chatScrollTop = elements.messages.scrollTop;
    state.activeView = 'documents';
    clearSourcePreview();
    elements.documentsWorkspace.hidden = false;
    if (elements.chatWorkspace) elements.chatWorkspace.hidden = true;
    if (elements.chatToolbar) elements.chatToolbar.hidden = true;
    closeExportMenu();
    closeSettingsPanel();
    setActiveTab(elements.documentsTab, true);
    setActiveTab(elements.chatTab, false);
    if (document.getElementById('explore-tab')?.dataset.workspaceMode === 'narrow') {
        window.VectoriaWorkspace?.showPane?.('documents', { skipWorkspaceTab: true, focus: false });
    }
}

function showChat({ restoreScroll = true } = {}) {
    if (!elements.chatWorkspace) return;
    state.activeView = 'chat';
    elements.documentsWorkspace.hidden = true;
    elements.chatWorkspace.hidden = false;
    if (elements.chatToolbar) elements.chatToolbar.hidden = false;
    setActiveTab(elements.documentsTab, false);
    setActiveTab(elements.chatTab, true);
    if (document.getElementById('explore-tab')?.dataset.workspaceMode === 'narrow') {
        window.VectoriaWorkspace?.showPane?.('chat', { skipWorkspaceTab: true, focus: false });
    }
    if (elements.backToList) elements.backToList.innerHTML = '<i class="fas fa-arrow-left"></i> Back to List';
    syncChatSettings();
    refreshAvailability();
    renderConversation();
    if (restoreScroll && elements.messages) {
        requestAnimationFrame(() => { elements.messages.scrollTop = state.chatScrollTop; });
    }
    focusChatInput();
}

function autoSizeComposer() {
    if (!elements.input) return;
    elements.input.style.height = 'auto';
    elements.input.style.height = `${Math.min(elements.input.scrollHeight, 132)}px`;
}

function getScopeSnapshot(preferred = elements.scope?.value || 'all') {
    const requested = preferred === 'current' ? 'current' : 'all';
    const info = typeof window.getCurrentRAGScope === 'function'
        ? window.getCurrentRAGScope(requested)
        : {
            scopeType: 'all',
            docIds: null,
            scopedCount: pipeline.currentDataset?.documents?.length || 0,
            totalDocuments: pipeline.currentDataset?.documents?.length || 0,
            label: 'All documents',
            details: ['Full dataset']
        };
    return {
        info,
        persisted: captureChatScope(info)
    };
}

function refreshScopeOption() {
    if (!elements.scope) return;
    const currentOption = elements.scope.querySelector('option[value="current"]');
    const snapshot = getScopeSnapshot('current');
    const hasScopedView = snapshot.info.scopeType !== 'all';
    if (currentOption) {
        const count = Number(snapshot.info.scopedCount) || 0;
        currentOption.textContent = hasScopedView ? `Current view (${count.toLocaleString()})` : 'Current view';
        currentOption.disabled = !hasScopedView || count === 0;
    }
    if ((!hasScopedView || snapshot.info.scopedCount === 0) && elements.scope.value === 'current') {
        elements.scope.value = 'all';
    }
}

function detectedMetadataNames() {
    const fields = typeof window.getDetectedMetadataFields === 'function'
        ? window.getDetectedMetadataFields()
        : (pipeline.currentDataset?.metadataSchema || []);
    return (Array.isArray(fields) ? fields : [])
        .filter(field => !field?.isTextColumn)
        .map(field => String(field?.name || field?.displayName || field || '').trim())
        .filter(Boolean);
}

function syncChatSettings(config = getConfig()) {
    const chat = config.chat || {};
    const selectedFields = new Set(Array.isArray(chat.metadata_fields) ? chat.metadata_fields.map(String) : []);
    const metadataMode = ['off', 'selected', 'all'].includes(chat.metadata_mode)
        ? chat.metadata_mode
        : (chat.include_metadata === false ? 'off' : selectedFields.size ? 'selected' : 'all');
    if (elements.hydeToggle) elements.hydeToggle.checked = config.ui_preferences?.hyde_enabled === true;
    if (elements.metadataMode) elements.metadataMode.value = metadataMode;
    if (elements.sourceCount) elements.sourceCount.value = String(Math.max(1, Math.min(20, Number(config.search?.num_results) || 5)));
    if (elements.similarityThreshold) {
        const threshold = Number(config.search?.similarity_threshold);
        elements.similarityThreshold.value = String(Math.max(0, Math.min(1, Number.isFinite(threshold) ? threshold : 0.7)));
    }
    if (elements.memoryMode) elements.memoryMode.value = ['adaptive', 'recent', 'none'].includes(chat.memory_mode) ? chat.memory_mode : 'adaptive';
    if (elements.memoryTurns) {
        elements.memoryTurns.value = String(Math.max(1, Math.min(50, Number(chat.max_memory_turns) || 8)));
        elements.memoryTurns.disabled = elements.memoryMode?.value === 'none';
    }
    if (elements.memoryTurnsGroup) elements.memoryTurnsGroup.hidden = elements.memoryMode?.value === 'none';
    if (elements.metadataFields) {
        const names = detectedMetadataNames();
        elements.metadataFields.replaceChildren(...names.map(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            option.selected = selectedFields.has(name);
            return option;
        }));
        elements.metadataFields.disabled = metadataMode !== 'selected' || names.length === 0;
    }
    if (elements.metadataFieldsGroup) elements.metadataFieldsGroup.hidden = metadataMode !== 'selected';
    refreshConfigSummary(config);
    refreshAvailability();
}

function refreshConfigSummary(config = getConfig()) {
    if (!elements.configSummary) return;
    const sourceCount = Number(config.search?.num_results) || 5;
    const rawWeight = Number(config.search?.vector_weight);
    const vectorPercent = Math.round((Number.isFinite(rawWeight) ? rawWeight : 0.6) * 100);
    const parts = [`Up to ${sourceCount} sources`, `${vectorPercent}/${100 - vectorPercent}`];
    if (config.ui_preferences?.hyde_enabled) parts.push('HyDE');
    const metadataMode = config.chat?.metadata_mode || (config.chat?.include_metadata === false ? 'off' : 'all');
    if (metadataMode !== 'off') {
        const count = config.chat?.metadata_fields?.length || 0;
        parts.push(metadataMode === 'selected' ? `${count} metadata` : 'All metadata');
    }
    elements.configSummary.textContent = parts.join(' · ');
    elements.configSummary.title = parts.join(' · ');
}

function updateChatConfig(updates) {
    window.ConfigManager?.updateConfig?.(updates);
    syncChatSettings();
}

function syncPopoverPresentation() {
    const settingsOpen = Boolean(elements.settingsPanel && !elements.settingsPanel.hidden);
    const exportOpen = Boolean(elements.exportMenu && !elements.exportMenu.hidden);
    const popoverOpen = settingsOpen || exportOpen;
    elements.workspaceCard?.classList.toggle('chat-popover-open', popoverOpen);

    // Keep the softened conversation out of the keyboard and pointer flow while
    // the foreground controls are open. The popovers live in the card header.
    if (elements.chatWorkspace) elements.chatWorkspace.inert = popoverOpen;
}

function syncSettingsPanelPortal() {
    if (!elements.settingsPanel || elements.settingsPanel.hidden) return;
    const usePortal = Boolean((settingsPanelBreakpoint || window.matchMedia?.('(max-width: 900px)'))?.matches);
    const isPortal = elements.settingsPanel.classList.contains('chat-popover-portal');
    if (usePortal && !isPortal) {
        document.body.append(elements.settingsPanel);
        elements.settingsPanel.classList.add('chat-popover-portal');
    } else if (!usePortal && isPortal) {
        elements.settingsPanel.classList.remove('chat-popover-portal');
        settingsPanelHome?.append(elements.settingsPanel);
    }

    if (usePortal) {
        elements.settingsPanel.style.removeProperty('max-height');
    } else {
        const top = elements.settingsPanel.getBoundingClientRect().top;
        const availableHeight = Math.max(240, Math.min(620, window.innerHeight - top - 16));
        elements.settingsPanel.style.maxHeight = `${availableHeight}px`;
    }
}

function closeSettingsPanel() {
    if (elements.settingsPanel) elements.settingsPanel.hidden = true;
    elements.settings?.setAttribute('aria-expanded', 'false');
    if (elements.settingsPanel?.classList.contains('chat-popover-portal')) {
        elements.settingsPanel.classList.remove('chat-popover-portal');
        settingsPanelHome?.append(elements.settingsPanel);
    }
    elements.settingsPanel?.style.removeProperty('max-height');
    syncPopoverPresentation();
}

function openAdvancedSettings(category) {
    closeSettingsPanel();
    if (typeof window.openAdvancedSettingsCategory === 'function') {
        window.openAdvancedSettingsCategory(category);
    }
}

function refreshAvailability() {
    const hasDataset = Boolean(currentDatasetId());
    const external = isExternalMode();
    const localModelReady = Boolean(pipeline.rag);
    const needsLocalSetup = hasDataset && !external && !localModelReady;
    const activeOperation = pipeline.rag?.activeOperation || null;
    const blockedByOtherAI = Boolean(activeOperation && activeOperation.owner !== 'chat');
    const cancellablePhases = new Set(['starting', 'loading-model', 'generating', 'awaiting-input', 'retrieving', 'reranking']);
    const chatCanStop = Boolean(state.isGenerating
        && activeOperation?.owner === 'chat'
        && cancellablePhases.has(activeOperation.phase));
    const processing = pipeline.isProcessing === true;
    const localUnavailable = blockedByOtherAI || processing;
    refreshScopeOption();
    if (elements.externalNote) elements.externalNote.hidden = !external;
    if (elements.modelNote) elements.modelNote.hidden = !needsLocalSetup;
    if (elements.localNote) elements.localNote.hidden = external;
    if (elements.input) {
        elements.input.disabled = !hasDataset || external || localUnavailable || state.isGenerating;
        elements.input.title = blockedByOtherAI
            ? `Local AI is currently ${pipeline.rag._operationLabel(activeOperation.owner)}.`
            : processing ? 'Wait for data processing to finish.' : '';
    }
    if (elements.scope) elements.scope.disabled = !hasDataset || external || state.isGenerating;
    if (elements.scope) elements.scope.title = '';
    if (elements.send) {
        elements.send.disabled = !hasDataset || external || localUnavailable || state.isGenerating || !elements.input?.value.trim();
        elements.send.hidden = state.isGenerating;
    }
    if (elements.stop) {
        elements.stop.hidden = !chatCanStop;
        elements.stop.disabled = !chatCanStop;
    }
    if (elements.newChat) elements.newChat.disabled = !hasDataset || state.isGenerating || !(state.conversation?.messages?.length);
    if (elements.exportChat) elements.exportChat.disabled = !hasDataset || !(state.conversation?.messages?.length);
    if (elements.trimMemory) elements.trimMemory.disabled = !hasDataset || state.isGenerating || !(state.conversation?.messages?.length);
    if (elements.memoryNote) elements.memoryNote.hidden = !state.conversation?.contextCutoffAt;
    document.querySelectorAll('[data-chat-suggestion]').forEach(button => {
        button.disabled = !hasDataset || external || localUnavailable || needsLocalSetup || state.isGenerating;
    });
}

function shouldStickToBottom() {
    if (!elements.messages) return true;
    return elements.messages.scrollHeight - elements.messages.scrollTop - elements.messages.clientHeight < 72;
}

function scrollToBottom() {
    if (!elements.messages) return;
    elements.messages.scrollTop = elements.messages.scrollHeight;
    state.chatScrollTop = elements.messages.scrollTop;
}

function appendFormattedText(container, value) {
    const tokenRegex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let cursor = 0;
    let match;
    const text = String(value || '');
    while ((match = tokenRegex.exec(text)) !== null) {
        if (match.index > cursor) container.append(document.createTextNode(text.slice(cursor, match.index)));
        const token = match[0];
        const element = token.startsWith('**') ? document.createElement('strong') : document.createElement('code');
        element.textContent = token.startsWith('**') ? token.slice(2, -2) : token.slice(1, -1);
        container.append(element);
        cursor = tokenRegex.lastIndex;
    }
    if (cursor < text.length) container.append(document.createTextNode(text.slice(cursor)));
}

function appendAssistantContent(container, message) {
    if (message.route?.resolved !== 'documents' && message.route?.resolved !== undefined) {
        appendFormattedText(container, message.content || '');
        return;
    }
    const sources = Array.isArray(message.sources) ? message.sources : [];
    for (const segment of tokenizeCitations(message.content || '', sources.length)) {
        if (segment.type === 'citation') {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'chat-inline-citation';
            button.textContent = segment.text;
            button.title = `Open source ${segment.sourceNumber}`;
            const source = sources[segment.sourceNumber - 1];
            bindSourcePreview(button, source);
            button.addEventListener('click', () => openSource(source));
            container.append(button);
        } else if (segment.type === 'unavailable-citation') {
            const unavailable = document.createElement('span');
            unavailable.className = 'chat-unavailable-citation';
            unavailable.textContent = segment.text;
            unavailable.title = 'This source was not available in the evidence supplied for this answer';
            unavailable.setAttribute('aria-label', `${segment.text}, unavailable source reference`);
            container.append(unavailable);
        } else {
            appendFormattedText(container, segment.text);
        }
    }
}

function routeBadgeText(message) {
    const resolved = message.route?.resolved || message.metadata?.route?.resolved;
    if (resolved === 'helper') return 'Vectoria help · No document search';
    if (resolved === 'conversation') return 'Legacy conversation · No document search';
    if (resolved === 'handoff') return 'Legacy handoff · Search not run';
    if (resolved === 'documents') return `Documents · ${message.scope?.label || 'All documents'}`;
    return '';
}

function sourceTitle(source) {
    const metadata = source?.metadata || {};
    return metadata.title || metadata.name || metadata.filename || metadata.file_name
        || (Number.isInteger(source?.documentIndex) ? `Item ${source.documentIndex + 1}` : `Document ${source?.sourceNumber || ''}`.trim());
}

function sourceScore(source, cited = false) {
    const rank = Number(source?.retrievalRank ?? source?.sourceNumber);
    const rankText = Number.isInteger(rank) && rank >= 1 ? `Rank ${rank}` : '';
    return [rankText, source?.rerankerApplied ? 'Reranked' : '', cited ? 'Cited' : 'Context'].filter(Boolean).join(' · ');
}

function createFilterSummary(message) {
    if ((message.route?.resolved || message.metadata?.route?.resolved) !== 'documents') return null;
    const filter = message.metadata?.filter;
    const filters = filter?.filters;
    const filterParts = filter?.applied && filters && typeof filters === 'object'
        ? Object.entries(filters).map(([field, value]) => {
            const values = Array.isArray(value) ? value : [value];
            return `${field}: ${values.map(item => String(item)).join(', ')}`;
        }).filter(Boolean)
        : [];
    const scopeParts = message.scope?.type === 'filters' && Array.isArray(message.scope?.details)
        ? message.scope.details.map(String).filter(Boolean)
        : [];
    const parts = filterParts.length ? filterParts : scopeParts;
    if (!parts.length) return null;
    const details = document.createElement('details');
    details.className = 'chat-filter-summary';
    const summary = document.createElement('summary');
    summary.textContent = `${filterParts.length ? 'Explicit filters' : 'Active filtered scope'} · ${parts.length}`;
    const body = document.createElement('p');
    body.textContent = parts.join(' · ');
    details.append(summary, body);
    return details;
}

function previewSource(source, owner = null) {
    if (!source || source.pointIndex < 0) return;
    state.sourcePreviewOwner = owner;
    window.mainVisualization?.previewChatPoint?.(source.pointIndex);
}

function clearSourcePreview(owner = null) {
    if (owner && state.sourcePreviewOwner && owner !== state.sourcePreviewOwner) return;
    state.sourcePreviewOwner = null;
    window.mainVisualization?.clearChatPreview?.();
}

function bindSourcePreview(element, source) {
    const clearAfterExit = event => {
        // pointerleave is normally sufficient, but pointerout/mouseleave cover
        // browsers that lose pointer capture while streamed content is replaced.
        if (event.relatedTarget && element.contains(event.relatedTarget)) return;
        clearSourcePreview(element);
    };
    element.addEventListener('pointerenter', () => previewSource(source, element));
    element.addEventListener('pointerleave', clearAfterExit);
    element.addEventListener('pointerout', clearAfterExit);
    element.addEventListener('pointercancel', () => clearSourcePreview(element));
    element.addEventListener('mouseleave', clearAfterExit);
    element.addEventListener('focus', () => previewSource(source, element));
    element.addEventListener('blur', () => clearSourcePreview(element));
}

function createSourcesDetails(message) {
    const sources = Array.isArray(message.sources) ? message.sources : [];
    if (!sources.length) return null;
    // Re-parse visible content so conversations saved by older citation
    // parsers immediately gain correct grouped-citation counts and links.
    const cited = new Set([
        ...(message.citations || []),
        ...extractCitations(message.content || '', sources.length)
    ]);
    const details = document.createElement('details');
    details.className = 'chat-sources';
    details.open = state.expandedSourceMessageIds.has(message.id);
    details.addEventListener('toggle', () => {
        if (details.open) state.expandedSourceMessageIds.add(message.id);
        else state.expandedSourceMessageIds.delete(message.id);
    });
    const summary = document.createElement('summary');
    summary.textContent = cited.size === sources.length
        ? `${cited.size} cited source${cited.size === 1 ? '' : 's'}`
        : cited.size
            ? `${cited.size} cited · ${sources.length} context sources`
            : `${sources.length} context source${sources.length === 1 ? '' : 's'} · no inline citations`;
    details.append(summary);
    const list = document.createElement('div');
    list.className = 'chat-source-list';
    sources.forEach((source) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chat-source-item';
        const heading = document.createElement('span');
        heading.className = 'chat-source-heading';
        heading.textContent = `[Doc ${source.sourceNumber}] ${sourceTitle(source)}`;
        const score = document.createElement('span');
        score.className = 'chat-source-score';
        score.textContent = sourceScore(source, cited.has(source.sourceNumber));
        score.title = 'Order returned by hybrid semantic and keyword retrieval';
        button.append(heading);
        if (score.textContent) button.append(score);
        bindSourcePreview(button, source);
        button.addEventListener('click', () => openSource(source));
        list.append(button);
    });
    details.append(list);
    details.addEventListener('pointerleave', () => clearSourcePreview());
    details.addEventListener('pointercancel', () => clearSourcePreview());
    return details;
}

function copyMessage(message, button) {
    navigator.clipboard.writeText(message.content || '').then(() => {
        const previous = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check" aria-hidden="true"></i>';
        setTimeout(() => { button.innerHTML = previous; }, 1200);
    }).catch(() => showToast('Unable to copy message', 'error'));
}

function createMessageElement(message) {
    const article = document.createElement('article');
    article.className = `chat-message chat-message-${message.role}`;
    article.dataset.messageId = message.id;

    const label = document.createElement('div');
    label.className = 'chat-message-label';
    label.textContent = message.role === 'assistant' ? 'Vectoria' : 'Question';
    const content = document.createElement('div');
    content.className = 'chat-message-content';
    if (message.role === 'assistant') appendAssistantContent(content, message);
    else content.textContent = message.content || '';
    if (!message.content && message.status === 'generating') {
        content.innerHTML = '<span class="chat-thinking"><span></span><span></span><span></span></span>';
    }
    if (message.status === 'error') content.classList.add('chat-message-error');

    const unavailableCitations = message.role === 'assistant'
        ? tokenizeCitations(message.content || '', Array.isArray(message.sources) ? message.sources.length : 0)
            .filter(token => token.type === 'unavailable-citation')
        : [];

    const actions = document.createElement('div');
    actions.className = 'chat-message-actions';
    if (message.content && message.status !== 'generating') {
        const copy = document.createElement('button');
        copy.type = 'button';
        copy.title = 'Copy message';
        copy.setAttribute('aria-label', 'Copy message');
        copy.innerHTML = '<i class="far fa-copy" aria-hidden="true"></i>';
        copy.addEventListener('click', () => copyMessage(message, copy));
        actions.append(copy);
    }
    if (message.role === 'assistant' && ['error', 'interrupted', 'stopped'].includes(message.status)) {
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'chat-retry-btn';
        retry.textContent = 'Retry';
        retry.disabled = state.isGenerating || isExternalMode();
        retry.addEventListener('click', () => retryAssistant(message));
        actions.append(retry);
    }

    article.append(label, content);
    if (unavailableCitations.length) {
        const warning = document.createElement('p');
        warning.className = 'chat-citation-warning';
        const count = new Set(unavailableCitations.map(token => token.sourceNumber)).size;
        warning.textContent = count === 1
            ? 'The answer referenced a source that was not available.'
            : `The answer referenced ${count} sources that were not available.`;
        article.append(warning);
    }
    if (message.role === 'assistant' && message.status !== 'generating') {
        const badgeText = routeBadgeText(message);
        if (badgeText) {
            const badge = document.createElement('span');
            badge.className = 'chat-route-badge';
            badge.textContent = badgeText;
            article.append(badge);
        }
        if (message.metadata?.resolvedQuery && (message.route?.resolved || message.metadata?.route?.resolved) === 'documents') {
            const searched = document.createElement('details');
            searched.className = 'chat-resolved-query';
            const summary = document.createElement('summary');
            summary.textContent = 'Searched for';
            const query = document.createElement('p');
            query.textContent = message.metadata.resolvedQuery;
            searched.append(summary, query);
            article.append(searched);
        }
        const filterSummary = createFilterSummary(message);
        if (filterSummary) article.append(filterSummary);
        const details = createSourcesDetails(message);
        if (details) article.append(details);
        const meta = document.createElement('div');
        meta.className = 'chat-message-meta';
        const parts = [];
        if (message.scope?.label && (message.route?.resolved || message.metadata?.route?.resolved) === 'documents') parts.push(message.scope.label);
        if (message.metadata?.model) parts.push(message.metadata.model.replace(/-q\w+.*$/i, ''));
        if (message.metadata?.generationTime) parts.push(`${Number(message.metadata.generationTime).toFixed(1)}s`);
        if (message.metadata?.hydeUsed) parts.push('HyDE');
        const reranker = message.metadata?.retrieval;
        if (reranker?.reranker_applied) parts.push(`Reranked ${reranker.reranker_candidates || ''}`.trim());
        else if (reranker?.reranker_fallback_reason) parts.push('Standard hybrid fallback');
        if (message.metadata?.memoryMode === 'none') parts.push('Memory off');
        else if (message.metadata?.summarizedTurns) parts.push(`${message.metadata.summarizedTurns} older turn${message.metadata.summarizedTurns === 1 ? '' : 's'} summarized`);
        else if (message.metadata?.omittedTurns) parts.push(`${message.metadata.omittedTurns} older turn${message.metadata.omittedTurns === 1 ? '' : 's'} omitted`);
        if (message.metadata?.truncatedPassages) parts.push(`${message.metadata.truncatedPassages} passage${message.metadata.truncatedPassages === 1 ? '' : 's'} shortened`);
        const requestedSources = Number(message.metadata?.requestedSourceCount) || 0;
        const retrievedSources = Number(message.metadata?.retrievedSourceCount) || 0;
        const includedSources = Number(message.metadata?.includedSourceCount) || 0;
        if (requestedSources && retrievedSources < requestedSources) parts.push(`${retrievedSources}/${requestedSources} sources retrieved`);
        if (retrievedSources && includedSources < retrievedSources) parts.push(`${includedSources}/${retrievedSources} retrieved sources fit`);
        const completionTokens = Number(message.metadata?.actualUsage?.completionTokens);
        if (completionTokens > 0) parts.push(`${completionTokens} output tokens`);
        if (message.metadata?.finishReason === 'length') parts.push('Output limit reached');
        if (message.metadata?.contextUsagePercent) parts.push(`${message.metadata.contextUsagePercent}% context plan`);
        if (message.status === 'stopped') parts.push('Stopped');
        if (message.status === 'interrupted') parts.push('Interrupted');
        meta.textContent = parts.join(' • ');
        if (meta.textContent) article.append(meta);
    }
    if (actions.children.length) article.append(actions);
    return article;
}

function renderConversation({ preservePosition = false } = {}) {
    if (!elements.messages || !state.conversation) return;
    // Replacing a hovered citation/source node does not reliably dispatch a
    // pointerleave event. Clear its transient visualization ownership first.
    clearSourcePreview();
    const stick = !preservePosition && shouldStickToBottom();
    const previousTop = elements.messages.scrollTop;
    elements.messages.replaceChildren(...state.conversation.messages.map(createMessageElement));
    const hasMessages = state.conversation.messages.length > 0;
    if (elements.empty) elements.empty.hidden = hasMessages;
    elements.messages.hidden = !hasMessages;
    if (stick) requestAnimationFrame(scrollToBottom);
    else elements.messages.scrollTop = previousTop;
    refreshAvailability();
}

function scheduleStreamingRender(message) {
    if (state.renderFrame) return;
    state.renderFrame = requestAnimationFrame(() => {
        state.renderFrame = null;
        const content = elements.messages?.querySelector(`[data-message-id="${message.id}"] .chat-message-content`);
        if (!content) return renderConversation();
        const stick = shouldStickToBottom();
        clearSourcePreview();
        content.replaceChildren();
        appendAssistantContent(content, message);
        if (stick) scrollToBottom();
    });
}

function compactMetadata(metadata) {
    const compact = {};
    Object.entries(metadata || {}).slice(0, 12).forEach(([key, value]) => {
        if (value === null || value === undefined) return;
        if (['string', 'number', 'boolean'].includes(typeof value)) compact[key] = value;
        else if (Array.isArray(value)) compact[key] = value.slice(0, 8).map(item => String(item));
    });
    return compact;
}

function normalizeSourceManifest(rawSources) {
    const documents = pipeline.currentDataset?.documents || [];
    const points = window.currentVisualizationData?.points || [];
    return (rawSources || []).map((source, index) => {
        const docId = String(source.doc_id ?? source.parent_id ?? source.id ?? source.metadata?.parent_id ?? source.metadata?.doc_id ?? '');
        let documentIndex = documents.findIndex(document => String(document.id ?? document.doc_id ?? document.metadata?.doc_id ?? '') === docId);
        if (documentIndex < 0 && Number.isInteger(source.index)) documentIndex = source.index;
        let pointIndex = points.findIndex(point => String(point.doc_id ?? point.id ?? '') === docId);
        if (pointIndex < 0 && documentIndex >= 0) pointIndex = documentIndex;
        const passageText = Array.isArray(source.chunks) && source.chunks.length
            ? source.chunks.map(chunk => chunk.text || chunk.metadata?.text || '').filter(Boolean).join('\n')
            : (source.text || source.content || source.metadata?.text || documents[documentIndex]?.text || '');
        const rawScore = source.score ?? source.maxScore;
        const score = rawScore === undefined || rawScore === null || !Number.isFinite(Number(rawScore))
            ? null
            : Number(rawScore);
        return {
            sourceNumber: Number(source.sourceNumber) || index + 1,
            docId,
            documentIndex,
            pointIndex,
            rank: Number(source.rank ?? source.vector_rank ?? index + 1),
            retrievalRank: index + 1,
            score,
            scoreKind: source.fusion_method === 'RRF' ? 'rrf' : 'retrieval',
            vectorScore: source.vector_score !== null && source.vector_score !== undefined && Number.isFinite(Number(source.vector_score))
                ? Number(source.vector_score)
                : null,
            bm25Score: source.bm25_score !== null && source.bm25_score !== undefined && Number.isFinite(Number(source.bm25_score))
                ? Number(source.bm25_score)
                : null,
            preRerankRank: Number(source.pre_rerank_rank) || null,
            rerankerRank: Number(source.reranker_rank) || null,
            rerankerScore: Number.isFinite(Number(source.reranker_score)) ? Number(source.reranker_score) : null,
            rerankerApplied: source.reranker_applied === true,
            rerankerModel: source.reranker_model || null,
            rerankerLatencyMs: Number(source.reranker_latency_ms) || 0,
            pre_rerank_rank: Number(source.pre_rerank_rank) || null,
            reranker_rank: Number(source.reranker_rank) || null,
            reranker_score: Number.isFinite(Number(source.reranker_score)) ? Number(source.reranker_score) : null,
            reranker_applied: source.reranker_applied === true,
            reranker_model: source.reranker_model || null,
            reranker_latency_ms: Number(source.reranker_latency_ms) || 0,
            reranker_fallback_reason: source.reranker_fallback_reason || null,
            excerpt: passageText.replace(/\s+/g, ' ').trim().slice(0, 420),
            chunkIds: Array.isArray(source.chunks)
                ? source.chunks.map(chunk => chunk.chunk_id || chunk.id).filter(Boolean).slice(0, 12)
                : [],
            metadata: compactMetadata({ ...(documents[documentIndex]?.metadata || {}), ...(source.metadata || {}) }),
            includedInContext: source.includedInContext !== false
        };
    });
}

function highlightAssistantSources(message) {
    if ((message.route?.resolved || message.metadata?.route?.resolved) !== 'documents') return;
    const sources = message.sources || [];
    const cited = new Set(message.citations || []);
    const selected = cited.size ? sources.filter(source => cited.has(source.sourceNumber)) : sources;
    const docIds = selected.map(source => source.docId).filter(Boolean);
    if (docIds.length && typeof window.mainVisualization?.pulseChatDocuments === 'function') {
        window.mainVisualization.pulseChatDocuments(docIds);
    }
}

function openSource(source) {
    if (!source) return;
    clearSourcePreview();
    const points = window.currentVisualizationData?.points || [];
    const documents = pipeline.currentDataset?.documents || [];
    const point = points[source.pointIndex]
        || points.find(candidate => String(candidate.doc_id ?? candidate.id ?? '') === String(source.docId))
        || (source.documentIndex >= 0 ? {
            index: source.documentIndex,
            doc_id: source.docId,
            text: documents[source.documentIndex]?.text || source.excerpt,
            metadata: documents[source.documentIndex]?.metadata || source.metadata,
            cluster: documents[source.documentIndex]?.metadata?.cluster ?? -1,
            cluster_probability: documents[source.documentIndex]?.metadata?.cluster_probability ?? 0,
            x: 0,
            y: 0
        } : null);
    if (!point || typeof window.showTextDetails !== 'function') {
        showToast('This source is no longer available in the active dataset', 'warning');
        return;
    }
    state.chatScrollTop = elements.messages?.scrollTop || 0;
    window.__returnToChatAfterReader = true;
    showDocuments();
    if (elements.backToList) elements.backToList.innerHTML = '<i class="fas fa-arrow-left"></i> Back to Ask';
    const index = source.pointIndex >= 0 ? source.pointIndex : source.documentIndex;
    window.showTextDetails(point, Math.max(0, index), { focusVisualization: true });
}

async function persistConversation() {
    if (!state.conversation || !state.datasetId) return;
    state.conversation.updatedAt = Date.now();
    await pipeline.chat.save(state.conversation);
}

function renderSuggestions(items) {
    if (!elements.suggestions) return;
    const list = items.length === 2 ? items : FALLBACK_SUGGESTIONS;
    elements.suggestions.replaceChildren(...list.map(item => {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.chatSuggestion = item.prompt;
        button.textContent = item.label;
        return button;
    }));
    refreshAvailability();
}

async function refreshSuggestedQuestions(datasetId) {
    if (!elements.suggestions) return;
    const requestId = ++state.suggestionRequestId;
    renderSuggestions(FALLBACK_SUGGESTIONS);
    if (!datasetId || !pipeline.rag) return;
    // Never contend with a question the user is actually waiting on.
    if (state.isGenerating || pipeline.rag.activeOperation) return;

    const sampled = sampleDocumentsForSuggestions(pipeline.currentDataset?.documents || []);
    if (sampled.length < 2) return;

    elements.suggestions.setAttribute('aria-busy', 'true');
    try {
        const text = await pipeline.rag.generateRaw(buildSuggestionPrompt(sampled), {
            owner: 'suggested-questions',
            temperature: 0.9,
            maxTokens: 80,
            datasetId
        });
        if (requestId !== state.suggestionRequestId || datasetId !== state.datasetId) return;
        const questions = parseSuggestionResponse(text);
        if (questions.length === 2) {
            renderSuggestions(questions.map(question => ({ label: question, prompt: question })));
        }
    } catch (error) {
        // Suggestions are a convenience: keep the static pair and stay quiet.
        console.debug('Suggested questions unavailable:', error?.message || error);
    } finally {
        if (requestId === state.suggestionRequestId) elements.suggestions.setAttribute('aria-busy', 'false');
    }
}

async function loadConversation(datasetId = currentDatasetId()) {
    const requestId = ++state.loadRequestId;
    const changedDataset = state.datasetId && String(state.datasetId) !== String(datasetId || '');
    if (changedDataset) {
        clearSourcePreview();
        state.expandedSourceMessageIds.clear();
        window.cancelHyDEReview?.();
        if (state.isGenerating) pipeline.abortRAG('chat');
        await pipeline.rag?.resetConversationState?.();
    }
    const conversation = datasetId
        ? await pipeline.chat.load(datasetId)
        : createEmptyConversation('none');
    if (requestId !== state.loadRequestId) return;
    if (state.renderFrame) {
        cancelAnimationFrame(state.renderFrame);
        state.renderFrame = null;
    }
    state.datasetId = datasetId;
    state.conversation = conversation;
    syncChatSettings();
    renderConversation();
    refreshAvailability();
    // Regenerate only when the empty state is actually on screen.
    if (!conversation.messages.length) void refreshSuggestedQuestions(datasetId);
}

async function runAssistant(userMessage, assistantMessage, historyBefore, scopeSnapshot, chatOptions = getChatOptionsSnapshot()) {
    state.isGenerating = true;
    assistantMessage.status = 'generating';
    assistantMessage.content = '';
    assistantMessage.sources = [];
    assistantMessage.citations = [];
    assistantMessage.route = null;
    assistantMessage.metadata = {};
    await persistConversation();
    renderConversation();
    setLiveStatus(chatOptions.useHyDE ? 'Generating a HyDE retrieval hypothesis…' : 'Preparing this turn…');

    try {
        const result = await pipeline.queryChat(userMessage.content, {
            history: historyBefore,
            scope: scopeSnapshot.persisted.type,
            allowedDocIds: scopeSnapshot.info.scopeType === 'all' ? null : scopeSnapshot.info.docIds,
            ...chatOptions,
            onRouteResolved: decision => {
                assistantMessage.route = { ...decision };
                assistantMessage.metadata = { ...(assistantMessage.metadata || {}), route: { ...decision } };
                void persistConversation();
                setLiveStatus(decision.resolved === 'helper'
                    ? 'Preparing document assistant guidance…'
                    : (chatOptions.useHyDE ? 'Generating a HyDE retrieval hypothesis…' : 'Retrieving relevant sources…'));
            },
            onStatus: (status, detail = {}) => {
                if (['hyde', 'awaiting-hyde', 'retrieving', 'reranking', 'generating', 'adjusting-context'].includes(status)) {
                    assistantMessage.metadata = { ...(assistantMessage.metadata || {}), phase: status, operationPhase: status };
                    if (status === 'awaiting-hyde') void persistConversation();
                }
                const rawProgress = Number(detail?.progress);
                const progress = Number.isFinite(rawProgress) && rawProgress > 0
                    ? ` ${Math.round(rawProgress > 1 ? rawProgress : rawProgress * 100)}%`
                    : '';
                setLiveStatus(status === 'hyde' ? 'Generating a HyDE retrieval hypothesis…'
                    : status === 'awaiting-hyde' ? 'Review the HyDE search draft to continue.'
                    : status === 'adjusting-context' ? 'Tightening context to fit the local model…'
                    : status === 'loading-model' ? 'Loading cached local AI…'
                    : status === 'model-progress' ? `Preparing local AI${progress}…`
                    : status === 'model-ready' ? 'Local AI ready. Continuing…'
                    : status === 'retrieving' ? 'Retrieving relevant sources…'
                    : status === 'reranking' ? `Improving evidence order${progress}…`
                    : status === 'generating' ? 'Generating locally…'
                    : status === 'stopped' ? 'Generation stopped.' : '');
            },
            onHyDEReview: async review => {
                assistantMessage.metadata = {
                    ...(assistantMessage.metadata || {}),
                    hyde: {
                        generated: review.generatedText,
                        contextualQuery: review.contextualQuery,
                        action: 'pending_review'
                    }
                };
                await persistConversation();
                if (typeof window.showHyDEReviewModal !== 'function') {
                    return { action: 'without_hyde' };
                }
                return window.showHyDEReviewModal(review.question, review.generatedText, { detailed: true });
            },
            onRetrievalComplete: (sources, telemetry) => {
                assistantMessage.sources = normalizeSourceManifest(sources);
                assistantMessage.metadata = { ...(assistantMessage.metadata || {}), ...telemetry };
            },
            onChunk: (_chunk, fullText) => {
                assistantMessage.content = fullText;
                scheduleStreamingRender(assistantMessage);
            }
        });

        if (!assistantMessage.sources.length) assistantMessage.sources = normalizeSourceManifest(result.sources || []);
        assistantMessage.content = result.answer || assistantMessage.content || 'No answer was generated.';
        assistantMessage.status = result.metadata?.wasStopped ? 'stopped' : 'complete';
        assistantMessage.route = { ...(result.route || result.metadata?.route || assistantMessage.route || {}) };
        assistantMessage.metadata = { ...(assistantMessage.metadata || {}), ...(result.metadata || {}) };
        if (assistantMessage.route.resolved === 'documents') {
            assistantMessage.citations = extractCitations(assistantMessage.content, assistantMessage.sources.length);
            highlightAssistantSources(assistantMessage);
        } else {
            assistantMessage.sources = [];
            assistantMessage.citations = [];
        }
    } catch (error) {
        const diagnostic = normalizeLocalAIError(error, {
            phase: assistantMessage.metadata?.phase || null,
            retrievalCompleted: assistantMessage.sources.length > 0,
            outputStreamed: Boolean(assistantMessage.content),
            model: pipeline.rag?.modelId || null
        });
        const stopped = diagnostic.code === 'local_ai_aborted';
        assistantMessage.status = stopped ? 'stopped' : 'error';
        assistantMessage.content = stopped
            ? (assistantMessage.content || 'Generation stopped.')
            : assistantMessage.sources.length
                ? 'Sources were found, but the local model could not generate the answer.'
                : 'The local model could not generate an answer. Retry this turn.';
        assistantMessage.metadata = {
            ...(assistantMessage.metadata || {}),
            error: !stopped,
            errorCode: diagnostic.code,
            errorDiagnostic: diagnostic,
            sourcesPreservedAfterError: !stopped && assistantMessage.sources.length > 0
        };
        if (!stopped) console.error('Local Ask turn failed:', diagnostic);
    } finally {
        state.isGenerating = false;
        await persistConversation();
        setLiveStatus('');
        renderConversation();
        focusChatInput();
    }
}

async function sendMessage() {
    const question = elements.input?.value.trim();
    if (!question || state.isGenerating || isExternalMode() || !currentDatasetId()) return;
    if (pipeline.isProcessing) {
        showToast('Wait for data processing to finish before using Ask', 'warning');
        return;
    }
    if (pipeline.rag?.activeOperation) {
        showToast(`Local AI is currently ${pipeline.rag._operationLabel(pipeline.rag.activeOperation.owner)}.`, 'warning');
        return;
    }
    const historyBefore = contextHistory(state.conversation.messages.slice());
    const pendingRoute = routeChatTurn(question, { history: historyBefore });
    if (pendingRoute.resolved === 'documents' && !pipeline.rag) {
        showToast('Set up the local AI model before asking a document question', 'warning');
        elements.openModelSetup?.focus();
        return;
    }
    clearSourcePreview();
    const scopeSnapshot = getScopeSnapshot(elements.scope.value);
    const chatOptions = getChatOptionsSnapshot();
    if (scopeSnapshot.persisted.type !== 'all'
        && (!scopeSnapshot.info.docIds || scopeSnapshot.info.docIds.length === 0)) {
        showToast('No documents are available in the current view', 'warning');
        return;
    }

    const userMessage = createChatMessage({ role: 'user', content: question, scope: scopeSnapshot.persisted });
    const assistantMessage = createChatMessage({
        role: 'assistant',
        content: '',
        status: 'generating',
        turnId: userMessage.turnId,
        scope: scopeSnapshot.persisted
    });
    state.conversation.messages.push(userMessage, assistantMessage);
    elements.input.value = '';
    autoSizeComposer();
    await runAssistant(userMessage, assistantMessage, historyBefore, scopeSnapshot, chatOptions);
}

async function retryAssistant(assistantMessage) {
    if (state.isGenerating || isExternalMode()) return;
    const assistantIndex = state.conversation.messages.findIndex(message => message.id === assistantMessage.id);
    const userIndex = state.conversation.messages.findIndex(message => message.turnId === assistantMessage.turnId && message.role === 'user');
    if (assistantIndex < 0 || userIndex < 0) return;
    const userMessage = state.conversation.messages[userIndex];
    const historyBefore = contextHistory(state.conversation.messages.slice(0, userIndex));
    const capturedScope = assistantMessage.scope || userMessage.scope || captureChatScope({ scopeType: 'all' });
    const fallbackDocIds = (assistantMessage.sources || []).map(source => source.docId).filter(Boolean);
    const scopeSnapshot = restoreChatScope(capturedScope, fallbackDocIds);
    const chatOptions = getChatOptionsSnapshot();
    await runAssistant(userMessage, assistantMessage, historyBefore, scopeSnapshot, chatOptions);
}

async function newChat() {
    if (!state.datasetId || state.isGenerating || !state.conversation?.messages?.length) return;
    if (!window.confirm('Start a new investigation? This clears the current Ask history for this dataset.')) return;
    state.conversation = await pipeline.chat.clear(state.datasetId);
    clearSourcePreview();
    state.expandedSourceMessageIds.clear();
    await pipeline.rag?.resetConversationState?.();
    state.chatScrollTop = 0;
    renderConversation();
    elements.input?.focus();
}

async function trimModelMemory() {
    if (!state.datasetId || state.isGenerating || !state.conversation?.messages?.length) return;
    if (!window.confirm('Forget earlier model context? Messages and exports stay intact, but future answers will not receive the existing conversation as memory.')) return;
    state.conversation.contextCutoffAt = Date.now();
    await persistConversation();
    await pipeline.rag?.resetConversationState?.();
    closeExportMenu();
    renderConversation({ preservePosition: true });
    showToast('Earlier messages were removed from future model context', 'info');
}

async function restoreModelMemory() {
    if (!state.conversation?.contextCutoffAt) return;
    state.conversation.contextCutoffAt = null;
    await persistConversation();
    await pipeline.rag?.resetConversationState?.();
    renderConversation({ preservePosition: true });
    showToast('Full conversation memory restored', 'success');
}

function closeExportMenu() {
    if (elements.exportMenu) elements.exportMenu.hidden = true;
    elements.exportChat?.setAttribute('aria-expanded', 'false');
    syncPopoverPresentation();
}

function toggleSettingsPanel() {
    if (!elements.settingsPanel) return;
    const willOpen = elements.settingsPanel.hidden;
    if (!willOpen) {
        closeSettingsPanel();
        return;
    }
    closeExportMenu();
    elements.settingsPanel.hidden = false;
    syncSettingsPanelPortal();
    elements.settings?.setAttribute('aria-expanded', 'true');
    syncChatSettings();
    elements.hydeToggle?.focus();
    syncPopoverPresentation();
}

function exportChat(format = 'json') {
    const messages = state.conversation?.messages || [];
    const entries = messages.filter(message => message.role === 'user').map((user) => {
        const assistant = messages.find(message => message.role === 'assistant' && message.turnId === user.turnId);
        return {
            timestamp: new Date(user.createdAt).toISOString(),
            query: user.content,
            answer: assistant?.content || '',
            sources: assistant?.sources || [],
            route: assistant?.route || assistant?.metadata?.route || null,
            metadata: { ...(assistant?.metadata || {}), scope: assistant?.scope || user.scope }
        };
    });
    if (!entries.length) return;
    if (typeof window.exportRAGConversation === 'function') window.exportRAGConversation(entries, format);
    closeExportMenu();
}

function bindEvents() {
    elements.documentsTab?.addEventListener('click', showDocuments);
    elements.chatTab?.addEventListener('click', () => showChat());
    [elements.documentsTab, elements.chatTab].forEach(tab => tab?.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const target = ['ArrowRight', 'End'].includes(event.key) ? elements.chatTab : elements.documentsTab;
        if (target === elements.chatTab) showChat();
        else showDocuments();
        target?.focus();
    }));
    elements.input?.addEventListener('input', () => { autoSizeComposer(); refreshAvailability(); });
    elements.input?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            void sendMessage();
        }
    });
    elements.send?.addEventListener('click', () => void sendMessage());
    elements.stop?.addEventListener('click', () => {
        if (!state.isGenerating || pipeline.rag?.activeOperation?.owner !== 'chat') return;
        if (!pipeline.abortRAG('chat')) return;
        window.cancelHyDEReview?.();
        setLiveStatus('Stopping generation…');
    });
    elements.newChat?.addEventListener('click', () => void newChat());
    elements.settings?.addEventListener('click', event => {
        event.stopPropagation();
        toggleSettingsPanel();
    });
    elements.settingsClose?.addEventListener('click', closeSettingsPanel);
    elements.hydeToggle?.addEventListener('change', () => {
        if (window.hydeMode) window.hydeMode.enabled = elements.hydeToggle.checked;
        updateChatConfig({ ui_preferences: { hyde_enabled: elements.hydeToggle.checked } });
    });
    elements.metadataMode?.addEventListener('change', () => {
        const mode = ['off', 'selected', 'all'].includes(elements.metadataMode.value)
            ? elements.metadataMode.value : 'off';
        updateChatConfig({ chat: { metadata_mode: mode, include_metadata: mode !== 'off' } });
    });
    elements.sourceCount?.addEventListener('change', () => {
        const count = Math.max(1, Math.min(20, Number(elements.sourceCount.value) || 5));
        updateChatConfig({ search: { num_results: count } });
    });
    elements.similarityThreshold?.addEventListener('change', () => {
        const threshold = Math.max(0, Math.min(1, Number(elements.similarityThreshold.value) || 0));
        updateChatConfig({ search: { similarity_threshold: threshold } });
    });
    elements.metadataFields?.addEventListener('change', () => {
        const selected = [...elements.metadataFields.selectedOptions].map(option => option.value);
        updateChatConfig({ chat: { metadata_fields: selected } });
    });
    elements.memoryMode?.addEventListener('change', () => {
        updateChatConfig({ chat: { memory_mode: elements.memoryMode.value } });
    });
    elements.memoryTurns?.addEventListener('change', () => {
        const turns = Math.max(1, Math.min(50, Number(elements.memoryTurns.value) || 8));
        updateChatConfig({ chat: { max_memory_turns: turns } });
    });
    elements.openRagSettings?.addEventListener('click', () => openAdvancedSettings('rag'));
    elements.openModelSettings?.addEventListener('click', () => openAdvancedSettings('models'));
    elements.enableLocal?.addEventListener('click', () => {
        if (window.VectoriaGenerationMode?.request) window.VectoriaGenerationMode.request('local');
        else openAdvancedSettings('mcp');
    });
    elements.openModelSetup?.addEventListener('click', () => {
        if (typeof window.openVectoriaModelSetup === 'function') {
            window.openVectoriaModelSetup();
        } else {
            openAdvancedSettings('models');
        }
    });
    elements.exportChat?.addEventListener('click', event => {
        event.stopPropagation();
        if (!elements.exportMenu) return;
        const willOpen = elements.exportMenu.hidden;
        closeSettingsPanel();
        elements.exportMenu.hidden = !willOpen;
        elements.exportChat.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
        syncPopoverPresentation();
        if (willOpen) elements.exportMenu.querySelector('[role="menuitem"]')?.focus();
    });
    elements.exportMenu?.querySelectorAll('[data-chat-export]').forEach(button => {
        button.addEventListener('click', () => exportChat(button.dataset.chatExport));
    });
    elements.trimMemory?.addEventListener('click', () => void trimModelMemory());
    elements.restoreMemory?.addEventListener('click', () => void restoreModelMemory());
    document.addEventListener('click', event => {
        if (!elements.exportMenu?.hidden && !elements.exportMenu.contains(event.target) && event.target !== elements.exportChat) closeExportMenu();
        if (!elements.settingsPanel?.hidden && !elements.settingsPanel.contains(event.target) && event.target !== elements.settings) closeSettingsPanel();
    });
    // Delegated: suggestion buttons are replaced whenever a dataset loads.
    elements.suggestions?.addEventListener('click', event => {
        const button = event.target.closest('[data-chat-suggestion]');
        if (!button || button.disabled) return;
        elements.input.value = button.dataset.chatSuggestion || '';
        autoSizeComposer();
        refreshAvailability();
        void sendMessage();
    });
    elements.messages?.addEventListener('scroll', () => {
        state.chatScrollTop = elements.messages.scrollTop;
        clearSourcePreview();
    }, { passive: true });
    elements.messages?.addEventListener('pointermove', event => {
        const owner = state.sourcePreviewOwner;
        if (owner && !owner.contains(event.target)) clearSourcePreview(owner);
    }, { passive: true });
    elements.messages?.addEventListener('mousemove', event => {
        const owner = state.sourcePreviewOwner;
        if (owner && !owner.contains(event.target)) clearSourcePreview(owner);
    }, { passive: true });
    elements.messages?.addEventListener('pointerleave', () => clearSourcePreview());
    elements.messages?.addEventListener('pointercancel', () => clearSourcePreview());
    window.addEventListener('blur', () => clearSourcePreview());
    window.addEventListener('resize', syncSettingsPanelPortal, { passive: true });
    settingsPanelBreakpoint = window.matchMedia?.('(max-width: 900px)') || null;
    settingsPanelBreakpoint?.addEventListener?.('change', syncSettingsPanelPortal);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) clearSourcePreview();
    });
    document.addEventListener('vectoria:dataset-changed', event => void loadConversation(event.detail?.datasetId || null));
    document.addEventListener('vectoria:generation-mode-changed', () => {
        if (isExternalMode() && state.isGenerating) {
            pipeline.abortRAG('chat');
            window.cancelHyDEReview?.();
        }
        refreshAvailability();
    });
    document.addEventListener('vectoria:mcp-state', refreshAvailability);
    document.addEventListener('vectoria:models-ready', () => {
        refreshAvailability();
        // The local model usually arrives after the dataset, so draft the
        // data-grounded suggestions once it is actually available.
        if (state.datasetId && !state.conversation?.messages?.length) {
            void refreshSuggestedQuestions(state.datasetId);
        }
    });
    document.addEventListener('vectoria:local-ai-operation', refreshAvailability);
    document.addEventListener('vectoria:scope-changed', refreshScopeOption);
    window.ConfigManager?.observeConfig?.(syncChatSettings);
}

async function initializeChatInterface() {
    if (state.initialized) return;
    cacheElements();
    if (!elements.chatWorkspace || !pipeline.chat) return;
    state.initialized = true;
    bindEvents();
    await loadConversation();
    showDocuments();
}

function closeTopSurface() {
    if (elements.exportMenu && !elements.exportMenu.hidden) {
        closeExportMenu();
        return true;
    }
    if (elements.settingsPanel && !elements.settingsPanel.hidden) {
        closeSettingsPanel();
        return true;
    }
    return false;
}

window.VectoriaChat = {
    showChat,
    showDocuments,
    closeTopSurface,
    submitQuestion: async (question, { scope = 'all' } = {}) => {
        const value = String(question || '').trim();
        if (!value) return false;
        showChat();
        if (elements.scope && ['all', 'current'].includes(scope)) elements.scope.value = scope;
        if (elements.input) elements.input.value = value;
        autoSizeComposer();
        await sendMessage();
        return true;
    },
    reload: () => loadConversation(),
    getState: () => ({ activeView: state.activeView, datasetId: state.datasetId, isGenerating: state.isGenerating })
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => void initializeChatInterface(), { once: true });
} else {
    void initializeChatInterface();
}

export { initializeChatInterface };
