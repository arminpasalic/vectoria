export const CHAT_SCHEMA_VERSION = 4;

function clone(value) {
    return JSON.parse(JSON.stringify(value));
}

export function createEmptyConversation(datasetId, now = Date.now()) {
    return {
        version: CHAT_SCHEMA_VERSION,
        datasetId: String(datasetId),
        createdAt: now,
        updatedAt: now,
        contextCutoffAt: null,
        messages: []
    };
}

export function createChatMessage({ role, content = '', status = 'complete', turnId = null, scope = null, route = null, sources = [], citations = [], metadata = {} }, now = Date.now()) {
    const random = globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
    return {
        id: `chat_${now}_${random}`,
        turnId: turnId || `turn_${now}_${random}`,
        role,
        content,
        status,
        createdAt: now,
        scope,
        route,
        sources,
        citations,
        metadata
    };
}

function normalizeConversation(datasetId, value, now, { interruptGenerating = false } = {}) {
    const supportedVersion = [1, 2, 3, CHAT_SCHEMA_VERSION].includes(value?.version);
    if (!value || !supportedVersion || !Array.isArray(value.messages)) {
        return createEmptyConversation(datasetId, now);
    }
    const messages = value.messages
        .filter(message => message && ['user', 'assistant'].includes(message.role))
        .map(message => {
            const sources = Array.isArray(message.sources) ? message.sources : [];
            const legacyAssistantRoute = message.role === 'assistant'
                ? { requested: 'auto', resolved: 'documents', reason: 'legacy_document_turn', handoffAvailable: false }
                : null;
            const metadata = message.metadata && typeof message.metadata === 'object' ? message.metadata : {};
            return {
                ...message,
                status: interruptGenerating && message.status === 'generating' ? 'interrupted' : (message.status || 'complete'),
                route: message.route && typeof message.route === 'object' ? message.route : legacyAssistantRoute,
                sources,
                citations: Array.isArray(message.citations) ? message.citations : [],
                metadata: {
                    ...metadata,
                    operationPhase: metadata.operationPhase || metadata.phase || null,
                    errorCode: metadata.errorCode || metadata.errorDiagnostic?.code || null,
                    recoveryAttempts: Array.isArray(metadata.recoveryAttempts) ? metadata.recoveryAttempts : [],
                    sourcesPreservedAfterError: metadata.sourcesPreservedAfterError === true
                }
            };
        });
    return {
        version: CHAT_SCHEMA_VERSION,
        datasetId: String(datasetId),
        createdAt: Number(value.createdAt) || now,
        updatedAt: Number(value.updatedAt) || now,
        contextCutoffAt: Number(value.contextCutoffAt) || null,
        messages
    };
}

export class ChatConversationStore {
    constructor(adapter, { now = () => Date.now() } = {}) {
        if (!adapter || typeof adapter.getItem !== 'function' || typeof adapter.setItem !== 'function') {
            throw new Error('ChatConversationStore requires a localforage-compatible adapter');
        }
        this.adapter = adapter;
        this.now = now;
        this.writeQueues = new Map();
    }

    key(datasetId) {
        return `chat:${String(datasetId)}`;
    }

    enqueueWrite(key, task) {
        const previous = this.writeQueues.get(key) || Promise.resolve();
        const operation = previous.catch(() => undefined).then(task);
        let tracked;
        tracked = operation.then(
            () => { if (this.writeQueues.get(key) === tracked) this.writeQueues.delete(key); },
            () => { if (this.writeQueues.get(key) === tracked) this.writeQueues.delete(key); }
        );
        this.writeQueues.set(key, tracked);
        return operation;
    }

    async waitForWrites(key) {
        await this.writeQueues.get(key);
    }

    async load(datasetId) {
        const key = this.key(datasetId);
        await this.waitForWrites(key);
        const raw = await this.adapter.getItem(key);
        const conversation = normalizeConversation(datasetId, raw, this.now(), { interruptGenerating: true });
        const needsRewrite = raw && (
            raw.version !== CHAT_SCHEMA_VERSION
            || !Array.isArray(raw.messages)
            || raw.messages.some(message => message?.status === 'generating')
        );
        if (needsRewrite) {
            await this.save(conversation);
        }
        return clone(conversation);
    }

    async save(conversation) {
        const normalized = normalizeConversation(conversation.datasetId, conversation, this.now());
        normalized.updatedAt = this.now();
        const snapshot = clone(normalized);
        await this.enqueueWrite(this.key(normalized.datasetId), () => this.adapter.setItem(this.key(normalized.datasetId), snapshot));
        return clone(snapshot);
    }

    async clear(datasetId) {
        const key = this.key(datasetId);
        await this.enqueueWrite(key, () => this.adapter.removeItem(key));
        return createEmptyConversation(datasetId, this.now());
    }

    async delete(datasetId) {
        const key = this.key(datasetId);
        await this.enqueueWrite(key, () => this.adapter.removeItem(key));
    }
}
