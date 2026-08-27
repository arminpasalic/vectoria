import test from 'node:test';
import assert from 'node:assert/strict';
import {
    ChatConversationStore,
    createChatMessage,
    createEmptyConversation
} from '../web_interface/static/js/browser-ml/chat-store.js';

class MemoryAdapter {
    constructor() { this.map = new Map(); }
    async getItem(key) { return this.map.get(key) ?? null; }
    async setItem(key, value) { this.map.set(key, structuredClone(value)); }
    async removeItem(key) { this.map.delete(key); }
}

test('conversation persists independently per dataset', async () => {
    const adapter = new MemoryAdapter();
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const conversation = createEmptyConversation('dataset-a', 1);
    conversation.messages.push(createChatMessage({ role: 'user', content: 'Hello' }, 2));
    await store.save(conversation);
    assert.equal((await store.load('dataset-a')).messages[0].content, 'Hello');
    assert.equal((await store.load('dataset-b')).messages.length, 0);
});

test('unfinished generations restore as interrupted', async () => {
    const adapter = new MemoryAdapter();
    await adapter.setItem('chat:dataset-a', {
        version: 1,
        datasetId: 'dataset-a',
        messages: [{ role: 'assistant', content: 'partial', status: 'generating' }]
    });
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const restored = await store.load('dataset-a');
    assert.equal(restored.messages[0].status, 'interrupted');
    assert.equal(restored.version, 4);
    assert.equal(restored.messages[0].route.resolved, 'documents');
});

test('schema v1 conversations migrate without losing messages', async () => {
    const adapter = new MemoryAdapter();
    await adapter.setItem('chat:dataset-a', {
        version: 1,
        datasetId: 'dataset-a',
        createdAt: 10,
        messages: [{ id: 'old', role: 'user', content: 'Keep me', status: 'complete', createdAt: 11 }]
    });
    const restored = await new ChatConversationStore(adapter, { now: () => 100 }).load('dataset-a');
    assert.equal(restored.version, 4);
    assert.equal(restored.messages[0].content, 'Keep me');
    assert.equal(restored.contextCutoffAt, null);
    assert.equal((await adapter.getItem('chat:dataset-a')).version, 4);
});

test('schema v2 gains document route provenance while preserving sources and cutoff', async () => {
    const adapter = new MemoryAdapter();
    await adapter.setItem('chat:dataset-a', {
        version: 2,
        datasetId: 'dataset-a',
        contextCutoffAt: 50,
        messages: [{ role: 'assistant', content: 'Answer [Doc 1]', status: 'complete', sources: [{ docId: 'a' }], citations: [1] }]
    });
    const restored = await new ChatConversationStore(adapter, { now: () => 100 }).load('dataset-a');
    assert.equal(restored.version, 4);
    assert.equal(restored.contextCutoffAt, 50);
    assert.equal(restored.messages[0].route.reason, 'legacy_document_turn');
    assert.deepEqual(restored.messages[0].sources, [{ docId: 'a' }]);
    assert.deepEqual(restored.messages[0].citations, [1]);
});

test('schema v3 preserves route and sources while adding reliability provenance', async () => {
    const adapter = new MemoryAdapter();
    await adapter.setItem('chat:dataset-a', {
        version: 3,
        datasetId: 'dataset-a',
        messages: [{
            role: 'assistant',
            content: 'Answer [Doc 1]',
            status: 'complete',
            route: { requested: 'auto', resolved: 'documents', reason: 'auto_documents_default' },
            sources: [{ docId: 'a' }],
            metadata: { phase: 'generation' }
        }]
    });
    const restored = await new ChatConversationStore(adapter, { now: () => 100 }).load('dataset-a');
    assert.equal(restored.version, 4);
    assert.equal(restored.messages[0].route.reason, 'auto_documents_default');
    assert.deepEqual(restored.messages[0].sources, [{ docId: 'a' }]);
    assert.equal(restored.messages[0].metadata.operationPhase, 'generation');
    assert.deepEqual(restored.messages[0].metadata.recoveryAttempts, []);
    assert.equal(restored.messages[0].metadata.sourcesPreservedAfterError, false);
});

test('generating placeholders persist before reload and dataset deletion removes them', async () => {
    const adapter = new MemoryAdapter();
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const conversation = createEmptyConversation('dataset-a', 1);
    conversation.messages.push(createChatMessage({ role: 'assistant', status: 'generating' }, 2));
    await store.save(conversation);
    assert.equal((await adapter.getItem('chat:dataset-a')).messages[0].status, 'generating');
    await store.delete('dataset-a');
    assert.equal(await adapter.getItem('chat:dataset-a'), null);
});

test('invalid schemas safely start a fresh conversation and clear removes it', async () => {
    const adapter = new MemoryAdapter();
    await adapter.setItem('chat:dataset-a', { version: 999, messages: [{ role: 'user', content: 'old' }] });
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    assert.equal((await store.load('dataset-a')).messages.length, 0);
    const cleared = await store.clear('dataset-a');
    assert.equal(cleared.messages.length, 0);
});

test('overlapping saves are serialized and snapshot mutable conversation state', async () => {
    let releaseFirstWrite;
    let writeCount = 0;
    class DelayedAdapter extends MemoryAdapter {
        async setItem(key, value) {
            writeCount++;
            if (writeCount === 1) {
                await new Promise(resolve => { releaseFirstWrite = resolve; });
            }
            await super.setItem(key, value);
        }
    }

    const adapter = new DelayedAdapter();
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const conversation = createEmptyConversation('dataset-a', 1);
    conversation.messages.push(createChatMessage({ role: 'user', content: 'First' }, 2));
    const firstSave = store.save(conversation);
    await Promise.resolve();

    conversation.messages[0].content = 'Second';
    const secondSave = store.save(conversation);
    await Promise.resolve();
    assert.equal(writeCount, 1);

    releaseFirstWrite();
    await Promise.all([firstSave, secondSave]);
    assert.equal(writeCount, 2);
    assert.equal((await store.load('dataset-a')).messages[0].content, 'Second');
});

test('loads and deletes wait for pending writes on the same dataset', async () => {
    let releaseWrite;
    class DelayedAdapter extends MemoryAdapter {
        async setItem(key, value) {
            await new Promise(resolve => { releaseWrite = resolve; });
            await super.setItem(key, value);
        }
    }

    const adapter = new DelayedAdapter();
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const conversation = createEmptyConversation('dataset-a', 1);
    conversation.messages.push(createChatMessage({ role: 'user', content: 'Saved' }, 2));
    const saving = store.save(conversation);
    await new Promise(resolve => setImmediate(resolve));
    const loading = store.load('dataset-a');
    const deleting = store.delete('dataset-a');

    releaseWrite();
    await saving;
    assert.equal((await loading).messages[0].content, 'Saved');
    await deleting;
    assert.equal(await adapter.getItem('chat:dataset-a'), null);
});

test('a failed write does not poison later conversation saves', async () => {
    let attempts = 0;
    class FlakyAdapter extends MemoryAdapter {
        async setItem(key, value) {
            attempts++;
            if (attempts === 1) throw new Error('quota unavailable');
            await super.setItem(key, value);
        }
    }

    const adapter = new FlakyAdapter();
    const store = new ChatConversationStore(adapter, { now: () => 100 });
    const conversation = createEmptyConversation('dataset-a', 1);
    conversation.messages.push(createChatMessage({ role: 'user', content: 'Retry me' }, 2));

    await assert.rejects(store.save(conversation), /quota unavailable/);
    await store.save(conversation);
    assert.equal((await store.load('dataset-a')).messages[0].content, 'Retry me');
});
