import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const exportImport = await readFile(new URL('../web_interface/static/js/export-import.js', import.meta.url), 'utf8');
const storage = await readFile(new URL('../web_interface/static/js/browser-ml/storage.js', import.meta.url), 'utf8');
const pipeline = await readFile(new URL('../web_interface/static/js/browser-ml/index.js', import.meta.url), 'utf8');
const chat = await readFile(new URL('../web_interface/static/js/chat-interface.js', import.meta.url), 'utf8');
const vectoria = await readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8');

test('imports become new persistent chat-capable datasets', () => {
    assert.match(exportImport, /const datasetId = `dataset_\$\{Date\.now\(\)\}_\$\{randomPart\}`/);
    assert.match(exportImport, /pipeline\.currentDatasetId = datasetId/);
    assert.match(exportImport, /pipeline\.storage\.saveDataset\(datasetId/);
    assert.match(exportImport, /detail: \{ datasetId, reason: 'imported' \}/);
    assert.match(exportImport, /pipeline\.chunkVectorSearch = new ChunkVectorSearch/);
    assert.match(exportImport, /pipeline\.chunkBM25Search = new ChunkBM25Search/);
});

test('three-tier embeddings and chunk indexes survive saved-dataset reloads', () => {
    assert.match(storage, /Array\.isArray\(raw\.parent\)/);
    assert.match(storage, /chunkToParentMap: raw\.chunkToParentMap \|\| raw\.chunk_map/);
    assert.match(pipeline, /const storedChunks = Array\.isArray\(resolvedEmbeddings\?\.chunks\)/);
    assert.match(pipeline, /this\.chunkVectorSearch = buildChunkIndex\(storedChunks/);
    assert.match(pipeline, /this\.rag\.setBM25Search\(this\.chunkBM25Search \|\| this\.bm25Search\)/);
});

test('chat availability permits model-free helpers but gates substantive suggestions', () => {
    assert.match(chat, /const needsLocalSetup = hasDataset && !external && !localModelReady/);
    assert.match(chat, /elements\.modelNote\.hidden = !needsLocalSetup/);
    assert.match(chat, /const localUnavailable = blockedByOtherAI \|\| processing/);
    assert.match(chat, /pendingRoute\.resolved === 'documents' && !pipeline\.rag/);
    assert.match(chat, /button\.disabled = !hasDataset \|\| external \|\| localUnavailable/);
    assert.match(chat, /localUnavailable \|\| needsLocalSetup \|\| state\.isGenerating/);
});

test('full reset clears live stores and cannot remain stuck indefinitely', () => {
    const clearStores = vectoria.indexOf("runResetStep('Vectoria IndexedDB stores'");
    const enumerateDatabases = vectoria.indexOf("runResetStep('IndexedDB databases'");
    assert.ok(clearStores >= 0 && enumerateDatabases > clearStores);
    assert.match(vectoria, /pipeline\?\.storage\?\.clearAll\?\.\(\)/);
    assert.match(vectoria, /const resetWatchdog = setTimeout/);
    assert.match(vectoria, /setTimeout\(\(\) => window\.location\.reload\(\), 500\)/);
});

test('retired RAG controls no longer emit startup warnings', () => {
    assert.doesNotMatch(vectoria, /loadRAGSettings not available; skipping RAG form sync/);
    assert.doesNotMatch(vectoria, /Missing RAG metadata control elements/);
});
