import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const rag = await readFile(new URL('../web_interface/static/js/browser-ml/llm-rag.js', import.meta.url), 'utf8');
const pipeline = await readFile(new URL('../web_interface/static/js/browser-ml/index.js', import.meta.url), 'utf8');
const analysis = await readFile(new URL('../web_interface/static/js/browser-ml/analysis.js', import.meta.url), 'utf8');
const integration = await readFile(new URL('../web_interface/static/js/browser-integration.js', import.meta.url), 'utf8');
const chat = await readFile(new URL('../web_interface/static/js/chat-interface.js', import.meta.url), 'utf8');
const vectoria = await readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8');
const index = await readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8');

test('all WebLLM generation paths share an exclusive operation lease', () => {
    assert.match(rag, /beginOperation\(owner = 'local-ai'/);
    assert.match(rag, /error\.code = 'local_ai_busy'/);
    assert.match(rag, /async withOperation\(owner, options, task\)/);
    assert.match(rag, /if \(supplied\) this\.throwIfOperationCancelled\(operation\)/);
    assert.match(rag, /generateHyDE[\s\S]*?this\.withOperation\(options\.owner \|\| 'hyde'/);
    assert.match(rag, /generateRaw[\s\S]*?this\.withOperation\(options\.owner \|\| 'cluster-label'/);
    assert.match(rag, /generateFromMessages[\s\S]*?this\.withOperation\(options\.owner \|\| 'chat'/);
    assert.match(pipeline, /const operationToken = this\.rag\.beginOperation\('chat', \{ datasetId \}\)/);
    assert.match(integration, /operationToken = pipeline\.rag\.beginOperation\('rag'/);
    assert.match(index, /pipeline\.rag\.beginOperation\('mcp'/);
});

test('operation ownership protects cancellation, suspension, processing, and mode changes', () => {
    assert.match(rag, /abort\(ownerOrOperation = null\)/);
    assert.match(rag, /if \(!this\._activeOperation\) return false/);
    assert.match(rag, /this\._activeOperation\.cancelled = true/);
    assert.match(rag, /phase === 'loading-model'[\s\S]*?this\.worker\.terminate\(\)/);
    assert.match(rag, /Promise\.race\(\[this\.initialize\(onProgress\), abortPromise\]\)/);
    assert.match(rag, /suspension skipped while/);
    assert.match(pipeline, /Cannot process a new file while local AI is/);
    assert.match(chat, /pipeline\.abortRAG\('chat'\)/);
    assert.match(integration, /pipeline\.abortRAG\('rag'\)/);
    assert.match(vectoria, /abortRAG\?\.\('cluster-label'\)/);
    // Mode blocking is exercised behaviorally in generation-mode-controller.test.js.
});

test('raw generation and HyDE always return to the shared idle lifecycle', () => {
    assert.match(rag, /finally \{[\s\S]*?if \(ownsOperation\) this\.endOperation\(operation\)/);
    assert.match(rag, /endOperation[\s\S]*?this\._scheduleIdleSuspend\(\)/);
    assert.match(rag, /ensureEngineReady[\s\S]*?this\._clearIdleTimer\(\)/);
    assert.match(rag, /phase === 'awaiting-input'\) this\._scheduleIdleSuspend\(\)/);
});

test('cluster labelling validates setup, dataset provenance, empty clusters, and partial failures', () => {
    assert.match(vectoria, /const localConfigured = window\.browserML\?\.isReady === true && Boolean\(pipeline\?\.rag\)/);
    assert.match(vectoria, /if \(!unique\.length\)[\s\S]*?phase: 'empty'/);
    assert.match(vectoria, /const operationToken = pipeline\.rag\.beginOperation\('cluster-label', \{ datasetId \}\)/);
    assert.match(vectoria, /failed\+\+/);
    assert.match(analysis, /The active dataset changed while this cluster label was being generated/);
    assert.match(index, /validateExternalClusterLabelTarget/);
});

test('model readiness follows the selected model and cold starts report progress', () => {
    assert.match(index, /MODEL_SIGNATURE_KEY = 'vectoria_models_ready_signature'/);
    assert.match(index, /MODEL_AUTOSTART_KEY = 'vectoria_model_setup_autostart'/);
    assert.match(vectoria, /localStorage\.removeItem\('vectoria_models_ready'\)/);
    assert.match(vectoria, /localStorage\.setItem\('vectoria_model_setup_autostart', 'true'\)/);
    assert.match(rag, /onStatus\?\.\('loading-model'/);
    assert.match(chat, /status === 'loading-model' \? 'Loading cached local AI/);
});

test('documents-only helper routing occurs before taking a WebLLM operation lease', () => {
    const routeIndex = pipeline.indexOf("const route = routeChatTurn(question, { history })");
    const helperIndex = pipeline.indexOf("if (route.resolved === 'helper')");
    const ragRequiredIndex = pipeline.indexOf("if (!this.rag) throw new Error('Set up the local AI model", helperIndex);
    const leaseIndex = pipeline.indexOf("const operationToken = this.rag.beginOperation('chat'", routeIndex);
    assert.ok(routeIndex >= 0 && helperIndex > routeIndex && ragRequiredIndex > helperIndex && leaseIndex > ragRequiredIndex);
    assert.match(pipeline, /if \(route\.resolved === 'helper'\)[\s\S]*?buildDocumentHelperReply/);
    assert.doesNotMatch(pipeline, /buildConversationChatPrompt/);
    assert.match(pipeline, /retrievalPerformed: false/);
});

test('HyDE review gates retrieval and keeps BM25 on the unchanged user question', () => {
    assert.match(pipeline, /generateHyDE\(retrievalQueries\.contextualSemanticQuery/);
    assert.match(pipeline, /setOperationPhase\(operationToken, 'awaiting-input'\)/);
    assert.match(pipeline, /await options\.onHyDEReview/);
    assert.match(pipeline, /if \(action === 'cancel'\)[\s\S]*?HyDE review cancelled/);
    assert.match(pipeline, /keywordQuery: retrievalQueries\.keywordQuery/);
    assert.match(chat, /showHyDEReviewModal\(review\.question, review\.generatedText, \{ detailed: true \}\)/);
    assert.match(pipeline, /buildDocumentChatPrompt\(\{[\s\S]*?maxSources: numResults/);
});

test('WebLLM streaming captures usage and retries context overflow at most once before output', () => {
    assert.match(rag, /stream_options: \{ include_usage: true \}/);
    assert.match(rag, /promptTokens: Number\(usage\.prompt_tokens\)/);
    assert.match(rag, /prefillTokensPerSecond/);
    assert.match(rag, /timeToFirstTokenSeconds/);
    assert.match(pipeline, /runChatGenerationWithRecovery\(\{/);
    assert.match(pipeline, /hasVisibleOutput: \(\) => streamState\.emitted/);
    assert.match(pipeline, /recoveryDiagnostics: planned\.recoveryDiagnostics/);
    assert.match(pipeline, /queryStream\(question, questionEmbedding, onChunk/);
    assert.match(pipeline, /applyCitationSafety\(answerResult, planned\.prompt\.includedSources\)/);
    assert.doesNotMatch(pipeline, /applyStrictGrounding|groundFreeTextAnswer|repairing-evidence/);
});

test('MCP remains one-shot and exposes no local chat or HyDE-review state', async () => {
    const ragTools = await readFile(new URL('../mcp-server/tools/rag.js', import.meta.url), 'utf8');
    const distributed = await readFile(new URL('../web_interface/static/mcp-server/tools/rag.js', import.meta.url), 'utf8');
    assert.equal(ragTools, distributed);
    assert.match(ragTools, /browser-local WebLLM model/);
    assert.match(ragTools, /MCP host owns conversational routing and memory/);
    assert.doesNotMatch(ragTools, /chat_history|routing_mode|hyde_review|conversation_store/);
});
