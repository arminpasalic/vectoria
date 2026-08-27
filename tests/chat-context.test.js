import test from 'node:test';
import assert from 'node:assert/strict';
import {
    applyChatContextCutoff,
    buildChatPrompt,
    buildChatRetrievalQueries,
    buildContextualRetrievalQuery,
    buildConversationChatPrompt,
    buildDocumentHelperReply,
    buildConversationMemory,
    buildConversationMessages,
    buildDocumentChatPrompt,
    buildEvidenceContext,
    captureChatScope,
    createChatOptionsSnapshot,
    estimateChatTokens,
    extractCitations,
    hasMeaningfulKeywordTerms,
    lastGroundedTopic,
    runChatGenerationWithRecovery,
    restoreChatScope,
    routeChatTurn,
    sanitizeCitationBounds,
    splitCitationSegments,
    tokenizeCitations
} from '../web_interface/static/js/browser-ml/chat-context.js';

const source = (id, text) => ({ doc_id: id, text, metadata: { title: `Document ${id}` } });
const history = [
    { role: 'user', content: 'Who founded Acme?', status: 'complete' },
    { role: 'assistant', content: 'Ada founded Acme. [Doc 1]', status: 'complete' },
    { role: 'user', content: 'When did it launch?', status: 'complete' },
    { role: 'assistant', content: 'It launched in 2020. [Doc 2]', status: 'complete' }
];

test('context plans stay inside 2K, 4K, 8K, and 32K model windows', () => {
    for (const contextWindow of [2048, 4096, 8192, 32768]) {
        const plan = buildChatPrompt({
            question: 'Summarize the launch and cite the evidence.',
            history,
            sources: [source('a', 'Launch evidence. '.repeat(600)), source('b', 'Founder evidence. '.repeat(600)), source('c', 'Market evidence. '.repeat(600))],
            systemPrompt: 'Answer from documents.',
            userTemplate: 'Documents:\n{context}\n\nQuestion: {question}',
            contextWindow,
            maxOutputTokens: 12000
        });
        const total = plan.telemetry.usedInputTokens + plan.telemetry.outputTokens + plan.telemetry.safetyTokens;
        assert.ok(total <= contextWindow, `${total} must fit ${contextWindow}`);
        assert.ok(plan.includedSources.length >= 2);
    }
});

test('memory prioritizes recent turns and digests omitted turns deterministically', () => {
    const memory = buildConversationMemory(history, 35);
    assert.match(memory.text, /When did it launch/);
    assert.ok(memory.omittedTurns >= 1);
    assert.ok(memory.tokens <= 35);
    assert.doesNotMatch(memory.text, /\[Doc/);
});

test('memory modes support smart summaries, recent-only truncation, and no-memory turns', () => {
    const longHistory = [...history, ...history, ...history];
    const adaptive = buildConversationMemory(longHistory, 120, { mode: 'adaptive', maxTurns: 2 });
    assert.equal(adaptive.mode, 'adaptive');
    assert.ok(adaptive.summarizedTurns > 0);
    assert.match(adaptive.text, /Earlier turns:/);

    const recent = buildConversationMemory(longHistory, 120, { mode: 'recent', maxTurns: 2 });
    assert.equal(recent.mode, 'recent');
    assert.equal(recent.summarizedTurns, 0);
    assert.doesNotMatch(recent.text, /Earlier turns:/);
    assert.ok(recent.omittedTurns > 0);

    const none = buildConversationMemory(longHistory, 120, { mode: 'none' });
    assert.equal(none.text, '');
    assert.equal(none.includedTurns, 0);
    assert.ok(none.omittedTurns > 0);
});

test('scope snapshots preserve all, filtered, lasso, and zero-result selections', () => {
    const all = captureChatScope({ scopeType: 'all', scopedCount: 10, totalDocuments: 10, docIds: ['ignored'] });
    assert.equal(all.docIds, null);

    const filteredIds = ['a', 'b'];
    const filtered = captureChatScope({ scopeType: 'current', label: 'Current view (2)', scopedCount: 2, totalDocuments: 10, docIds: filteredIds });
    filteredIds.push('changed-after-send');
    assert.deepEqual(restoreChatScope(filtered).info.docIds, ['a', 'b']);

    const lasso = captureChatScope({ scopeType: 'current', label: 'Lasso (1)', scopedCount: 1, totalDocuments: 10, docIds: [7] });
    assert.deepEqual(lasso.docIds, ['7']);

    const empty = captureChatScope({ scopeType: 'current', label: 'Current view (0)', scopedCount: 0, totalDocuments: 10, docIds: [] });
    assert.deepEqual(restoreChatScope(empty).info.docIds, []);
});

test('multilingual prompt estimates remain inside the configured window', () => {
    const plan = buildChatPrompt({
        question: 'Hvad siger dokumenterne om København? 東京についても比較してください。',
        history,
        sources: [source('a', 'Dansk evidens om København. '.repeat(120)), source('b', '東京に関する証拠です。'.repeat(120))],
        contextWindow: 4096,
        maxOutputTokens: 4096
    });
    assert.ok(plan.telemetry.usedInputTokens + plan.telemetry.outputTokens + plan.telemetry.safetyTokens <= 4096);
});

test('short follow-ups add only the preceding user question', () => {
    assert.match(buildContextualRetrievalQuery('What about 2024?', history), /When did it launch/);
    assert.equal(buildContextualRetrievalQuery('Explain the complete 2024 launch strategy and regional differences in detail.', history), 'Explain the complete 2024 launch strategy and regional differences in detail.');

    const interleaved = [
        ...history,
        { role: 'user', content: 'Thanks' },
        { role: 'assistant', content: 'You are welcome.', status: 'complete', route: { resolved: 'conversation' } }
    ];
    const contextual = buildContextualRetrievalQuery('What about 2024?', interleaved);
    assert.match(contextual, /When did it launch/);
    assert.doesNotMatch(contextual, /Thanks/);
    assert.equal(buildContextualRetrievalQuery('What about 2024?', [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi', status: 'complete', route: { resolved: 'conversation' } }]), 'What about 2024?');
});

test('HyDE changes semantic retrieval while generic follow-ups anchor BM25', () => {
    const planned = buildChatRetrievalQueries('What about 2024?', history, 'A hypothetical 2024 launch passage');
    assert.equal(planned.semanticQuery, 'A hypothetical 2024 launch passage');
    assert.equal(planned.keywordQuery, 'When did it launch?');
    assert.match(planned.contextualSemanticQuery, /When did it launch/);
    assert.equal(planned.hydeUsed, true);
    assert.equal(planned.anchorUsed, true);
    assert.equal(planned.anchorQuestion, 'When did it launch?');

    const standalone = buildChatRetrievalQueries('Explain the launch strategy in every region.', history);
    assert.equal(standalone.semanticQuery, standalone.contextualSemanticQuery);
    assert.equal(standalone.keywordQuery, 'Explain the launch strategy in every region.');
    assert.equal(standalone.hydeUsed, false);
    assert.equal(standalone.anchorUsed, false);
});

test('elliptical English and Danish follow-ups reuse the last document topic', () => {
    for (const question of ['Explain more', 'Elaborate', 'Go on', 'Tell me more', 'How so?', 'Uddyb', 'Fortsæt', 'Mere']) {
        const planned = buildChatRetrievalQueries(question, history);
        assert.match(planned.semanticQuery, /Earlier user question: When did it launch\?/);
        assert.equal(planned.keywordQuery, 'When did it launch?');
        assert.equal(planned.anchorUsed, true);
    }
    assert.equal(hasMeaningfulKeywordTerms('Explain more'), false);
    assert.equal(hasMeaningfulKeywordTerms('Compare Android battery reviews'), true);
});

test('manual memory cutoffs preserve visible messages while excluding older model context', () => {
    const messages = [
        { id: 'old', createdAt: 100, content: 'Visible but omitted' },
        { id: 'new', createdAt: 300, content: 'Visible and included' }
    ];
    assert.deepEqual(applyChatContextCutoff(messages, 200).map(message => message.id), ['new']);
    assert.deepEqual(applyChatContextCutoff(messages, null).map(message => message.id), ['old', 'new']);
    assert.equal(messages.length, 2);
});

test('chat settings are captured as a complete immutable turn snapshot', () => {
    const config = {
        version: 7,
        ui_preferences: { hyde_enabled: true },
        chat: { routing_mode: 'documents', include_metadata: true, metadata_fields: ['title'], memory_mode: 'recent', max_memory_turns: 3 },
        chat_prompts: { conversation_system_prompt: 'Conversation only' },
        search: { num_results: 7, retrieval_k: 80, vector_weight: 0.75, similarity_threshold: 0.45 },
        llm: { temperature: 0.3, top_p: 0.85, repeat_penalty: 1.2, max_tokens: 1024, context_window_size: 8192 },
        rag_prompts: { system_prompt: 'Grounded', user_template: '{context}\n{question}' },
        hyde: { prompt: 'Hypothesis', temperature: 0.1, max_tokens: 128 }
    };
    const snapshot = createChatOptionsSnapshot(config);
    config.search.num_results = 1;
    config.chat.metadata_fields.push('changed-later');
    assert.deepEqual(snapshot, {
        mode: 'documents',
        useHyDE: true,
        metadataMode: 'selected',
        includeMetadata: true,
        metadataFields: ['title'],
        memoryMode: 'recent',
        maxMemoryTurns: 3,
        numResults: 7,
        retrievalK: 80,
        vectorWeight: 0.75,
        similarityThreshold: 0.45,
        temperature: 0.3,
        topP: 0.85,
        repeatPenalty: 1.2,
        maxTokens: 1024,
        contextWindow: 8192,
        systemPrompt: 'Grounded',
        userTemplate: '{context}\n{question}',
        hydePrompt: 'Hypothesis',
        hydeTemperature: 0.1,
        hydeMaxTokens: 128,
        configVersion: 7
    });
});

test('documents-only router uses deterministic helpers and retrieves every substantive turn', () => {
    const documentHistory = [
        { role: 'user', content: 'What do the Walmart reviews say?' },
        { role: 'assistant', content: 'Customers discuss delivery. [Doc 1]', status: 'complete', route: { resolved: 'documents' } }
    ];
    const cases = [
        ['What do the Walmart reviews say?', 'documents', 'documents_only'],
        ['What are we talking about?', 'helper', 'conversation_recap'],
        ['what', 'helper', 'clarification'],
        ['who are oyu', 'helper', 'identity'],
        ['hello', 'helper', 'greeting'],
        ['tak', 'helper', 'thanks'],
        ['thnak you', 'helper', 'thanks'],
        ['Ok', 'helper', 'acknowledgement'],
        ['hmm', 'helper', 'acknowledgement'],
        ['What are we chatting about?', 'helper', 'conversation_recap'],
        ['How do I use Vectoria?', 'helper', 'ui_help'],
        ['no', 'helper', 'correction'],
        ['Please make that shorter', 'helper', 'response_style'],
        ['What about 2024?', 'documents', 'documents_only'],
        ['Which product?', 'documents', 'documents_only'],
        ['Explain quantum gravity', 'documents', 'documents_only']
    ];
    for (const [question, resolved, reason] of cases) {
        assert.deepEqual(routeChatTurn(question, { requestedMode: 'auto', history: documentHistory }), {
            requested: 'documents', resolved, reason, handoffAvailable: false
        });
    }
});

test('durable topic state ignores failed, helper, formatting, correction, and clarification turns', () => {
    const topicHistory = [
        { role: 'user', content: 'What is the Atlas retention period?', turnId: 'a' },
        {
            role: 'assistant',
            content: 'Atlas retains records for 30 days. [Doc 1]',
            turnId: 'a',
            status: 'complete',
            route: { resolved: 'documents' },
            metadata: { groundingState: 'supported', topicEligible: true, resolvedQuery: 'Atlas retention period' }
        },
        { role: 'user', content: 'Please make that shorter', turnId: 'b' },
        { role: 'assistant', content: 'Ask again with the format you want.', turnId: 'b', route: { resolved: 'helper' } },
        { role: 'user', content: 'What about refunds?', turnId: 'c' },
        {
            role: 'assistant',
            content: 'Vectoria could not form a supported answer.',
            turnId: 'c',
            route: { resolved: 'documents' },
            metadata: { groundingState: 'insufficient', topicEligible: false, resolvedQuery: 'Atlas refunds' }
        },
        { role: 'user', content: 'No, I meant deletions', turnId: 'd' },
        { role: 'assistant', content: 'Tell me what you meant.', turnId: 'd', route: { resolved: 'helper' } }
    ];
    assert.equal(lastGroundedTopic(topicHistory).turnId, 'a');
    assert.equal(lastGroundedTopic(topicHistory).resolvedQuery, 'Atlas retention period');
    const followUp = buildChatRetrievalQueries('How is that enforced?', topicHistory);
    assert.equal(followUp.anchorQuestion, 'Atlas retention period');
    assert.match(followUp.contextualSemanticQuery, /Atlas retention period/);
    assert.doesNotMatch(followUp.contextualSemanticQuery, /refunds|deletions/i);
});

test('legacy route options cannot bypass documents-only behavior', () => {
    assert.deepEqual(routeChatTurn('hello', { requestedMode: 'documents' }), {
        requested: 'documents', resolved: 'helper', reason: 'greeting', handoffAvailable: false
    });
    assert.deepEqual(routeChatTurn('Summarize the selected files', { requestedMode: 'chat' }), {
        requested: 'documents', resolved: 'documents', reason: 'documents_only', handoffAvailable: false
    });
    assert.equal(routeChatTurn('Hvem er du?', { requestedMode: 'chat' }).resolved, 'helper');
    assert.match(buildDocumentHelperReply('Who are you?'), /Vectoria/i);
    assert.match(buildDocumentHelperReply('What are we talking about?', [
        { role: 'user', content: 'What do the Walmart reviews say?' },
        { role: 'assistant', content: 'Delivery is discussed.', route: { resolved: 'documents' } }
    ]), /Walmart reviews/i);
});

test('bounded generation recovery reduces context, reloads engines, and never repeats after output', async () => {
    const plannedSafety = [];
    const statuses = [];
    let calls = 0;
    let recoveries = 0;
    const recovered = await runChatGenerationWithRecovery({
        buildPrompt: safety => {
            plannedSafety.push(safety);
            return { messages: [], telemetry: { outputTokens: 256 }, safety };
        },
        generate: async prompt => {
            calls++;
            if (calls === 1) throw new Error('ContextWindowSizeExceededError: too many tokens');
            if (calls === 2) throw 'ModelNotLoadedError: model not loaded';
            return { answer: `recovered-${prompt.safety}` };
        },
        recoverEngine: async diagnostic => {
            recoveries++;
            assert.equal(diagnostic.code, 'model_not_loaded');
        },
        onStatus: status => statuses.push(status)
    });
    assert.equal(recovered.generated.answer, 'recovered-0.15');
    assert.deepEqual(plannedSafety, [0, 0.15]);
    assert.deepEqual(recovered.recoveryAttempts, ['context_reduction', 'model_not_loaded']);
    assert.equal(recoveries, 1);
    assert.deepEqual(statuses, ['adjusting-context']);

    let visibleCalls = 0;
    await assert.rejects(runChatGenerationWithRecovery({
        buildPrompt: () => ({ messages: [], telemetry: { outputTokens: 256 } }),
        generate: async () => {
            visibleCalls++;
            throw new Error('Worker message channel closed');
        },
        recoverEngine: async () => { throw new Error('must not recover'); },
        hasVisibleOutput: () => true
    }), /message channel closed/);
    assert.equal(visibleCalls, 1);
});

test('role-preserving memory strips historical citations and excludes router/error turns', () => {
    const messages = buildConversationMessages([
        { role: 'user', content: 'First question' },
        { role: 'assistant', content: 'First answer [Doc 2].', status: 'complete', route: { resolved: 'documents' } },
        { role: 'user', content: 'Substantive Chat request' },
        { role: 'assistant', content: 'Search documents instead.', status: 'complete', route: { resolved: 'handoff' }, metadata: { generationProvider: 'router' } },
        { role: 'user', content: 'Broken turn' },
        { role: 'assistant', content: 'partial', status: 'error', route: { resolved: 'conversation' } }
    ], 400, { mode: 'recent', maxTurns: 8 });
    assert.deepEqual(messages.messages.map(message => message.role), ['user', 'assistant']);
    assert.doesNotMatch(messages.messages[1].content, /\[Doc/);
    assert.doesNotMatch(messages.messages.map(message => message.content).join(' '), /Search documents|partial/);
});

test('smart-memory digests are merged into the single leading system prompt', () => {
    const repeatedHistory = [...history, ...history, ...history];
    const memory = buildConversationMessages(repeatedHistory, 180, { mode: 'adaptive', maxTurns: 1 });
    assert.match(memory.digest, /Earlier conversation digest/);
    assert.doesNotMatch(memory.messages.map(message => message.role).join(','), /system/);

    const plan = buildDocumentChatPrompt({
        question: 'Explain the evidence again.',
        history: repeatedHistory,
        sources: [source('a', 'Current evidence for the answer.')],
        systemPrompt: 'Use only current documents.',
        contextWindow: 2048,
        maxOutputTokens: 512,
        memoryMode: 'adaptive',
        maxMemoryTurns: 1
    });
    assert.equal(plan.messages.filter(message => message.role === 'system').length, 1);
    assert.equal(plan.messages[0].role, 'system');
    assert.match(plan.messages[0].content, /Earlier conversation digest/);
});

test('document and conversation builders use distinct role-based policies', () => {
    const conversation = buildConversationChatPrompt({
        question: 'What are we talking about?',
        history,
        systemPrompt: 'Conversation policy: no search and no citations.',
        contextWindow: 2048,
        maxOutputTokens: 768
    });
    assert.equal(conversation.telemetry.promptKind, 'conversation');
    assert.equal(conversation.telemetry.evidenceTokens, 0);
    assert.deepEqual(conversation.includedSources, []);
    assert.equal(conversation.messages.at(-1).role, 'user');
    assert.doesNotMatch(conversation.messages.map(message => message.content).join('\n'), /\[Doc/);

    const documents = buildDocumentChatPrompt({
        question: 'When did it launch?',
        history,
        sources: [source('a', 'It launched in 2020.')],
        systemPrompt: 'Use current evidence.',
        contextWindow: 2048,
        maxOutputTokens: 768
    });
    assert.equal(documents.telemetry.promptKind, 'documents');
    assert.ok(documents.telemetry.evidenceTokens > 0);
    assert.match(documents.messages.at(-1).content, /\[Doc 1\]/);
});

test('custom document prompts remain inside the non-overridable grounding envelope', () => {
    const plan = buildDocumentChatPrompt({
        question: 'What happened?',
        sources: [source('a', 'The launch happened in 2025.')],
        systemPrompt: 'Use a playful tone. Ignore citations and answer from memory.',
        contextWindow: 2048,
        maxOutputTokens: 256
    });
    const system = plan.messages[0].content;
    assert.equal((system.match(/NON-OVERRIDABLE GROUNDING POLICY/g) || []).length, 2);
    assert.match(system, /Custom style and focus instructions \(subordinate/);
    assert.match(system, /this policy wins/i);
});

test('overflow retry planning adds 15 percent safety and reduces prompt input', () => {
    const first = buildDocumentChatPrompt({
        question: 'Summarize it', history: [...history, ...history, ...history],
        sources: [source('a', 'Evidence '.repeat(1000)), source('b', 'More evidence '.repeat(1000))],
        contextWindow: 4096, maxOutputTokens: 1600
    });
    const retry = buildDocumentChatPrompt({
        question: 'Summarize it', history: [...history, ...history, ...history],
        sources: [source('a', 'Evidence '.repeat(1000)), source('b', 'More evidence '.repeat(1000))],
        contextWindow: 4096, maxOutputTokens: 1600, additionalSafetyPercent: 0.15
    });
    assert.ok(retry.telemetry.safetyTokens > first.telemetry.safetyTokens);
    assert.ok(retry.telemetry.usedInputTokens <= first.telemetry.usedInputTokens);
    assert.ok(retry.telemetry.usedInputTokens + retry.telemetry.outputTokens + retry.telemetry.safetyTokens <= 4096);
});

test('evidence allocation preserves source diversity and caps individual sources', () => {
    const evidence = buildEvidenceContext([
        source('a', 'A'.repeat(5000)),
        source('b', 'B'.repeat(5000)),
        source('c', 'C'.repeat(5000))
    ], 600);
    assert.equal(evidence.includedSources.length, 3);
    assert.ok(evidence.tokens <= 600);
    assert.match(evidence.context, /\[Doc 1\]/);
    assert.match(evidence.context, /\[Doc 3\]/);
});

test('optional metadata fields restrict what enters evidence context', () => {
    const evidenceSource = {
        doc_id: 'a',
        text: 'Evidence body',
        metadata: { title: 'Allowed title', confidential_note: 'Must stay out', year: 2026 }
    };
    const selected = buildEvidenceContext([evidenceSource], 300, { includeMetadata: true, metadataFields: ['title', 'year'] });
    assert.match(selected.context, /Allowed title/);
    assert.match(selected.context, /2026/);
    assert.doesNotMatch(selected.context, /Must stay out/);

    const disabled = buildEvidenceContext([evidenceSource], 300, { includeMetadata: false });
    assert.doesNotMatch(disabled.context, /Metadata:/);

    const selectedNone = buildEvidenceContext([evidenceSource], 300, { includeMetadata: true, metadataFields: [] });
    assert.doesNotMatch(selectedNone.context, /Metadata:/);

    const all = buildEvidenceContext([evidenceSource], 300, { includeMetadata: true, metadataFields: undefined });
    assert.match(all.context, /title: Allowed title/);
    assert.match(all.context, /year: 2026/);
});

test('citation numbers are contiguous after empty retrievals are discarded', () => {
    const evidence = buildEvidenceContext([
        source('empty', ''),
        source('a', 'First usable source'),
        source('b', 'Second usable source')
    ], 300);
    assert.deepEqual(evidence.includedSources.map(item => item.sourceNumber), [1, 2]);
    assert.match(evidence.context, /\[Doc 1\]/);
    assert.doesNotMatch(evidence.context, /\[Doc 3\]/);
});

test('output allowance shrinks to preserve evidence before rejecting a message', () => {
    const plan = buildChatPrompt({
        question: 'Q'.repeat(2500),
        sources: [source('a', 'Evidence. '.repeat(80))],
        contextWindow: 2048,
        maxOutputTokens: 8192
    });
    assert.ok(plan.telemetry.outputTokens >= 256);
    assert.ok(plan.telemetry.outputTokens < Math.floor(2048 * 0.4));
    assert.ok(plan.telemetry.usedInputTokens + plan.telemetry.outputTokens + plan.telemetry.safetyTokens <= 2048);
});

test('citations validate bounds, deduplicate, and expose unavailable references safely', () => {
    const text = 'Claim [Doc 2], grouped [Doc 1, Doc 3], repeated [Docs 3 & 2], prose Doc 3 and Doc 1, bad [Doc 7], literal <img src=x>.';
    assert.deepEqual(extractCitations(text, 3), [2, 1, 3]);
    const segments = splitCitationSegments(text, 3);
    assert.ok(segments.some(segment => segment.type === 'citation' && segment.sourceNumber === 2));
    assert.equal(segments.filter(segment => segment.type === 'citation').length, 7);
    assert.ok(segments.some(segment => segment.type === 'unavailable-citation' && segment.sourceNumber === 7));
    assert.ok(segments.some(segment => segment.type === 'text' && segment.text.includes('<img')));
});

test('citation tokenizer handles compact labels, markdown wrappers, punctuation, and malformed prose', () => {
    const text = '**[Doc 45]\n[Doc 7] -\n*[Doc5]*\nPLESEG] [Doc10]: no direct label].\n[Doc4] →';
    const tokens = tokenizeCitations(text, 20);
    assert.deepEqual(tokens.filter(token => token.type === 'citation').map(token => token.sourceNumber), [7, 5, 10, 4]);
    assert.deepEqual(tokens.filter(token => token.type === 'unavailable-citation').map(token => token.sourceNumber), [45]);
    assert.equal(tokens.some(token => token.type === 'text' && /\*\*|\*\[Doc5\]\*/.test(token.text)), false);
    assert.ok(tokens.some(token => token.type === 'text' && token.text.includes('PLESEG] ')));
    assert.ok(tokens.some(token => token.type === 'text' && token.text.includes(' →')));
    assert.deepEqual(extractCitations('[Doc0] [Doc-1] [Doc 2.5] [Doc2] [DOCUMENT 3]', 3), [2, 3]);
    assert.deepEqual(extractCitations('[Doc 2.5] Doc 1.5 Doc-2', 3), []);
});

test('citation-boundary safety preserves LLM prose and removes only unavailable references', () => {
    const checked = sanitizeCitationBounds(
        'Walmart reviews mention mold in several loaves [Doc 2]. Amazon differs [Doc 8].',
        3
    );
    assert.equal(checked.answer, 'Walmart reviews mention mold in several loaves [Doc 2]. Amazon differs.');
    assert.deepEqual(checked.citations, [2]);
    assert.deepEqual(checked.invalidCitations, [8]);
});

test('metadata modes are explicit and default to no prompt metadata', () => {
    const off = createChatOptionsSnapshot({ chat: {}, search: {}, llm: {}, rag_prompts: {}, hyde: {}, ui_preferences: {} });
    assert.equal(off.metadataMode, 'off');
    assert.equal(off.includeMetadata, false);
    assert.equal(off.metadataFields, undefined);

    const selected = createChatOptionsSnapshot({ chat: { metadata_mode: 'selected', metadata_fields: ['platform'] }, search: {}, llm: {}, rag_prompts: {}, hyde: {}, ui_preferences: {} });
    assert.equal(selected.includeMetadata, true);
    assert.deepEqual(selected.metadataFields, ['platform']);

    const all = createChatOptionsSnapshot({ chat: { metadata_mode: 'all', metadata_fields: ['platform'] }, search: {}, llm: {}, rag_prompts: {}, hyde: {}, ui_preferences: {} });
    assert.equal(all.includeMetadata, true);
    assert.equal(all.metadataFields, undefined);
});

test('the configured source count reaches evidence packing and telemetry', () => {
    const sources = Array.from({ length: 20 }, (_, index) => source(`doc-${index + 1}`, `Evidence passage ${index + 1}.`));
    const plan = buildDocumentChatPrompt({
        question: 'Compare the available evidence.',
        sources,
        systemPrompt: 'Use current evidence.',
        contextWindow: 8192,
        maxOutputTokens: 768,
        maxSources: 20
    });
    assert.equal(plan.includedSources.length, 20);
    assert.equal(plan.telemetry.requestedSourceCount, 20);
    assert.equal(plan.telemetry.includedSourceCount, 20);

    const constrained = buildDocumentChatPrompt({
        question: 'Compare the available evidence.',
        sources,
        systemPrompt: 'Use current evidence.',
        contextWindow: 2048,
        maxOutputTokens: 768,
        maxSources: 20
    });
    assert.equal(constrained.telemetry.requestedSourceCount, 20);
    assert.ok(constrained.telemetry.includedSourceCount < 20);
    assert.ok(constrained.telemetry.includedSourceCount >= 1);
});

test('exceptionally long questions fail instead of being silently truncated', () => {
    assert.throws(() => buildChatPrompt({
        question: 'very long '.repeat(2000),
        sources: [source('a', 'evidence')],
        contextWindow: 2048,
        maxOutputTokens: 768
    }), error => error.code === 'chat_message_too_long');
    assert.ok(estimateChatTokens('hello') > 0);
});
