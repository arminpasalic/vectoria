import test from 'node:test';
import assert from 'node:assert/strict';

import { buildChatCsvRows, buildChatMarkdown } from '../web_interface/static/js/browser-ml/chat-export.js';

const conversation = [{
    timestamp: '2026-08-08T10:00:00.000Z',
    query: 'What happened?',
    answer: 'The launch succeeded. [Doc 1]',
    sources: [{ sourceNumber: 1, docId: 'launch-1', metadata: { title: 'Launch\nreport' } }],
    route: { requested: 'auto', resolved: 'documents' },
    metadata: {
        model: 'local-model', temperature: 0, retrievalPerformed: true,
        hydeUsed: true, hydeEdited: true, actualUsage: { promptTokens: 420, completionTokens: 80 }, finishReason: 'stop',
        resolvedQuery: 'What happened during the launch?', topicAnchorQuestion: 'Launch outcome',
        groundingState: 'supported', removedClaimCount: 0, repairCount: 0,
        candidate_parent_count: 7, selected_count: 1, dropped_count: 6, post_fusion: 'neural_rrf_coverage_mmr_v1',
        reranker_applied: true,
        reranker_model: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        reranker_candidates: 7,
        reranker_latency_ms: 84,
        filter: { applied: true, filters: { region: ['EU'] } }
    }
}];

test('Markdown export keeps turns, citations, and source identifiers readable', () => {
    const markdown = buildChatMarkdown(conversation, { exportedAt: '2026-08-08T11:00:00.000Z' });
    assert.match(markdown, /^# Vectoria Ask history/);
    assert.match(markdown, /### Question\n\nWhat happened\?/);
    assert.match(markdown, /### Vectoria\n\nThe launch succeeded\./);
    assert.match(markdown, /Mode: Documents · HyDE \(edited\)/);
    assert.match(markdown, /The launch succeeded\. \[Doc 1\]/);
    assert.match(markdown, /- \[Doc 1\] Launch report — ID: launch-1/);
    assert.match(markdown, /Export format: 3/);
    assert.match(markdown, /Resolved query: What happened during the launch\?/);
    assert.match(markdown, /Grounding: supported/);
    assert.match(markdown, /Reranker: cross-encoder\/mmarco-mMiniLMv2-L12-H384-v1 · 7 candidates · 84 ms/);
    assert.ok(markdown.endsWith('\n'));
});

test('CSV export rows preserve zero-valued generation settings and source ids', () => {
    assert.deepEqual(buildChatCsvRows(conversation), [{
        timestamp: '2026-08-08T10:00:00.000Z',
        query: 'What happened?',
        answer: 'The launch succeeded. [Doc 1]',
        num_sources: 1,
        source_ids: 'launch-1',
        model: 'local-model',
        temperature: 0,
        requested_mode: 'auto',
        resolved_route: 'documents',
        retrieval_performed: true,
        hyde_used: true,
        hyde_edited: true,
        prompt_tokens: 420,
        completion_tokens: 80,
        resolved_query: 'What happened during the launch?',
        topic_anchor: 'Launch outcome',
        explicit_filters: '{"region":["EU"]}',
        retrieval_diagnostics: '{"candidate_parent_count":7,"selected_count":1,"dropped_count":6,"post_fusion":"neural_rrf_coverage_mmr_v1","reranker_applied":true,"reranker_model":"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1","reranker_candidates":7,"reranker_latency_ms":84,"reranker_fallback_reason":null}',
        reranker_applied: true,
        reranker_model: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        reranker_candidates: 7,
        reranker_latency_ms: 84,
        reranker_fallback_reason: '',
        grounding_state: 'supported',
        rejected_claim_count: 0,
        repair_count: 0,
        finish_reason: 'stop'
    }]);
});

test('conversation Markdown has no empty source section and JSON provenance remains untouched', () => {
    const entry = {
        timestamp: '2026-08-08T12:00:00.000Z', query: 'Who are you?', answer: 'I am Vectoria.', sources: [],
        route: { requested: 'auto', resolved: 'conversation', reason: 'identity' },
        metadata: { retrievalPerformed: false, finishReason: 'stop' }
    };
    const markdown = buildChatMarkdown([entry]);
    assert.match(markdown, /Mode: Conversation/);
    assert.doesNotMatch(markdown, /#### Sources/);
    assert.equal(JSON.parse(JSON.stringify(entry)).route.reason, 'identity');
});
