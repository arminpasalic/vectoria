import test from 'node:test';
import assert from 'node:assert/strict';
import { rerankAndDiversify } from '../web_interface/static/js/browser-ml/retrieval-ranking.js';
import { buildChatRetrievalQueries, lastGroundedTopic } from '../web_interface/static/js/browser-ml/chat-context.js';

// Synthetic and dataset-agnostic: policies, reviews, research, notes, support,
// multilingual text, duplicates, conflicts, and absent answers.
const corpus = [
    { doc_id: 'policy-a', score: 1, text: 'The retention policy keeps records for 30 days and then deletes them.' },
    { doc_id: 'policy-duplicate', score: 0.98, text: 'The retention policy keeps records for 30 days and then deletes them.' },
    { doc_id: 'research', score: 0.76, text: 'Forsøget omfattede 42 deltagere i København og sluttede i 2025.' },
    { doc_id: 'support', score: 0.7, text: 'Support ticket ACME-2048 was resolved by replacing the adapter.' },
    { doc_id: 'review', score: 0.66, text: 'The review says battery life lasted eleven hours in ordinary use.' },
    { doc_id: 'notes', score: 0.62, text: 'Meeting notes list launch preparation but contain no product price.' }
];

test('synthetic retrieval benchmark retains recall while reducing duplicate rate and padding', () => {
    const selected = rerankAndDiversify(corpus, 'retention policy 30 days', { maxResults: 5 });
    const ids = selected.map(item => item.doc_id);
    const recallAt5 = ids.includes('policy-a') || ids.includes('policy-duplicate') ? 1 : 0;
    const reciprocalRank = 1 / (ids.findIndex(id => id.startsWith('policy-')) + 1);
    const duplicateRate = ids.includes('policy-a') && ids.includes('policy-duplicate') ? 1 / ids.length : 0;
    const sourceDiversity = new Set(ids).size / ids.length;
    assert.equal(recallAt5, 1);
    assert.equal(reciprocalRank, 1);
    assert.equal(duplicateRate, 0);
    assert.equal(sourceDiversity, 1);
    assert.ok(selected.length < 5, 'weak tail candidates should not pad the configured maximum');
});

test('topic benchmark preserves supported anchors across meta, correction, and failed turns', () => {
    const history = [
        { role: 'user', content: 'What is the retention period?', turnId: '1' },
        { role: 'assistant', content: '30 days [Doc 1]', turnId: '1', status: 'complete', route: { resolved: 'documents' }, metadata: { groundingState: 'supported', topicEligible: true, resolvedQuery: 'retention period' } },
        { role: 'user', content: 'Thanks', turnId: '2' },
        { role: 'assistant', content: 'You are welcome.', turnId: '2', route: { resolved: 'helper' } },
        { role: 'user', content: 'What is the price?', turnId: '3' },
        { role: 'assistant', content: 'Insufficient evidence.', turnId: '3', route: { resolved: 'documents' }, metadata: { groundingState: 'insufficient', topicEligible: false } },
        { role: 'user', content: 'No, shorter', turnId: '4' },
        { role: 'assistant', content: 'Ask again with the format.', turnId: '4', route: { resolved: 'helper' } }
    ];
    assert.equal(lastGroundedTopic(history).resolvedQuery, 'retention period');
    const resolved = buildChatRetrievalQueries('How is that enforced?', history);
    assert.equal(resolved.anchorQuestion, 'retention period');
    assert.match(resolved.resolvedQuery, /retention period/);
});
