import test from 'node:test';
import assert from 'node:assert/strict';
import {
    lexicalSimilarity,
    queryCoverageScore,
    rerankAndDiversify
} from '../web_interface/static/js/browser-ml/retrieval-ranking.js';

test('coverage rewards exact terms, numbers, dates, and identifiers', () => {
    const query = 'What happened to ticket ACME-2048 on 2026-08-24?';
    const exact = queryCoverageScore(query, { text: 'Ticket ACME-2048 was resolved on 2026-08-24.' });
    const generic = queryCoverageScore(query, { text: 'A support request was resolved recently.' });
    assert.ok(exact > generic);
    assert.ok(exact > 0.6);
});

test('lexical similarity is Unicode-aware and detects near duplicates', () => {
    const duplicate = lexicalSimilarity(
        { text: 'Forsøget i København omfattede 42 deltagere.' },
        { text: 'Forsøget i København omfattede præcis 42 deltagere.' }
    );
    const unrelated = lexicalSimilarity(
        { text: 'Forsøget i København omfattede 42 deltagere.' },
        { text: 'The policy expires after thirty days.' }
    );
    assert.ok(duplicate > unrelated);
});

test('post-fusion selection keeps relevant evidence and removes duplicate padding', () => {
    const results = [
        { id: 'a', score: 1, text: 'The retention period is 30 days and records are deleted.' },
        { id: 'a-copy', score: 0.99, text: 'The retention period is 30 days and records are deleted.' },
        { id: 'b', score: 0.8, text: 'A separate policy says archived logs remain for 60 days.' },
        { id: 'noise', score: 0.01, text: 'The office cafeteria opens at noon.' }
    ];
    const selected = rerankAndDiversify(results, 'How long are records retained?', { maxResults: 5 });
    assert.equal(selected[0].id, 'a');
    assert.ok(selected.some(result => result.id === 'b'));
    assert.equal(selected.filter(result => result.id === 'a-copy').length, 0);
    assert.equal(selected.filter(result => result.id === 'noise').length, 0);
    assert.ok(selected[0].retrieval_quality);
});

test('source count is a maximum rather than a padding requirement', () => {
    const selected = rerankAndDiversify([
        { id: 'relevant', score: 0.9, text: 'Mars is known as the red planet.' },
        { id: 'weak', score: 0.01, text: 'Bread should be stored in a cool place.' }
    ], 'Which planet is known as the red planet?', { maxResults: 8 });
    assert.deepEqual(selected.map(result => result.id), ['relevant']);
});
