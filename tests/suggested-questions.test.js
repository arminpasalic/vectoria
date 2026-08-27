import test from 'node:test';
import assert from 'node:assert/strict';

import {
    FALLBACK_SUGGESTIONS,
    MAX_QUESTION_CHARS,
    SUGGESTION_EXCERPT_CHARS,
    SUGGESTION_SAMPLE_SIZE,
    buildSuggestionPrompt,
    parseSuggestionResponse,
    sampleDocumentsForSuggestions
} from '../web_interface/static/js/browser-ml/suggested-questions.js';

const corpus = Array.from({ length: 40 }, (_, index) => ({
    text: `Document number ${index} with enough substantive body text to clear the length filter.`
}));

test('sampling is random, deduplicated, and skips documents too short to be useful', () => {
    const withNoise = [...corpus, { text: 'too short' }, { text: '' }, {}, null];

    const sample = sampleDocumentsForSuggestions(withNoise);
    assert.equal(sample.length, SUGGESTION_SAMPLE_SIZE);
    assert.equal(new Set(sample).size, sample.length, 'a document must not be sampled twice');
    assert.ok(!sample.some(document => String(document?.text || '').length <= 40));

    // Re-sampling the same dataset should vary, which is what makes the
    // suggestions differ between runs.
    const signatures = new Set();
    for (let attempt = 0; attempt < 20; attempt += 1) {
        signatures.add(sampleDocumentsForSuggestions(withNoise).map(d => d.text).join('|'));
    }
    assert.ok(signatures.size > 1, 'sampling must not be deterministic');
});

test('sampling degrades safely on tiny, empty, and malformed corpora', () => {
    assert.deepEqual(sampleDocumentsForSuggestions([]), []);
    assert.deepEqual(sampleDocumentsForSuggestions(null), []);
    assert.deepEqual(sampleDocumentsForSuggestions([{ text: 'short' }]), []);
    assert.equal(sampleDocumentsForSuggestions([corpus[0]]).length, 1);
    assert.equal(sampleDocumentsForSuggestions(corpus, 3).length, 3);
});

test('the prompt carries numbered excerpts and caps how much of each document it sends', () => {
    const long = { text: `${'lorem ipsum '.repeat(200)}` };
    const prompt = buildSuggestionPrompt([long, corpus[0]]);
    assert.match(prompt, /\[1\] lorem ipsum/);
    assert.match(prompt, /\[2\] Document number 0/);
    assert.match(prompt, /exactly 2 short questions/i);
    assert.match(prompt, /Maximum 8 words each/);

    const firstExcerpt = prompt.split('\n').find(line => line.startsWith('[1] '));
    assert.ok(firstExcerpt.length <= SUGGESTION_EXCERPT_CHARS + 8);
    // Newlines inside a document must not fake extra excerpts.
    assert.doesNotMatch(buildSuggestionPrompt([{ text: 'a\nb\nc '.repeat(20) }]), /\n\[2\]/);
});

test('parsing recovers two questions from the shapes small models actually emit', () => {
    const expected = ['What drove adoption?', 'Which regions lagged?'];
    for (const [label, raw] of [
        ['plain', 'What drove adoption?\nWhich regions lagged?'],
        ['numbered', '1. What drove adoption?\n2. Which regions lagged?'],
        ['bulleted', '- What drove adoption?\n- Which regions lagged?'],
        ['quoted', '"What drove adoption?"\n"Which regions lagged?"'],
        ['missing question marks', 'What drove adoption\nWhich regions lagged'],
        ['preamble', 'Here are two questions:\nWhat drove adoption?\nWhich regions lagged?'],
        ['chatty preamble', 'Sure! Below you go:\nWhat drove adoption?\nWhich regions lagged?'],
        ['trailing junk', 'What drove adoption?\nWhich regions lagged?\nok\n!!!']
    ]) {
        assert.deepEqual(parseSuggestionResponse(raw), expected, label);
    }
});

test('parsing rejects unusable output rather than surfacing noise', () => {
    assert.deepEqual(parseSuggestionResponse(''), []);
    assert.deepEqual(parseSuggestionResponse(null), []);
    assert.deepEqual(parseSuggestionResponse('!!!\n???\n...'), []);
    assert.deepEqual(parseSuggestionResponse('ok\nyes'), []);
    // Over-long lines are dropped instead of overflowing the button.
    assert.deepEqual(parseSuggestionResponse(`${'x'.repeat(MAX_QUESTION_CHARS + 10)}?`), []);
    // Duplicates never fill both slots.
    assert.deepEqual(parseSuggestionResponse('What drove adoption?\nWhat drove adoption?'), ['What drove adoption?']);
});

test('a usable pair always has two entries so the empty state stays populated', () => {
    assert.equal(FALLBACK_SUGGESTIONS.length, 2);
    for (const item of FALLBACK_SUGGESTIONS) {
        assert.ok(item.label.length > 0 && item.label.length <= MAX_QUESTION_CHARS);
        assert.ok(item.prompt.length > item.label.length - 1);
    }
});
