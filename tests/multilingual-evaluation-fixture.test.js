import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const fixture = JSON.parse(await readFile(
    new URL('./fixtures/multilingual-retrieval.json', import.meta.url),
    'utf8'
));

test('public retrieval evaluation fixture is synthetic, multilingual, and includes edge cases', () => {
    assert.equal(fixture.license, 'CC0-1.0');
    const documentIds = new Set(fixture.documents.map(document => document.id));
    const languages = new Set(fixture.queries.map(query => query.language));
    assert.deepEqual([...languages].sort(), ['da', 'de', 'en', 'es', 'fr', 'zh']);
    assert.ok(fixture.queries.some(query => query.relevant.length === 0), 'absent-answer query is required');
    assert.ok(fixture.queries.some(query => query.conflicting?.length), 'conflicting evidence is required');
    assert.ok(fixture.queries.some(query => query.relevant.length > 1), 'duplicate/redundant evidence is required');
    for (const query of fixture.queries) {
        for (const id of [...query.relevant, ...(query.conflicting || [])]) {
            assert.ok(documentIds.has(id), `${query.id} references known document ${id}`);
        }
    }
});
