import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    buildStructuralRanges,
    detectCodeLanguage,
    normalizeCodeLanguage
} from '../web_interface/static/js/browser-ml/chunking/browserCodeBackend.js';

const index = await readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8');
const chunker = await readFile(new URL('../web_interface/static/js/browser-ml/chunking/chonkieChunker.js', import.meta.url), 'utf8');
const pipeline = await readFile(new URL('../web_interface/static/js/browser-ml/index.js', import.meta.url), 'utf8');
const vectoria = await readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8');

test('all seven ChonkieJS chunkers are imported and exposed in settings', () => {
    for (const className of [
        'TokenChunker', 'RecursiveChunker', 'SentenceChunker', 'SemanticChunker',
        'CodeChunker', 'TableChunker', 'FastChunker'
    ]) assert.match(chunker, new RegExp(`\\b${className}\\b`));

    for (const strategy of ['token', 'recursive', 'sentence', 'semantic', 'code', 'table', 'fast']) {
        assert.match(index, new RegExp(`<option value="${strategy}"`));
    }
});

test('semantic, code, and table settings reach the processing pipeline', () => {
    for (const field of [
        'semantic_threshold', 'semantic_similarity_window', 'semantic_filter_window',
        'semantic_filter_polyorder', 'semantic_filter_tolerance', 'semantic_skip_window',
        'code_language', 'table_mode', 'table_rows_per_chunk'
    ]) {
        assert.match(vectoria, new RegExp(`\\b${field}\\b`), `${field} is not persisted`);
        assert.match(pipeline, new RegExp(`chunkConfig\\.${field}\\b`), `${field} does not reach chunking`);
    }
    assert.match(pipeline, /semanticEmbeddings:\s*textsToEmbed\s*=>\s*this\.embeddings\.embed/);
});

test('code-language resolution honors explicit values, metadata, fences, and content', () => {
    assert.equal(normalizeCodeLanguage('C#'), 'c_sharp');
    assert.equal(normalizeCodeLanguage('.tsx'), 'tsx');
    assert.equal(normalizeCodeLanguage('not-a-language'), null);

    assert.equal(detectCodeLanguage('console.log(1)', {}, 'python'), 'python');
    assert.equal(detectCodeLanguage('ambiguous()', { file_name: 'worker.rs' }), 'rust');
    assert.equal(detectCodeLanguage('```ts\nconst count: number = 1;\n```'), 'typescript');
    assert.equal(detectCodeLanguage('def greet(name):\n    return name'), 'python');
    assert.equal(detectCodeLanguage('{"valid": true}'), 'json');
});

test('AST-derived ranges stay structural, contiguous, and cover every byte', () => {
    const leaf = (startIndex, endIndex) => ({
        startIndex,
        endIndex,
        namedChildCount: 0,
        namedChildren: []
    });
    const oversizedFunction = {
        startIndex: 0,
        endIndex: 120,
        namedChildCount: 3,
        namedChildren: [leaf(0, 40), leaf(40, 80), leaf(80, 120)]
    };
    const root = {
        startIndex: 0,
        endIndex: 120,
        namedChildCount: 1,
        namedChildren: [oversizedFunction]
    };

    const ranges = buildStructuralRanges(root, 120, 50);
    assert.deepEqual(ranges, [[0, 40], [40, 80], [80, 120]]);
    assert.equal(ranges[0][0], 0);
    assert.equal(ranges.at(-1)[1], 120);
    for (let index = 1; index < ranges.length; index++) {
        assert.equal(ranges[index - 1][1], ranges[index][0]);
    }
});

