import test from 'node:test';
import assert from 'node:assert/strict';

import {
    createMetadataFilterScope,
    matchesMetadataFilters,
    mergeMetadataFilters,
    normalizeMetadataFilters,
    serializeMetadataFilterScope
} from '../web_interface/static/js/browser-ml/metadata-filters.js';
import { BM25Search, BrowserVectorSearch } from '../web_interface/static/js/browser-ml/vector-search.js';

globalThis.window = {};
const { AnalysisService } = await import('../web_interface/static/js/browser-ml/analysis.js');

const documents = [
    { id: 'w1', text: 'delivery delivery refund', metadata: { platform: 'walmart', rating: 1, sentiment: 'Negative', active: true, created: '2026-01-10' } },
    { id: 'w2', text: 'delivery order', metadata: { platform: 'walmart', rating: 4, sentiment: 'Positive', active: false, created: '2026-02-10' } },
    { id: 'a1', text: 'delivery', metadata: { platform: 'amazon', rating: 1, sentiment: 'Negative', active: 'yes', created: '2026-03-10' } }
];

test('normalizes and matches scalar, array, and multi-field filters', () => {
    const scalar = normalizeMetadataFilters({ platform: 'AMAZON' }, documents);
    assert.equal(matchesMetadataFilters(documents[2], scalar), true);
    assert.equal(matchesMetadataFilters(documents[0], scalar), false);

    const array = normalizeMetadataFilters({ platform: ['amazon', 'target'] }, documents);
    assert.equal(matchesMetadataFilters(documents[2], array), true);

    const multi = normalizeMetadataFilters({ platform: 'amazon', rating: 1 }, documents);
    assert.equal(matchesMetadataFilters(documents[2], multi), true);
    assert.equal(matchesMetadataFilters(documents[0], multi), false);
});

test('supports numeric, date, boolean, and text filters', () => {
    const numeric = normalizeMetadataFilters({ rating: { min: 1, max: 1 } }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, numeric).indices, [0, 2]);

    const date = normalizeMetadataFilters({
        created: { type: 'date', value: { min: '2026-02-01', max: '2026-03-31' } }
    }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, date).indices, [1, 2]);

    const boolean = normalizeMetadataFilters({ active: { type: 'boolean', value: true } }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, boolean).indices, [0, 2]);

    const text = normalizeMetadataFilters({ sentiment: { type: 'text', value: 'neg' } }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, text).indices, [0, 2]);
});

test('supports legacy UI condition and range filter shapes', () => {
    const numeric = normalizeMetadataFilters({
        rating: { type: 'number', range: { min: 1, max: 1 }, conditions: [] }
    }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, numeric).indices, [0, 2]);

    const text = normalizeMetadataFilters({
        sentiment: { type: 'text', conditions: ['neg'] }
    }, documents);
    assert.deepEqual(createMetadataFilterScope(documents, text).indices, [0, 2]);
});

test('rejects unknown fields and malformed filters', () => {
    assert.throws(
        () => normalizeMetadataFilters({ missing: 'value' }, documents),
        /Unknown metadata filter field/
    );
    assert.throws(
        () => normalizeMetadataFilters({ platform: {} }, documents),
        /unsupported object shape/
    );
    assert.throws(
        () => normalizeMetadataFilters({ platform: [] }, documents),
        /empty array/
    );
});

test('inline filters override matching persistent fields and preserve other fields', () => {
    const persistent = normalizeMetadataFilters({ platform: 'walmart', rating: 1 }, documents);
    const inline = normalizeMetadataFilters({ platform: 'amazon' }, documents);
    const merged = mergeMetadataFilters(persistent, inline);
    const scope = createMetadataFilterScope(documents, merged);
    assert.deepEqual(scope.indices, [2]);
    assert.deepEqual(serializeMetadataFilterScope(scope), {
        applied: true,
        active_filters: ['platform', 'rating'],
        matched_documents: 1,
        total_documents: 3,
        metadata_filters: merged
    });
});

test('zero-match filters return an empty scope', () => {
    const filters = normalizeMetadataFilters({ platform: 'target' }, documents);
    const scope = createMetadataFilterScope(documents, filters);
    assert.equal(scope.applied, true);
    assert.equal(scope.matchedDocuments, 0);
    assert.deepEqual(scope.documents, []);
    assert.deepEqual(scope.indices, []);
});

test('BM25 applies filters before top-k selection', () => {
    const index = new BM25Search();
    index.buildIndex(documents, documents.map(document => document.id));
    const filters = normalizeMetadataFilters({ platform: 'amazon' }, documents);
    const results = index.search('delivery', 1, {
        filter: document => matchesMetadataFilters(document, filters)
    });
    assert.equal(results.length, 1);
    assert.equal(results[0].doc_id, 'a1');
});

test('semantic vector search applies filters before top-k selection', async () => {
    const index = new BrowserVectorSearch(2);
    await index.buildIndex(
        [[1, 0], [0.9, 0.1], [0.5, 0.5]],
        documents.map(document => document.id),
        documents
    );
    const filters = normalizeMetadataFilters({ platform: 'amazon' }, documents);
    const results = index.search([1, 0], 1, {
        filter: metadata => matchesMetadataFilters(metadata, filters)
    });
    assert.equal(results.length, 1);
    assert.equal(results[0].doc_id, 'a1');
});

test('analysis aggregate and subset use persistent filters with inline override', () => {
    const pipeline = createAnalysisPipeline(documents, { platform: 'amazon' });
    const analysis = new AnalysisService(pipeline);

    const aggregate = analysis.aggregate({
        group_by: 'platform',
        use_persistent_filters: true
    });
    assert.deepEqual(aggregate.groups, [{ key: 'amazon', value: 1, n: 1 }]);
    assert.equal(aggregate.filter_metadata.matched_documents, 1);

    const subset = analysis.filterToSubset({
        filters: { platform: 'walmart', rating: 1 },
        use_persistent_filters: true
    });
    assert.deepEqual(subset.doc_indices, [0]);
    assert.equal(subset.count, 1);
});

test('multi-vector search passes the effective predicate into every vector ranking', async () => {
    const pipeline = createAnalysisPipeline(documents, { platform: 'walmart' });
    const seen = [];
    pipeline.embeddings = {
        embedSingle: async () => [1, 0]
    };
    pipeline.vectorSearch = {
        isBuilt: true,
        multiVectorSearch(vectors, options) {
            seen.push(options.filter({ platform: 'amazon' }));
            seen.push(options.filter({ platform: 'walmart', rating: 1 }));
            return { results: [{ doc_id: 'w1', score: 1 }] };
        }
    };
    const analysis = new AnalysisService(pipeline);
    const result = await analysis.multiVectorSearch({
        queries: ['delivery'],
        metadata_filters: { rating: 1 },
        use_persistent_filters: true
    });

    assert.deepEqual(seen, [false, true]);
    assert.equal(result.filter_metadata.matched_documents, 1);
    assert.equal(result.results[0].doc_id, 'w1');
});

test('Amazon reproducer returns 87 of 1511 documents instead of the full dataset', () => {
    const reviewDocuments = [
        ...Array.from({ length: 1424 }, (_, index) => ({
            id: `w${index}`,
            text: 'walmart review',
            metadata: {
                platform: 'walmart',
                rating: index % 5 + 1,
                sentiment: index % 5 === 0 ? 'Negative' : 'Positive'
            }
        })),
        ...Array.from({ length: 87 }, (_, index) => ({
            id: `a${index}`,
            text: 'amazon review',
            metadata: {
                platform: 'amazon',
                rating: index % 5 + 1,
                sentiment: index % 5 === 0 ? 'Negative' : 'Positive'
            }
        }))
    ];
    const pipeline = createAnalysisPipeline(reviewDocuments, {});
    const analysis = new AnalysisService(pipeline);

    const subset = analysis.filterToSubset({ filters: { platform: 'amazon' } });
    assert.equal(subset.count, 87);
    assert.equal(subset.doc_indices[0], 1424);
    assert.equal(subset.doc_indices.at(-1), 1510);

    const aggregate = analysis.aggregate({
        group_by: 'platform',
        filter: { platform: 'amazon' }
    });
    assert.deepEqual(aggregate.groups, [{ key: 'amazon', value: 87, n: 87 }]);

    const lowRatingSentiment = analysis.aggregate({
        group_by: 'sentiment',
        filter: { rating: 1 }
    });
    assert.equal(
        lowRatingSentiment.groups.reduce((sum, group) => sum + group.n, 0),
        reviewDocuments.filter(document => document.metadata.rating === 1).length
    );
    assert.deepEqual(lowRatingSentiment.groups.map(group => group.key), ['Negative']);
});

function createAnalysisPipeline(docs, persistentFilters) {
    const pipeline = {
        currentDataset: {
            documents: docs,
            clusters: docs.map(() => 0)
        },
        mcpMetadataFilters: normalizeMetadataFilters(persistentFilters, docs),
        subsets: new Map(),
        metrics: {
            has: () => false
        },
        createMetadataFilterScope(inline = {}, { includePersistent = false } = {}) {
            const normalizedInline = normalizeMetadataFilters(inline || {}, docs);
            const effective = includePersistent
                ? mergeMetadataFilters(this.mcpMetadataFilters, normalizedInline)
                : normalizedInline;
            return createMetadataFilterScope(docs, effective);
        },
        serializeMetadataFilterScope
    };
    return pipeline;
}
