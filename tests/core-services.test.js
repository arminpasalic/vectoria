import test from 'node:test';
import assert from 'node:assert/strict';

import { AnnotationsStore } from '../web_interface/static/js/browser-ml/annotations-store.js';
import { MetricsRegistry } from '../web_interface/static/js/browser-ml/metrics-registry.js';
import { SessionsStore, hashCanonical } from '../web_interface/static/js/browser-ml/sessions-store.js';
import {
    buildContingency,
    chiSquare,
    cramersV,
    jsDivergence,
    quantile,
    welchT
} from '../web_interface/static/js/browser-ml/statistics.js';

function createPipeline() {
    const pipeline = {
        currentDataset: {
            id: 'dataset-a',
            documents: [
                { id: 'a', text: 'alpha', metadata: { likes: 4, shares: 3 } },
                { id: 'b', text: 'beta', metadata: { likes: 2, shares: 1 } }
            ]
        },
        annotations: new Map(),
        registeredMetrics: new Map(),
        analysisSessions: [],
        customClusterLabels: new Map(),
        subsets: new Map(),
        mcpMetadataFilters: {},
        setCustomClusterLabel(clusterId, label, source) {
            this.customClusterLabels.set(Number(clusterId), { label, source });
        },
        setMcpMetadataFilters(filters) {
            this.mcpMetadataFilters = { ...filters };
        },
        clearMcpMetadataFilters() {
            this.mcpMetadataFilters = {};
        }
    };
    pipeline.annotationsApi = new AnnotationsStore(pipeline);
    pipeline.metrics = new MetricsRegistry(pipeline);
    pipeline.sessions = new SessionsStore(pipeline);
    return pipeline;
}

test('statistical helpers cover empty, interpolated, independent, and associated data', () => {
    assert.equal(quantile([], 0.5), null);
    assert.equal(quantile([1, 3, 9], 0.25), 2);

    const independent = chiSquare([[10, 10], [20, 20]]);
    assert.equal(independent.chi2, 0);
    assert.equal(independent.p_value, 1);

    const associated = chiSquare([[30, 0], [0, 30]]);
    assert.ok(associated.chi2 > 50);
    assert.ok(associated.p_value < 0.001);
    assert.ok(cramersV(associated.chi2, associated.n, 2, 2) > 0.9);

    assert.equal(jsDivergence({ a: 1 }, { a: 1 }), 0);
    assert.equal(jsDivergence({ a: 1 }, { b: 1 }), 1);

    const welch = welchT([10, 11, 12, 13], [1, 2, 3, 4]);
    assert.ok(welch.t > 9);
    assert.ok(welch.p_value < 0.001);
});

test('contingency construction remains deterministic and ignores unmatched tails', () => {
    assert.deepEqual(buildContingency(['a', 'a', 'b'], ['x', 'y', 'x']), {
        matrix: [[1, 1], [1, 0]],
        rowKeys: ['a', 'b'],
        colKeys: ['x', 'y']
    });
    assert.deepEqual(buildContingency(['a', 'b'], ['x']), {
        matrix: [[1], [0]],
        rowKeys: ['a', 'b'],
        colKeys: ['x']
    });
});

test('metric formulas honor precedence, unary values, missing fields, and division safety', () => {
    const pipeline = createPipeline();
    pipeline.metrics.register('engagement', 'likes + 2 * shares');
    pipeline.metrics.register('adjusted', '-(likes - shares) / 0');

    assert.equal(pipeline.metrics.evaluate('engagement', pipeline.currentDataset.documents[0]), 10);
    assert.equal(pipeline.metrics.evaluate('engagement', { metadata: {} }), 0);
    assert.equal(pipeline.metrics.evaluate('adjusted', pipeline.currentDataset.documents[0]), 0);
    assert.throws(() => pipeline.metrics.evaluate('missing', {}), /Unknown metric/);
});

test('metric parser rejects incomplete numbers and expressions instead of silently evaluating them', () => {
    const registry = new MetricsRegistry({});
    for (const formula of ['1 +', '1e', '1e+', '.', '2 ** 3', '1 2']) {
        assert.throws(() => registry.register(`bad_${formula}`, formula), /Invalid|Unexpected|Expected/);
    }
});

test('annotations validate indices, deduplicate requests, and maintain the reverse index', () => {
    const pipeline = createPipeline();
    const added = pipeline.annotationsApi.add({ doc_indices: [0, 0, 1], tag: ' review ', note: 'check' });
    assert.equal(added.added, 2);
    assert.equal(pipeline.annotationsApi.list({ tag: 'review' }).count, 2);
    assert.equal(pipeline.annotationsApi.forDoc(0).length, 1);

    const [firstId] = added.annotation_ids;
    assert.equal(pipeline.annotationsApi.remove(firstId), true);
    assert.deepEqual(pipeline.annotationsApi.forDoc(0), []);
    assert.equal(pipeline.annotationsApi.remove(firstId), false);

    assert.throws(() => pipeline.annotationsApi.add({ doc_indices: [], tag: 'x' }), /non-empty/);
    assert.throws(() => pipeline.annotationsApi.add({ doc_indices: [-1], tag: 'x' }), /non-negative integer/);
    assert.throws(() => pipeline.annotationsApi.add({ doc_indices: [2], tag: 'x' }), /outside the active dataset/);
    assert.throws(() => pipeline.annotationsApi.add({ doc_indices: [0], tag: '   ' }), /non-empty tag/);
});

test('analysis sessions round-trip state and reject tampered payloads', async () => {
    const pipeline = createPipeline();
    pipeline.metrics.register('engagement', 'likes + shares');
    pipeline.annotationsApi.add({ doc_indices: [0], tag: 'review' });
    pipeline.customClusterLabels.set(1, { label: 'Topic', source: 'test' });
    pipeline.subsets.set('subset-1', { name: 'First', doc_indices: [0] });
    pipeline.mcpMetadataFilters = { likes: { type: 'number', min: 2 } };

    const saved = await pipeline.sessions.save('Audit', 'Findings', false);
    assert.equal(pipeline.sessions.list().length, 1);

    pipeline.metrics.hydrate([]);
    pipeline.annotationsApi.hydrate([]);
    pipeline.customClusterLabels.clear();
    pipeline.subsets.clear();
    pipeline.mcpMetadataFilters = {};

    const restored = await pipeline.sessions.load(saved.session_id);
    assert.equal(restored.ok, true);
    assert.equal(pipeline.metrics.has('engagement'), true);
    assert.equal(pipeline.annotationsApi.list().count, 1);
    assert.equal(pipeline.customClusterLabels.get(1).label, 'Topic');
    assert.equal(pipeline.subsets.get('subset-1').name, 'First');
    assert.deepEqual(pipeline.mcpMetadataFilters, { likes: { type: 'number', min: 2 } });

    const record = pipeline.analysisSessions[0];
    await assert.rejects(
        pipeline.sessions.load({ payload: { ...record.payload, findings: 'changed' }, sha256: record.sha256 }),
        /signature mismatch/
    );
});

test('analysis sessions fail clearly when only an exported list stub remains', async () => {
    const pipeline = createPipeline();
    pipeline.sessions.hydrate([{ id: 'stub', name: 'Old session', dataset_id: 'dataset-a' }]);
    await assert.rejects(pipeline.sessions.load('stub'), /does not contain a restorable payload/);
});

test('analysis sessions cannot restore dataset-scoped state onto another dataset', async () => {
    const source = createPipeline();
    const saved = await source.sessions.save('Scoped', '', false);
    const record = source.analysisSessions.find(session => session.id === saved.session_id);

    const target = createPipeline();
    target.currentDataset = {
        id: 'dataset-b',
        documents: [{ id: 'z', text: 'different', metadata: {} }]
    };
    await assert.rejects(target.sessions.load({ payload: record.payload, sha256: record.sha256 }), /different dataset/);
});

test('canonical hashes ignore object key order and retain array order', async () => {
    assert.equal(await hashCanonical({ b: 2, a: 1 }), await hashCanonical({ a: 1, b: 2 }));
    assert.notEqual(await hashCanonical([1, 2]), await hashCanonical([2, 1]));
});
