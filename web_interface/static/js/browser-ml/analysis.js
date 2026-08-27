/**
 * AnalysisService — advanced analytical operations for MCP tools and the UI.
 * Glue between the existing pipeline (clustering, vector search, RAG) and
 * higher-level analytical primitives (summarize, outliers, cross-tab, etc.).
 */

import { selectQuantileExemplars, selectAdaptiveExemplarCount } from './clustering.js';
import { buildContingency, chiSquare, cramersV, jsDivergence, welchT } from './statistics.js';

export class AnalysisService {
    constructor(pipeline) {
        this.pipeline = pipeline;
    }

    // ---- helpers ----------------------------------------------------------

    _docs() {
        const ds = this.pipeline.currentDataset;
        if (!ds?.documents?.length) throw new Error('No dataset loaded');
        return ds;
    }

    _readField(doc, field) {
        if (field === '__cluster__') {
            const cid = doc?.metadata?.cluster;
            const lbl = this.pipeline.getCustomClusterLabel?.(cid);
            return lbl?.label || (cid === -1 ? 'Outlier' : `Cluster ${cid}`);
        }
        if (field === '__count__') return 1;
        if (this.pipeline.metrics?.has?.(field)) {
            try { return this.pipeline.metrics.evaluate(field, doc); } catch (_) { return null; }
        }
        if (doc?.metadata && Object.prototype.hasOwnProperty.call(doc.metadata, field)) {
            return doc.metadata[field];
        }
        return doc?.[field];
    }

    _filterScope(filters, usePersistentFilters = false) {
        return this.pipeline.createMetadataFilterScope(filters || {}, {
            includePersistent: usePersistentFilters
        });
    }

    _filterMetadata(scope) {
        return this.pipeline.serializeMetadataFilterScope(scope);
    }

    // ---- summarize_cluster ------------------------------------------------

    async summarizeCluster({
        cluster_id,
        summarizer = 'external',
        n_exemplars,
        persist_label = true,
        operationToken = null,
        onStatus = null
    } = {}) {
        if (summarizer === 'local' && this.pipeline.isProcessing) {
            throw new Error('Wait for data processing to finish before labelling clusters.');
        }
        if (summarizer === 'local' && typeof window !== 'undefined'
            && window.__vectoriaGenerationMode === 'external') {
            throw new Error('Local cluster labelling is disabled in AI client mode. Use summarizer="external" or enable the local browser model.');
        }
        const ds = this._docs();
        const datasetId = this.pipeline.currentDatasetId || ds.id || null;
        const cid = Number(cluster_id);
        const indices = [];
        for (let i = 0; i < ds.documents.length; i++) {
            if (ds.clusters?.[i] === cid) indices.push(i);
        }
        if (!indices.length) {
            return { cluster_id: cid, exemplars: [], keywords: [], coverage: 0, error: 'Cluster has no members' };
        }

        const probabilities = ds.documents.map(d => d?.metadata?.cluster_probability ?? 0);
        const n = n_exemplars && Number(n_exemplars) > 0
            ? Math.min(Number(n_exemplars), indices.length)
            : selectAdaptiveExemplarCount(indices.length);

        const picks = selectQuantileExemplars(indices, probabilities, n);
        const exemplars = picks.map(p => ({
            index: p.index,
            doc_id: ds.documents[p.index]?.id,
            text: ds.documents[p.index]?.text || '',
            probability: p.probability,
            quantile: p.quantile,
            band: p.band
        }));

        // Keywords from clustering module
        const kwMap = this.pipeline.clustering?.getClusterKeywords?.();
        const keywords = kwMap?.get(cid)?.slice(0, 10) || [];

        // Coverage = fraction of cluster members with probability >= 0.5
        const conf = indices.map(i => probabilities[i]).filter(p => p >= 0.5).length;
        const coverage = indices.length ? conf / indices.length : 0;

        const result = {
            cluster_id: cid,
            cluster_size: indices.length,
            n_exemplars: exemplars.length,
            exemplars,
            keywords,
            coverage,
            summarizer
        };

        if (summarizer === 'local') {
            const label = await this._localLabel(exemplars, keywords, cid, {
                operationToken,
                datasetId,
                onStatus
            });
            if (this.pipeline.currentDataset !== ds
                || String(this.pipeline.currentDatasetId || '') !== String(datasetId || '')) {
                throw new Error('The active dataset changed while this cluster label was being generated.');
            }
            result.label = label.label;
            result.summary = label.summary;
            if (persist_label && label.label) {
                this.pipeline.setCustomClusterLabel?.(cid, label.label, 'local');
            }
        } else {
            // External path — caller's MCP AI synthesizes. Provide a ready-made prompt.
            result.prompt_template = buildExternalPrompt(exemplars, keywords, cid);
            result.persist_via = 'set_cluster_label';
        }

        return result;
    }

    async _localLabel(exemplars, keywords, cid, options = {}) {
        if (!this.pipeline.rag) {
            throw new Error('Local AI models are not set up. Download them before labelling clusters.');
        }
        const prompt = buildLocalPrompt(exemplars, keywords, cid);
        try {
            const text = await this.pipeline.rag.generateRaw(prompt, {
                temperature: 0.4,
                maxTokens: 180,
                owner: 'cluster-label',
                operationToken: options.operationToken,
                datasetId: options.datasetId,
                onStatus: options.onStatus
            });
            return parseLabelResponse(text, cid);
        } catch (e) {
            console.warn('Local cluster summary failed:', e.message);
            throw new Error(`Local cluster labelling failed: ${e.message}`);
        }
    }

    // ---- get_outliers -----------------------------------------------------

    getOutliers({ threshold = 0.5, k = 50, include_text = false } = {}) {
        const ds = this._docs();
        const out = [];
        for (let i = 0; i < ds.documents.length; i++) {
            const cluster = ds.clusters?.[i];
            const prob = ds.documents[i]?.metadata?.cluster_probability ?? 0;
            if (cluster === -1 || prob < threshold) {
                out.push({
                    index: i,
                    cluster,
                    probability: prob,
                    reason: cluster === -1
                        ? 'HDBSCAN noise point (no dense neighbourhood)'
                        : `Probability ${prob.toFixed(2)} below threshold ${threshold}`,
                    ...(include_text ? { text: ds.documents[i]?.text || '' } : {}),
                    metadata: include_text ? ds.documents[i]?.metadata : undefined
                });
            }
        }
        out.sort((a, b) => a.probability - b.probability);
        return { outliers: out.slice(0, k), count: out.length, threshold };
    }

    // ---- cross_tabulate ---------------------------------------------------

    crossTabulate({ row_field, col_field, normalize = 'none', filter = null, use_persistent_filters = false } = {}) {
        if (!row_field || !col_field) throw new Error('row_field and col_field required');
        this._docs();
        const scope = this._filterScope(filter, use_persistent_filters);
        const docs = scope.documents;

        const rowVals = docs.map(d => this._readField(d, row_field));
        const colVals = docs.map(d => this._readField(d, col_field));
        const { matrix, rowKeys, colKeys } = buildContingency(rowVals, colVals);

        const { chi2, dof, p_value, n } = chiSquare(matrix);
        const v = cramersV(chi2, n, rowKeys.length, colKeys.length);

        const rowTotals = matrix.map(r => r.reduce((s, x) => s + x, 0));
        const colTotals = colKeys.map((_, j) => matrix.reduce((s, r) => s + r[j], 0));

        const normalize_one = (val, ri, ci) => {
            if (normalize === 'row') return rowTotals[ri] ? val / rowTotals[ri] : 0;
            if (normalize === 'col') return colTotals[ci] ? val / colTotals[ci] : 0;
            if (normalize === 'total') return n ? val / n : 0;
            return val;
        };

        const table = {};
        rowKeys.forEach((rk, ri) => {
            table[rk] = {};
            colKeys.forEach((ck, ci) => {
                table[rk][ck] = normalize_one(matrix[ri][ci], ri, ci);
            });
        });

        return {
            row_field,
            col_field,
            normalize,
            row_keys: rowKeys,
            col_keys: colKeys,
            table,
            row_totals: Object.fromEntries(rowKeys.map((k, i) => [k, rowTotals[i]])),
            col_totals: Object.fromEntries(colKeys.map((k, i) => [k, colTotals[i]])),
            n,
            chi_square: chi2,
            dof,
            p_value,
            cramers_v: v,
            filter_metadata: this._filterMetadata(scope)
        };
    }

    // ---- aggregate --------------------------------------------------------

    aggregate({ group_by, metric = '__count__', agg = 'count', filter = null, use_persistent_filters = false } = {}) {
        if (!group_by) throw new Error('group_by required');
        this._docs();
        const scope = this._filterScope(filter, use_persistent_filters);
        const docs = scope.documents;

        const groups = new Map();
        for (const doc of docs) {
            const key = this._readField(doc, group_by);
            const keyStr = key === undefined || key === null ? '(null)' : String(key);
            if (!groups.has(keyStr)) groups.set(keyStr, []);
            let val;
            if (metric === '__count__') {
                val = 1;
            } else {
                val = this._readField(doc, metric);
                val = typeof val === 'number' ? val : parseFloat(val);
                if (!Number.isFinite(val)) val = 0;
            }
            groups.get(keyStr).push(val);
        }

        const aggFn = aggFunction(agg);
        const result = [];
        for (const [key, values] of groups.entries()) {
            result.push({
                key,
                value: aggFn(values),
                n: values.length
            });
        }
        result.sort((a, b) => b.value - a.value);
        return {
            group_by,
            metric,
            agg,
            groups: result,
            filter_metadata: this._filterMetadata(scope)
        };
    }

    // ---- compare_clusters -------------------------------------------------

    compareClusters({ cluster_ids, fields = null } = {}) {
        if (!Array.isArray(cluster_ids) || cluster_ids.length < 2) {
            throw new Error('Provide at least 2 cluster_ids');
        }
        const ds = this._docs();
        const groups = cluster_ids.map(cid => ({
            cluster_id: cid,
            docs: ds.documents.filter((_, i) => ds.clusters?.[i] === Number(cid))
        }));

        // Auto-discover comparable fields if not provided
        if (!fields) {
            const sample = ds.documents[0]?.metadata || {};
            fields = Object.keys(sample).filter(k =>
                !['cluster', 'cluster_label', 'cluster_probability',
                  'cluster_keywords', 'cluster_keyword_scores', 'cluster_keywords_viz'].includes(k)
            ).slice(0, 8);
        }

        const per_field = {};
        for (const field of fields) {
            const series = groups.map(g => g.docs.map(d => this._readField(d, field)));
            const isNumeric = series.flat().every(v => v === null || v === undefined || v === '' || Number.isFinite(Number(v)));

            if (isNumeric) {
                const numeric = series.map(s => s.map(Number).filter(Number.isFinite));
                const stats = numeric.map(arr => ({
                    n: arr.length,
                    mean: arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
                }));
                let welch = null;
                if (numeric.length === 2 && numeric[0].length && numeric[1].length) {
                    welch = welchT(numeric[0], numeric[1]);
                }
                per_field[field] = { type: 'numeric', stats, welch };
            } else {
                const dists = series.map(s => {
                    const dist = {};
                    for (const v of s) {
                        const key = v === undefined || v === null ? '(null)' : String(v);
                        dist[key] = (dist[key] || 0) + 1;
                    }
                    return dist;
                });
                let divergence = null;
                if (dists.length === 2) divergence = jsDivergence(dists[0], dists[1]);
                per_field[field] = { type: 'categorical', distributions: dists, js_divergence: divergence };
            }
        }

        return {
            cluster_ids,
            sizes: groups.map(g => g.docs.length),
            per_field
        };
    }

    // ---- filter_to_subset -------------------------------------------------

    filterToSubset({ filters = {}, name = null, use_persistent_filters = false } = {}) {
        this._docs();
        const scope = this._filterScope(filters, use_persistent_filters);
        const indices = scope.indices;
        const subset_id = `subset_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        if (!(this.pipeline.subsets instanceof Map)) this.pipeline.subsets = new Map();
        this.pipeline.subsets.set(subset_id, {
            name: name || subset_id,
            doc_indices: indices,
            filters: scope.filters,
            created_at: Date.now()
        });
        return {
            subset_id,
            name: name || subset_id,
            count: indices.length,
            doc_indices: indices,
            filter_metadata: this._filterMetadata(scope)
        };
    }

    // ---- multi_vector_search ---------------------------------------------

    async multiVectorSearch({
        queries,
        k = 10,
        rrf_k = 60,
        fuse = 'rrf',
        metadata_filters = null,
        use_persistent_filters = false
    } = {}) {
        if (!Array.isArray(queries) || !queries.length) {
            throw new Error('queries[] required');
        }
        if (!this.pipeline.embeddings) throw new Error('Embeddings unavailable');
        if (!this.pipeline.vectorSearch?.isBuilt) throw new Error('Vector index not built');

        const vectors = [];
        for (const q of queries) {
            if (q && typeof q === 'object' && Array.isArray(q.vector)) {
                vectors.push({ vector: q.vector, label: q.label || q.text || null });
            } else {
                const text = typeof q === 'string' ? q : q?.text;
                if (!text) throw new Error('Query must be a string or {text} or {vector}');
                const vec = await this.pipeline.embeddings.embedSingle(text, { mode: 'query' });
                vectors.push({ vector: vec, label: text });
            }
        }

        const scope = this._filterScope(metadata_filters, use_persistent_filters);
        const filter = scope.predicate;
        if (scope.applied && scope.matchedDocuments === 0) {
            return {
                n_queries: vectors.length,
                queries: vectors.map(v => v.label),
                results: [],
                filter_metadata: this._filterMetadata(scope)
            };
        }

        const fused = this.pipeline.vectorSearch.multiVectorSearch(vectors, {
            k,
            rrfK: rrf_k,
            fuse,
            filter,
            includeMetadata: true
        });

        return {
            n_queries: vectors.length,
            queries: vectors.map(v => v.label),
            ...fused,
            filter_metadata: this._filterMetadata(scope)
        };
    }
}

// --- prompt helpers -----------------------------------------------------

function buildLocalPrompt(exemplars, keywords, cid) {
    const kws = keywords?.length ? `Top keywords: ${keywords.slice(0, 8).join(', ')}.\n\n` : '';
    const ex = exemplars.slice(0, 9).map((e, i) =>
        `[${i + 1}] (${e.band}, p=${e.probability.toFixed(2)}): ${truncate(e.text, 280)}`
    ).join('\n\n');
    return `You are labelling a topic cluster (#${cid}) for a researcher.

${kws}Below are exemplar documents drawn from low/mid/high HDBSCAN confidence levels so you see the cluster's range:

${ex}

Respond in EXACTLY this format and nothing else:
LABEL: <3-5 word topic label>
SUMMARY: <one sentence summary of what unites these documents>`;
}

function buildExternalPrompt(exemplars, keywords, cid) {
    const kws = keywords?.length ? `Top keywords: ${keywords.slice(0, 8).join(', ')}.\n\n` : '';
    const ex = exemplars.map((e, i) =>
        `[${i + 1}] (confidence band: ${e.band}, probability: ${e.probability.toFixed(2)}): ${truncate(e.text, 400)}`
    ).join('\n\n');
    return `You are labelling cluster #${cid}. The exemplars below were sampled across confidence quantiles so you see the cluster's range, not just its core.

${kws}${ex}

When done, call the \`set_cluster_label\` tool with cluster_id=${cid} and a 3-5 word label.`;
}

function parseLabelResponse(text, cid) {
    const labelMatch = text.match(/LABEL:\s*(.+)/i);
    const summaryMatch = text.match(/SUMMARY:\s*([\s\S]+)/i);
    const label = labelMatch
        ? cleanLabel(labelMatch[1].split('\n')[0])
        : '';
    const summary = summaryMatch
        ? summaryMatch[1].trim().replace(/LABEL:.*/i, '').trim()
        : text.trim();
    return {
        label: label || `Cluster ${cid}`,
        summary
    };
}

/**
 * Strip markdown emphasis (**bold**, *italic*, ***, `code`, _underscore_),
 * surrounding quotes, and a stray "LABEL:" echo from an AI-produced label.
 */
export function cleanLabel(raw) {
    if (raw === undefined || raw === null) return '';
    let s = String(raw).trim();
    s = s.replace(/^label:\s*/i, '');           // stray "LABEL:" echo
    s = s.replace(/[*_`]+/g, '');                // markdown emphasis / code ticks
    s = s.replace(/^["'“”‘’]+|["'“”‘’]+$/g, ''); // wrapping quotes
    s = s.replace(/\s+/g, ' ').trim();           // collapse whitespace
    return s;
}

function truncate(s, n) {
    if (!s) return '';
    return s.length > n ? s.slice(0, n) + '…' : s;
}

function aggFunction(agg) {
    switch (agg) {
        case 'sum':    return arr => arr.reduce((a, b) => a + b, 0);
        case 'mean':   return arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        case 'median': return arr => {
            if (!arr.length) return 0;
            const s = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(s.length / 2);
            return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
        };
        case 'min':    return arr => arr.length ? Math.min(...arr) : 0;
        case 'max':    return arr => arr.length ? Math.max(...arr) : 0;
        case 'count':
        default:       return arr => arr.length;
    }
}
