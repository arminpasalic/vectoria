/**
 * Analysis session store. A "session" snapshots:
 *   - active metadata filters
 *   - registered metrics
 *   - annotations
 *   - custom cluster labels
 *   - subsets
 *   - caller-supplied findings text
 * It is tamper-evidenced with a SHA-256 over the canonical JSON payload and
 * can be downloaded as a standalone artifact.
 */
export class SessionsStore {
    constructor(pipeline) {
        this.pipeline = pipeline;
        if (!Array.isArray(pipeline.analysisSessions)) pipeline.analysisSessions = [];
    }

    async save(name, findings = '', download = true) {
        const ds = this.pipeline.currentDataset;
        const dataset_id = ds?.id || null;
        const dataset_hash = ds ? await hashCanonical(ds.documents?.map(d => d.id) ?? []) : null;

        const filters = (typeof window !== 'undefined' && window.currentMetadataFilters)
            ? window.currentMetadataFilters
            : {};

        const customClusterLabels = [...(this.pipeline.customClusterLabels?.entries() ?? [])]
            .map(([cluster_id, v]) => ({ cluster_id, ...v }));

        const subsets = [...(this.pipeline.subsets?.entries() ?? [])]
            .map(([subset_id, v]) => ({ subset_id, ...v }));

        const payload = {
            dataset_id,
            dataset_hash,
            timestamp: Date.now(),
            saved_at: new Date().toISOString(),
            name: name || `session_${Date.now()}`,
            findings,
            filters,
            metrics: this.pipeline.metrics?.serialize() ?? [],
            annotations: this.pipeline.annotationsApi?.serialize() ?? [],
            customClusterLabels,
            subsets
        };

        const sha256 = await hashCanonical(payload);
        const session_id = `sess_${payload.timestamp}_${Math.random().toString(36).slice(2, 8)}`;
        const record = { id: session_id, name: payload.name, saved_at: payload.saved_at, sha256, dataset_id, payload };
        this.pipeline.analysisSessions.push(record);

        if (download && typeof document !== 'undefined') {
            try {
                const wrapped = { version: 'session-v1', session_id, sha256, payload };
                const blob = new Blob([JSON.stringify(wrapped, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `vectoria-session-${payload.name.replace(/[^a-z0-9-_]+/gi, '_')}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            } catch (e) {
                console.warn('Session download failed:', e.message);
            }
        }

        return { session_id, sha256, saved_at: payload.saved_at };
    }

    async load(idOrPayload) {
        let record = null;

        if (typeof idOrPayload === 'string') {
            record = this.pipeline.analysisSessions.find(s => s.id === idOrPayload);
            if (!record) throw new Error(`Session not found: ${idOrPayload}`);
        } else if (idOrPayload && typeof idOrPayload === 'object') {
            // Accept either wrapped {version, payload, sha256} or raw payload
            const wrapped = idOrPayload.payload && idOrPayload.sha256
                ? idOrPayload
                : { payload: idOrPayload, sha256: null };
            const payload = wrapped.payload;
            const computed = await hashCanonical(payload);
            if (wrapped.sha256 && wrapped.sha256 !== computed) {
                throw new Error(`Session signature mismatch (tampered or corrupted). expected=${wrapped.sha256} got=${computed}`);
            }
            record = {
                id: `sess_imported_${Date.now()}`,
                name: payload.name || 'imported',
                saved_at: payload.saved_at || new Date().toISOString(),
                sha256: computed,
                dataset_id: payload.dataset_id || null,
                payload
            };
            this.pipeline.analysisSessions.push(record);
        } else {
            throw new Error('load_analysis_session requires session_id or json_payload');
        }

        const p = record.payload;
        if (p.metrics) this.pipeline.metrics?.hydrate(p.metrics);
        if (p.annotations) this.pipeline.annotationsApi?.hydrate(p.annotations);
        if (Array.isArray(p.customClusterLabels)) {
            for (const { cluster_id, label, source } of p.customClusterLabels) {
                this.pipeline.setCustomClusterLabel?.(cluster_id, label, source || 'session');
            }
        }
        if (Array.isArray(p.subsets) && this.pipeline.subsets) {
            this.pipeline.subsets.clear();
            for (const { subset_id, ...rest } of p.subsets) this.pipeline.subsets.set(subset_id, rest);
        }
        if (p.filters && typeof window !== 'undefined') {
            window.currentMetadataFilters = p.filters;
        }

        return {
            ok: true,
            session_id: record.id,
            restored: {
                filters: p.filters || {},
                metrics: p.metrics || [],
                annotations: p.annotations || [],
                customClusterLabels: p.customClusterLabels || [],
                findings: p.findings || ''
            }
        };
    }

    list() {
        return this.pipeline.analysisSessions.map(({ id, name, saved_at, sha256, dataset_id }) =>
            ({ id, name, saved_at, sha256, dataset_id })
        );
    }

    hydrate(arr) {
        if (!Array.isArray(arr)) return;
        // arr is the list() output; we keep stub records (no payload) for visibility.
        this.pipeline.analysisSessions = arr.map(r => ({ ...r, payload: r.payload || null }));
    }
}

export async function hashCanonical(obj) {
    const canonical = canonicalJSON(obj);
    if (typeof crypto?.subtle?.digest !== 'function') {
        // Fallback (deterministic non-crypto). Acceptable for non-secure contexts.
        let h = 5381;
        for (let i = 0; i < canonical.length; i++) h = ((h << 5) + h + canonical.charCodeAt(i)) | 0;
        return `nohash:${(h >>> 0).toString(16)}`;
    }
    const buf = new TextEncoder().encode(canonical);
    const digest = await crypto.subtle.digest('SHA-256', buf);
    return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
}

function canonicalJSON(value) {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(',')}]`;
    const keys = Object.keys(value).sort();
    return `{${keys.map(k => `${JSON.stringify(k)}:${canonicalJSON(value[k])}`).join(',')}}`;
}
