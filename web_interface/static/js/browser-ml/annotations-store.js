/**
 * In-memory document annotation store.
 * Annotations attach tags/notes/colors to documents so a multi-step analysis
 * (often driven by an external MCP AI) can accumulate labelled state.
 * Persists into vectoria.json via serialize/hydrate.
 */
export class AnnotationsStore {
    constructor(pipeline) {
        this.pipeline = pipeline;
        if (!(pipeline.annotations instanceof Map)) {
            pipeline.annotations = new Map();
        }
        this._byDoc = new Map(); // docIndex → Set(annotationId)
    }

    add({ doc_indices, tag, note = '', color = null }) {
        if (!Array.isArray(doc_indices) || !tag) {
            throw new Error('annotate_documents requires doc_indices[] and tag');
        }
        const created_at = Date.now();
        const ids = [];
        for (const idx of doc_indices) {
            const id = `ann_${created_at}_${Math.random().toString(36).slice(2, 8)}_${idx}`;
            const entry = { id, doc_index: idx, tag, note, color, created_at };
            this.pipeline.annotations.set(id, entry);
            if (!this._byDoc.has(idx)) this._byDoc.set(idx, new Set());
            this._byDoc.get(idx).add(id);
            ids.push(id);
        }
        return { added: ids.length, annotation_ids: ids };
    }

    list({ tag, doc_index } = {}) {
        const out = [];
        for (const ann of this.pipeline.annotations.values()) {
            if (tag && ann.tag !== tag) continue;
            if (doc_index !== undefined && ann.doc_index !== doc_index) continue;
            out.push(ann);
        }
        return { annotations: out, count: out.length };
    }

    forDoc(idx) {
        const ids = this._byDoc.get(idx);
        if (!ids) return [];
        return [...ids].map(id => this.pipeline.annotations.get(id)).filter(Boolean);
    }

    remove(id) {
        const ann = this.pipeline.annotations.get(id);
        if (!ann) return false;
        this.pipeline.annotations.delete(id);
        this._byDoc.get(ann.doc_index)?.delete(id);
        return true;
    }

    serialize() {
        return [...this.pipeline.annotations.values()];
    }

    hydrate(arr) {
        this.pipeline.annotations.clear();
        this._byDoc.clear();
        if (!Array.isArray(arr)) return;
        for (const ann of arr) {
            if (!ann?.id) continue;
            this.pipeline.annotations.set(ann.id, ann);
            if (!this._byDoc.has(ann.doc_index)) this._byDoc.set(ann.doc_index, new Set());
            this._byDoc.get(ann.doc_index).add(ann.id);
        }
    }
}
