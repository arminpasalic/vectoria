/**
 * User-defined derived metrics. A metric is a small whitelisted formula over
 * a document's metadata fields and numeric literals — e.g. "likes + 2*shares".
 *
 * Grammar (recursive descent, no eval):
 *   Expr   = Term (('+'|'-') Term)*
 *   Term   = Factor (('*'|'/') Factor)*
 *   Factor = Number | Ident | '(' Expr ')' | '-' Factor
 *   Ident  = [A-Za-z_][A-Za-z0-9_]*   → resolves to doc.metadata[ident]
 */
export class MetricsRegistry {
    constructor(pipeline) {
        this.pipeline = pipeline;
        if (!(pipeline.registeredMetrics instanceof Map)) {
            pipeline.registeredMetrics = new Map();
        }
    }

    register(name, formula, description = '') {
        if (!name || typeof name !== 'string') throw new Error('Metric name is required');
        if (name.startsWith('__')) throw new Error('Metric name cannot start with "__" (reserved)');
        const ast = this._parse(formula);
        const created_at = Date.now();
        this.pipeline.registeredMetrics.set(name, { name, formula, ast, description, created_at });
        return { ok: true, name, ast };
    }

    has(name) {
        return this.pipeline.registeredMetrics.has(name);
    }

    evaluate(name, doc) {
        const entry = this.pipeline.registeredMetrics.get(name);
        if (!entry) throw new Error(`Unknown metric: ${name}`);
        return this._eval(entry.ast, doc?.metadata || {});
    }

    list() {
        const sampleDoc = this.pipeline.currentDataset?.documents?.[0];
        const metrics = [];
        for (const entry of this.pipeline.registeredMetrics.values()) {
            let sample_value = null;
            if (sampleDoc) {
                try { sample_value = this._eval(entry.ast, sampleDoc.metadata || {}); } catch (_) { sample_value = null; }
            }
            metrics.push({
                name: entry.name,
                formula: entry.formula,
                description: entry.description,
                created_at: entry.created_at,
                sample_value
            });
        }
        return { metrics };
    }

    serialize() {
        return [...this.pipeline.registeredMetrics.values()].map(({ name, formula, description, created_at }) =>
            ({ name, formula, description, created_at })
        );
    }

    hydrate(arr) {
        this.pipeline.registeredMetrics.clear();
        if (!Array.isArray(arr)) return;
        for (const m of arr) {
            try {
                this.register(m.name, m.formula, m.description || '');
                const stored = this.pipeline.registeredMetrics.get(m.name);
                if (stored && m.created_at) stored.created_at = m.created_at;
            } catch (e) {
                console.warn(`Failed to re-register metric ${m.name}:`, e.message);
            }
        }
    }

    // ------------- parser -------------
    _parse(src) {
        if (typeof src !== 'string' || !src.trim()) throw new Error('Formula must be a non-empty string');
        this._src = src;
        this._pos = 0;
        const ast = this._parseExpr();
        this._skipWs();
        if (this._pos < this._src.length) {
            throw new Error(`Unexpected token at position ${this._pos}: "${this._src.slice(this._pos)}"`);
        }
        return ast;
    }
    _peek() { this._skipWs(); return this._src[this._pos]; }
    _skipWs() { while (this._pos < this._src.length && /\s/.test(this._src[this._pos])) this._pos++; }

    _parseExpr() {
        let left = this._parseTerm();
        while (true) {
            const c = this._peek();
            if (c === '+' || c === '-') {
                this._pos++;
                const right = this._parseTerm();
                left = { type: 'bin', op: c, left, right };
            } else break;
        }
        return left;
    }

    _parseTerm() {
        let left = this._parseFactor();
        while (true) {
            const c = this._peek();
            if (c === '*' || c === '/') {
                this._pos++;
                const right = this._parseFactor();
                left = { type: 'bin', op: c, left, right };
            } else break;
        }
        return left;
    }

    _parseFactor() {
        this._skipWs();
        if (this._pos >= this._src.length) {
            throw new Error(`Unexpected end of formula at position ${this._pos}`);
        }
        const c = this._src[this._pos];
        if (c === '(') {
            this._pos++;
            const inner = this._parseExpr();
            this._skipWs();
            if (this._src[this._pos] !== ')') throw new Error(`Expected ")" at position ${this._pos}`);
            this._pos++;
            return inner;
        }
        if (c === '-') {
            this._pos++;
            return { type: 'neg', value: this._parseFactor() };
        }
        // Number
        if (/[0-9.]/.test(c)) {
            const start = this._pos;
            const match = this._src.slice(start).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?/);
            if (!match) throw new Error(`Invalid number at position ${start}`);
            this._pos += match[0].length;
            const num = Number(match[0]);
            if (!Number.isFinite(num)) throw new Error(`Invalid number at position ${start}`);
            return { type: 'num', value: num };
        }
        // Identifier
        if (/[A-Za-z_]/.test(c)) {
            const start = this._pos;
            while (this._pos < this._src.length && /[A-Za-z0-9_]/.test(this._src[this._pos])) this._pos++;
            const name = this._src.slice(start, this._pos);
            return { type: 'ident', name };
        }
        throw new Error(`Unexpected character "${c}" at position ${this._pos}`);
    }

    _eval(node, metadata) {
        if (!node) return 0;
        if (node.type === 'num') return node.value;
        if (node.type === 'ident') {
            const v = metadata[node.name];
            if (v === undefined || v === null || v === '') return 0;
            const n = typeof v === 'number' ? v : parseFloat(v);
            return Number.isFinite(n) ? n : 0;
        }
        if (node.type === 'neg') return -this._eval(node.value, metadata);
        if (node.type === 'bin') {
            const a = this._eval(node.left, metadata);
            const b = this._eval(node.right, metadata);
            switch (node.op) {
                case '+': return a + b;
                case '-': return a - b;
                case '*': return a * b;
                case '/': return b === 0 ? 0 : a / b;
            }
        }
        return 0;
    }
}
