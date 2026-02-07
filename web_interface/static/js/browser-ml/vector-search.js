/**
 * Production-grade browser retrieval primitives with backward compatibility.
 * - Vector store: contiguous Float32Array + precomputed norms + fast cosine.
 * - BM25: true inverted index using posting lists; query-time touches only matched docs.
 * - Serialization: binary via base64 for vectors/norms; still accepts legacy JSON arrays.
 * - API kept stable: buildIndex, search, getDocument, getAllIds, getStats, serialize, deserialize, clear.
 * - No fake HNSW flags. Exact search is optimized; ANN can be plugged later without breaking API.
 */

//////////////////////////
// Utility: base64 <-> ArrayBuffer
//////////////////////////
const _b64 = {
  fromArrayBuffer(buf) {
    const bytes = new Uint8Array(buf);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  },
  toArrayBuffer(b64) {
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }
};

//////////////////////////
// BrowserVectorSearch
//////////////////////////
export class BrowserVectorSearch {
  constructor(dimension = 384, options = {}) {
    this.dimension = dimension;

    // Storage (contiguous)
    this._matrix = null;        // Float32Array length = numVectors * dimension
    this._norms = null;         // Float32Array length = numVectors
    this.ids = [];
    this.metadata = [];

    // Flags
    this.isBuilt = false;

    // Options (kept for compatibility; not used by exact search)
    this.maxElements = options.maxElements || 100000;
    this.efConstruction = options.efConstruction || 200;
    this.M = options.M || 16;
    this.ef = options.ef || 50;

    // Legacy flag retained; now just means "exact search"
    this.useFlatIndex = true;
  }

  /**
   * Build the vector index
   * @param {number[][]} embeddings - Array of embedding vectors
   * @param {string[]} docIds - Document IDs corresponding to embeddings
   * @param {Object[]} documents - Document metadata
   */
  async buildIndex(embeddings, docIds, documents) {
    if (!embeddings || embeddings.length === 0) throw new Error("Cannot build index with zero embeddings");
    if (embeddings.length !== docIds.length) throw new Error("Embeddings and IDs must have same length");

    const n = embeddings.length;
    const d = this.dimension;

    // Accept mismatched dims but fail fast if detected
    const firstLen = embeddings[0].length;
    if (firstLen !== d) throw new Error(`Embedding dimension mismatch: got ${firstLen}, expected ${d}`);

    // Allocate contiguous matrix and norms
    const mat = new Float32Array(n * d);
    const norms = new Float32Array(n);

    // Fill matrix and norms
    let offset = 0;
    for (let i = 0; i < n; i++) {
      const row = embeddings[i];
      let sum = 0;
      for (let j = 0; j < d; j++) {
        const v = row[j];
        mat[offset + j] = v;
        sum += v * v;
      }
      norms[i] = Math.sqrt(sum) || 1e-12; // avoid zero division
      offset += d;
    }

    this._matrix = mat;
    this._norms = norms;
    this.ids = docIds.slice(0);
    if (documents && Array.isArray(documents)) {
      const processedMetadata = new Array(n);
      for (let i = 0; i < n; i++) {
        const doc = documents[i];
        if (doc && typeof doc === "object") {
          const flattened = {};

          if (doc.metadata && typeof doc.metadata === "object") {
            for (const [key, value] of Object.entries(doc.metadata)) {
              flattened[key] = value;
            }
          }

          for (const [key, value] of Object.entries(doc)) {
            if (key === "metadata") continue;
            if (key === "text") {
              flattened.text = value;
            } else if (key === "id") {
              flattened.doc_id = value;
            } else if (!(key in flattened)) {
              flattened[key] = value;
            }
          }

          processedMetadata[i] = flattened;
        } else {
          processedMetadata[i] = doc ?? {};
        }
      }
      this.metadata = processedMetadata;
    } else {
      this.metadata = Array.from({ length: n }, () => ({}));
    }
    this.isBuilt = true;

    return {
      numVectors: n,
      dimension: d,
      indexType: "flat"
    };
  }

  /**
   * Search for similar vectors (cosine similarity exact, optimized)
   * @param {number[]} queryEmbedding - Query vector
   * @param {number} k - Number of results to return
   * @param {Object} options - { filter, minScore, includeMetadata }
   * @returns {Object[]} Search results with scores
   */
  search(queryEmbedding, k = 10, options = {}) {
    if (!this.isBuilt) throw new Error("Index not built. Call buildIndex() first.");

    const { filter = null, minScore = 0.0, includeMetadata = true } = options;

    const d = this.dimension;
    if (!queryEmbedding || queryEmbedding.length !== d) throw new Error(`Query dim mismatch: got ${queryEmbedding?.length}, expected ${d}`);

    const q = new Float32Array(queryEmbedding);
    let qnorm = 0;
    for (let i = 0; i < d; i++) qnorm += q[i] * q[i];
    qnorm = Math.sqrt(qnorm) || 1e-12;

    // Compute cosine similarities with partial selection (nth_element emulation)
    const n = this.ids.length;
    const mat = this._matrix;
    const norms = this._norms;

    // Pre-allocate result buffers
    const scores = new Float32Array(n);
    // Fast dot product
    for (let row = 0, base = 0; row < n; row++, base += d) {
      let dot = 0;
      // Unroll by 4 for a minor speed win
      let j = 0;
      const limit = d - (d % 4);
      for (; j < limit; j += 4) {
        dot += q[j] * mat[base + j]
             + q[j + 1] * mat[base + j + 1]
             + q[j + 2] * mat[base + j + 2]
             + q[j + 3] * mat[base + j + 3];
      }
      for (; j < d; j++) dot += q[j] * mat[base + j];

      const denom = qnorm * norms[row];
      scores[row] = denom > 0 ? dot / denom : 0;
    }

    // Optional filtering; gather candidates
    const candidates = [];
    for (let i = 0; i < n; i++) {
      const s = scores[i];
      if (s < minScore) continue;
      if (filter && !filter(this.metadata[i])) continue;
      candidates.push(i);
    }

    // Select top-k without sorting all if large
    if (candidates.length > 2 * k && k > 0) {
      // Quickselect approach: pick threshold by partial sorting a copy of scores
      const arr = candidates.slice(0);
      arr.sort((a, b) => scores[b] - scores[a]); // simple sort; acceptable since filtered set is smaller
      candidates.length = Math.min(k, arr.length);
      for (let i = 0; i < candidates.length; i++) candidates[i] = arr[i];
    } else {
      candidates.sort((a, b) => scores[b] - scores[a]);
      if (k > 0) candidates.length = Math.min(k, candidates.length);
    }

    // Build results
    const out = new Array(candidates.length);
    for (let r = 0; r < candidates.length; r++) {
      const idx = candidates[r];
      if (includeMetadata) {
        out[r] = {
          index: idx,
          score: scores[idx],
          doc_id: this.ids[idx],
          metadata: this.metadata[idx] || {},
          text: (this.metadata[idx] && this.metadata[idx].text) || ""
        };
      } else {
        out[r] = { index: idx, score: scores[idx], doc_id: this.ids[idx] };
      }
    }

    if (out.length > 0) {
    }

    return out;
  }

  /**
   * Get document by ID
   */
  getDocument(docId) {
    const index = this.ids.indexOf(docId);
    if (index === -1) return null;

    const d = this.dimension;
    const start = index * d;
    const embedding = Array.from(this._matrix.subarray(start, start + d));
    return {
      doc_id: docId,
      embedding,
      metadata: this.metadata[index]
    };
  }

  getAllIds() {
    return this.ids.slice(0);
  }

  getStats() {
    const n = this.ids.length;
    const bytes = n * this.dimension * 4 + n * 4;
    return {
      numVectors: n,
      dimension: this.dimension,
      indexType: "flat",
      memoryEstimate: `~${Math.round(bytes / 1024 / 1024)} MB`
    };
  }

  /**
   * Serialize to binary-friendly JSON (vectors/norms as base64)
   * Backward compatibility: legacy consumer can still parse ids/metadata; vectors are base64.
   */
  serialize() {
    const vecB64 = this._matrix ? _b64.fromArrayBuffer(this._matrix.buffer) : null;
    const normsB64 = this._norms ? _b64.fromArrayBuffer(this._norms.buffer) : null;

    return {
      dimension: this.dimension,
      ids: this.ids,
      metadata: this.metadata,
      vectors_b64: vecB64,
      norms_b64: normsB64,
      options: {
        maxElements: this.maxElements,
        efConstruction: this.efConstruction,
        M: this.M,
        ef: this.ef
      },
      // Legacy interop hint
      format: "contiguous_f32_b64_v1"
    };
  }

  /**
   * Deserialize with backward compatibility:
   * - Preferred: { vectors_b64, norms_b64 }
   * - Legacy: { vectors: number[][] }
   */
  deserialize(data) {
    this.dimension = data.dimension;

    if (data.vectors_b64 && data.norms_b64) {
      const vecBuf = _b64.toArrayBuffer(data.vectors_b64);
      const normsBuf = _b64.toArrayBuffer(data.norms_b64);
      this._matrix = new Float32Array(vecBuf);
      this._norms = new Float32Array(normsBuf);
    } else if (Array.isArray(data.vectors)) {
      // Legacy path: vectors as array of arrays
      const vectors = data.vectors;
      const n = vectors.length;
      if (n > 0 && vectors[0].length !== this.dimension) {
        console.warn(`⚠️ Vector dimension mismatch: expected ${this.dimension}, got ${vectors[0].length}. Adjusting.`);
        this.dimension = vectors[0].length;
      }
      const d = this.dimension;
      const mat = new Float32Array(n * d);
      const norms = new Float32Array(n);
      let off = 0;
      for (let i = 0; i < n; i++) {
        const row = vectors[i];
        let sum = 0;
        for (let j = 0; j < d; j++) {
          const v = row[j];
          mat[off + j] = v;
          sum += v * v;
        }
        norms[i] = Math.sqrt(sum) || 1e-12;
        off += d;
      }
      this._matrix = mat;
      this._norms = norms;
    } else {
      throw new Error("Invalid serialized data: missing vectors");
    }

    this.ids = data.ids || [];
    this.metadata = data.metadata || [];

    if (data.options) {
      this.maxElements = data.options.maxElements ?? this.maxElements;
      this.efConstruction = data.options.efConstruction ?? this.efConstruction;
      this.M = data.options.M ?? this.M;
      this.ef = data.options.ef ?? this.ef;
    }

    this.isBuilt = true;
  }

  clear() {
    this._matrix = null;
    this._norms = null;
    this.ids = [];
    this.metadata = [];
    this.isBuilt = false;
  }
}

//////////////////////////
// BM25Search with inverted index
//////////////////////////
export class BM25Search {
  constructor(options = {}) {
    this.k1 = options.k1 || 1.5;
    this.b = options.b || 0.75;

    // Core data
    this.docCount = 0;
    this.docLengths = [];                 // number[]
    this.avgDocLength = 0;

    // Inverted index: term -> [{doc, tf}]
    this.postings = new Map();

    // ID mapping and document storage
    this.docIds = [];
    this.documents = [];                  // parallel to docIds; store metadata objects

    this.isBuilt = false;
  }

  tokenize(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  /**
   * Build inverted index from documents or metadata objects that include .text
   * @param {Array<Object|string>} documents
   * @param {string[]} docIds
   */
  buildIndex(documents, docIds) {
    if (!documents || documents.length === 0) throw new Error("Cannot build BM25 with zero documents");
    if (documents.length !== docIds.length) throw new Error("Documents and IDs must have same length");

    this.docCount = documents.length;
    this.docIds = docIds.slice(0);
    this.documents = documents.slice(0);

    const postings = new Map();
    const docLengths = new Array(this.docCount);
    let totalLen = 0;

    for (let i = 0; i < this.docCount; i++) {
      const doc = documents[i];
      const text = typeof doc === "string" ? doc : (doc.text || "");
      const terms = this.tokenize(text);

      // Build per-doc tf map
      const tfMap = new Map();
      for (const t of terms) tfMap.set(t, (tfMap.get(t) || 0) + 1);

      // Update inverted index
      for (const [t, tf] of tfMap.entries()) {
        if (!postings.has(t)) postings.set(t, []);
        postings.get(t).push({ doc: i, tf });
      }

      const len = terms.length;
      docLengths[i] = len;
      totalLen += len;
    }

    this.docLengths = docLengths;
    this.avgDocLength = Math.max(1, totalLen / (this.docCount || 1));
    this.postings = postings;
    this.isBuilt = true;
  }

  /**
   * BM25 scoring touching only posted docs
   */
  search(query, k = 10) {
    if (!this.isBuilt) throw new Error("BM25 index not built");

    const qTerms = this.tokenize(query);
    if (qTerms.length === 0) return [];

    // Accumulate scores per doc using a dense float array for speed
    const scores = new Float32Array(this.docCount);
    const touched = new Set();

    // Unique query terms to avoid duplicate work
    const uniq = Array.from(new Set(qTerms));

    for (const term of uniq) {
      const plist = this.postings.get(term);
      if (!plist) continue;

      const df = plist.length;
      // BM25 IDF with +1
      const idf = Math.log((this.docCount - df + 0.5) / (df + 0.5) + 1);

      for (let i = 0; i < plist.length; i++) {
        const { doc, tf } = plist[i];
        const dl = this.docLengths[doc];
        const denom = tf + this.k1 * (1 - this.b + this.b * (dl / this.avgDocLength));
        const s = idf * (tf * (this.k1 + 1)) / denom;
        scores[doc] += s;
        touched.add(doc);
      }
    }

    // Collect touched docs and rank
    const candidates = Array.from(touched);
    candidates.sort((a, b) => scores[b] - scores[a]);

    const top = candidates.slice(0, k).map((idx) => {
      const doc = this.documents[idx];
      const docMetadata = (typeof doc === "object" && doc !== null) ? (doc.metadata || doc) : {};

      return {
        index: idx,
        score: scores[idx],
        doc_id: this.docIds[idx],
        text: (typeof doc === "string" ? doc : (doc?.text || "")),
        // Flatten parent_id to top level for consistency with vector search
        parent_id: docMetadata.parent_id || doc?.parent_id,
        metadata: docMetadata
      };
    });

    return top;
  }
}

// Legacy HybridSearch removed; retrieval now uses dedicated semantic or keyword paths.
