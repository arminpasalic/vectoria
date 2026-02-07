/**
 * Tier 3 Chunk Embeddings
 * Generates E5 passage-mode embeddings for RAG retrieval chunks
 */

/**
 * EmbeddedChunkRecord structure
 * @typedef {Object} EmbeddedChunkRecord
 * @property {string} docId - Parent document ID
 * @property {string} chunkId - Unique chunk identifier
 * @property {number} chunkIndex - Zero-based chunk index within parent
 * @property {string} text - Chunk text content
 * @property {number[]} embedding - 384-dim embedding vector
 * @property {Object} metadata - Chunk metadata
 */

/**
 * Embed chunks using E5 passage mode for asymmetric retrieval
 *
 * @param {Array} chunks - Array of chunk objects from chonkieChunker
 * @param {Object} embedder - BrowserEmbeddings instance
 * @param {Object} options - Embedding options
 * @param {Function} options.onProgress - Progress callback
 * @returns {Promise<EmbeddedChunkRecord[]>} Chunks with embeddings
 */
export async function embedChunks(chunks, embedder, options = {}) {
    const {
        onProgress = null
    } = options;

    const startTime = performance.now();

    // Extract chunk texts for batch embedding
    const chunkTexts = chunks.map(chunk => chunk.text);

    // Embed all chunks in passage mode (E5 asymmetric retrieval)
    // The BrowserEmbeddings class will automatically prepend "passage: " prefix
    const embeddings = await embedder.embed(chunkTexts, {
        mode: 'passage',        // E5 asymmetric retrieval mode
        showProgress: true,
        useCache: true,
        maxLength: 512,         // Chunks are already ~512 chars
        onProgress: (embProgress) => {
            if (!onProgress) return;
            onProgress({
                stage: 'chunk_embedding',
                progress: embProgress.progress,
                batch: embProgress.batch,
                totalBatches: embProgress.totalBatches,
                message: `Chunk embeddings batch ${embProgress.batch}/${embProgress.totalBatches} (${Math.round((embProgress.progress || 0) * 100)}%)`
            });
        }
    });

    // Validate all chunks got valid embeddings
    if (!embeddings || embeddings.length !== chunkTexts.length) {
        throw new Error(`Chunk embedding count mismatch: expected ${chunkTexts.length}, got ${embeddings?.length || 0}`);
    }

    // Combine chunks with their embeddings
    const embeddedChunks = chunks.map((chunk, index) => ({
        docId: chunk.parent_id,
        chunkId: chunk.chunk_id,
        chunkIndex: chunk.position,
        text: chunk.text,
        embedding: embeddings[index],
        metadata: chunk.metadata
    }));

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    return embeddedChunks;
}

/**
 * Build a vector search index for chunk embeddings
 *
 * @param {Array<EmbeddedChunkRecord>} embeddedChunks - Chunks with embeddings
 * @param {Object} vectorSearchClass - BrowserVectorSearch constructor
 * @returns {Object} Configured vector search index
 */
export function buildChunkIndex(embeddedChunks, vectorSearchClass) {
    const dim = embeddedChunks[0]?.embedding?.length || 384;
    const chunkIndex = new vectorSearchClass(dim);

    // Extract data for index building
    const embeddings = embeddedChunks.map(c => c.embedding);
    const chunkIds = embeddedChunks.map(c => c.chunkId);
    const chunkMetadata = embeddedChunks.map(c => ({
        text: c.text,
        doc_id: c.chunkId,
        parent_id: c.docId,
        chunk_index: c.chunkIndex,
        ...c.metadata
    }));

    // Build flat vector index
    chunkIndex.buildIndex(embeddings, chunkIds, chunkMetadata);

    return chunkIndex;
}

/**
 * Group chunk search results by parent document
 *
 * @param {Array} chunkResults - Raw chunk search results from vector index
 * @param {number} topK - Number of parent documents to return
 * @returns {Array} Grouped results sorted by relevance
 */
export function groupChunksByParent(chunkResults, topK = 5) {
    const parentGroups = new Map();

    // Group chunks by parent document
    chunkResults.forEach(chunk => {
        const parentId = chunk.metadata?.parent_id || chunk.parent_id;

        if (!parentGroups.has(parentId)) {
            parentGroups.set(parentId, {
                parent_id: parentId,
                chunks: [],
                maxScore: 0,
                avgScore: 0
            });
        }

        const group = parentGroups.get(parentId);
        group.chunks.push(chunk);
        group.maxScore = Math.max(group.maxScore, chunk.score);
    });

    // Calculate average scores and sort chunks within each group
    parentGroups.forEach(group => {
        const totalScore = group.chunks.reduce((sum, c) => sum + c.score, 0);
        group.avgScore = totalScore / group.chunks.length;

        // Sort chunks by position (reading order)
        group.chunks.sort((a, b) => {
            const posA = a.metadata?.chunk_index ?? a.chunk_index ?? 0;
            const posB = b.metadata?.chunk_index ?? b.chunk_index ?? 0;
            return posA - posB;
        });
    });

    // Sort parent groups by max score (best chunk wins)
    const sortedGroups = Array.from(parentGroups.values())
        .sort((a, b) => b.maxScore - a.maxScore)
        .slice(0, topK);

    return sortedGroups;
}
