/**
 * Tier 3 Chunking Layer using ChonkieJS
 * Implements RAG-optimized document chunking with character-based splitting
 */

import { TokenChunker } from 'https://esm.run/@chonkiejs/core';

/**
 * ChunkRecord structure
 * @typedef {Object} ChunkRecord
 * @property {string} docId - Parent document ID
 * @property {number} chunkIndex - Zero-based chunk index
 * @property {string} text - Chunk text content
 */

/**
 * Chunk a single document into overlapping passages for RAG retrieval
 *
 * @param {string} docId - Parent document identifier
 * @param {string} text - Full document text to chunk
 * @param {Object} options - Chunking configuration
 * @param {number} options.chunkSize - Target chunk size in characters (default: 512)
 * @param {number} options.chunkOverlap - Overlap between chunks in characters (default: 128)
 * @param {number} options.minChunkSize - Minimum characters per chunk (default: 50)
 * @returns {Promise<ChunkRecord[]>} Array of chunk records
 */
export async function chunkDocument(docId, text, options = {}) {
    const {
        chunkSize = 512,
        chunkOverlap = 128,
        minChunkSize = 50
    } = options;

    // Validate inputs
    if (!docId || typeof docId !== 'string') {
        throw new Error('docId must be a non-empty string');
    }

    if (!text || typeof text !== 'string') {
        console.warn(`⚠️ Empty text for document ${docId}, returning empty chunks`);
        return [];
    }

    // Short documents don't need chunking
    if (text.length <= chunkSize * 1.2) {
        return [{
            docId: docId,
            chunkIndex: 0,
            text: text.trim()
        }];
    }

    try {
        // Create TokenChunker with character-based tokenization
        // TokenChunker supports chunkOverlap, RecursiveChunker does not
        const chunker = await TokenChunker.create({
            chunkSize: chunkSize,
            chunkOverlap: chunkOverlap,
            minCharactersPerChunk: minChunkSize
        });

        // Chunk the document
        const chonkieChunks = await chunker.chunk(text);

        // Transform Chonkie output to our ChunkRecord format
        const chunkRecords = chonkieChunks
            .map((chunk, index) => ({
                docId: docId,
                chunkIndex: index,
                text: chunk.text.trim()
            }))
            .filter(chunk => chunk.text.length >= minChunkSize); // Filter tiny chunks

        return chunkRecords;

    } catch (error) {
        console.error(`❌ Chunking failed for document ${docId}:`, error);
        // Fallback: return entire document as single chunk
        return [{
            docId: docId,
            chunkIndex: 0,
            text: text.trim()
        }];
    }
}

/**
 * Batch chunk multiple documents
 *
 * @param {Array<{id: string, text: string}>} documents - Array of documents to chunk
 * @param {Object} options - Chunking configuration (passed to chunkDocument)
 * @returns {Promise<{chunks: ChunkRecord[], chunkToParentMap: Object}>} Chunked results with parent mapping
 */
export async function chunkDocuments(documents, options = {}) {
    const startTime = performance.now();

    const allChunks = [];
    const chunkToParentMap = {};

    // Process documents in parallel batches for performance
    const BATCH_SIZE = 50;
    for (let i = 0; i < documents.length; i += BATCH_SIZE) {
        const batch = documents.slice(i, i + BATCH_SIZE);
        const batchResults = await Promise.all(
            batch.map(doc => chunkDocument(doc.id, doc.text, options))
        );

        batchResults.forEach((docChunks, batchIdx) => {
            const doc = batch[batchIdx];

            // Add chunks to global array and build parent mapping
            docChunks.forEach(chunk => {
                const chunkId = `${doc.id}_chunk_${chunk.chunkIndex}`;

                allChunks.push({
                    chunk_id: chunkId,
                    parent_id: doc.id,
                    text: chunk.text,
                    position: chunk.chunkIndex,
                    totalChunks: docChunks.length,
                    metadata: {
                        ...doc.metadata,
                        parent_id: doc.id,
                        chunk_position: `${chunk.chunkIndex + 1}/${docChunks.length}`,
                        chunk_chars: chunk.text.length
                    }
                });

                chunkToParentMap[chunkId] = doc.id;
            });
        });
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    return {
        chunks: allChunks,
        chunkToParentMap: chunkToParentMap
    };
}
