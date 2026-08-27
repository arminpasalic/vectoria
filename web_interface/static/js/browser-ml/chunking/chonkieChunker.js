/**
 * Tier 3 Chunking Layer using ChonkieJS
 * Implements all seven local ChonkieJS chunking strategies for RAG retrieval.
 */

// Pin the browser build to a tested release. An unversioned CDN import can
// introduce breaking changes between deployments.
import {
    CodeChunker,
    FastChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TableChunker,
    TokenChunker
} from 'https://esm.run/@chonkiejs/core@0.0.11';
import { createBrowserCodeBackend, detectCodeLanguage } from './browserCodeBackend.js';

const SUPPORTED_STRATEGIES = new Set([
    'token', 'recursive', 'sentence', 'semantic', 'code', 'table', 'fast'
]);

function clampNumber(value, fallback, min, max) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(min, Math.min(max, numeric));
}

function positiveInteger(value, fallback, max = Number.MAX_SAFE_INTEGER) {
    return Math.round(clampNumber(value, fallback, 1, max));
}

function wholeDocumentRecord(docId, text, strategy) {
    return [{
        docId,
        chunkIndex: 0,
        // Leading indentation is meaningful when a short code document or a
        // code-chunking fallback is returned whole.
        text: strategy === 'code' ? text : text.trim()
    }];
}

function coalesceSmallCodeChunks(chunks, minChunkSize) {
    const records = [];
    let pendingText = '';

    for (const chunk of chunks) {
        pendingText += chunk.text;
        if (pendingText.length >= minChunkSize) {
            records.push({ ...chunk, text: pendingText });
            pendingText = '';
        }
    }
    if (pendingText) {
        if (records.length) records[records.length - 1].text += pendingText;
        else records.push({ docId: chunks[0]?.docId, text: pendingText });
    }
    return records.map((chunk, index) => ({ ...chunk, chunkIndex: index }));
}

async function createChunker(options) {
    const strategy = SUPPORTED_STRATEGIES.has(options.strategy) ? options.strategy : 'token';
    const safeOverlap = Math.max(0, Math.min(options.chunkOverlap, options.chunkSize - 1));

    switch (strategy) {
        case 'recursive':
            return RecursiveChunker.create({
                tokenizer: 'character',
                chunkSize: options.chunkSize,
                minCharactersPerChunk: options.minChunkSize
            });
        case 'sentence':
            return SentenceChunker.create({
                tokenizer: 'character',
                chunkSize: options.chunkSize,
                chunkOverlap: safeOverlap,
                minSentencesPerChunk: options.sentenceMinSentences,
                minCharactersPerSentence: options.sentenceMinCharacters,
                delim: options.sentenceDelimiters,
                includeDelim: options.sentenceIncludeDelimiter
            });
        case 'semantic': {
            if (typeof options.semanticEmbeddings !== 'function') {
                throw new Error('SemanticChunker requires Vectoria\'s local embedding model');
            }
            let filterWindow = positiveInteger(options.semanticFilterWindow, 5, 99);
            if (filterWindow < 3) filterWindow = 3;
            if (filterWindow % 2 === 0) filterWindow += 1;
            const filterPolyorder = Math.min(
                positiveInteger(options.semanticFilterPolyorder, 3, 20),
                filterWindow - 1
            );
            return SemanticChunker.create({
                embeddings: options.semanticEmbeddings,
                tokenizer: 'character',
                chunkSize: options.chunkSize,
                threshold: clampNumber(options.semanticThreshold, 0.8, 0.01, 0.99),
                similarityWindow: positiveInteger(options.semanticSimilarityWindow, 3, 20),
                minSentencesPerChunk: positiveInteger(options.sentenceMinSentences, 1, 20),
                minCharactersPerSentence: positiveInteger(options.sentenceMinCharacters, 12, 500),
                delimiters: options.sentenceDelimiters,
                includeDelim: options.sentenceIncludeDelimiter,
                filterWindow,
                filterPolyorder,
                filterTolerance: clampNumber(options.semanticFilterTolerance, 0.2, 0.01, 0.99),
                skipWindow: Math.round(clampNumber(options.semanticSkipWindow, 0, 0, 20))
            });
        }
        case 'code': {
            const language = detectCodeLanguage(options.text, options.metadata, options.codeLanguage);
            const backend = await createBrowserCodeBackend(language);
            return CodeChunker.create({
                tokenizer: 'character',
                chunkSize: options.chunkSize,
                language,
                backend
            });
        }
        case 'table': {
            const tableMode = options.tableMode === 'character' ? 'character' : 'row';
            return TableChunker.create({
                tokenizer: tableMode,
                chunkSize: tableMode === 'row'
                    ? positiveInteger(options.tableRowsPerChunk, 10, 1000)
                    : options.chunkSize
            });
        }
        case 'fast':
            return FastChunker.create({
                chunkSize: options.chunkSize,
                delimiters: options.fastDelimiters,
                prefix: options.fastPrefix,
                consecutive: options.fastConsecutive,
                forwardFallback: options.fastForwardFallback
            });
        default:
            return TokenChunker.create({
                tokenizer: 'character',
                chunkSize: options.chunkSize,
                chunkOverlap: safeOverlap
            });
    }
}

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
 * @param {Object} metadata - Parent document metadata (used for code-language detection)
 * @returns {Promise<ChunkRecord[]>} Array of chunk records
 */
export async function chunkDocument(docId, text, options = {}, metadata = {}) {
    const {
        strategy = 'token',
        chunkSize = 512,
        chunkOverlap = 128,
        minChunkSize = 50,
        sentenceMinSentences = 1,
        sentenceMinCharacters = 12,
        sentenceDelimiters = ['. ', '! ', '? ', '\n'],
        sentenceIncludeDelimiter = 'prev',
        semanticEmbeddings = null,
        semanticThreshold = 0.8,
        semanticSimilarityWindow = 3,
        semanticFilterWindow = 5,
        semanticFilterPolyorder = 3,
        semanticFilterTolerance = 0.2,
        semanticSkipWindow = 0,
        codeLanguage = 'auto',
        tableMode = 'row',
        tableRowsPerChunk = 10,
        fastDelimiters = '\n.?',
        fastPrefix = false,
        fastConsecutive = false,
        fastForwardFallback = true
    } = options;

    // Validate inputs
    if (!docId || typeof docId !== 'string') {
        throw new Error('docId must be a non-empty string');
    }

    if (!text || typeof text !== 'string') {
        console.warn(`⚠️ Empty text for document ${docId}, returning empty chunks`);
        return [];
    }

    const safeStrategy = SUPPORTED_STRATEGIES.has(strategy) ? strategy : 'token';

    // Fast counts UTF-8 bytes and Table can count rows, so character length is
    // not a valid early-exit signal for those strategies.
    if (!['fast', 'table'].includes(safeStrategy) && text.length <= chunkSize) {
        return wholeDocumentRecord(docId, text, safeStrategy);
    }

    try {
        const chunker = await createChunker({
            strategy: safeStrategy, chunkSize, chunkOverlap, minChunkSize,
            sentenceMinSentences, sentenceMinCharacters, sentenceDelimiters,
            sentenceIncludeDelimiter, semanticEmbeddings, semanticThreshold,
            semanticSimilarityWindow, semanticFilterWindow, semanticFilterPolyorder,
            semanticFilterTolerance, semanticSkipWindow, codeLanguage, tableMode,
            tableRowsPerChunk, fastDelimiters, fastPrefix, fastConsecutive,
            fastForwardFallback, text, metadata
        });

        // Chunk the document
        const chonkieChunks = await chunker.chunk(text);

        if (!Array.isArray(chonkieChunks) || chonkieChunks.length === 0) {
            console.warn(`⚠️ ${safeStrategy} chunking produced no chunks for ${docId}; keeping the document whole`);
            return wholeDocumentRecord(docId, text, safeStrategy);
        }

        // Transform Chonkie output to our ChunkRecord format
        const transformedChunks = chonkieChunks
            .map((chunk, index) => ({
                docId: docId,
                chunkIndex: index,
                text: safeStrategy === 'code' ? String(chunk.text || '') : chunk.text.trim()
            }))
            .filter(chunk => chunk.text.length > 0);
        const chunkRecords = safeStrategy === 'code'
            ? coalesceSmallCodeChunks(transformedChunks, minChunkSize)
            : safeStrategy === 'table'
                ? transformedChunks // Never discard table rows solely because their text is short.
                : transformedChunks.filter(chunk => chunk.text.length >= minChunkSize);

        return chunkRecords.length
            ? chunkRecords
            : wholeDocumentRecord(docId, text, safeStrategy);

    } catch (error) {
        console.error(`❌ Chunking failed for document ${docId}:`, error);
        // Fallback: return entire document as single chunk
        return wholeDocumentRecord(docId, text, safeStrategy);
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
    // Semantic chunking invokes the local embedding worker for sentence
    // windows. Keep it sequential so multiple documents do not compete for the
    // same worker or initialize Chonkie's WASM runtime concurrently.
    const BATCH_SIZE = options.strategy === 'semantic' ? 1 : 50;
    for (let i = 0; i < documents.length; i += BATCH_SIZE) {
        const batch = documents.slice(i, i + BATCH_SIZE);
        const batchResults = await Promise.all(
            batch.map(doc => chunkDocument(doc.id, doc.text, options, doc.metadata || {}))
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
