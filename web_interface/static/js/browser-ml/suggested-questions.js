/**
 * Data-grounded starter questions for the Ask empty state.
 *
 * A random sample of the dataset is shown to the local model, which drafts two
 * short questions. Sampling is deliberately random so reprocessing the same
 * data can surface different questions.
 */

export const SUGGESTION_SAMPLE_SIZE = 8;
export const SUGGESTION_EXCERPT_CHARS = 240;
export const MIN_DOCUMENT_CHARS = 40;
export const MAX_QUESTION_CHARS = 90;

export const FALLBACK_SUGGESTIONS = [
    { label: 'What are the main themes?', prompt: 'What are the main themes in these documents?' },
    { label: 'Summarize key findings', prompt: 'Summarize the most important findings and cite the sources.' }
];

const QUESTION_OPENERS = /^(what|which|how|why|when|where|who|whose|whom|does|do|did|is|are|was|were|can|could|should|would|will|has|have|had|list|name|compare|summar)/i;
const PREAMBLE_OPENERS = /^(here|these|below|sure|okay|certainly|answer|question)\b/i;

/**
 * Draw a random sample of usable documents. Documents too short to carry
 * subject matter are skipped so the model sees real material.
 */
export function sampleDocumentsForSuggestions(documents, size = SUGGESTION_SAMPLE_SIZE) {
    const usable = (Array.isArray(documents) ? documents : [])
        .filter(document => String(document?.text || '').trim().length > MIN_DOCUMENT_CHARS);
    if (!usable.length) return [];
    const pool = usable.slice();
    const picked = [];
    while (pool.length && picked.length < size) {
        picked.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
    }
    return picked;
}

export function buildSuggestionPrompt(sampled) {
    const excerpts = sampled.map((document, index) => {
        const text = String(document?.text || '').replace(/\s+/g, ' ').trim();
        return `[${index + 1}] ${text.slice(0, SUGGESTION_EXCERPT_CHARS)}`;
    }).join('\n');
    return `Below are excerpts sampled from a document collection.

${excerpts}

Write exactly 2 short questions a reader could ask about this collection. Rules:
- Each question must be answerable from material like these excerpts.
- Maximum 8 words each. No preamble, no numbering, no quotes.
- Use the collection's own subject matter and terminology.
- Return only the two questions, one per line.`;
}

/**
 * Pull two usable questions out of a small model's response, which frequently
 * arrives numbered, quoted, or wrapped in a "Here are two questions:" preamble.
 */
export function parseSuggestionResponse(text) {
    const seen = new Set();
    return String(text || '')
        .split('\n')
        .map(line => line
            .replace(/^\s*(?:[-*\d.)\]]+\s*)+/, '')
            .replace(/^["'`]+|["'`]+$/g, '')
            .trim())
        .filter(line => line.length > 6 && line.length <= MAX_QUESTION_CHARS && line.includes(' '))
        .filter(line => line.endsWith('?') || QUESTION_OPENERS.test(line))
        .filter(line => !PREAMBLE_OPENERS.test(line))
        .map(line => (line.endsWith('?') ? line : `${line}?`))
        .filter(line => {
            const key = line.toLowerCase();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        })
        .slice(0, 2);
}
