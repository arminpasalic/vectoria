/** Pure post-fusion coverage scoring and diversity selection. */

function normalize(value) {
    return String(value ?? '').normalize('NFKC').toLocaleLowerCase().replace(/\s+/gu, ' ').trim();
}

function tokens(value) {
    return normalize(value).match(/[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu) || [];
}

function meaningfulQueryTerms(query) {
    return [...new Set(tokens(query).filter(token => token.length > 2 || /\d/u.test(token)))];
}

function resultText(result) {
    const chunks = Array.isArray(result?.chunks)
        ? result.chunks.map(chunk => chunk?.text || chunk?.metadata?.text || '').filter(Boolean)
        : [];
    return chunks.length ? chunks.join(' ') : String(result?.text || result?.metadata?.text || '');
}

export function queryCoverageScore(query, result) {
    const queryText = normalize(query);
    const text = normalize(resultText(result));
    if (!queryText || !text) return 0;
    const terms = meaningfulQueryTerms(queryText);
    const textTokens = new Set(tokens(text));
    const termCoverage = terms.length ? terms.filter(term => textTokens.has(term)).length / terms.length : 0;
    const protectedTerms = queryText.match(/(?:[$€£¥]\s*)?\b\d[\d.,:/-]*(?:\s*%)?|\b(?=[\p{L}\p{N}_-]*\p{L})(?=[\p{L}\p{N}_-]*\p{N})[\p{L}\p{N}_-]{3,}\b/gu) || [];
    const protectedCoverage = protectedTerms.length
        ? protectedTerms.filter(term => text.includes(normalize(term))).length / protectedTerms.length
        : termCoverage;
    const phrases = queryText.match(/["“][^"”]{2,}["”]/gu) || [];
    const phraseCoverage = phrases.length
        ? phrases.filter(phrase => text.includes(normalize(phrase).replace(/^['"“]|['"”]$/gu, ''))).length / phrases.length
        : termCoverage;
    return Math.max(0, Math.min(1, (termCoverage * 0.6) + (protectedCoverage * 0.25) + (phraseCoverage * 0.15)));
}

export function lexicalSimilarity(left, right) {
    const a = new Set(tokens(resultText(left)).filter(token => token.length > 2));
    const b = new Set(tokens(resultText(right)).filter(token => token.length > 2));
    if (!a.size || !b.size) return 0;
    let intersection = 0;
    for (const token of a) if (b.has(token)) intersection++;
    return intersection / (a.size + b.size - intersection);
}

function normalizedRankScore(result, index, results) {
    const numeric = Number(result?.score);
    const scores = results.map(entry => Number(entry?.score)).filter(Number.isFinite);
    if (Number.isFinite(numeric) && scores.length > 1) {
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        if (max > min) return (numeric - min) / (max - min);
    }
    return 1 - (index / Math.max(1, results.length));
}

export function rerankAndDiversify(results, query, {
    maxResults = 5,
    relevanceWeight = 0.82,
    weakTailRatio = 0.35,
    duplicateThreshold = 0.9
} = {}) {
    const source = Array.isArray(results) ? results.filter(Boolean) : [];
    if (!source.length) return [];
    const scored = source.map((result, index) => {
        const coverage = queryCoverageScore(query, result);
        const rankScore = normalizedRankScore(result, index, source);
        const relevance = (rankScore * 0.75) + (coverage * 0.25);
        return { result, coverage, rankScore, relevance, originalRank: index };
    }).sort((a, b) => b.relevance - a.relevance || a.originalRank - b.originalRank);

    const strongest = scored[0]?.relevance || 0;
    const eligible = scored.filter((entry, index) => index === 0
        || entry.coverage > 0
        || entry.relevance >= strongest * weakTailRatio);
    const selected = [];
    while (eligible.length && selected.length < Math.max(1, maxResults)) {
        let bestIndex = -1;
        let bestScore = -Infinity;
        eligible.forEach((entry, index) => {
            const maxSimilarity = selected.length
                ? Math.max(...selected.map(chosen => lexicalSimilarity(entry.result, chosen.result)))
                : 0;
            if (maxSimilarity >= duplicateThreshold && selected.length) return;
            const mmr = (relevanceWeight * entry.relevance) - ((1 - relevanceWeight) * maxSimilarity);
            if (mmr > bestScore) {
                bestScore = mmr;
                bestIndex = index;
            }
        });
        if (bestIndex < 0) break;
        const [chosen] = eligible.splice(bestIndex, 1);
        if (!chosen) break;
        chosen.result.retrieval_quality = {
            coverage: chosen.coverage,
            normalized_rank: chosen.rankScore,
            relevance: chosen.relevance,
            original_rank: chosen.originalRank
        };
        selected.push(chosen);
    }
    return selected.map(entry => entry.result);
}
