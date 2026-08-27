function cleanLine(value) {
    return String(value ?? '').replace(/\r?\n/g, ' ').trim();
}

export function buildChatMarkdown(conversationHistory, { exportedAt = new Date().toISOString() } = {}) {
    const entries = Array.isArray(conversationHistory) ? conversationHistory : [];
    const sections = ['# Vectoria Ask history', '', 'Export format: 3', `Exported: ${exportedAt}`, ''];
    entries.forEach((entry, index) => {
        const retrieval = entry?.metadata?.retrieval || entry?.metadata || {};
        const resolvedRoute = entry?.route?.resolved || entry?.metadata?.route?.resolved || 'documents';
        const modeLabel = resolvedRoute === 'helper' ? 'Helper'
            : resolvedRoute === 'conversation' ? 'Conversation'
            : resolvedRoute === 'handoff' ? 'Handoff' : 'Documents';
        sections.push(
            `## Turn ${index + 1}`,
            '',
            `Mode: ${modeLabel}${entry?.metadata?.hydeUsed ? ` · HyDE${entry?.metadata?.hydeEdited ? ' (edited)' : ''}` : ''}`,
            '',
            '### Question',
            '',
            String(entry?.query || ''),
            '',
            '### Vectoria',
            '',
            String(entry?.answer || '')
        );
        const diagnostics = [
            entry?.metadata?.resolvedQuery ? `Resolved query: ${cleanLine(entry.metadata.resolvedQuery)}` : '',
            entry?.metadata?.topicAnchorQuestion ? `Topic anchor: ${cleanLine(entry.metadata.topicAnchorQuestion)}` : '',
            entry?.metadata?.model ? `Model: ${cleanLine(entry.metadata.model)}` : '',
            entry?.metadata?.groundingState ? `Grounding: ${cleanLine(entry.metadata.groundingState)}` : '',
            Number.isFinite(Number(entry?.metadata?.removedClaimCount))
                ? `Rejected claims: ${Number(entry.metadata.removedClaimCount)}` : '',
            Number.isFinite(Number(entry?.metadata?.repairCount))
                ? `Repair passes: ${Number(entry.metadata.repairCount)}` : '',
            entry?.metadata?.filter?.applied
                ? `Explicit filters: ${cleanLine(JSON.stringify(entry.metadata.filter.filters || entry.metadata.filter))}` : '',
            retrieval?.post_fusion || retrieval?.postFusion
                ? `Retrieval post-fusion: ${cleanLine(retrieval.post_fusion || retrieval.postFusion)}` : '',
            retrieval?.reranker_applied
                ? `Reranker: ${cleanLine(retrieval.reranker_model || 'multilingual MiniLM')} · ${Number(retrieval.reranker_candidates) || 0} candidates · ${Number(retrieval.reranker_latency_ms) || 0} ms` : '',
            retrieval?.reranker_fallback_reason
                ? `Reranker fallback: ${cleanLine(retrieval.reranker_fallback_reason)}` : ''
        ].filter(Boolean);
        if (diagnostics.length) sections.push('', '#### Diagnostics', '', ...diagnostics.map(value => `- ${value}`));
        if (entry?.sources?.length) {
            sections.push('', '#### Sources', '');
            entry.sources.forEach((source, sourceIndex) => {
                const number = source.sourceNumber || sourceIndex + 1;
                const metadata = source.metadata || {};
                const title = metadata.title || metadata.name || metadata.filename || metadata.file_name || `Document ${number}`;
                const identifier = source.docId ?? source.id ?? source.documentIndex ?? source.index;
                sections.push(`- [Doc ${number}] ${cleanLine(title)}${identifier !== undefined && identifier !== null ? ` — ID: ${cleanLine(identifier)}` : ''}`);
            });
        }
        sections.push('');
    });
    return `${sections.join('\n').trim()}\n`;
}

export function buildChatCsvRows(conversationHistory) {
    const entries = Array.isArray(conversationHistory) ? conversationHistory : [];
    return entries.map(entry => {
        const retrieval = entry?.metadata?.retrieval || entry?.metadata || {};
        return ({
        timestamp: entry?.timestamp || '',
        query: entry?.query || '',
        answer: entry?.answer || '',
        num_sources: entry?.sources?.length || 0,
        source_ids: entry?.sources?.map(source => source.docId ?? source.id ?? source.documentIndex ?? source.index).join('|') || '',
        model: entry?.metadata?.model || '',
        temperature: entry?.metadata?.temperature ?? '',
        requested_mode: entry?.route?.requested || entry?.metadata?.route?.requested || '',
        resolved_route: entry?.route?.resolved || entry?.metadata?.route?.resolved || '',
        retrieval_performed: entry?.metadata?.retrievalPerformed ?? '',
        hyde_used: entry?.metadata?.hydeUsed ?? false,
        hyde_edited: entry?.metadata?.hydeEdited ?? false,
        prompt_tokens: entry?.metadata?.actualUsage?.promptTokens ?? '',
        completion_tokens: entry?.metadata?.actualUsage?.completionTokens ?? '',
        resolved_query: entry?.metadata?.resolvedQuery || '',
        topic_anchor: entry?.metadata?.topicAnchorQuestion || '',
        explicit_filters: entry?.metadata?.filter?.applied
            ? JSON.stringify(entry.metadata.filter.filters || entry.metadata.filter) : '',
        retrieval_diagnostics: JSON.stringify({
            candidate_parent_count: retrieval?.candidate_parent_count ?? null,
            selected_count: retrieval?.selected_count ?? null,
            dropped_count: retrieval?.dropped_count ?? null,
            post_fusion: retrieval?.post_fusion || retrieval?.postFusion || null,
            reranker_applied: retrieval?.reranker_applied ?? false,
            reranker_model: retrieval?.reranker_model || null,
            reranker_candidates: retrieval?.reranker_candidates ?? 0,
            reranker_latency_ms: retrieval?.reranker_latency_ms ?? 0,
            reranker_fallback_reason: retrieval?.reranker_fallback_reason || null
        }),
        reranker_applied: retrieval?.reranker_applied ?? false,
        reranker_model: retrieval?.reranker_model || '',
        reranker_candidates: retrieval?.reranker_candidates ?? 0,
        reranker_latency_ms: retrieval?.reranker_latency_ms ?? 0,
        reranker_fallback_reason: retrieval?.reranker_fallback_reason || '',
        grounding_state: entry?.metadata?.groundingState || '',
        rejected_claim_count: entry?.metadata?.removedClaimCount ?? '',
        repair_count: entry?.metadata?.repairCount ?? '',
        finish_reason: entry?.metadata?.finishReason || ''
        });
    });
}
