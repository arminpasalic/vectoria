// Enhanced Fast Lightweight Search System for RAG-Vectoria
// Provides instant, lightweight text search with highlighting and scoring
// This is now the DEFAULT search mode for better UX

class FastSearch {
    constructor(data = []) {
        this.data = data;
        this.searchIndex = new Map();
        this.wordIndex = new Map(); // Word-to-documents index
        this.lastQuery = '';
        this.lastResults = [];
        this.isReady = false;
        this.isIndexing = false;

        // Enhanced search configuration
        this.config = {
            minQueryLength: 1,
            maxResults: Infinity, // Unlimited results
            fuzzyThreshold: 0.8,
            highlightClass: 'fast-highlight',
            boostExactMatch: 50,
            boostStartsWith: 30,
            boostWordBoundary: 20,
            caseSensitive: false,
            stemming: false // Simple stemming disabled for performance
        };

        if (data && data.length > 0) {
            this.buildSearchIndex();
        }
    }

    // Build an enhanced search index for instant text matching
    // Optimized: Reduced memory footprint by storing only index references
    buildSearchIndex() {
        if (this.isIndexing) return;

        // console.log(' Building enhanced fast search index...');
        this.isIndexing = true;
        this.searchIndex.clear();
        this.wordIndex.clear();

        try {
            this.data.forEach((item, index) => {
                const text = this.extractSearchableText(item);
                const normalizedText = text.toLowerCase().trim();
                const words = normalizedText.split(/\s+/);

                // Store minimal metadata - compute text on-demand from this.data
                this.searchIndex.set(index, {
                    index: index,
                    // item reference removed - access via this.data[index]
                    // text removed - compute via normalization when needed
                    // originalText removed - access via this.data[index]
                    wordCount: words.length,
                    firstWords: words.slice(0, 5).join(' '), // Keep for quick phrase boosting
                    textLength: normalizedText.length // For quick access
                });

                // Build word index for fast lookup
                const uniqueWords = this._extractWords(normalizedText);
                uniqueWords.forEach(word => {
                    if (word.length >= 2) { // Include 2-letter words
                        if (!this.wordIndex.has(word)) {
                            this.wordIndex.set(word, new Set());
                        }
                        this.wordIndex.get(word).add(index);
                    }
                });
            });

            this.isReady = true;
            // console.log(`✅ Enhanced search index built: ${this.searchIndex.size} documents, ${this.wordIndex.size} unique words`);
        } catch (error) {
            console.error('❌ Failed to build search index:', error);
            this.isReady = false;
        } finally {
            this.isIndexing = false;
        }
    }

    // Helper: Get normalized text for a document (on-demand)
    _getNormalizedText(docIndex) {
        const item = this.data[docIndex];
        if (!item) return '';
        return this.extractSearchableText(item).toLowerCase().trim();
    }

    // Helper: Get original text for a document (on-demand)
    _getOriginalText(docIndex) {
        const item = this.data[docIndex];
        if (!item) return '';
        return this.extractSearchableText(item);
    }

    // Extract all searchable text from data item
    extractSearchableText(item) {
        const textParts = [];

        // Primary text field (highest priority)
        if (item.text && typeof item.text === 'string') {
            textParts.push(item.text);
        }

        // Include other meaningful string fields
        const searchableFields = ['title', 'content', 'description', 'summary', 'name'];
        searchableFields.forEach(field => {
            if (item[field] && typeof item[field] === 'string' && item[field].length > 0) {
                textParts.push(item[field]);
            }
        });

        // Include any other string fields (but lower priority)
        Object.keys(item).forEach(key => {
            if (!['text', 'title', 'content', 'description', 'summary', 'name', 'x', 'y', 'index', 'cluster', 'doc_id', 'chunk_id'].includes(key)) {
                const value = item[key];
                if (typeof value === 'string' && value.length > 0 && value.length < 500) { // Avoid very long fields
                    textParts.push(value);
                }
            }
        });

        return textParts.join(' ').trim();
    }

    // Extract and normalize words for indexing
    _extractWords(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s'-]/g, ' ') // Keep letters, numbers, apostrophes, hyphens
            .split(/\s+/)
            .filter(word => word.length > 0)
            .map(word => word.replace(/^[-']+|[-']+$/g, '')); // Trim leading/trailing punctuation
    }

    // Enhanced fast text search with improved ranking and performance
    search(query, options = {}) {
        if (!query || query.length < this.config.minQueryLength) {
            return { results: [], query: query, totalMatches: 0, searchTime: 0 };
        }

        if (!this.isReady) {
            console.warn('⚠️ Search index not ready');
            return { results: [], query: query, totalMatches: 0, searchTime: 0, error: 'Index not ready' };
        }

        const startTime = performance.now();

        // Check if this is a boolean query
        const isBooleanQuery = this._isBooleanQuery(query);
        if (isBooleanQuery) {
            return this._performBooleanSearch(query, options, startTime);
        }

        // Cache check
        if (query === this.lastQuery && this.lastResults) {
            return {
                results: this.lastResults,
                query: query,
                totalMatches: this.lastResults.length,
                searchTime: performance.now() - startTime,
                cached: true
            };
        }

        const maxResults = options.maxResults || this.config.maxResults;
        const fuzzy = options.fuzzy !== false;

        try {
            // Normalize and parse query
            const normalizedQuery = query.toLowerCase().trim();
            const queryWords = this._extractWords(normalizedQuery);

            if (queryWords.length === 0) {
                return { results: [], query: query, totalMatches: 0, searchTime: performance.now() - startTime };
            }

            // Find candidate documents using word index
            const candidateIds = this._findCandidateDocuments(queryWords, fuzzy);

            // Score and rank candidates
            const scoredResults = this._scoreDocuments(candidateIds, queryWords, normalizedQuery);

            // Sort by score (no limit - return all results)
            const sortedResults = scoredResults
                .sort((a, b) => b.score - a.score);

            // Prepare final results with highlighting
            const results = sortedResults.map((result, rank) => ({
                index: result.index,
                item: result.item,
                originalText: result.originalText,
                text: result.text,
                score: result.score,
                rank: rank + 1,
                query: query,
                matchedText: this.highlightMatches(result.originalText, queryWords),
                searchType: 'fast',
                matchCount: result.matchCount,
                cluster: result.item.cluster,
                coordinates: [result.item.x, result.item.y],
                doc_id: result.item.doc_id,
                chunk_id: result.item.chunk_id
            }));

            const searchTime = performance.now() - startTime;

            // Cache results
            this.lastQuery = query;
            this.lastResults = results;

            // console.log(`⚡ Enhanced fast search: "${query}" → ${results.length} results (${searchTime.toFixed(1)}ms)`);

            return {
                results: results,
                query: query,
                totalMatches: results.length,
                searchTime: searchTime
            };

        } catch (error) {
            console.error('❌ Search error:', error);
            return { results: [], query: query, totalMatches: 0, searchTime: performance.now() - startTime, error: error.message };
        }
    }

    // Find candidate documents that might match the query
    _findCandidateDocuments(queryWords, fuzzy = true) {
        const candidateIds = new Set();

        queryWords.forEach(word => {
            // Exact word matches
            if (this.wordIndex.has(word)) {
                this.wordIndex.get(word).forEach(id => candidateIds.add(id));
            }

            // Prefix matches for partial typing
            if (word.length >= 2) {
                for (const [indexedWord, docIds] of this.wordIndex.entries()) {
                    if (indexedWord.startsWith(word) && indexedWord !== word) {
                        docIds.forEach(id => candidateIds.add(id));
                    }
                }
            }

            // Fuzzy matches for typos
            if (fuzzy && word.length >= 3) {
                for (const [indexedWord, docIds] of this.wordIndex.entries()) {
                    if (this._isFuzzyMatch(word, indexedWord)) {
                        docIds.forEach(id => candidateIds.add(id));
                    }
                }
            }
        });

        return Array.from(candidateIds);
    }

    // Score documents based on query match quality
    _scoreDocuments(candidateIds, queryWords, fullQuery) {
        const results = [];

        candidateIds.forEach(id => {
            const doc = this.searchIndex.get(id);
            if (!doc) return;

            // Get text on-demand (reduced memory footprint)
            const normalizedText = this._getNormalizedText(id);

            let totalScore = 0;
            let matchCount = 0;

            // Score individual words
            queryWords.forEach(word => {
                const wordScore = this._scoreWord(word, normalizedText);
                if (wordScore > 0) {
                    totalScore += wordScore;
                    matchCount++;
                }
            });

            // Bonus for matching multiple words
            if (queryWords.length > 1 && matchCount > 1) {
                totalScore += matchCount * 10;
            }

            // Bonus for exact phrase matches
            if (fullQuery.length > 3 && normalizedText.includes(fullQuery)) {
                totalScore += this.config.boostExactMatch;
            }

            // Bonus for matches at the beginning of text
            if (doc.firstWords.includes(queryWords[0])) {
                totalScore += this.config.boostStartsWith;
            }

            // Penalty for very long documents (prefer focused matches)
            if (doc.wordCount > 100) {
                totalScore *= 0.9;
            }

            if (totalScore > 0) {
                results.push({
                    index: doc.index,
                    item: this.data[id],
                    originalText: this._getOriginalText(id),
                    text: normalizedText,
                    score: Math.round(totalScore),
                    matchCount: matchCount
                });
            }
        });

        return results;
    }

    // Score a single word match within a document
    // Optimized: Single-pass scanning instead of triple regex
    _scoreWord(word, text) {
        let score = 0;
        const escapedWord = this._escapeRegExp(word);

        // Single regex pass to find all matches with context
        const globalRegex = new RegExp(escapedWord, 'gi');
        let match;
        let firstIndex = -1;
        let exactCount = 0;
        let prefixCount = 0;
        let partialCount = 0;

        // Build word boundary regex once
        const wordBoundaryRegex = new RegExp(`\\b${escapedWord}\\b`, 'i');
        const prefixBoundaryRegex = new RegExp(`\\b${escapedWord}`, 'i');

        while ((match = globalRegex.exec(text)) !== null) {
            const matchIndex = match.index;
            const matchText = match[0];

            // Track first match position for bonus
            if (firstIndex === -1) {
                firstIndex = matchIndex;
            }

            // Get surrounding context to check boundaries
            const before = matchIndex > 0 ? text[matchIndex - 1] : ' ';
            const after = matchIndex + matchText.length < text.length ? text[matchIndex + matchText.length] : ' ';

            // Check match type based on word boundaries
            const isWordBoundaryBefore = /\W/.test(before);
            const isWordBoundaryAfter = /\W/.test(after);

            if (isWordBoundaryBefore && isWordBoundaryAfter) {
                // Exact word match (highest priority)
                exactCount++;
            } else if (isWordBoundaryBefore) {
                // Prefix match (word starts with query)
                prefixCount++;
            } else {
                // Partial match (substring within word)
                partialCount++;
            }
        }

        // Apply scoring (mutually exclusive now)
        score += exactCount * 30;      // Exact word boundaries
        score += prefixCount * 20;     // Starts word
        score += partialCount * 15;    // Substring match

        // Position bonus (earlier matches score higher)
        if (firstIndex !== -1) {
            const positionBonus = Math.max(0, 25 - Math.floor(firstIndex / 20));
            score += positionBonus;
        }

        return score;
    }

    // Improved fuzzy matching with better performance
    _isFuzzyMatch(query, target) {
        if (Math.abs(query.length - target.length) > 2) return false;
        if (query === target) return false; // Skip exact matches (already handled)
        if (target.includes(query)) return false; // Skip substring matches (already handled)

        // Use a faster approximate edit distance
        return this._approximateEditDistance(query, target) <= 1;
    }

    // Fast approximate edit distance (only for small differences)
    _approximateEditDistance(a, b) {
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;

        let distance = 0;
        let i = 0, j = 0;

        while (i < a.length && j < b.length) {
            if (a[i] === b[j]) {
                i++;
                j++;
            } else {
                distance++;
                if (distance > 1) return distance; // Early exit

                // Try different operations
                if (i + 1 < a.length && a[i + 1] === b[j]) {
                    i += 2; j++; // Deletion
                } else if (j + 1 < b.length && a[i] === b[j + 1]) {
                    i++; j += 2; // Insertion
                } else {
                    i++; j++; // Substitution
                }
            }
        }

        distance += Math.abs(a.length - i) + Math.abs(b.length - j);
        return distance;
    }

    // Highlight matching terms in text
    highlightMatches(text, queryWords) {
        if (!text || !queryWords || queryWords.length === 0) return text;

        let highlightedText = text;

        queryWords.forEach(word => {
            const regex = new RegExp(`(${this._escapeRegExp(word)})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark class="fast-highlight">$1</mark>');
        });

        return highlightedText;
    }

    // Escape special regex characters
    _escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Check if query contains boolean operators
    _isBooleanQuery(query) {
        if (!query) return false;
        const upper = query.toUpperCase();
        return upper.includes(' AND ') ||
            upper.includes(' OR ') ||
            upper.includes(' NOT ') ||
            /^NOT\s+/i.test(query) ||
            /^AND\s+/i.test(query) ||
            /^OR\s+/i.test(query) ||
            query.includes('+') ||
            query.includes('-') ||
            query.includes('"');
    }

    // Perform boolean search
    _performBooleanSearch(query, options = {}, startTime) {
        const maxResults = options.maxResults || this.config.maxResults;

        try {
            // Parse boolean query into components
            const parsedQuery = this._parseBooleanQuery(query);

            // Find documents that match the boolean logic
            const matchingDocs = this._evaluateBooleanQuery(parsedQuery);

            // Score the matching documents
            const scoredResults = matchingDocs.map((docIndex, rank) => {
                const doc = this.searchIndex.get(docIndex);
                return {
                    index: doc.index,
                    item: this.data[docIndex],
                    originalText: this._getOriginalText(docIndex),
                    text: this._getNormalizedText(docIndex),
                    score: 100 - rank, // Simple ranking based on order
                    matchCount: 1
                };
            }); // No slice - return all results

            const highlightTerms = [];
            parsedQuery.forEach(token => {
                if (['term', 'phrase', 'required'].includes(token.type)) {
                    const value = token.value.replace(/^"|"$/g, '').trim();
                    if (value) {
                        highlightTerms.push(value);
                        value.split(/\s+/).forEach(part => {
                            if (part && !highlightTerms.includes(part)) {
                                highlightTerms.push(part);
                            }
                        });
                    }
                }
            });

            // Prepare final results
            const results = scoredResults.map((result, rank) => ({
                index: result.index,
                item: result.item,
                originalText: result.originalText,
                text: result.text,
                score: result.score,
                rank: rank + 1,
                query: query,
                matchedText: highlightTerms.length > 0 ? this.highlightMatches(result.originalText, highlightTerms) : result.originalText,
                searchType: 'boolean',
                matchCount: result.matchCount,
                cluster: result.item.cluster,
                coordinates: [result.item.x, result.item.y],
                doc_id: result.item.doc_id,
                chunk_id: result.item.chunk_id
            }));

            const searchTime = performance.now() - startTime;

            return {
                results: results,
                query: query,
                totalMatches: results.length,
                searchTime: searchTime,
                searchType: 'boolean'
            };

        } catch (error) {
            console.error('❌ Boolean search error:', error);
            return { results: [], query: query, totalMatches: 0, searchTime: performance.now() - startTime, error: error.message };
        }
    }

    // Parse boolean query into structured format
    _parseBooleanQuery(query) {
        // Simple boolean query parser
        // Supports: "phrase", +required, -excluded, AND, OR, NOT

        const tokens = [];
        let current = '';
        let inQuotes = false;
        let i = 0;

        while (i < query.length) {
            const char = query[i];

            if (char === '"') {
                if (inQuotes) {
                    tokens.push({ type: 'phrase', value: current.trim() });
                    current = '';
                    inQuotes = false;
                } else {
                    if (current.trim()) {
                        tokens.push({ type: 'term', value: current.trim() });
                    }
                    current = '';
                    inQuotes = true;
                }
            } else if (!inQuotes && (char === '+' || char === '-')) {
                if (current.trim()) {
                    tokens.push({ type: 'term', value: current.trim() });
                }
                current = '';
                const modifier = char;
                i++;
                while (i < query.length && query[i] !== ' ') {
                    current += query[i];
                    i++;
                }
                if (current.trim()) {
                    tokens.push({ type: modifier === '+' ? 'required' : 'excluded', value: current.trim() });
                }
                current = '';
                i--;
            } else if (!inQuotes && char === ' ') {
                if (current.trim()) {
                    // Check for boolean operators
                    const upper = current.trim().toUpperCase();
                    if (upper === 'AND' || upper === 'OR' || upper === 'NOT') {
                        tokens.push({ type: 'operator', value: upper });
                    } else {
                        tokens.push({ type: 'term', value: current.trim() });
                    }
                }
                current = '';
            } else {
                current += char;
            }
            i++;
        }

        if (current.trim()) {
            if (inQuotes) {
                tokens.push({ type: 'phrase', value: current.trim() });
            } else {
                const upper = current.trim().toUpperCase();
                if (upper === 'AND' || upper === 'OR' || upper === 'NOT') {
                    tokens.push({ type: 'operator', value: upper });
                } else {
                    tokens.push({ type: 'term', value: current.trim() });
                }
            }
        }

        return tokens;
    }

    // Evaluate boolean query against documents
    _evaluateBooleanQuery(tokens) {
        const allDocIndices = new Set(this.searchIndex.keys());
        let result = new Set(allDocIndices);
        let currentOperation = 'AND';
        let nextSet = new Set();

        for (let i = 0; i < tokens.length; i++) {
            const token = tokens[i];

            if (token.type === 'operator') {
                currentOperation = token.value;
                continue;
            }

            // Find documents matching this token
            const matchingDocs = this._findDocsForBooleanToken(token);

            if (token.type === 'excluded') {
                // Remove excluded terms
                for (const docIndex of matchingDocs) {
                    result.delete(docIndex);
                }
            } else {
                // Apply boolean logic
                if (currentOperation === 'AND' || token.type === 'required') {
                    if (i === 0 || token.type === 'required') {
                        result = new Set(matchingDocs.filter(doc => result.has(doc)));
                    } else {
                        result = new Set([...result].filter(doc => matchingDocs.includes(doc)));
                    }
                } else if (currentOperation === 'OR') {
                    result = new Set([...result, ...matchingDocs]);
                } else if (currentOperation === 'NOT') {
                    result = new Set([...result].filter(doc => !matchingDocs.includes(doc)));
                }
            }
        }

        return Array.from(result);
    }

    // Find documents matching a boolean token
    _findDocsForBooleanToken(token) {
        const matchingDocs = [];

        for (const [docIndex, doc] of this.searchIndex.entries()) {
            let matches = false;

            if (token.type === 'phrase') {
                // Exact phrase match - get text on-demand
                const normalizedText = this._getNormalizedText(docIndex);
                matches = normalizedText.includes(token.value.toLowerCase());
            } else {
                // Term match - use WORD BOUNDARIES to avoid substring false positives
                // (e.g., "mold" should NOT match "moldy")
                const words = this._extractWords(token.value.toLowerCase());
                matches = words.some(word => {
                    // Check word index first (fast - already uses word boundaries)
                    if (this.wordIndex.has(word) && this.wordIndex.get(word).has(docIndex)) {
                        return true;
                    }
                    // Fallback to word boundary regex (not substring)
                    const normalizedText = this._getNormalizedText(docIndex);
                    const wordBoundaryRegex = new RegExp(`\\b${this._escapeRegExp(word)}\\b`, 'i');
                    return wordBoundaryRegex.test(normalizedText);
                });
            }

            if (matches) {
                matchingDocs.push(docIndex);
            }
        }

        return matchingDocs;
    }

    // Update search data with new dataset
    updateData(newData) {
        // console.log(' Updating fast search with new data:', newData ? newData.length : 0, 'items');
        this.data = newData || [];
        this.lastQuery = '';
        this.lastResults = [];
        this.isReady = false;

        if (this.data.length > 0) {
            this.buildSearchIndex();
        } else {
            this.searchIndex.clear();
            this.wordIndex.clear();
        }
    }

    // Clear search cache
    clearCache() {
        this.lastQuery = '';
        this.lastResults = [];
    }

    // Get search statistics and performance metrics
    getStats() {
        return {
            totalItems: this.data.length,
            documentsIndexed: this.searchIndex.size,
            uniqueWords: this.wordIndex.size,
            isReady: this.isReady,
            isIndexing: this.isIndexing,
            lastQuery: this.lastQuery,
            lastResultCount: this.lastResults ? this.lastResults.length : 0,
            configuredMaxResults: this.config.maxResults,
            searchMode: 'enhanced_fast'
        };
    }
}

// Enhanced search interface integration
class SearchInterface {
    constructor() {
        this.fastSearch = null;
        this.isInitialized = false;
        this.searchTimeout = null;
        this.currentMode = 'fast'; // 'fast' is now the default mode

        this.initializeInterface();
    }

    initializeInterface() {
        // Update search UI to show both options with fast as default
        this.setupSearchModeSelector();
        this.setupFastSearchHandlers();
        this.isInitialized = true;

        // console.log('✅ Enhanced search interface initialized with fast search as default');
    }

    setupSearchModeSelector() {
        const searchTypeSelect = document.getElementById('search-type');
        const resultCountGroup = document.getElementById('result-count-group');
        const resultCountSelect = document.getElementById('result-count');
        if (searchTypeSelect) {
            // Clear existing options
            searchTypeSelect.innerHTML = '';

            // Add fast search as default (first option)
            const fastOption = document.createElement('option');
            fastOption.value = 'fast';
            fastOption.textContent = 'Keyword search';
            searchTypeSelect.appendChild(fastOption);

            // Add semantic vector search option
            const semanticOption = document.createElement('option');
            semanticOption.value = 'semantic';
            semanticOption.textContent = 'Semantic search';
            searchTypeSelect.appendChild(semanticOption);

            // Add RAG option
            const ragOption = document.createElement('option');
            ragOption.value = 'rag';
            ragOption.textContent = 'Ask AI (RAG)';
            searchTypeSelect.appendChild(ragOption);

            // Always default to keyword search on load
            const defaultSearchType = 'fast';
            searchTypeSelect.value = defaultSearchType;
            this.currentMode = defaultSearchType;
            if (window.ConfigManager) {
                window.ConfigManager.updateConfig({
                    ui_preferences: { search_type: defaultSearchType }
                });
            }

            const updateResultCountVisibility = (mode) => {
                if (!resultCountGroup || !resultCountSelect) return;
                if (mode === 'fast') {
                    resultCountGroup.style.display = 'none';
                    resultCountSelect.disabled = true;
                } else {
                    resultCountGroup.style.display = '';
                    resultCountSelect.disabled = false;
                    if (mode === 'semantic') {
                        const allowedValues = ['5', '10', '20', '50'];
                        // Remove any non-allowed options
                        Array.from(resultCountSelect.options).forEach(opt => {
                            if (!allowedValues.includes(opt.value)) {
                                resultCountSelect.removeChild(opt);
                            }
                        });
                        // Ensure all allowed values exist
                        allowedValues.forEach(value => {
                            if (!Array.from(resultCountSelect.options).some(opt => opt.value === value)) {
                                const option = document.createElement('option');
                                option.value = value;
                                option.textContent = `${value} results`;
                                resultCountSelect.appendChild(option);
                            }
                        });
                        if (!allowedValues.includes(resultCountSelect.value)) {
                            resultCountSelect.value = '10';
                        }
                    }
                }
            };

            // Update mode on change + AUTO-SAVE
            searchTypeSelect.addEventListener('change', (e) => {
                if (e.target.value === 'fast') {
                    this.currentMode = 'fast';
                } else if (e.target.value === 'rag') {
                    this.currentMode = 'rag';
                } else {
                    this.currentMode = 'semantic';
                }
                this.updateSearchPlaceholder();
                updateResultCountVisibility(e.target.value);
                // Show/hide RAG mode UI
                const isRAG = e.target.value === 'rag';
                this.updateRAGModeUI(isRAG);

                // Update result count dropdown based on mode
                if (resultCountSelect && window.ConfigManager) {
                    const config = window.ConfigManager.getConfig();
                    if (isRAG) {
                        // RAG mode: use search.num_results (default 5)
                        const ragResults = config?.search?.num_results ?? 5;
                        resultCountSelect.value = String(ragResults);
                    } else {
                        // Non-RAG modes: use ui_preferences.result_count (default 10)
                        const normalResults = config?.ui_preferences?.result_count ?? 10;
                        resultCountSelect.value = String(normalResults);
                    }
                }

                // AUTO-SAVE: Save search type to config
                if (window.ConfigManager) {
                    window.ConfigManager.updateConfig({
                        ui_preferences: { search_type: e.target.value }
                    });
                }
            });

            // Initialize visibility state
            updateResultCountVisibility(searchTypeSelect.value || 'fast');

            // Ensure RAG UI is hidden initially
            const initialMode = searchTypeSelect.value || 'fast';
            const isRAG = initialMode === 'rag';
            this.updateRAGModeUI(isRAG);
            // Wire up RAG settings button
            const ragSettingsBtn = document.getElementById('rag-settings-btn');
            if (ragSettingsBtn) {
                ragSettingsBtn.addEventListener('click', () => {
                    const modal = document.getElementById('quick-settings-modal');
                    if (modal) {
                        modal.style.display = 'flex';
                        // Initialize modal interactivity when opened
                        if (typeof initializeQuickSettingsModal === 'function') {
                            initializeQuickSettingsModal();
                        }
                        // Reload settings from localStorage when modal opens
                        if (typeof loadQuickSettingsFromStorage === 'function') {
                            loadQuickSettingsFromStorage();
                        }
                    }
                });
            }

            // Wire up HyDE mode toggle + AUTO-SAVE
            const hydeModeToggle = document.getElementById('hyde-mode-toggle');
            if (hydeModeToggle) {
                // Initialize from config or existing toggle-hyde-mode button state
                const config = window.ConfigManager ? window.ConfigManager.getConfig() : null;
                const savedHydeEnabled = config?.ui_preferences?.hyde_enabled ?? false;

                const existingHydeBtn = document.getElementById('toggle-hyde-mode');
                if (existingHydeBtn) {
                    const indicator = document.getElementById('hyde-mode-indicator');
                    const isActive = indicator && indicator.style.display !== 'none';
                    hydeModeToggle.checked = savedHydeEnabled || isActive;
                } else {
                    hydeModeToggle.checked = savedHydeEnabled;
                }

                hydeModeToggle.addEventListener('change', (e) => {
                    // Sync with existing toggle-hyde-mode button
                    const existingHydeBtn = document.getElementById('toggle-hyde-mode');
                    if (existingHydeBtn) {
                        existingHydeBtn.click();
                    }

                    // AUTO-SAVE: Save HyDE enabled state to config
                    if (window.ConfigManager) {
                        window.ConfigManager.updateConfig({
                            ui_preferences: { hyde_enabled: e.target.checked }
                        });
                    }
                });
            }

            // Wire up result count selector + AUTO-SAVE
            if (resultCountSelect) {
                // Load initial value from config based on current search mode
                const config = window.ConfigManager ? window.ConfigManager.getConfig() : null;
                const currentSearchType = searchTypeSelect ? searchTypeSelect.value : 'fast';
                const isRAG = currentSearchType === 'rag';
                // RAG mode uses search.num_results (default 5), other modes use ui_preferences.result_count (default 10)
                const savedResultCount = isRAG
                    ? (config?.search?.num_results ?? 5)
                    : (config?.ui_preferences?.result_count ?? 10);
                resultCountSelect.value = String(savedResultCount);

                resultCountSelect.addEventListener('change', (e) => {
                    const count = parseInt(e.target.value, 10);
                    // AUTO-SAVE: Save result count to config
                    if (window.ConfigManager) {
                        window.ConfigManager.updateConfig({
                            ui_preferences: { result_count: count }
                        });
                    }
                });
            }

            const highlightToggle = document.getElementById('highlight-results');
            if (highlightToggle) {
                highlightToggle.addEventListener('change', () => {
                    const enabled = this.isHighlightEnabled();
                    if (!enabled) {
                        this.clearVisualizationHighlight();
                    }
                    this.updateSearchResultsFooterMessages(enabled);
                });
                this.updateSearchResultsFooterMessages(this.isHighlightEnabled());
            }
        }
    }

    isHighlightEnabled() {
        const highlightToggle = document.getElementById('highlight-results');
        return highlightToggle ? highlightToggle.checked : true;
    }

    updateSearchResultsFooterMessages(highlightEnabled) {
        const messages = document.querySelectorAll('.search-results-footer-message');
        messages.forEach((message) => {
            const onText = message.dataset.highlightOn;
            const offText = message.dataset.highlightOff;
            if (!onText || !offText) return;
            message.textContent = highlightEnabled ? onText : offText;
        });
    }

    setupFastSearchHandlers() {
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');

        if (searchInput) {
            // No live search; searches trigger on Enter or button

            // Enter key for all modes
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.performSearch();
                }
            });

            // Auto-clear search results when input becomes empty
            searchInput.addEventListener('keyup', (e) => {
                const query = e.target.value.trim();
                if (query === '') {
                    // User cleared the input - automatically clear search results
                    this.clearSearchResults();
                }
            });

            // Also handle paste/cut events that might clear the input
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.trim();
                if (query === '') {
                    // Input was cleared via paste/cut - automatically clear search results
                    this.clearSearchResults();
                }
            });

            // Add focus/blur animations
            searchInput.addEventListener('focus', () => {
                searchInput.parentElement.classList.add('focused');
            });

            searchInput.addEventListener('blur', () => {
                searchInput.parentElement.classList.remove('focused');
            });
        }

        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.performSearch());
        }

        this.updateSearchPlaceholder();
    }

    updateSearchPlaceholder() {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            const placeholders = {
                fast: 'Search by keyword (use + to require, - to exclude, or "quotes" for phrases)...',
                semantic: 'Enter semantic query...',
                rag: 'Ask questions about your data — the AI will find the most relevant parts and answer…'
            };
            searchInput.placeholder = placeholders[this.currentMode] || 'Search...';
        }
    }

    updateRAGModeUI(isActive) {
        const ragModeUI = document.getElementById('rag-mode-ui');
        if (ragModeUI) {
            ragModeUI.style.display = isActive ? 'block' : 'none';
        }
        document.body.classList.toggle('rag-mode-active', isActive);

        // Update scope text and metadata chips when RAG is activated
        if (isActive) {
            this.updateRAGScopeText();
            if (typeof populateRAGMetadataFields === 'function') {
                populateRAGMetadataFields();
            }
        }

    }

    updateRAGScopeText() {
        const scopeText = document.getElementById('rag-scope-text');
        if (!scopeText) return;

        // Get scope info from the determineRAGScope function if available
        if (typeof determineRAGScope === 'function') {
            const scopeInfo = determineRAGScope();
            if (scopeInfo && scopeInfo.label) {
                if (scopeInfo.scopeType === 'all') {
                    const count = scopeInfo.scopedCount.toLocaleString();
                    scopeText.textContent = `Analyzing ${count} document${scopeInfo.scopedCount === 1 ? '' : 's'} from the entire dataset`;
                } else {
                    // Label already includes count (e.g., "Lasso selection (16)")
                    scopeText.textContent = `Analyzing ${scopeInfo.label.toLowerCase()}`;
                }
            } else {
                scopeText.textContent = 'Analyzing your entire dataset';
            }
        }
    }


    performSearch() {
        const searchInput = document.getElementById('search-input');
        const searchTypeSelect = document.getElementById('search-type');
        const searchBtn = document.getElementById('search-btn');

        if (!searchInput) return;

        const query = searchInput.value.trim();
        const searchType = searchTypeSelect ? searchTypeSelect.value : 'fast';

        if (!query) {
            this.showToast('Please enter a search query', 'warning');
            return;
        }

        // Check if search is ready
        if (searchType === 'fast' && !window.fastSearchReady) {
            // Try to initialize with existing data if available
            if (window.currentVisualizationData && window.currentVisualizationData.points && !this.fastSearch) {
                this.initializeWithData(window.currentVisualizationData.points);
                if (this.fastSearch) {
                    // Retry the search now that we've initialized
                    this.performSearch();
                    return;
                }
            }
            this.showToast('No data available for search. Please upload and process a file first.', 'warning');
            return;
        }

        if (searchType === 'fast') {
            // For fast search, show loading state briefly
            if (searchBtn) {
                searchBtn.disabled = true;
                const originalText = searchBtn.innerHTML;
                searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';

                // Perform fast search
                this.performFastSearch(query);

                // Restore button after fast search completes (shorter timeout for fast search)
                setTimeout(() => {
                    searchBtn.disabled = false;
                    searchBtn.innerHTML = originalText;
                }, 800);
            } else {
                this.performFastSearch(query);
            }
        } else {
            // Delegate to existing semantic search (let it handle its own loading state)
            if (window.performSearch) {
                window.performSearch();
            }
        }
    }

    performFastSearch(query) {
        if (!this.fastSearch) {
            console.warn('Fast search not ready - search engine not initialized');
            this.showToast('No data available for search. Please upload and process a file first.', 'warning');

            // Try to initialize with existing data if available
            if (window.currentVisualizationData && window.currentVisualizationData.points) {
                this.initializeWithData(window.currentVisualizationData.points);

                // Retry search if initialization succeeded
                if (this.fastSearch) {
                    this.performFastSearch(query);
                }
            }
            return;
        }

        if (!window.currentVisualizationData) {
            console.warn('Fast search not ready - no visualization data available');
            this.showToast('No data available for search. Please upload and process a file first.', 'warning');
            return;
        }

        // Check if we need to search in filtered data only
        const activeFilters = (window.getActiveMetadataFilters && window.getActiveMetadataFilters()) ||
            (window.collectMetadataFiltersForSearch && window.collectMetadataFiltersForSearch()) || {};
        const hasActiveFilters = activeFilters && Object.keys(activeFilters).length > 0;
        let searchCorpus = window.currentVisualizationData.points;

        if (hasActiveFilters) {
            // Filter the data first, then search only in the filtered subset
            searchCorpus = window.currentVisualizationData.points.filter(point =>
                this.matchesMetadataFilters(point, activeFilters)
            );
            // Temporarily reinitialize search with filtered data
            const tempFastSearch = new FastSearch(searchCorpus);

            try {
                const results = tempFastSearch.search(query, {
                    maxResults: Infinity,
                    fuzzy: true
                });

                if (results.results.length > 0) {
                    this.displayFastSearchResults(results);
                    this.highlightFastSearchResults(results.results);
                    this.showToast(`Found ${results.results.length} results in filtered data (${searchCorpus.length} points)`, 'success');
                } else {
                    this.displayNoResults(query);
                    this.clearVisualizationHighlight();
                    this.showToast(`No results found in filtered data (searched ${searchCorpus.length} points)`, 'info');
                }
            } catch (error) {
                console.error('Fast search error in filtered data:', error);
                this.showToast('Search failed', 'error');
            }
        } else {
            // No filters active, search in all data using the existing search engine
            try {
                const results = this.fastSearch.search(query, {
                    maxResults: Infinity,
                    fuzzy: true
                });

                if (results.results.length > 0) {
                    this.displayFastSearchResults(results);
                    this.highlightFastSearchResults(results.results);
                    this.showToast(`Found ${results.results.length} results`, 'success');
                } else {
                    this.displayNoResults(query);
                    this.clearVisualizationHighlight();
                    this.showToast('No results found', 'info');
                }
            } catch (error) {
                console.error('Fast search error:', error);
                this.showToast('Search failed', 'error');
            }
        }
    }

    displayFastSearchResults(searchData) {
        const resultsDiv = document.getElementById('search-results');
        const resultsCount = document.getElementById('results-count');
        const hasActiveFilters = (window.getActiveMetadataFilters && Object.keys(window.getActiveMetadataFilters() || {}).length > 0);

        // Update results count
        if (resultsCount) {
            resultsCount.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>${searchData.results.length} results found • Search</span>
                    ${hasActiveFilters ? '<span class="filter-badge" title="Results limited to filtered data">Filtered</span>' : ''}
                </div>
            `;
        }

        // Show results section
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }

        // Update text list with results
        this.updateTextListWithFastResults(searchData.results, searchData.query);
    }

    displayNoResults(query) {
        const resultsCount = document.getElementById('results-count');
        const hasActiveFilters = (window.getActiveMetadataFilters && Object.keys(window.getActiveMetadataFilters() || {}).length > 0);
        if (resultsCount) {
            resultsCount.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px; color: #ff9800;">
                    <span>No results found • Search</span>
                    ${hasActiveFilters ? '<span class="filter-badge" title="Results limited to filtered data">Filtered</span>' : ''}
                </div>
            `;
        }

        const resultsDiv = document.getElementById('search-results');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
    }

    updateTextListWithFastResults(results, query) {
        const textList = document.getElementById('text-list');
        if (!textList) return;

        // Store current search results globally
        window.currentSearchResults = {
            results: results,
            searchType: 'fast',
            query: query
        };

        // Clear current text list
        textList.innerHTML = '';

        // Add search results header
        const header = document.createElement('div');
        header.className = 'search-results-header';
        const hasActiveFilters = (window.getActiveMetadataFilters && Object.keys(window.getActiveMetadataFilters() || {}).length > 0);
        header.innerHTML = `
            <div style="padding: 12px; background: linear-gradient(135deg, #ffd700 0%, #ff8f00 100%); color: #000; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);">
                <h3 style="margin: 0 0 8px 0; font-size: 1.1em; color: #000;">
                    Search Results
                </h3>
                <div style="font-size: 0.9em; opacity: 0.9; color: #000; display:flex; align-items:center; gap:10px;">
                    ${results.length} matches for "${query}"
                    ${hasActiveFilters ? '<span class="filter-badge" title="Results limited to filtered data">Filtered</span>' : ''}
                </div>
            </div>
        `;
        textList.appendChild(header);

        // Show search results with enhanced styling
        results.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'text-item search-match fast-result';
            item.dataset.index = index;
            item.dataset.searchResult = 'true';

            // Cluster can be at top level (regular search) or in metadata (RAG/vector search)
            const cluster = result.cluster !== undefined ? result.cluster : (result.metadata?.cluster ?? 0);
            const clusterColor = this.getClusterColor(cluster);
            const clusterName = this.getClusterName(cluster);

            const displayText = result.originalText || result.text || 'No content available';
            const scoreDisplay = `<span class="fast-search-score">${result.score.toFixed(0)}</span>`;

            item.innerHTML = `
                <div class="text-item-header">
                    <span class="cluster-indicator" style="background-color: ${clusterColor}; box-shadow: 0 0 0 2px rgba(255,215,0,0.6);"></span>
                    <span class="item-title">Result ${index + 1} - ${clusterName}</span>
                    <div class="result-badges">
                        ${scoreDisplay}
                    </div>
                </div>
                <div class="text-preview enhanced-preview">
                    ${result.matchedText ? result.matchedText.substring(0, 200) : displayText.substring(0, 200)}${displayText.length > 200 ? '...' : ''}
                </div>
            `;

            item.addEventListener('click', () => {
                const pointData = {
                    index: result.index,
                    cluster: cluster,
                    cluster_color: clusterColor,
                    text: displayText,
                    coordinates: result.coordinates,
                    score: result.score,
                    search_type: 'fast',
                    search_rank: index + 1,
                    search_query: query,
                    is_search_result: true,
                    ...result.item
                };

                if (window.showTextDetails) {
                    window.showTextDetails(pointData, index);
                }

                document.querySelectorAll('.text-item').forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');

                if (window.mainVisualization) {
                    window.mainVisualization.highlightPoint(result.index, { focus: true, revealTooltip: true });
                }
            });

            textList.appendChild(item);
        });

        // Add footer
        const footer = document.createElement('div');
        footer.className = 'search-results-footer';
        const highlightEnabled = this.isHighlightEnabled();
        const highlightOnText = 'Search results • Highlighted in visualization';
        const highlightOffText = 'Search results';
        const footerText = highlightEnabled ? highlightOnText : highlightOffText;
        footer.innerHTML = `
            <div style="padding: 8px 12px; background: rgba(255,215,0,0.1); border-radius: 6px; margin-top: 16px; text-align: center; font-size: 0.85em; color: #666;">
                <span class="search-results-footer-message" data-highlight-on="${highlightOnText}" data-highlight-off="${highlightOffText}">${footerText}</span>
            </div>
        `;
        textList.appendChild(footer);
    }

    highlightFastSearchResults(results) {
        if (!this.isHighlightEnabled()) {
            this.clearVisualizationHighlight();
            return;
        }

        if (!window.mainVisualization || !results || results.length === 0) {
            return;
        }

        // Convert to format expected by visualization
        const searchResultsData = results.map((result, index) => ({
            coordinates: result.coordinates,
            index: result.index,
            score: result.score,
            rank: index,
            text: result.originalText || result.text,
            cluster: result.cluster,
            isValid: true
        }));

        // Store globally for persistence
        window.currentVisualizationSearchResults = searchResultsData;

        // Enable search highlighting mode
        window.mainVisualization.enableSearchHighlightMode();
        window.mainVisualization.highlightSearchResults(searchResultsData);

        // Force re-render
        setTimeout(() => {
            if (window.mainVisualization) {
                window.mainVisualization.requestRender();
            }
        }, 50);
    }

    clearVisualizationHighlight() {
        if (window.mainVisualization) {
            window.mainVisualization.disableSearchHighlightMode();
            window.mainVisualization.clearSearchHighlight();
        }
    }

    matchesMetadataFilters(point, metadataFilters) {
        // Use the same filtering logic as the main metadata filtering system
        if (!metadataFilters || Object.keys(metadataFilters).length === 0) {
            return true;
        }

        for (const [fieldName, filterConfig] of Object.entries(metadataFilters)) {
            const filterType = filterConfig.type;
            const filterValue = filterConfig.value;

            // Resolve the actual value from multiple locations to match backend logic
            let actualValue = undefined;
            if (point && Object.prototype.hasOwnProperty.call(point, fieldName)) {
                actualValue = point[fieldName];
            } else if (point && point.metadata && Object.prototype.hasOwnProperty.call(point.metadata, fieldName)) {
                actualValue = point.metadata[fieldName];
            } else if (point && point.data && Object.prototype.hasOwnProperty.call(point.data, fieldName)) {
                actualValue = point.data[fieldName];
            }

            // Skip if field doesn't exist on this point
            if (actualValue === null || actualValue === undefined || actualValue === '') {
                return false;
            }

            // Apply type-specific filtering (same logic as vectoria.js)
            if (filterType === 'category') {
                // Handle both single values and arrays of values
                if (Array.isArray(filterValue)) {
                    if (!filterValue.includes(String(actualValue))) {
                        return false;
                    }
                } else {
                    if (String(actualValue) !== String(filterValue)) {
                        return false;
                    }
                }
            } else if (filterType === 'number') {
                const actualNum = Number(actualValue);
                if (isNaN(actualNum)) return false;

                if (filterValue.min !== undefined && actualNum < Number(filterValue.min)) {
                    return false;
                }
                if (filterValue.max !== undefined && actualNum > Number(filterValue.max)) {
                    return false;
                }
            } else if (filterType === 'boolean') {
                const expectedBool = filterValue === 'true' || filterValue === true;
                const actualBool = Boolean(actualValue);
                if (actualBool !== expectedBool) {
                    return false;
                }
            } else if (filterType === 'text') {
                const searchText = String(filterValue).toLowerCase();
                const actualText = String(actualValue).toLowerCase();
                if (!actualText.includes(searchText)) {
                    return false;
                }
            } else if (filterType === 'date') {
                try {
                    const actualDate = new Date(actualValue);
                    if (filterValue.min) {
                        const minDate = new Date(filterValue.min);
                        if (actualDate < minDate) return false;
                    }
                    if (filterValue.max) {
                        const maxDate = new Date(filterValue.max);
                        if (actualDate > maxDate) return false;
                    }
                } catch (error) {
                    return false;
                }
            }
        }

        return true;
    }

    clearSearchResults() {
        const resultsDiv = document.getElementById('search-results');
        if (resultsDiv) {
            resultsDiv.style.display = 'none';
        }

        // Reset text list if no search active
        if (window.currentVisualizationData && window.currentVisualizationData.points) {
            if (window.updateTextList) {
                if (typeof window.unlockTextList === 'function') {
                    window.unlockTextList('fast search cleared');
                }
                window.updateTextList(window.currentVisualizationData.points, { force: true });
            }
        }

        this.clearVisualizationHighlight();

        // Clear global search state
        window.currentSearchResults = null;
        window.currentVisualizationSearchResults = null;
    }

    initializeWithData(data) {
        try {
            this.fastSearch = new FastSearch(data);
            window.fastSearchReady = true;
        } catch (error) {
            console.error('Failed to initialize fast search:', error);
            window.fastSearchReady = false;
        }
    }

    getClusterColor(clusterId) {
        if (window.getClusterColor) {
            return window.getClusterColor(clusterId);
        }
        // Fallback - should use centralized color manager from vectoria.js
        if (window.VectoriaColorManager) {
            return window.VectoriaColorManager.getColor(clusterId);
        }
        // Last resort fallback
        return clusterId === -1 ? '#9CA3AF' : '#3B82F6';
    }

    getClusterName(clusterId) {
        return clusterId === -1 ? 'Outlier' : `Cluster ${clusterId}`;
    }

    showToast(message, type = 'info') {
        if (window.showToast) {
            window.showToast(message, type);
        } else {
        }
    }
}

// Initialize fast search interface
let globalSearchInterface = null;

// Integration hook for when visualization data is loaded
function initializeFastSearch(data) {
    try {
        if (!globalSearchInterface) {
            globalSearchInterface = new SearchInterface();
        }

        if (data && data.length > 0) {
            globalSearchInterface.initializeWithData(data);
            window.fastSearchReady = true;
        } else {
            console.warn('⚠️ Fast search initialized without data');
            window.fastSearchReady = false;
        }

        // Update global reference
        window.globalSearchInterface = globalSearchInterface;

    } catch (error) {
        console.error('❌ Failed to initialize fast search:', error);
        window.fastSearchReady = false;
    }
}

// ============================================================================
// Quick Settings Modal Handler
// ============================================================================

function initializeQuickSettingsModal() {
    const modal = document.getElementById('quick-settings-modal');
    const closeBtn = document.getElementById('close-quick-settings');
    const applyBtn = document.getElementById('apply-quick-settings');
    const resetBtn = document.getElementById('reset-quick-settings');

    // New controls for hybrid search
    const vectorWeightSlider = document.getElementById('quick-vector-weight');
    const retrievalKSlider = document.getElementById('quick-retrieval-k');

    // Existing LLM controls
    const temperatureSlider = document.getElementById('quick-temperature');
    const maxTokensSlider = document.getElementById('quick-max-tokens');
    const topPSlider = document.getElementById('quick-top-p');
    const repeatPenaltySlider = document.getElementById('quick-repeat-penalty');
    const systemPromptTextarea = document.getElementById('quick-system-prompt');
    const userTemplateTextarea = document.getElementById('quick-user-template');

    // HyDE controls
    const hydeTemperatureSlider = document.getElementById('hyde-temperature');
    const hydeMaxTokensSlider = document.getElementById('hyde-max-tokens');
    const hydePromptTextarea = document.getElementById('hyde-prompt');

    if (!modal) {
        console.error('⚠️ Quick settings modal not found - initialization aborted');
        return;
    }

    // Check if already initialized to prevent duplicate listeners
    if (modal.dataset.quickSettingsInitialized === 'true') {
        return;
    }
    modal.dataset.quickSettingsInitialized = 'true';

    // Auto-save timeout for debouncing
    let autoSaveTimeout = null;

    // Load saved settings from localStorage via ConfigManager
    async function loadQuickSettings() {
        try {
            // Get config from ConfigManager (uses localStorage, merged with defaults)
            const config = window.ConfigManager ? window.ConfigManager.getConfig() : window.ConfigManager.DEFAULT_CONFIG;
            const defaults = window.ConfigManager ? window.ConfigManager.DEFAULT_CONFIG : {};

            // Extract values from config (with defaults from DEFAULT_CONFIG)
            const settings = {
                vectorWeight: config.search?.vector_weight ?? defaults.search?.vector_weight ?? 0.6,
                retrievalK: config.search?.retrieval_k ?? defaults.search?.retrieval_k ?? 60,
                temperature: config.llm?.temperature ?? defaults.llm?.temperature ?? 0.5,
                maxTokens: config.llm?.max_tokens ?? defaults.llm?.max_tokens ?? 768,
                topP: config.llm?.top_p ?? defaults.llm?.top_p ?? 0.9,
                repeatPenalty: config.llm?.repeat_penalty ?? defaults.llm?.repeat_penalty ?? 1.15,
                systemPrompt: config.rag_prompts?.system_prompt ?? defaults.rag_prompts?.system_prompt ?? '',
                userTemplate: config.rag_prompts?.user_template ?? defaults.rag_prompts?.user_template ?? '',
                hydePrompt: config.hyde?.prompt ?? defaults.hyde?.prompt ?? '',
                hydeTemperature: config.hyde?.temperature ?? defaults.hyde?.temperature ?? 0.2,
                hydeMaxTokens: config.hyde?.max_tokens ?? defaults.hyde?.max_tokens ?? 256
            };

            // Set UI values for RAG controls
            if (vectorWeightSlider) vectorWeightSlider.value = settings.vectorWeight * 100;
            if (retrievalKSlider) retrievalKSlider.value = settings.retrievalK;
            if (temperatureSlider) temperatureSlider.value = settings.temperature;
            if (maxTokensSlider) maxTokensSlider.value = settings.maxTokens;
            if (topPSlider) topPSlider.value = settings.topP;
            if (repeatPenaltySlider) repeatPenaltySlider.value = settings.repeatPenalty;
            if (systemPromptTextarea) systemPromptTextarea.value = settings.systemPrompt;
            if (userTemplateTextarea) userTemplateTextarea.value = settings.userTemplate;

            // Set UI values for HyDE controls
            if (hydeTemperatureSlider) hydeTemperatureSlider.value = settings.hydeTemperature;
            if (hydeMaxTokensSlider) hydeMaxTokensSlider.value = settings.hydeMaxTokens;
            if (hydePromptTextarea) hydePromptTextarea.value = settings.hydePrompt;

            updateSliderValues();

            // Show/hide think-mode hint based on current model
            const thinkHint = document.getElementById('think-mode-hint');
            if (thinkHint) {
                const modelId = config.llm?.model_id;
                const constraints = typeof getModelConstraints === 'function' ? getModelConstraints(modelId) : null;
                thinkHint.style.display = constraints?.hasThinkMode ? 'block' : 'none';
            }

        } catch (error) {
            console.error('⚠️ Failed to load settings:', error);
            updateSliderValues();
        }
    }

    // Make loadQuickSettings available globally so it can be called when modal opens
    window.loadQuickSettingsFromStorage = loadQuickSettings;

    // Auto-save function (debounced, saves to localStorage via ConfigManager)
    function autoSaveQuickSettings() {
        // Clear existing timeout
        if (autoSaveTimeout) {
            clearTimeout(autoSaveTimeout);
        }

        // Debounce for 300ms
        autoSaveTimeout = setTimeout(() => {
            try {
                if (!window.ConfigManager) {
                    console.warn('⚠️ ConfigManager not available for auto-save');
                    return;
                }

                // Gather current values from UI
                const updates = {
                    search: {
                        vector_weight: vectorWeightSlider ? parseFloat(vectorWeightSlider.value) / 100 : 0.6,
                        retrieval_k: retrievalKSlider ? parseInt(retrievalKSlider.value, 10) : 60
                    },
                    llm: {
                        temperature: temperatureSlider ? parseFloat(temperatureSlider.value) : 0.5,
                        max_tokens: maxTokensSlider ? parseInt(maxTokensSlider.value, 10) : 768,
                        top_p: topPSlider ? parseFloat(topPSlider.value) : 0.9,
                        repeat_penalty: repeatPenaltySlider ? parseFloat(repeatPenaltySlider.value) : 1.15
                    },
                    rag_prompts: {
                        system_prompt: systemPromptTextarea ? systemPromptTextarea.value : '',
                        user_template: userTemplateTextarea ? userTemplateTextarea.value : ''
                    },
                    hyde: {
                        prompt: hydePromptTextarea ? hydePromptTextarea.value : '',
                        temperature: hydeTemperatureSlider ? parseFloat(hydeTemperatureSlider.value) : 0.2,
                        max_tokens: hydeMaxTokensSlider ? parseInt(hydeMaxTokensSlider.value, 10) : 256
                    }
                };

                // Save via ConfigManager (deep merge with existing config)
                window.ConfigManager.updateConfig(updates);
            } catch (error) {
                console.error('❌ Auto-save failed:', error);
            }
        }, 300);
    }

    // Update slider value displays
    function updateSliderValues() {
        const weightVal = document.getElementById('quick-weight-value');
        const tempVal = document.getElementById('quick-temperature-value');
        const tokensVal = document.getElementById('quick-max-tokens-value');
        const topPVal = document.getElementById('quick-top-p-value');
        const repeatPenaltyVal = document.getElementById('quick-repeat-penalty-value');
        const retrievalVal = document.getElementById('quick-retrieval-k-value');
        const hydeTempVal = document.getElementById('hyde-temperature-value');
        const hydeTokensVal = document.getElementById('hyde-max-tokens-value');

        if (weightVal && vectorWeightSlider) {
            const vectorPercent = parseInt(vectorWeightSlider.value);
            const bm25Percent = 100 - vectorPercent;
            weightVal.textContent = `${vectorPercent} / ${bm25Percent}`;
        }

        if (tempVal && temperatureSlider) {
            tempVal.textContent = temperatureSlider.value;
        }

        if (tokensVal && maxTokensSlider) {
            tokensVal.textContent = maxTokensSlider.value;
        }

        if (topPVal && topPSlider) {
            topPVal.textContent = topPSlider.value;
        }

        if (repeatPenaltyVal && repeatPenaltySlider) {
            repeatPenaltyVal.textContent = repeatPenaltySlider.value;
        }

        if (retrievalVal && retrievalKSlider) {
            retrievalVal.textContent = retrievalKSlider.value;
        }

        if (hydeTempVal && hydeTemperatureSlider) {
            hydeTempVal.textContent = hydeTemperatureSlider.value;
        }

        if (hydeTokensVal && hydeMaxTokensSlider) {
            hydeTokensVal.textContent = hydeMaxTokensSlider.value;
        }
    }

    // Tab switching
    const settingsTabs = document.querySelectorAll('.settings-tab');
    const ragTabContent = document.getElementById('rag-tab');
    const hydeTabContent = document.getElementById('hyde-tab');

    settingsTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Update tab button styles
            settingsTabs.forEach(t => {
                t.classList.remove('active');
                t.style.borderBottomColor = '';
                t.style.color = '';
            });
            tab.classList.add('active');

            // Show/hide tab content
            if (targetTab === 'rag-tab') {
                if (ragTabContent) ragTabContent.style.display = 'block';
                if (hydeTabContent) hydeTabContent.style.display = 'none';
            } else if (targetTab === 'hyde-tab') {
                if (ragTabContent) ragTabContent.style.display = 'none';
                if (hydeTabContent) hydeTabContent.style.display = 'block';
            }
        });
    });

    // Close modal handlers
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    } else {
        console.error('⚠️ Close button not found!');
    }

    // Close on Escape key
    const handleEscapeKey = (event) => {
        if (event.key === 'Escape' && modal.style.display !== 'none') {
            modal.style.display = 'none';
        }
    };
    document.addEventListener('keydown', handleEscapeKey);

    // Update value displays when sliders change + AUTO-SAVE
    const sliders = [retrievalKSlider, vectorWeightSlider, temperatureSlider, maxTokensSlider, topPSlider, repeatPenaltySlider, hydeTemperatureSlider, hydeMaxTokensSlider].filter(Boolean);
    sliders.forEach(slider => {
        slider.addEventListener('input', () => {
            updateSliderValues();
            autoSaveQuickSettings(); // Auto-save on change
        });
    });

    // AUTO-SAVE: Add listeners to textareas
    const textareas = [systemPromptTextarea, userTemplateTextarea, hydePromptTextarea].filter(Boolean);
    textareas.forEach(textarea => {
        textarea.addEventListener('input', () => {
            autoSaveQuickSettings(); // Auto-save on change
        });
    });

    // Apply settings (now uses ConfigManager - auto-save makes this optional but kept for explicit saves)
    if (applyBtn) {
        applyBtn.addEventListener('click', async () => {
            try {
                if (!window.ConfigManager) {
                    throw new Error('ConfigManager not available');
                }

                // Gather current values from UI
                const updates = {
                    search: {
                        vector_weight: vectorWeightSlider ? parseFloat(vectorWeightSlider.value) / 100 : 0.6,
                        retrieval_k: retrievalKSlider ? parseInt(retrievalKSlider.value, 10) : 60
                    },
                    llm: {
                        temperature: temperatureSlider ? parseFloat(temperatureSlider.value) : 0.5,
                        max_tokens: maxTokensSlider ? parseInt(maxTokensSlider.value, 10) : 768,
                        top_p: topPSlider ? parseFloat(topPSlider.value) : 0.9,
                        repeat_penalty: repeatPenaltySlider ? parseFloat(repeatPenaltySlider.value) : 1.15
                    },
                    rag_prompts: {
                        system_prompt: systemPromptTextarea ? systemPromptTextarea.value : '',
                        user_template: userTemplateTextarea ? userTemplateTextarea.value : ''
                    },
                    hyde: {
                        prompt: hydePromptTextarea ? hydePromptTextarea.value : '',
                        temperature: hydeTemperatureSlider ? parseFloat(hydeTemperatureSlider.value) : 0.2,
                        max_tokens: hydeMaxTokensSlider ? parseInt(hydeMaxTokensSlider.value, 10) : 256
                    }
                };

                // Save via ConfigManager
                window.ConfigManager.updateConfig(updates);
                if (typeof showToast === 'function') {
                    showToast('RAG settings saved successfully', 'success');
                }

                modal.style.display = 'none';
            } catch (error) {
                console.error('❌ Failed to save RAG settings:', error);
                if (typeof showToast === 'function') {
                    showToast(`Failed to save settings: ${error.message}`, 'error');
                }
            }
        });
    } else {
        console.error('⚠️ Apply button not found!');
    }

    // Reset to defaults (uses ConfigManager.resetConfig())
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            try {
                if (!window.ConfigManager) {
                    throw new Error('ConfigManager not available');
                }

                // Reset only Quick Settings categories (not entire config)
                const defaults = window.ConfigManager.DEFAULT_CONFIG;
                const currentConfig = window.ConfigManager.getConfig();

                // Partial reset: only RAG Search Settings categories
                const partialReset = {
                    search: {
                        ...currentConfig.search,
                        vector_weight: defaults.search.vector_weight,
                        retrieval_k: defaults.search.retrieval_k
                    },
                    llm: {
                        ...currentConfig.llm,
                        temperature: defaults.llm.temperature,
                        max_tokens: defaults.llm.max_tokens,
                        top_p: defaults.llm.top_p,
                        repeat_penalty: defaults.llm.repeat_penalty
                    },
                    rag_prompts: { ...defaults.rag_prompts },
                    hyde: { ...defaults.hyde }
                };

                window.ConfigManager.updateConfig(partialReset);

                // Reload UI from reset defaults
                if (vectorWeightSlider) vectorWeightSlider.value = (defaults.search?.vector_weight ?? 0.6) * 100;
                if (retrievalKSlider) retrievalKSlider.value = defaults.search?.retrieval_k ?? 60;
                if (temperatureSlider) temperatureSlider.value = defaults.llm?.temperature ?? 0.5;
                if (maxTokensSlider) maxTokensSlider.value = defaults.llm?.max_tokens ?? 768;
                if (topPSlider) topPSlider.value = defaults.llm?.top_p ?? 0.9;
                if (repeatPenaltySlider) repeatPenaltySlider.value = defaults.llm?.repeat_penalty ?? 1.15;
                if (systemPromptTextarea) systemPromptTextarea.value = defaults.rag_prompts?.system_prompt ?? '';
                if (userTemplateTextarea) userTemplateTextarea.value = defaults.rag_prompts?.user_template ?? '';
                if (hydeTemperatureSlider) hydeTemperatureSlider.value = defaults.hyde?.temperature ?? 0.2;
                if (hydeMaxTokensSlider) hydeMaxTokensSlider.value = defaults.hyde?.max_tokens ?? 256;
                if (hydePromptTextarea) hydePromptTextarea.value = defaults.hyde?.prompt ?? '';

                updateSliderValues();

                if (typeof showToast === 'function') {
                    showToast('RAG Search Settings reset to defaults', 'info');
                }

            } catch (error) {
                console.error('❌ Failed to reset settings:', error);
                if (typeof showToast === 'function') {
                    showToast(`Failed to reset settings: ${error.message}`, 'error');
                }
            }
        });
    } else {
        console.error('⚠️ Reset button not found!');
    }

    // Initialize with saved settings
    loadQuickSettings();

}

// ============================================================================

// Initialize search interface on page load
document.addEventListener('DOMContentLoaded', function () {
    if (!globalSearchInterface) {
        globalSearchInterface = new SearchInterface();
        window.globalSearchInterface = globalSearchInterface;
    }
});

// Make it globally available
window.FastSearch = FastSearch;
window.SearchInterface = SearchInterface;
window.initializeFastSearch = initializeFastSearch;
window.initializeQuickSettingsModal = initializeQuickSettingsModal;
window.globalSearchInterface = globalSearchInterface;
