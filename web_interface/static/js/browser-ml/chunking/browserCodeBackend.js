/**
 * Browser Tree-sitter backend for ChonkieJS CodeChunker.
 *
 * ChonkieJS's default code backend is a native Node.js module. Vectoria uses
 * Chonkie's supported custom-backend hook so code chunking stays in-browser.
 * The parser runtime and one selected grammar are downloaded lazily, only when
 * the Code strategy is used.
 */

// tree-sitter-wasms 0.1.13 grammars were built with Tree-sitter CLI 0.20.x.
// The runtime must stay on the matching dynamic-linking ABI.
const WEB_TREE_SITTER_VERSION = '0.20.8';
const TREE_SITTER_WASMS_VERSION = '0.1.13';
const WEB_TREE_SITTER_MODULE_URL = `https://cdn.jsdelivr.net/npm/web-tree-sitter@${WEB_TREE_SITTER_VERSION}/+esm`;
const WEB_TREE_SITTER_WASM_URL = `https://cdn.jsdelivr.net/npm/web-tree-sitter@${WEB_TREE_SITTER_VERSION}/tree-sitter.wasm`;
const GRAMMAR_BASE_URL = `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@${TREE_SITTER_WASMS_VERSION}/out`;

export const SUPPORTED_CODE_LANGUAGES = Object.freeze([
    'bash', 'c', 'cpp', 'c_sharp', 'css', 'dart', 'elixir', 'elm', 'go',
    'html', 'java', 'javascript', 'json', 'kotlin', 'lua', 'objc', 'ocaml',
    'php', 'python', 'ruby', 'rust', 'scala', 'solidity', 'swift', 'toml',
    'tsx', 'typescript', 'vue', 'yaml', 'zig'
]);

const SUPPORTED_LANGUAGE_SET = new Set(SUPPORTED_CODE_LANGUAGES);
const LANGUAGE_ALIASES = Object.freeze({
    'c#': 'c_sharp',
    csharp: 'c_sharp',
    cs: 'c_sharp',
    'c++': 'cpp',
    cc: 'cpp',
    cxx: 'cpp',
    hpp: 'cpp',
    shell: 'bash',
    sh: 'bash',
    zsh: 'bash',
    js: 'javascript',
    jsx: 'javascript',
    mjs: 'javascript',
    cjs: 'javascript',
    ts: 'typescript',
    py: 'python',
    rb: 'ruby',
    rs: 'rust',
    golang: 'go',
    kt: 'kotlin',
    kts: 'kotlin',
    yml: 'yaml',
    objectivec: 'objc',
    'objective-c': 'objc',
    ex: 'elixir',
    exs: 'elixir'
});

const EXTENSION_LANGUAGES = Object.freeze({
    bash: 'bash', sh: 'bash', zsh: 'bash',
    c: 'c', h: 'c', cc: 'cpp', cpp: 'cpp', cxx: 'cpp', hpp: 'cpp',
    cs: 'c_sharp', css: 'css', dart: 'dart', ex: 'elixir', exs: 'elixir',
    elm: 'elm', go: 'go', html: 'html', htm: 'html', java: 'java',
    js: 'javascript', jsx: 'javascript', mjs: 'javascript', cjs: 'javascript',
    json: 'json', kt: 'kotlin', kts: 'kotlin', lua: 'lua', m: 'objc', mm: 'objc',
    ml: 'ocaml', mli: 'ocaml', php: 'php', py: 'python', rb: 'ruby', rs: 'rust',
    scala: 'scala', sc: 'scala', sol: 'solidity', swift: 'swift', toml: 'toml',
    ts: 'typescript', tsx: 'tsx', vue: 'vue', yaml: 'yaml', yml: 'yaml', zig: 'zig'
});

let runtimePromise = null;
const backendPromises = new Map();

export function normalizeCodeLanguage(value) {
    const normalized = String(value || '').trim().toLowerCase().replace(/^\./, '');
    const resolved = LANGUAGE_ALIASES[normalized] || normalized;
    return SUPPORTED_LANGUAGE_SET.has(resolved) ? resolved : null;
}

function languageFromPath(value) {
    const path = String(value || '').trim().toLowerCase().split(/[?#]/, 1)[0];
    const match = path.match(/\.([a-z0-9_+-]+)$/i);
    return match ? (EXTENSION_LANGUAGES[match[1]] || normalizeCodeLanguage(match[1])) : null;
}

function languageFromMetadata(metadata = {}) {
    if (!metadata || typeof metadata !== 'object') return null;

    const normalizedEntries = Object.entries(metadata).map(([key, value]) => [
        String(key).trim().toLowerCase().replace(/[\s-]+/g, '_'),
        value
    ]);
    const directKeys = new Set(['language', 'lang', 'programming_language', 'code_language']);
    const pathKeys = new Set(['file', 'filename', 'file_name', 'path', 'source_path', 'extension', 'ext']);

    for (const [key, value] of normalizedEntries) {
        if (directKeys.has(key)) {
            const direct = normalizeCodeLanguage(value);
            if (direct) return direct;
        }
    }
    for (const [key, value] of normalizedEntries) {
        if (pathKeys.has(key)) {
            const fromPath = languageFromPath(value) || normalizeCodeLanguage(value);
            if (fromPath) return fromPath;
        }
    }
    return null;
}

function languageFromFence(text) {
    const match = String(text || '').match(/^\s*```\s*([a-zA-Z0-9_+#.-]+)/);
    return match ? normalizeCodeLanguage(match[1]) : null;
}

/**
 * Resolve an explicit language or infer one from metadata, fenced-code labels,
 * and conservative content signatures. Ambiguous code defaults to JavaScript,
 * which has a permissive grammar and is Vectoria's most common code input.
 */
export function detectCodeLanguage(text, metadata = {}, preferred = 'auto') {
    if (preferred && String(preferred).toLowerCase() !== 'auto') {
        return normalizeCodeLanguage(preferred) || 'javascript';
    }

    const metadataLanguage = languageFromMetadata(metadata);
    if (metadataLanguage) return metadataLanguage;

    const fenceLanguage = languageFromFence(text);
    if (fenceLanguage) return fenceLanguage;

    const source = String(text || '').trim();
    if (/^<\?php\b/i.test(source)) return 'php';
    if (/^#!.*\b(?:bash|sh|zsh)\b/m.test(source)) return 'bash';
    if (/^\s*(?:\{|\[)/.test(source)) {
        try {
            JSON.parse(source);
            return 'json';
        } catch (_) { /* continue with code signatures */ }
    }
    if (/<(?:!doctype\s+html|html|head|body|div|section|template)\b/i.test(source)) {
        return /<template\b[\s\S]*<script\b/i.test(source) ? 'vue' : 'html';
    }
    if (/^\s*(?:def|class)\s+[A-Za-z_]\w*.*:\s*$/m.test(source)
        || /^\s*(?:from\s+\S+\s+import|import\s+[A-Za-z_][\w.]*)\s*$/m.test(source)) return 'python';
    if (/^\s*pragma\s+solidity\b/m.test(source)) return 'solidity';
    if (/^\s*package\s+main\b/m.test(source) && /\bfunc\s+\w+\s*\(/.test(source)) return 'go';
    if (/\bfn\s+\w+\s*\([^)]*\)\s*(?:->[^\{]+)?\{/m.test(source) || /^\s*use\s+\w+(?:::\w+)*/m.test(source)) return 'rust';
    if (/^\s*#\s*include\s*[<"]/m.test(source)) {
        return /\b(?:std::|namespace\s+\w+|template\s*<)/.test(source) ? 'cpp' : 'c';
    }
    if (/^\s*using\s+System\b/m.test(source) || /\bnamespace\s+\w+[\s\S]*\bclass\s+\w+/.test(source)) return 'c_sharp';
    if (/^\s*package\s+[\w.]+\s*;/m.test(source) || /\bpublic\s+(?:final\s+)?class\s+\w+/.test(source)) return 'java';
    if (/^\s*(?:fun|val|var)\s+\w+/m.test(source)) return 'kotlin';
    if (/^\s*import\s+(?:Swift|Foundation|UIKit)\b/m.test(source)) return 'swift';
    if (/^\s*(?:require|class|module)\s+['":A-Z]/m.test(source) && /^\s*def\s+\w+[!?=]?/m.test(source)) return 'ruby';
    if (/^\s*(?:local\s+)?function\s+\w+/m.test(source) || /^\s*local\s+\w+\s*=/m.test(source)) return 'lua';
    if (/^\s*\[[\w.-]+\]\s*$/m.test(source) && /^\s*[\w.-]+\s*=\s*.+$/m.test(source)) return 'toml';
    if (/^\s*[\w.-]+:\s*(?:[^{};]+)?$/m.test(source) && !/[;{}]/.test(source)) return 'yaml';
    if (/\binterface\s+\w+\s*(?:<[^>]+>)?\s*\{|\btype\s+\w+\s*=|:\s*(?:string|number|boolean)\b/.test(source)) {
        return /<\w+[\s>][\s\S]*<\/\w+>/.test(source) ? 'tsx' : 'typescript';
    }
    if (/^\s*(?:@charset|@import|@media)\b/m.test(source)
        || /(?:^|\})\s*[.#]?[\w-]+(?:\s+[.#]?[\w-]+)*\s*\{\s*[\w-]+\s*:/m.test(source)) return 'css';
    return 'javascript';
}

function collectStructuralBoundaries(node, maxBytes, boundaries, depth = 0) {
    if (!node || depth > 64) return;
    const children = Array.isArray(node.namedChildren) ? node.namedChildren : [];
    if (!children.length) return;

    for (const child of children) {
        if (!Number.isFinite(child.startIndex) || !Number.isFinite(child.endIndex)) continue;
        if (child.startIndex > node.startIndex) boundaries.add(child.startIndex);
        if (child.endIndex - child.startIndex > maxBytes && child.namedChildCount > 1) {
            collectStructuralBoundaries(child, maxBytes, boundaries, depth + 1);
        }
    }
}

/**
 * Turn AST boundaries into contiguous source ranges near the requested byte
 * budget. Ranges cover the complete input, including comments and whitespace.
 */
export function buildStructuralRanges(rootNode, totalBytes, maxBytes) {
    if (!rootNode || totalBytes <= 0) return [];
    const safeMax = Math.max(1, Number(maxBytes) || totalBytes);
    const boundaries = new Set([0, totalBytes]);
    collectStructuralBoundaries(rootNode, safeMax, boundaries);
    const points = [...boundaries]
        .filter(point => Number.isFinite(point) && point >= 0 && point <= totalBytes)
        .sort((a, b) => a - b);

    const ranges = [];
    let chunkStart = points[0] || 0;
    for (let index = 1; index < points.length; index++) {
        const candidateEnd = points[index];
        const previousBoundary = points[index - 1];
        if (candidateEnd - chunkStart > safeMax && previousBoundary > chunkStart) {
            ranges.push([chunkStart, previousBoundary]);
            chunkStart = previousBoundary;
        }
    }
    if (chunkStart < totalBytes) ranges.push([chunkStart, totalBytes]);
    return ranges.length ? ranges : [[0, totalBytes]];
}

async function loadRuntime() {
    if (!runtimePromise) {
        runtimePromise = import(WEB_TREE_SITTER_MODULE_URL).then(async module => {
            const Parser = module.default || module.Parser;
            if (!Parser?.init) throw new Error('The browser Tree-sitter runtime did not expose Parser.init');
            await Parser.init({
                locateFile: file => file.endsWith('.wasm') ? WEB_TREE_SITTER_WASM_URL : file
            });
            if (!Parser.Language) {
                throw new Error('The browser Tree-sitter runtime did not expose Parser.Language after initialization');
            }
            return { Parser, Language: Parser.Language };
        }).catch(error => {
            runtimePromise = null;
            throw error;
        });
    }
    return runtimePromise;
}

class BrowserTreeSitterBackend {
    constructor(Parser, languageName, grammar) {
        this.Parser = Parser;
        this.languageName = languageName;
        this.grammar = grammar;
    }

    hasLanguage(name) {
        return normalizeCodeLanguage(name) === this.languageName;
    }

    detectLanguageFromContent() {
        return this.languageName;
    }

    downloadedLanguages() {
        return [this.languageName];
    }

    process(source, config = {}) {
        const parser = new this.Parser();
        parser.setLanguage(this.grammar);
        const tree = parser.parse(source);
        if (!tree) {
            parser.delete();
            throw new Error(`Tree-sitter could not parse ${this.languageName} source`);
        }

        try {
            const root = tree.rootNode;
            const sourceBytes = new TextEncoder().encode(source);
            const decoder = new TextDecoder();
            const maxBytes = Math.max(1, Number(config.chunkMaxSize) || sourceBytes.length);
            const ranges = buildStructuralRanges(root, sourceBytes.length, maxBytes);
            const topLevel = root.namedChildren || [];

            return {
                metrics: {
                    errorCount: root.descendantsOfType('ERROR').length,
                    totalLines: source.split(/\r?\n/).length
                },
                structure: topLevel.map(node => ({
                    type: node.type,
                    startByte: node.startIndex,
                    endByte: node.endIndex
                })),
                imports: topLevel
                    .filter(node => /(?:import|include|require|use)/i.test(node.type))
                    .map(node => node.type),
                chunks: ranges.map(([startByte, endByte]) => ({
                    content: decoder.decode(sourceBytes.slice(startByte, endByte)),
                    startByte,
                    endByte
                }))
            };
        } finally {
            tree.delete();
            parser.delete();
        }
    }
}

export async function createBrowserCodeBackend(language) {
    const normalized = normalizeCodeLanguage(language);
    if (!normalized) throw new Error(`Unsupported code language: ${language}`);
    if (!backendPromises.has(normalized)) {
        backendPromises.set(normalized, (async () => {
            const { Parser, Language } = await loadRuntime();
            const grammarUrl = `${GRAMMAR_BASE_URL}/tree-sitter-${normalized}.wasm`;
            const grammar = await Language.load(grammarUrl);
            return new BrowserTreeSitterBackend(Parser, normalized, grammar);
        })().catch(error => {
            backendPromises.delete(normalized);
            throw new Error(`Could not load the ${normalized} Tree-sitter grammar: ${error.message}`);
        }));
    }
    return backendPromises.get(normalized);
}
