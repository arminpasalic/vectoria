<p align="center">
  <img src="web_interface/static/img/wordmark_light.svg" alt="Vectoria" width="300">
</p>

<p align="center">
  <strong>Browser-first text exploration, clustering, and semantic search</strong>
</p>

<p align="center">
  <a href="https://vectoria.app/">
    <img src="https://img.shields.io/badge/Launch-App-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Launch App" />
  </a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

Use AI to analyze, search, and ask questions of your private documents without uploading them—all running on your own device.

Vectoria helps you **search, map, compare, and understand text collections locally**, with every generated conclusion connected to inspectable evidence. The optional AI-client mode returns only retrieved excerpts or cluster exemplars to the AI provider connected through MCP.

🚀 **Built on Open-source** | 🔒 **Privacy-preserving** | 📊 **Interactive Visualization & Exploration** | 🤖 **AI-powered RAG**

<p align="center">
  <img src="web_interface/static/img/example.gif" alt="Vectoria demo" width="800">
</p>

---

## What is Vectoria?

Vectoria is a tool for **browser-first text exploration, clustering, and semantic search** that combines:
- **Document embedding** and semantic search
- **Interactive 2D visualization & exploration** via UMAP clustering
- **RAG (Retrieval-Augmented Generation)** for intelligent Q&A
- **Hybrid search** (vector + keyword)

Core processing runs **in your browser using your device's hardware** with zero hosted processing backend. The optional MCP integration adds a local relay so an AI client can use Vectoria's tools.

Unlike [Embedding Atlas](https://github.com/apple/embedding-atlas), Vectoria is centered on evidence-linked text investigation: multilingual hybrid retrieval, optional HyDE and reranking, local grounded synthesis, inspectable source passages, and MCP access. It does not position itself as a larger-scale embedding-map renderer.

## Potential use cases

- **Personal knowledge bases** - Organize notes, papers, documents.
- **Research data exploration** - Explore corpora and qualitative data.
- **Corporate document analysis** - Privacy-preserving semantic search.
- **Educational tools** - Learn about embeddings, RAG, and clustering.
- **Multilingual content** - Analyze text in multiple languages.

### Key features

- **Multi-format support** - CSV, Excel, JSON, TXT
- **Multilingual embeddings** - Multilingual-e5-small (100+ languages)
- **Browser-based LLM** - Multiple models (Gemma 2, Llama 3.2, Qwen 3, etc.) for RAG via WebGPU
- **Two generation modes** - Use a local browser model or answer through an MCP-connected AI client
- **UMAP visualization** - Interactive 2D semantic maps
- **Hybrid search** - Vector similarity + BM25 keyword matching
- **Optional multilingual reranking** - An advanced, browser-local MiniLM cross-encoder improves final search and evidence ordering
- **Cluster labelling** - Manual, Local AI, or AI client via MCP
- **Validated metadata filtering** - Shared filters across search, RAG, and analysis
- **Persistent storage** - IndexedDB caching for instant reload
- **Privacy-first** - Local by default, with explicit disclosure when excerpts are shared with an AI provider

The optional reranker has one fixed, reviewed model rather than a model picker. See [third-party model notices](docs/THIRD_PARTY_MODELS.md) for the pinned revision, browser artifacts, and license.

---

> [!WARNING]
> **System requirements & first load**
>
> * **Storage:** Models and processed datasets use Vectoria's browser-origin storage quota, not a separate desktop application folder. Required space depends on the selected model; larger local LLMs can require several gigabytes.
> * **Hardware:** A device with a dedicated GPU or modern integrated graphics (Apple Silicon M1/M2/M3) is strongly recommended.
>
> **Browser requirements:**
> * **Chrome/Edge 120+** (WebGPU support required)
> * **8GB+ RAM** (16GB recommended for large datasets)
> * **Stable internet** (first load only - cached thereafter)

---

## How to use

### 1. Load your data
Select and load:
- CSV files (with header row)
- Excel spreadsheets (.xlsx, .xls)
- JSON files (array of objects)
- Plain text files (.txt)

### 2. Process & visualize
1. Select the text column to analyze.
2. Click "Process Data".
3. **Watch progress:** Parsing → Embedding → UMAP dimensionality reduction → HDBSCAN clustering → Done.
4. The interactive 2D visualization will appear automatically - ready for exploration and search.

### 3. Search & explore

**Fast search**
- Type a query: `"machine learning applications"`
- Get instant results with text highlighting. Supports keywords and Boolean logic.

**Semantic search**
- Type a conceptual query.
- Get results that are semantically similar to your query, even if they don't share keywords.

**Ask your documents**
- Open **Ask** beside **Documents** in the right-hand workspace.
- Ask a question, then continue with useful follow-ups in the same local investigation. Each answer retrieves fresh evidence and preserves its sources.
- Choose **All documents** or **Current view** before sending. Citations such as `[Doc 1]` open the full source and focus its point in the visualization.
- Use Ask settings to toggle HyDE, choose evidence metadata, set the source count and similarity threshold, and select Smart summary, Recent turns only, or This question only memory.
- The full Search & answers and Models controls remain available from Ask, including prompts, retrieval weighting, generation parameters, model choice, and context-window size. Settings are captured when a question is sent.
- Ask history is stored locally per dataset and can be copied or exported as Markdown, JSON, or CSV. **Forget earlier model context** excludes existing turns from future prompts without deleting them or removing them from exports.
- In **AI client via MCP** mode, website Ask is read-only and the investigation continues in the connected client. Existing RAG MCP tools remain available, and callers can request `scope: "current"`.

Vectoria budgets each local Ask prompt against the configured model context window. Current retrieved evidence has priority over follow-up continuity; older turns are deterministically condensed or omitted according to the selected memory mode. An oversized question is rejected with a size explanation rather than silently truncated.

### Cluster labels

- **Manual** — rename clusters directly at any time.
- **Local AI** — use the browser model to create labels.
- **AI client via MCP** — Vectoria prepares exemplars and an instruction; the connected AI client calls `summarize_cluster` and `set_cluster_label`.
- Labels show lightweight provenance such as Manual, Local AI, or the connected client.
- Labels are scoped to the active dataset and are cleared when the dataset changes. Matching analysis sessions or exports can restore them.

### Advanced Settings

The current tab order is:

1. **General**
2. **Models**
3. **Data preparation**
4. **Search & answers**
5. **Projection & clustering**
6. **Integrations**

Document chunking is configured under **Data preparation**. Available Chonkie-based strategies are:

- **Token** — fixed size with overlap
- **Recursive** — preserves paragraphs and sentences
- **Sentence-aware** — configurable sentence boundaries and delimiters
- **Semantic** — finds topic boundaries with the selected local embedding model
- **Code** — preserves source structure with browser Tree-sitter WASM
- **Table** — splits Markdown or HTML tables and repeats their headers
- **Fast/WASM** — UTF-8 byte boundary detection

All seven chunkers from `@chonkiejs/core` are available. Code grammars are
loaded lazily on first use, and all chunking remains in the browser. Chunking
changes apply when the next dataset is processed.

---

## Project structure

```
vectoria/
├── web_interface/                  # Main browser application
│   ├── index.html                  # Entry point and MCP browser routes
│   ├── static/
│   │   ├── css/                   # Styling
│   │   ├── js/
│   │   │   ├── browser-ml/        # Embeddings, search, RAG, clustering, storage
│   │   │   │   ├── chunking/      # Chonkie chunking strategies
│   │   │   │   ├── analysis.js
│   │   │   │   ├── metadata-filters.js
│   │   │   │   ├── sessions-store.js
│   │   │   │   ├── vector-search.js
│   │   │   │   └── index.js
│   │   │   ├── browser-integration.js
│   │   │   ├── vectoria.js
│   │   │   └── viz.js
│   │   ├── install-mcp.sh         # macOS/Linux MCP installer
│   │   ├── install-mcp.ps1        # Windows PowerShell MCP installer
│   │   └── mcp-server/            # Downloadable MCP bundle
│   └── sw.js                      # Service worker
├── mcp-server/                    # MCP source and 33 tool registrations
│   ├── bridge.js
│   ├── index.js
│   └── tools/
├── scripts/                       # MCP build and cache-version stamping
├── tests/                         # Node regression and security-safety tests
├── package.json
└── vercel.json
```

### Browser ML modules

| Module | Purpose | Technology |
|--------|---------|------------|
| `embeddings.js` | Text → vectors | @huggingface/transformers (ONNX) |
| `vector-search.js` | Similarity + keyword search | Exact flat cosine search + BM25 |
| `llm-rag.js` | RAG Q&A | WebLLM (multi-model) |
| `file-processor.js` | Parse CSV/Excel/JSON/TXT | PapaParse, SheetJS |
| `clustering.js` | UMAP + HDBSCAN clustering | UMAP-WASM, Pyodide |
| `metadata-filters.js` | Validated filters shared by search, RAG, and analysis | JavaScript |
| `analysis.js` | Aggregation, cross-tabs, outliers, subsets, and multi-query search | JavaScript |
| `storage.js` | Persistent caching | localforage (IndexedDB) |
| `chunking/` | Document chunking for RAG | Chonkie.js |
| `index.js` | Pipeline orchestration | Coordinates all modules |

---

## Tech stack

### Browser ML
- **Embeddings**: @huggingface/transformers + `multilingual-e5-small` (384D, 100+ languages)
- **LLM**: WebLLM with multi-model support (Gemma 2, Llama 3.2, Qwen 3, DeepSeek R1, SmolLM2, Phi 3.5)
- **Vector search**: Exact flat cosine similarity with pre-ranking predicates
- **Keyword search**: BM25 (TF-IDF ranking)
- **Optional reranker**: multilingual mMiniLMv2 cross-encoder, q8 ONNX/WASM, disabled by default
- **Dimensionality reduction**: UMAP-WASM (WebAssembly accelerated)
- **Clustering**: HDBSCAN via Pyodide (scikit-learn in browser)
- **Document chunking**: Chonkie.js
- **Storage**: IndexedDB via localforage

### Frontend
- **UI**: Vanilla JavaScript
- **Visualization**: WebGL (hardware-accelerated)
- **Styling**: Custom CSS with glassmorphism
- **Icons**: Font Awesome

### Infrastructure
- **Hosting**: Vercel (edge network)
- **Models**: Loaded from HuggingFace CDN

---

## Privacy & security

**Processing is local by default.**
- Files, embeddings, retrieval, filtering, clustering, and visualization stay in the browser.
- Local browser-model generation stays on the device.
- In **AI client via MCP** mode, only retrieved excerpts or cluster exemplars needed for the requested operation are returned to the connected AI client and its configured provider.
- Vectoria's hosted application does not receive your dataset for processing.
- No tracking or analytics.
- Browser state and cached data use localStorage, Cache Storage, and IndexedDB in the `vectoria.app` browser origin.

**Clear data anytime**
- **Check browser storage** reports estimated usage and remaining quota for the Vectoria browser origin. It is not your computer's total disk usage.
- **Reset and clear all data** removes Vectoria's stored datasets, cached models, settings, sessions, and service-worker caches, then reloads the page.
- Or clear via your browser's "Clear Browsing Data" settings.

---

## Troubleshooting

> **Note:** Processing speed and analysis capabilities are entirely dependent on your device's power.

### Models won't load
Enable WebGPU in your browser:
1. Go to `chrome://flags`
2. Search "WebGPU"
3. Set to **Enabled**

### Out of Memory
- Use a smaller dataset (<10K docs).
- Close other resource-heavy tabs.
- Use a desktop browser (mobile browsers allocate less memory).

### Slow Performance
- Ensure Hardware Acceleration is enabled in browser settings.
- Reduce the number of UMAP neighbors in settings.

---

## Acknowledgments

**Technologies**:
- [@huggingface/transformers](https://huggingface.co/docs/transformers.js) by Hugging Face
- [WebLLM](https://github.com/mlc-ai/web-llm) by MLC AI
- [UMAP](https://apple.github.io/embedding-atlas/algorithms.html#umap) WebAssembly implementation by Apple
- [localforage](https://github.com/localForage/localForage) by Mozilla
- [Chonkie](https://github.com/chonkie-inc/chonkiejs) for document chunking
- [Pyodide](https://pyodide.org/en/stable/) for Python distribution in the browser

Thank you for making this possible.

---

## What makes this special

1.  **Zero Infrastructure & Zero Cost** - No servers, no databases, no backend.
2.  **Privacy-First** - Processing stays local unless the user explicitly chooses AI client mode, where bounded excerpts or exemplars are shared with that provider.
3.  **Offline-Capable** - Works without an internet connection after initial load.
4.  **Production-Ready** - Complete, tested, and documented.

This proves that sophisticated AI applications can run entirely in the browser.

---

## Development checks

From the repository root, install dependencies and run the complete local
verification suite before a release:

```bash
npm ci
npm ci --prefix web_interface/static/js/browser-ml/umap-wasm
npm ci --prefix mcp-server
npm run check
```

This runs the Node tests, the UMAP-WASM tests, JavaScript syntax checks, the
MCP distribution build, and the service-worker stamp check. Focused commands
remain available as `npm run test:filters`, `npm run test:coverage`, and
`npm run test:wasm`.

Browser runtime libraries are version-pinned at their import sites and cached
by the browser. The root npm install contains development tooling only; MCP and
UMAP-WASM keep their own independently locked dependencies.

---

## MCP Server

Exposes Vectoria's semantic search, RAG, visualization data, clustering, metadata, and config as MCP tools to Claude Desktop, OpenCode, Cursor, Zed, Continue.dev, and any other MCP-compatible AI client.

### Prerequisites

- **Node.js v18+** — runs the MCP server. Install from [nodejs.org](https://nodejs.org).
- **macOS/Linux:** Python 3 is used by the shell installer to update detected AI-client configuration files. macOS users can run `xcode-select --install`; Linux users can install the `python3` package.
- **Windows:** PowerShell 5.1+ is required. The PowerShell installer does not require Python.

### Setup

**1. Install dependencies**

```bash
cd mcp-server
npm install
```

**2. Run the installer (auto-configures all detected clients)**

macOS / Linux:

```bash
curl -fsSL https://vectoria.app/static/install-mcp.sh | bash
```

Windows PowerShell:

```powershell
$script = Invoke-RestMethod 'https://vectoria.app/static/install-mcp.ps1'; & ([scriptblock]::Create($script))
```

The script detects which AI clients are installed on your machine and configures all of them automatically. Example output:

```
✅ Configured:  Claude Desktop, Cursor
⏭  Not found:   OpenCode, Zed, Continue.dev
```

**3. Enable in Vectoria**

1. Open Vectoria in your browser (e.g. `https://vectoria.app` or your local `http://localhost:5050`)
2. Go to **Settings → Integrations** → enable the MCP Bridge
3. If prompted, allow Vectoria to access devices on your local network
4. Status shows **● Connected · Claude Desktop** (or whichever client connected)

> The MCP bridge runs locally on your machine (`ws://127.0.0.1:3700`). The hosted Vectoria tab connects directly to that local relay. Dataset processing remains in the browser; AI-client mode can return retrieved excerpts or cluster exemplars to the connected client/provider.

### Generation modes

- **Local browser model** — generation stays in Vectoria and uses local GPU/RAM.
- **AI client via MCP** — unloads the local LLM while preserving cached model files. Retrieval stays in Vectoria, and the AI client answers in its own chat using `query_rag_external`.

MCP tool calls are initiated by the AI client. Vectoria cannot independently push a prompt into every possible MCP client, so UI handoffs prepare an instruction that the user runs in the connected client.

### Supported Clients

| Client | Auto-configured | Config path |
|---|---|---|
| Claude Desktop | ✅ | macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`; Windows: `%APPDATA%\Claude\claude_desktop_config.json` |
| Cursor | ✅ | `~/.cursor/mcp.json` |
| OpenCode | ✅ | `~/.config/opencode/config.json` |
| Zed | ✅ | macOS/Linux: `~/.config/zed/settings.json`; Windows: `%APPDATA%\Zed\settings.json` |
| Continue.dev | ✅ | `~/.continue/config.json` |

**Manual config (if needed)**

Claude Desktop / Cursor (`mcpServers` object):
```json
{
  "mcpServers": {
    "vectoria": {
      "command": "/absolute/path/to/node",
      "args": ["/Users/you/.vectoria-mcp/index.js", "--allowed-origin", "https://vectoria.app"]
    }
  }
}
```

OpenCode (`~/.config/opencode/config.json`):
```json
{
  "mcp": {
    "vectoria": {
      "type": "local",
      "command": ["/absolute/path/to/node", "/Users/you/.vectoria-mcp/index.js", "--allowed-origin", "https://vectoria.app"]
    }
  }
}
```

Zed (`~/.config/zed/settings.json`):
```json
{
  "context_servers": {
    "vectoria": {
      "command": {
        "path": "/absolute/path/to/node",
        "args": ["/Users/you/.vectoria-mcp/index.js", "--allowed-origin", "https://vectoria.app"]
      }
    }
  }
}
```

Continue.dev (`~/.continue/config.json`):
```json
{
  "mcpServers": [
    {
      "name": "vectoria",
      "command": "/absolute/path/to/node",
      "args": ["/Users/you/.vectoria-mcp/index.js", "--allowed-origin", "https://vectoria.app"]
    }
  ]
}
```

### Available Tools (33)

| Category | Tools |
|---|---|
| Search | `search`, `hybrid_search`, `get_document`, `get_documents_by_cluster` |
| RAG | `query_rag_local` (local WebLLM), `query_rag_external` (AI client answers) |
| Data | `get_visualization_data`, `get_cluster_summary`, `get_dataset_stats`, `get_point_neighbors` |
| Metadata | `get_metadata_schema`, `set_metadata_filters`, `clear_metadata_filters` |
| Config | `get_config`, `set_config` |
| Dataset | `list_datasets`, `get_dataset_info` |
| Analysis | `summarize_cluster`, `get_outliers`, `cross_tabulate`, `aggregate`, `compare_clusters`, `multi_vector_search`, `query_with_citations`, `filter_to_subset` |
| Annotations | `annotate_documents`, `list_annotations` |
| Cluster labels | `set_cluster_label` |
| Metrics | `register_metric`, `list_metrics` |
| Sessions | `save_analysis_session`, `list_sessions`, `load_analysis_session` |

### Metadata filtering

- `set_metadata_filters` validates and stores persistent filters for the active dataset.
- Inline filters on a tool call override persistent values for the same fields and combine with the remaining persistent fields.
- Persistent filters reset when the active dataset changes or is cleared. Browser UI filters are separate from MCP filter state.
- Filters are applied before BM25, semantic, hybrid, RAG, and multi-vector ranking.
- Filter-aware search and analysis responses report the active fields plus matched and total document counts.
- Invalid fields and malformed filters return errors; zero matches return valid empty results rather than unfiltered data.

### RAG scope and cluster labelling

- Website Ask uses **All documents** by default and can capture the active filtered or lasso view for a turn.
- Persistent Ask investigations remain local and are not exposed through MCP; AI-client history stays in the client.
- `query_rag_external` can explicitly use `scope: "current"` to retrieve from the active selection or filtered view.
- Cluster labels can be Manual, Local AI, or AI client generated. AI-client labelling uses `summarize_cluster` followed by `set_cluster_label`.
- Label provenance is displayed in the UI. Labels belong to the current dataset and do not carry into another dataset.

### Architecture

```
AI Client (stdio) → ~/.vectoria-mcp/index.js → BrowserBridge (ws://127.0.0.1:3700)
                                                       ↕
                                           SharedWorker in browser tab
                                                       ↕
                                           window.browserMLFetch() (existing pipeline)
```

### MCP troubleshooting

**Waiting for local MCP client**

- Fully quit the AI client, including any background process, then reopen it.
- Confirm the client has a Vectoria MCP entry and launches `~/.vectoria-mcp/index.js` (Windows: `%USERPROFILE%\.vectoria-mcp\index.js`).
- Keep Vectoria open and enable **Settings → Integrations → MCP Bridge**.

**Local relay not reachable**

- If the browser requests Local Network Access for `vectoria.app`, choose **Allow**.
- If access was previously blocked, enable it in the browser's site settings, then toggle the MCP Bridge off and on.
- The relay listens only on `ws://127.0.0.1:3700`; it is launched automatically by the configured AI client.

**MCP relay update required**

- Re-run the installer to replace the local MCP files with the current relay protocol.
- Fully quit and restart the AI client after reinstalling.
- Reload Vectoria and re-enable the bridge.

macOS/Linux reinstall:

```bash
curl -fsSL https://vectoria.app/static/install-mcp.sh | bash -s -- --base-url https://vectoria.app
```

Windows PowerShell reinstall:

```powershell
$script = Invoke-RestMethod 'https://vectoria.app/static/install-mcp.ps1'; & ([scriptblock]::Create($script)) -BaseUrl 'https://vectoria.app'
```

### Uninstall

macOS / Linux:

```bash
curl -fsSL https://vectoria.app/static/install-mcp.sh | bash -s -- --uninstall
```

Windows PowerShell:

```powershell
$script = Invoke-RestMethod 'https://vectoria.app/static/install-mcp.ps1'; & ([scriptblock]::Create($script)) -Uninstall
```

Removes `~/.vectoria-mcp/` (or `%USERPROFILE%\.vectoria-mcp` on Windows) and cleans the Vectoria entry from all detected client configs.

---

## Contributing

Contributions are incredibly welcome! Whether you're fixing a bug, improving the docs, or adding a new feature, every bit helps.

Feel free to:
- **Fork** the repository and experiment.
- **Submit Pull Requests** for bug fixes or new features.
- **Report Issues** if you find bugs or have ideas.

**Support the project:**
If you find Vectoria interesting or useful, **please give it a star ⭐**! It helps more people discover the project.

---

**Vectoria** - Semantic exploration without limits.

MIT License - Free for personal and commercial use.
