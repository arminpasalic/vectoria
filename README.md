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

A fully browser-native semantic search, exploration, and RAG system that runs 100% client-side.

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

All running **entirely in your browser using your device's hardware** with zero backend infrastructure.

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
- **UMAP visualization** - Interactive 2D semantic maps
- **Hybrid search** - Vector similarity + BM25 keyword matching
- **Persistent storage** - IndexedDB caching for instant reload
- **Privacy-first** - All data processing happens on your device

---

> [!WARNING]
> **System requirements & first load**
>
> * **Storage:** The first load will download AI models and requires **at least 5GB** of free disk space.
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

**RAG query (AI-powered)**
- Switch to "Semantic Search (RAG)".
- Ask: `"What are the main topics discussed?"`
- The local AI generates an answer with source citations.

---

## Project structure

```
vectoria/
├── web_interface/              # Main application
│   ├── index.html             # Entry point
│   ├── static/
│   │   ├── css/               # Styling
│   │   └── js/
│   │       ├── browser-ml/    # Core ML modules
│   │       │   ├── chunking/          # Document chunking (Chonkie.js)
│   │       │   ├── embedding/         # Embedding utilities
│   │       │   ├── umap-wasm/         # UMAP WebAssembly
│   │       │   ├── embeddings.js
│   │       │   ├── embedding-worker.js
│   │       │   ├── vector-search.js
│   │       │   ├── llm-rag.js
│   │       │   ├── llm-worker.js
│   │       │   ├── file-processor.js
│   │       │   ├── clustering.js
│   │       │   ├── hdbscan-pyodide.js
│   │       │   ├── pyodide-hdbscan-worker.js
│   │       │   ├── umap-wasm-adapter.js
│   │       │   ├── storage.js
│   │       │   ├── export.js
│   │       │   └── index.js
│   │       ├── browser-integration.js
│   │       ├── browser-capabilities.js
│   │       ├── config-manager.js
│   │       ├── export-import.js
│   │       ├── fast-search.js
│   │       ├── hyde-handler.js
│   │       ├── main.js
│   │       ├── model-constraints.js
│   │       ├── search-enhancement.js
│   │       ├── vectoria.js
│   │       ├── viz.js
│   │       └── webgl-renderer.js
│   └── sw.js                  # Service worker
├── package.json
└── vercel.json
```

### Browser ML modules

| Module | Purpose | Technology |
|--------|---------|------------|
| `embeddings.js` | Text → vectors | @huggingface/transformers (ONNX) |
| `vector-search.js` | Similarity + keyword search | JS HNSW + BM25 |
| `llm-rag.js` | RAG Q&A | WebLLM (multi-model) |
| `file-processor.js` | Parse CSV/Excel/JSON/TXT | PapaParse, SheetJS |
| `clustering.js` | UMAP + HDBSCAN clustering | UMAP-WASM, Pyodide |
| `storage.js` | Persistent caching | localforage (IndexedDB) |
| `chunking/` | Document chunking for RAG | Chonkie.js |
| `index.js` | Pipeline orchestration | Coordinates all modules |

---

## Tech stack

### Browser ML
- **Embeddings**: @huggingface/transformers + `multilingual-e5-small` (384D, 100+ languages)
- **LLM**: WebLLM with multi-model support (Gemma 2, Llama 3.2, Qwen 3, DeepSeek R1, SmolLM2, Phi 3.5)
- **Vector search**: Pure JavaScript HNSW implementation
- **Keyword search**: BM25 (TF-IDF ranking)
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

**All processing happens in your browser and on your device.**
- No data uploaded to servers.
- No tracking or analytics.
- No cookies (except localStorage/IndexedDB for caching).

**Clear data anytime**
- Use the "Delete Data & Cache" button in the app to remove indexed data and cached AI models.
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
2.  **Privacy-First** - Data never leaves the user's device.
3.  **Offline-Capable** - Works without an internet connection after initial load.
4.  **Production-Ready** - Complete, tested, and documented.

This proves that sophisticated AI applications can run entirely in the browser.

---

## MCP Server

Exposes Vectoria's semantic search, RAG, visualization data, clustering, metadata, and config as MCP tools to Claude Desktop, OpenCode, Cursor, Zed, Continue.dev, and any other MCP-compatible AI client.

### Prerequisites

- **Node.js v18+** — runs the MCP server. Install from [nodejs.org](https://nodejs.org).
- **Python 3** — used by the install script to configure your AI clients. macOS: `xcode-select --install`; Linux: install the `python3` package; or [python.org](https://python.org).

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
2. Go to **Advanced Settings → MCP Bridge** → enable the toggle
3. If prompted, allow Vectoria to access devices on your local network
4. Status shows **● Connected · Claude Desktop** (or whichever client connected)

> The MCP bridge runs locally on your machine (`ws://127.0.0.1:3700`). Your browser tab connects to it directly — no data leaves your computer, regardless of where Vectoria itself is hosted.

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

### Available Tools (16)

| Category | Tools |
|---|---|
| Search | `search`, `hybrid_search`, `get_document`, `get_documents_by_cluster` |
| RAG | `query_rag_local` (local ONNX), `query_rag_external` (AI client answers) |
| Data | `get_visualization_data`, `get_cluster_summary`, `get_dataset_stats`, `get_point_neighbors` |
| Metadata | `get_metadata_schema`, `set_metadata_filters`, `clear_metadata_filters` |
| Config | `get_config`, `set_config` |
| Dataset | `list_datasets`, `get_dataset_info` |

### Architecture

```
AI Client (stdio) → ~/.vectoria-mcp/index.js → BrowserBridge (ws://127.0.0.1:3700)
                                                       ↕
                                           SharedWorker in browser tab
                                                       ↕
                                           window.browserMLFetch() (existing pipeline)
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
