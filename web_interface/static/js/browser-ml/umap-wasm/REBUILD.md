# Rebuilding UMAP-WASM with Increased Memory

The current `runtime.js` WASM binary has memory limits that may cause "memory access out of bounds" errors for datasets with many dimensions or points.

## Quick Fix: Rebuild with More Memory

### 1. Install Emscripten

```bash
# Clone and setup Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### 2. Download Dependencies

```bash
cd /path/to/vectoria-browser/web_interface/static/js/browser-ml/umap-wasm/third_party
./download_dependencies.sh
```

### 3. Rebuild WASM Module

```bash
cd /path/to/vectoria-browser/web_interface/static/js/browser-ml/umap-wasm
make clean
make
```

This will create a new `runtime.js` with:
- **Initial Memory**: 128MB (was ~16MB)
- **Maximum Memory**: 2GB (allows growth as needed)

### Memory Requirements

For reference, approximate memory needed:
- **Small datasets** (<1K points): ~10MB
- **Medium datasets** (1K-10K points): ~50-200MB
- **Large datasets** (10K-100K points): ~500MB-1GB
- **Very large datasets** (>100K points): >1GB

### What Changed in Makefile

Added these flags to the `emcc` command (in bytes):
```makefile
-sINITIAL_MEMORY=134217728 \    # 128MB
-sMAXIMUM_MEMORY=2147483648 \   # 2GB
```

Note: Emscripten 4.x requires byte values, not MB/GB suffix notation.

## Alternative: Use Pre-built Binary

If you cannot install Emscripten, you can:

1. Ask the maintainers to provide a pre-built `runtime.js` with increased memory
2. Or temporarily use a smaller subset of your data until the rebuild is available

## Technical Details

The "memory access out of bounds" error occurs because:
1. UMAP builds internal data structures (HNSW index, neighbor graphs)
2. These structures grow with dataset size and dimensionality
3. The original WASM binary had a conservative memory limit
4. Modern browsers support WASM modules up to 4GB

The fix ensures the WASM module can grow to handle real-world datasets.
