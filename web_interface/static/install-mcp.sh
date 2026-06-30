#!/usr/bin/env bash
# Vectoria MCP Server installer
# Usage:     curl -fsSL https://your-vectoria-url/static/install-mcp.sh | bash -s -- --base-url https://your-vectoria-url
# Uninstall: curl -fsSL https://your-vectoria-url/static/install-mcp.sh | bash -s -- --uninstall

set -euo pipefail

INSTALL_DIR="$HOME/.vectoria-mcp"
BASE_URL="https://vectoria.vercel.app"
UNINSTALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uninstall) UNINSTALL=true; shift ;;
    --base-url)  BASE_URL="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ── Client config definitions ─────────────────────────────────────────────────
# Each entry: "display_name|config_path|format"
# Formats: mcp_object (Claude Desktop / Cursor), opencode, zed, continue

declare -a CLIENTS=(
  "Claude Desktop|$HOME/Library/Application Support/Claude/claude_desktop_config.json|mcp_object"
  "Cursor|$HOME/.cursor/mcp.json|mcp_object"
  "OpenCode|$HOME/.config/opencode/config.json|opencode"
  "Zed|$HOME/.config/zed/settings.json|zed"
  "Continue.dev|$HOME/.continue/config.json|continue"
)

patch_add() {
  local cfg_path="$1"
  local format="$2"
  local entry="$3"

  python3 - "$cfg_path" "$format" "$entry" <<'EOF'
import sys, json, os

path, fmt, entry = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {}
if os.path.exists(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        pass

if fmt == 'mcp_object':
    cfg.setdefault('mcpServers', {})['vectoria'] = {
        'command': 'node', 'args': [entry]
    }
elif fmt == 'opencode':
    cfg.setdefault('mcp', {})['vectoria'] = {
        'type': 'local', 'command': ['node', entry]
    }
elif fmt == 'zed':
    cfg.setdefault('context_servers', {})['vectoria'] = {
        'command': {'path': 'node', 'args': [entry]}
    }
elif fmt == 'continue':
    servers = cfg.setdefault('mcpServers', [])
    servers = [s for s in servers if s.get('name') != 'vectoria']
    servers.append({'name': 'vectoria', 'command': 'node', 'args': [entry]})
    cfg['mcpServers'] = servers

with open(path, 'w') as f:
    json.dump(cfg, f, indent=2)
EOF
}

patch_remove() {
  local cfg_path="$1"
  local format="$2"

  [[ -f "$cfg_path" ]] || return 0

  python3 - "$cfg_path" "$format" <<'EOF'
import sys, json, os

path, fmt = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    sys.exit(0)
try:
    with open(path) as f:
        cfg = json.load(f)
except Exception:
    sys.exit(0)

removed = False
if fmt in ('mcp_object', 'opencode', 'zed'):
    key = {'mcp_object': 'mcpServers', 'opencode': 'mcp', 'zed': 'context_servers'}[fmt]
    if key in cfg and 'vectoria' in cfg[key]:
        del cfg[key]['vectoria']
        removed = True
elif fmt == 'continue':
    before = len(cfg.get('mcpServers', []))
    cfg['mcpServers'] = [s for s in cfg.get('mcpServers', []) if s.get('name') != 'vectoria']
    removed = before != len(cfg['mcpServers'])

if removed:
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✅ Removed from {path}")
else:
    print(f"  ℹ️  Not found in {path}")
EOF
}

# ── UNINSTALL ─────────────────────────────────────────────────────────────────
if $UNINSTALL; then
  echo "🗑  Uninstalling Vectoria MCP server..."

  if [[ -d "$INSTALL_DIR" ]]; then
    rm -rf "$INSTALL_DIR"
    echo "✅ Removed $INSTALL_DIR"
  else
    echo "ℹ️  $INSTALL_DIR not found, already removed."
  fi

  echo "Cleaning up client configs..."
  for entry in "${CLIENTS[@]}"; do
    IFS='|' read -r name cfg_path format <<< "$entry"
    patch_remove "$cfg_path" "$format" && true
  done

  echo ""
  echo "✅ Uninstall complete. Restart your AI clients to apply changes."
  exit 0
fi

# ── INSTALL ───────────────────────────────────────────────────────────────────
echo "🚀 Installing Vectoria MCP server to $INSTALL_DIR"

if ! command -v node &>/dev/null; then
  echo "❌ Node.js not found. Install v18+ from https://nodejs.org and re-run."
  exit 1
fi
if ! node -e "if(parseInt(process.versions.node)<18)process.exit(1)" 2>/dev/null; then
  echo "❌ Node.js v18+ required (you have $(node -v)). Upgrade at https://nodejs.org"
  exit 1
fi
echo "✅ Node.js $(node -v)"

if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found (used to configure your AI clients)."
  echo "   macOS: run 'xcode-select --install', or install from https://python.org"
  echo "   Linux: install the 'python3' package, then re-run."
  exit 1
fi
echo "✅ python3 $(python3 --version 2>&1 | awk '{print $2}')"

mkdir -p "$INSTALL_DIR/tools"

dl() {
  local file="$1"
  local url="$BASE_URL/static/mcp-server/$file"
  local dest="$INSTALL_DIR/$file"
  echo "   ↓ $file"
  curl -fsSL "$url" -o "$dest" || { echo "❌ Failed to download $url"; exit 1; }
}

echo "📦 Downloading MCP server files..."
dl "package.json"
dl "index.js"
dl "bridge.js"
dl "tools/search.js"
dl "tools/rag.js"
dl "tools/data.js"
dl "tools/metadata.js"
dl "tools/config.js"
dl "tools/dataset.js"
dl "tools/analysis.js"
dl "tools/annotations.js"
dl "tools/clusters.js"
dl "tools/metrics.js"
dl "tools/sessions.js"

# Verify package.json is valid JSON
if ! python3 -c "import json; json.load(open('$INSTALL_DIR/package.json'))" 2>/dev/null; then
  echo "❌ Downloaded package.json is not valid JSON."
  echo "   Make sure $BASE_URL is accessible and serving the correct files."
  rm -rf "$INSTALL_DIR"
  exit 1
fi

echo "📦 Installing Node dependencies..."
cd "$INSTALL_DIR"
npm install 2>&1 | grep -E "added|warn|error" || true
echo "✅ Dependencies installed"

# ── Patch all detected clients ────────────────────────────────────────────────
echo ""
echo "🔧 Configuring AI clients..."

CONFIGURED=()
SKIPPED=()

for entry in "${CLIENTS[@]}"; do
  IFS='|' read -r name cfg_path format <<< "$entry"
  cfg_dir="$(dirname "$cfg_path")"

  if [[ -d "$cfg_dir" || -f "$cfg_path" ]]; then
    mkdir -p "$cfg_dir"
    patch_add "$cfg_path" "$format" "$INSTALL_DIR/index.js"
    CONFIGURED+=("$name")
    echo "   ✅ $name"
  else
    SKIPPED+=("$name")
  fi
done

echo ""
if [[ ${#CONFIGURED[@]} -gt 0 ]]; then
  echo "✅ Configured:  $(IFS=', '; echo "${CONFIGURED[*]}")"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "⏭  Not found:   $(IFS=', '; echo "${SKIPPED[*]}")"
fi

echo ""
echo "✅ Vectoria MCP server installed!"
echo ""
echo "   Next steps:"
echo "   1. Restart your AI client(s): $(IFS=', '; echo "${CONFIGURED[*]}")"
echo "   2. Open Vectoria in your browser"
echo "   3. Advanced Settings → MCP Bridge → enable the toggle"
echo "   4. Status turns 🟢 Connected · <client name>"
echo ""
echo "   To uninstall:"
echo "   curl -fsSL $BASE_URL/static/install-mcp.sh | bash -s -- --uninstall --base-url $BASE_URL"
