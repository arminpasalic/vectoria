#!/usr/bin/env bash
# Installs a pre-commit hook that auto-stamps the cache-busting version
# (sw.js BUILD_ID + index.html ?v=) whenever web assets are part of the commit.
# Run once:  bash scripts/install-git-hooks.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$ROOT/.git/hooks/pre-commit"

cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# Auto-stamp the cache-busting version when web assets change, so sw.js's
# BUILD_ID and index.html's ?v= strings can never drift out of sync.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"

# Only stamp if something under web_interface/ or scripts/ is staged.
if git diff --cached --name-only | grep -qE '^(web_interface/|scripts/)'; then
  node "$ROOT/scripts/stamp-version.js"
  # Re-stage the files the stamper may have touched.
  git add "$ROOT/web_interface/sw.js" "$ROOT/web_interface/index.html"
fi
HOOK_EOF

chmod +x "$HOOK"
echo "✅ Installed pre-commit hook at .git/hooks/pre-commit"
echo "   It runs 'npm run stamp' automatically when web assets are committed."
