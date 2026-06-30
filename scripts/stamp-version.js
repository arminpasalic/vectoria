#!/usr/bin/env node
/**
 * Stamps a single cache-busting version into every asset that needs one, so
 * the service worker's BUILD_ID and the ?v= query strings in index.html can
 * never drift out of sync (that drift caused browsers to keep loading old,
 * immutable-cached JS/CSS even after a deploy).
 *
 * Version format: YYYY-MM-DD-<short git sha>  e.g. 2026-06-30-7ecaa2f
 *   - date     → human-readable, sortable
 *   - git sha  → unique per commit (no same-day collisions, traceable)
 *
 * Falls back to a timestamp-ish suffix if git is unavailable.
 *
 * Runs LOCALLY (pre-commit hook or `npm run stamp`), never on Vercel —
 * Vercel ignores buildCommand when a `builds` array is present, so the
 * stamped files must be committed.
 *
 * Usage:
 *   node scripts/stamp-version.js        # stamp using current HEAD
 *   node scripts/stamp-version.js --check # exit 1 if anything is unstamped (CI guard)
 */
import { readFileSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const web = join(root, 'web_interface');

const CHECK = process.argv.includes('--check');

function gitVersion() {
  try {
    const date = execSync('git show -s --format=%cs HEAD', { cwd: root })
      .toString().trim(); // committer date, YYYY-MM-DD
    const sha = execSync('git rev-parse --short HEAD', { cwd: root })
      .toString().trim();
    if (date && sha) return `${date}-${sha}`;
  } catch (_) { /* fall through */ }
  // Fallback: no git (e.g. shallow checkout). Use a stable-ish suffix.
  const d = new Date();
  const iso = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
  return `${iso}-nogit`;
}

const VERSION = gitVersion();
const VPATTERN = /\d{4}-\d{2}-\d{2}-[0-9a-z]+/g; // matches both old (-01) and new (-<sha>) styles

// Each target: file + a regex whose capture group 1 is the version token.
const targets = [
  {
    file: join(web, 'index.html'),
    // ?v=<version> on any asset URL
    re: /(\?v=)\d{4}-\d{2}-\d{2}-[0-9a-z]+/g,
    apply: (s) => s.replace(/(\?v=)\d{4}-\d{2}-\d{2}-[0-9a-z]+/g, `$1${VERSION}`),
    find: /\?v=\d{4}-\d{2}-\d{2}-[0-9a-z]+/g,
  },
  {
    file: join(web, 'sw.js'),
    // const BUILD_ID = '<version>';
    apply: (s) => s.replace(/(const BUILD_ID = ')\d{4}-\d{2}-\d{2}-[0-9a-z]+(';)/, `$1${VERSION}$2`),
    find: /const BUILD_ID = '\d{4}-\d{2}-\d{2}-[0-9a-z]+';/g,
  },
];

let changed = [];
let stale = [];

for (const t of targets) {
  const before = readFileSync(t.file, 'utf8');
  const after = t.apply(before);

  if (CHECK) {
    // In check mode, flag any version token that isn't the current VERSION.
    const found = before.match(t.find) || [];
    const wrong = found.filter((m) => !m.includes(VERSION));
    if (wrong.length) stale.push(`${t.file}: ${wrong.join(', ')}`);
    continue;
  }

  if (after !== before) {
    writeFileSync(t.file, after);
    const count = (before.match(t.find) || []).length;
    changed.push(`${t.file} (${count} ref${count === 1 ? '' : 's'})`);
  }
}

if (CHECK) {
  if (stale.length) {
    console.error(`✗ Unstamped/stale version tokens (expected ${VERSION}):`);
    stale.forEach((s) => console.error(`   ${s}`));
    console.error(`Run: npm run stamp`);
    process.exit(1);
  }
  console.log(`✓ All version tokens are current (${VERSION}).`);
  process.exit(0);
}

if (changed.length) {
  console.log(`Stamped version ${VERSION}:`);
  changed.forEach((c) => console.log(`  ${c}`));
} else {
  console.log(`Version already ${VERSION} — nothing to stamp.`);
}
