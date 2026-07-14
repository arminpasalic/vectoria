#!/usr/bin/env node
/**
 * Stamps a single cache-busting version into every asset that needs one, so
 * the service worker's BUILD_ID and the ?v= query strings in index.html can
 * never drift out of sync (that drift caused browsers to keep loading old,
 * immutable-cached JS/CSS even after a deploy).
 *
 * Version format: YYYY-MM-DD-<content hash>  e.g. 2026-07-10-7ecaa2f
 *   - date     → human-readable, sortable
 *   - hash     → changes whenever the deployable web app changes
 *
 * Runs LOCALLY (pre-commit hook or `npm run stamp`), never on Vercel —
 * Vercel ignores buildCommand when a `builds` array is present, so the
 * stamped files must be committed.
 *
 * Usage:
 *   node scripts/stamp-version.js         # stamp the current worktree
 *   node scripts/stamp-version.js --check # exit 1 if anything is unstamped (CI guard)
 */
import { readFileSync, readdirSync, statSync, writeFileSync } from 'fs';
import { createHash } from 'crypto';
import { join, dirname, relative } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const web = join(root, 'web_interface');

const CHECK = process.argv.includes('--check');

const VERSION_TOKEN = /\d{4}-\d{2}-\d{2}-[0-9a-z]+/g;
const HASHED_EXTENSIONS = /\.(?:css|html|js|json|py|svg)$/;

function deployableFiles(dir = web) {
  return readdirSync(dir, { withFileTypes: true })
    .flatMap((entry) => {
      const path = join(dir, entry.name);
      const rel = relative(web, path);
      if (entry.isDirectory()) {
        if (rel === 'static/samples' || entry.name === 'node_modules') return [];
        return deployableFiles(path);
      }
      return HASHED_EXTENSIONS.test(entry.name) && statSync(path).isFile() ? [path] : [];
    })
    .sort();
}

function contentHash() {
  const hash = createHash('sha256');
  for (const file of deployableFiles()) {
    // Ignore existing cache stamps so running this script reaches a fixed point.
    const content = readFileSync(file, 'utf8').replace(VERSION_TOKEN, '<VERSION>');
    hash.update(relative(web, file));
    hash.update('\0');
    hash.update(content);
    hash.update('\0');
  }
  return hash.digest('hex').slice(0, 7);
}

function worktreeVersion() {
  const d = new Date();
  const iso = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
  return `${iso}-${contentHash()}`;
}

const VERSION = worktreeVersion();

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
    // The date records when stamping happened; the suffix must match today's
    // deployable content, and every target must carry the same full token.
    const found = before.match(t.find) || [];
    const tokens = found.flatMap((m) => m.match(VERSION_TOKEN) || []);
    const wrong = tokens.filter((token) => !token.endsWith(`-${contentHash()}`));
    if (wrong.length || !tokens.length) stale.push(`${t.file}: ${wrong.join(', ') || 'missing version token'}`);
    t.tokens = tokens;
    continue;
  }

  if (after !== before) {
    writeFileSync(t.file, after);
    const count = (before.match(t.find) || []).length;
    changed.push(`${t.file} (${count} ref${count === 1 ? '' : 's'})`);
  }
}

if (CHECK) {
  const allTokens = targets.flatMap((t) => t.tokens || []);
  if (new Set(allTokens).size > 1) stale.push(`Version tokens do not match: ${[...new Set(allTokens)].join(', ')}`);
  if (stale.length) {
    console.error(`✗ Unstamped/stale version tokens (content hash ${contentHash()}):`);
    stale.forEach((s) => console.error(`   ${s}`));
    console.error(`Run: npm run stamp`);
    process.exit(1);
  }
  console.log(`✓ All version tokens match current content (${allTokens[0]}).`);
  process.exit(0);
}

if (changed.length) {
  console.log(`Stamped version ${VERSION}:`);
  changed.forEach((c) => console.log(`  ${c}`));
} else {
  console.log(`Version already ${VERSION} — nothing to stamp.`);
}
