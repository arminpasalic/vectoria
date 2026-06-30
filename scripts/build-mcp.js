import { copyFileSync, mkdirSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const src = join(root, 'mcp-server');
const dest = join(root, 'web_interface', 'static', 'mcp-server');

function copyDir(from, to) {
  mkdirSync(to, { recursive: true });
  for (const entry of readdirSync(from)) {
    if (entry === 'node_modules' || entry === 'README.md' || entry === '.DS_Store') continue;
    const s = join(from, entry);
    const d = join(to, entry);
    if (statSync(s).isDirectory()) {
      copyDir(s, d);
    } else {
      copyFileSync(s, d);
      console.log(`  copied: static/mcp-server/${entry.replace(from, '')}`);
    }
  }
}

console.log('Building MCP server distribution...');
copyDir(src, dest);
console.log('Done.');
