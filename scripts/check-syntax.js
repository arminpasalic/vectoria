#!/usr/bin/env node

import { readdirSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = resolve(import.meta.dirname, '..');
const sourceRoots = ['web_interface/static/js', 'mcp-server', 'scripts', 'tests'];
const extensions = new Set(['.js', '.mjs']);

function collectJavaScript(directory) {
    return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
        if (entry.name === 'node_modules') return [];
        const path = join(directory, entry.name);
        if (entry.isDirectory()) return collectJavaScript(path);
        return extensions.has(extname(entry.name)) ? [path] : [];
    });
}

const files = sourceRoots.flatMap(directory => collectJavaScript(join(root, directory))).sort();
const failures = [];

for (const file of files) {
    const result = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
    if (result.status !== 0) failures.push(result.stderr || result.stdout || file);
}

if (failures.length) {
    console.error(failures.join('\n'));
    process.exit(1);
}

console.log(`Syntax check passed for ${files.length} JavaScript files.`);
