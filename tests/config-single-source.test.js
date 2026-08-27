import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { DEFAULT_CONFIG } from '../web_interface/static/js/config-manager.js';

const read = path => readFile(new URL(`../web_interface/${path}`, import.meta.url), 'utf8');

const [configManager, llmRag, fastSearch, vectoria, browserIntegration, index] = await Promise.all([
    read('static/js/config-manager.js'),
    read('static/js/browser-ml/llm-rag.js'),
    read('static/js/fast-search.js'),
    read('static/js/vectoria.js'),
    read('static/js/browser-integration.js'),
    read('index.html')
]);

/**
 * config-manager.js is the single source of truth for shipped defaults. These
 * tests fail if a default is duplicated elsewhere, which is how the prompt and
 * model-id copies previously drifted apart.
 */

test('the shipped model id appears only in config-manager and authoritative model metadata', async () => {
    const modelId = DEFAULT_CONFIG.llm.model_id;
    const modelConstraints = await read('static/js/model-constraints.js');

    // The picker is rendered from model metadata, so the default id itself only
    // needs to exist in configuration and the authoritative catalog.
    assert.ok(configManager.includes(modelId));
    assert.match(index, /<select id="llm-model-id"[^>]*><\/select>/);
    assert.ok(modelConstraints.includes(`'${modelId}'`));

    // Not allowed: a hardcoded copy in runtime code.
    for (const [name, source] of [
        ['llm-rag.js', llmRag],
        ['fast-search.js', fastSearch],
        ['vectoria.js', vectoria],
        ['browser-integration.js', browserIntegration]
    ]) {
        assert.ok(!source.includes(modelId), `${name} must read the model id from DEFAULT_CONFIG, not hardcode it`);
    }

    // index.html must not carry a separate scripted fallback.
    const scriptFallback = new RegExp(`\\|\\|\\s*['"\`]${modelId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`);
    assert.doesNotMatch(index, scriptFallback, 'index.html must not hardcode the default model as a fallback');
});

test('shipped prompts are defined once, in config-manager', () => {
    const prompts = [
        DEFAULT_CONFIG.rag_prompts.system_prompt,
        DEFAULT_CONFIG.rag_prompts.user_template,
        DEFAULT_CONFIG.hyde.prompt,
        DEFAULT_CONFIG.chat_prompts.conversation_system_prompt
    ];
    for (const prompt of prompts) {
        // Compare on a distinctive slice: whole prompts contain newlines and
        // backticks that make substring checks brittle.
        const fingerprint = prompt.split('\n')[0].slice(0, 60);
        assert.ok(configManager.includes(fingerprint), 'prompt must be defined in config-manager.js');
        for (const [name, source] of [['llm-rag.js', llmRag], ['fast-search.js', fastSearch]]) {
            assert.ok(!source.includes(fingerprint), `${name} must not carry its own copy of a shipped prompt`);
        }
    }
});

test('llm-rag derives every default from DEFAULT_CONFIG rather than literals', () => {
    assert.match(llmRag, /import \{ DEFAULT_CONFIG \} from "\.\.\/config-manager\.js"/);

    // The constructor previously re-declared ~13 defaults as literals.
    const constructorBody = llmRag.slice(llmRag.indexOf('const savedConfig = this.loadSavedConfig();'), llmRag.indexOf('getModelConstraints() {'));
    for (const field of ['temperature', 'max_tokens', 'top_p', 'repeat_penalty', 'num_results', 'retrieval_k']) {
        const literalFallback = new RegExp(`savedConfig\\.${field}\\s*(\\?\\?|\\|\\|)\\s*[\\d'"\`]`);
        assert.doesNotMatch(constructorBody, literalFallback, `savedConfig.${field} must fall back to DEFAULT_CONFIG`);
    }
});

test('every default llm-rag reads actually exists in DEFAULT_CONFIG', () => {
    // A typo here would silently make a runtime value undefined.
    for (const [section, keys] of Object.entries({
        llm: ['model_id', 'temperature', 'max_tokens', 'top_p', 'repeat_penalty', 'context_window_size'],
        search: ['num_results', 'similarity_threshold', 'retrieval_k', 'vector_weight', 'max_chunks_per_parent'],
        rag_prompts: ['system_prompt', 'user_template'],
        hyde: ['prompt', 'temperature', 'max_tokens']
    })) {
        for (const key of keys) {
            assert.notEqual(DEFAULT_CONFIG[section]?.[key], undefined, `DEFAULT_CONFIG.${section}.${key} is missing`);
        }
    }
});

test('the storage version and migration chain stay in step', () => {
    const version = DEFAULT_CONFIG.version;
    assert.equal(Number.isInteger(version) && version > 0, true);
    assert.match(configManager, new RegExp(`const STORAGE_VERSION = ${version};`));

    // Every migration branch must target a version at or below the current one,
    // otherwise it would never run.
    const branches = [...configManager.matchAll(/parsedVersion < (\d+)/g)].map(match => Number(match[1]));
    assert.ok(branches.length > 0);
    assert.ok(Math.max(...branches) <= version, 'a migration targets a version above STORAGE_VERSION');

    // The newest default change should have a matching migration branch.
    assert.ok(branches.includes(version), `no migration branch exists for v${version}`);
});

test('config-manager documents how to change a default', () => {
    assert.match(configManager, /CHANGING A DEFAULT/);
    assert.match(configManager, /EDITABLE DEFAULTS/);
    assert.match(configManager, /SUPERSEDED DEFAULTS/);
});
