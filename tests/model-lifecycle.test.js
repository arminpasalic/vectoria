import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const pipeline = await readFile(new URL('../web_interface/static/js/browser-ml/index.js', import.meta.url), 'utf8');
const embeddings = await readFile(new URL('../web_interface/static/js/browser-ml/embeddings.js', import.meta.url), 'utf8');
const rag = await readFile(new URL('../web_interface/static/js/browser-ml/llm-rag.js', import.meta.url), 'utf8');
const integration = await readFile(new URL('../web_interface/static/js/browser-integration.js', import.meta.url), 'utf8');
const index = await readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8');
const mainCss = await readFile(new URL('../web_interface/static/css/main.css', import.meta.url), 'utf8');
const catalogModule = await import('../web_interface/static/js/model-constraints.js?catalog-lifecycle');

test('initial setup caches models and releases both runtimes', () => {
    assert.match(integration, /releaseAfterInitialize:\s*true/);
    assert.match(pipeline, /this\.embeddings\.suspendWorker\('models-cached'\)/);
    assert.match(pipeline, /this\.rag\.suspendWorker\('models-cached'\)/);
    assert.ok(
        pipeline.indexOf("this.embeddings.suspendWorker('models-cached')")
            < pipeline.indexOf('await this.rag.initialize(onLLMProgress)'),
        'embeddings should be released before WebLLM is initialized'
    );
});

test('repeat visits prepare the pipeline without loading cached models into memory', () => {
    assert.match(integration, /prepareBrowserMLFromCache[\s\S]*?deferModels:\s*true/);
    assert.match(index, /if \(alreadyReady\)[\s\S]*?await prepareBrowserMLFromCache\(\)/);
    assert.doesNotMatch(index, /if \(alreadyReady\)[\s\S]{0,180}?await startModelInitialization\(\)/);
});

test('the upload page welcomes returning visitors and existing model installs', () => {
    assert.match(index, /id="upload-hero-title"><span class="first-visit-title">Load your data<\/span><span class="returning-visitor-title">Welcome back!<\/span><\/h2>/);
    assert.match(index, /const hasVisited = localStorage\.getItem\('vectoria_has_visited'\) === 'true'/);
    assert.match(index, /const hasHandledSetup = localStorage\.getItem\('vectoria_models_prompted'\) === 'true'/);
    assert.match(index, /const modelsReady = localStorage\.getItem\('vectoria_models_ready'\) === 'true'/);
    assert.match(index, /const hasModels = modelsReady[\s\S]*?vectoria_models_cached/);
    assert.match(index, /if \(hasModels \|\| \(hasVisited && hasHandledSetup\)\) \{[\s\S]*?classList\.add\('returning-visitor'\)/);
    assert.match(index, /localStorage\.setItem\('vectoria_has_visited', 'true'\)/);
    assert.ok(
        index.indexOf("classList.add('returning-visitor')") < index.indexOf('static/css/main.css'),
        'returning-visitor state must be applied before the stylesheet can paint the page'
    );
    assert.match(mainCss, /\.returning-visitor-title\s*\{\s*display:\s*none;/);
    assert.match(mainCss, /html\.returning-visitor \.first-visit-title\s*\{\s*display:\s*none;/);
    assert.match(mainCss, /html\.returning-visitor \.returning-visitor-title\s*\{\s*display:\s*inline;/);
    assert.doesNotMatch(index, /uploadHeroTitle\.textContent = 'Welcome back!'/);
});

test('cached model setup UI is hidden before first paint', () => {
    assert.match(index, /if \(modelsReady\) \{\s*document\.documentElement\.classList\.add\('models-ready'\)/);
    assert.ok(
        index.indexOf("classList.add('models-ready')") < index.indexOf('static/css/main.css'),
        'model readiness must be applied before the stylesheet can paint the setup UI'
    );
    assert.match(mainCss, /html\.models-ready #model-setup-panel,[\s\S]*?html\.models-ready #model-loading-status\s*\{\s*display:\s*none;/);
    assert.match(index, /classList\.toggle\('models-ready', state === 'ready'\)/);
});

test('processing loads embeddings while the local LLM remains cold', () => {
    const processStart = pipeline.indexOf('async processFile(');
    const parseStart = pipeline.indexOf("updateProgress('parsing'", processStart);
    const embeddingReady = pipeline.indexOf('await this.embeddings.initialize((modelProgress)', processStart);
    assert.ok(processStart >= 0 && embeddingReady > processStart && embeddingReady < parseStart);
    assert.match(pipeline, /LLM stays[\s\S]*?cold for the entire processing pipeline/);
});

test('semantic work and document chat restore their cached runtimes lazily', () => {
    assert.match(embeddings, /if \(this\.useWorker[\s\S]*?!this\.workerReady \|\| !this\.worker[\s\S]*?await this\.initialize/);
    assert.match(rag, /if \(!this\.isInitialized\)[\s\S]*?LLM is cached — loading for first query[\s\S]*?await this\._initializeWithAbort\(progress =>/);
});

test('setup copy explains the disk-versus-memory lifecycle', () => {
    assert.match(index, /Download once to browser storage\. Models load into memory only when needed\./);
    assert.match(index, /Stays unloaded during document processing and loads for your first document question\./);
});

test('model catalog only references ids that exist in the pinned web-llm build', async () => {
    const options = Object.entries(catalogModule.MODEL_CONSTRAINTS)
        .filter(([, model]) => model.catalogGroup)
        .map(([id]) => id);

    assert.ok(options.length >= 15, 'expected the instruction-model catalog to be populated');
    for (const id of options) {
        const model = catalogModule.MODEL_CONSTRAINTS[id];
        assert.equal(model.instructionTuned, true, `${id} must be instruction tuned`);
        assert.equal(model.modelType, 'text-generation', `${id} must be text-only generation`);
        assert.ok(model.downloadBytes > model.weightBytes, `${id} must include download overhead`);
        assert.ok(model.vramRequiredMB4K > 0, `${id} must expose separate runtime VRAM`);
    }
    assert.ok(!options.includes('DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC'));
    assert.deepEqual(options.filter(id => id.includes('q4f32_1')), []);
    assert.match(index, /<select id="llm-model-id"[^>]*><\/select>/);
    assert.match(index, /id="llm-reasoning-mode"/);
});

test('download sizes use audited bytes and never infer bytes from shard count', () => {
    const model = catalogModule.MODEL_CONSTRAINTS['Llama-3.2-3B-Instruct-q4f16_1-MLC'];
    assert.equal(model.weightBytes, 1807423488);
    assert.equal(model.downloadBytes, 1822644325);
    assert.equal(catalogModule.formatDecimalBytes(model.downloadBytes), '1.82 GB');
    assert.equal(model.vramRequiredMB4K, 2263.69);
    assert.notEqual(model.downloadBytes, 58 * 100 * 1024 * 1024);
    assert.doesNotMatch(integration, /100\s*\*\s*1024\s*\*\s*1024[\s\S]{0,120}llm/i);
    assert.doesNotMatch(integration, /vectoria_model_download_sizes|__webllmRealDownloadSizes/);
    assert.deepEqual(catalogModule.getModelDownloadProgress(model.isDefault ? 'Llama-3.2-3B-Instruct-q4f16_1-MLC' : '', 1), {
        loaded: model.downloadBytes,
        total: model.downloadBytes
    });
});

test('no-thinking soft switch is only claimed where the vendor documents it', async () => {
    const { MODEL_CONSTRAINTS } = await import('../web_interface/static/js/model-constraints.js?no-think');
    const withSwitch = Object.entries(MODEL_CONSTRAINTS)
        .filter(([, value]) => value.noThinkSwitch)
        .map(([id]) => id);

    assert.ok(withSwitch.length > 0);
    // Qwen3 documents /no_think; Qwen3.5 documents only enable_thinking=False,
    // which MLC's chat template does not expose.
    assert.ok(withSwitch.every(id => id.startsWith('Qwen3-')), withSwitch.join(', '));
    assert.ok(withSwitch.every(id => MODEL_CONSTRAINTS[id].thinkSwitch === '/think'));
    for (const id of ['Qwen3.5-4B-q4f16_1-MLC', 'Qwen3.5-9B-q4f16_1-MLC']) {
        assert.equal(MODEL_CONSTRAINTS[id].noThinkSwitch, undefined);
        assert.equal(MODEL_CONSTRAINTS[id].hasThinkMode, true, `${id} still needs the think filter`);
    }
});

test('every supported model exposes a non-blocking RAG tier and verified-schema flag', async () => {
    const { MODEL_CONSTRAINTS } = await import('../web_interface/static/js/model-constraints.js?rag-capabilities');
    for (const [id, constraints] of Object.entries(MODEL_CONSTRAINTS)) {
        assert.ok(['limited', 'recommended', 'quality'].includes(constraints.ragTier), `${id} needs a RAG tier`);
        assert.equal(typeof constraints.schemaVerified, 'boolean', `${id} needs schemaVerified metadata`);
    }
    assert.equal(MODEL_CONSTRAINTS['SmolLM2-1.7B-Instruct-q4f16_1-MLC'].ragTier, 'limited');
    assert.equal(MODEL_CONSTRAINTS['Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC'].ragTier, 'recommended');
    assert.equal(MODEL_CONSTRAINTS['Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC'].schemaVerified, false);
});
