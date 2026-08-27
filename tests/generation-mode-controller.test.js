import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../web_interface/static/js/generation-mode-controller.js', import.meta.url), 'utf8');

function loadController() {
    const context = {};
    vm.createContext(context);
    vm.runInContext(source, context);
    return { Controller: context.VectoriaGenerationModeController, context };
}

function storageWith(initial = {}) {
    const values = new Map(Object.entries(initial));
    return {
        values,
        getItem: key => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, String(value))
    };
}

test('external mode to immediate Enable local chat uses one atomic controller transition', () => {
    const { Controller, context } = loadController();
    const storage = storageWith({ vectoria_generation_mode: 'external' });
    let reloads = 0;
    let confirmations = 0;
    const published = [];
    const controller = new Controller({
        storage,
        confirmChange: mode => { confirmations++; return mode === 'local'; },
        reload: () => { reloads++; },
        publish: mode => published.push(mode)
    });
    assert.equal(controller.mode, 'external');
    assert.equal(controller.request('local'), true);
    assert.equal(controller.mode, 'local');
    assert.equal(context.__vectoriaGenerationMode, 'local');
    assert.equal(context.__vectoriaLLMUnloaded, false);
    assert.equal(storage.values.get('vectoria_generation_mode'), 'local');
    assert.equal(storage.values.get('vectoria_mcp_llm_unloaded'), 'false');
    assert.equal(confirmations, 1);
    assert.equal(reloads, 1);
    assert.deepEqual(published, ['local']);
});

test('external transitions publish the same controller state before unloading', () => {
    const { Controller } = loadController();
    const storage = storageWith({ vectoria_generation_mode: 'local' });
    const order = [];
    const controller = new Controller({
        storage,
        confirmChange: () => true,
        publish: mode => order.push(`publish:${mode}`),
        unloadLocal: () => order.push('unload')
    });
    assert.equal(controller.request('external'), true);
    assert.deepEqual(order, ['publish:external', 'unload']);
});

test('active local work blocks mode mutation and subscribers derive unchanged UI state', () => {
    const { Controller } = loadController();
    const storage = storageWith({ vectoria_generation_mode: 'local' });
    const observed = [];
    let blocked = 0;
    const controller = new Controller({
        storage,
        activeOperation: () => ({ owner: 'chat' }),
        alertBlocked: () => { blocked++; },
        confirmChange: () => { throw new Error('confirmation must not run while blocked'); }
    });
    controller.subscribe(mode => observed.push(mode));
    assert.equal(controller.request('external'), false);
    assert.equal(controller.mode, 'local');
    assert.equal(storage.values.get('vectoria_generation_mode'), 'local');
    assert.equal(blocked, 1);
    assert.deepEqual(observed, ['local', 'local']);
});
