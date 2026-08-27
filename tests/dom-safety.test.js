import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../web_interface/static/js/dom-safety.js', import.meta.url), 'utf8');

function loadSafety(cssSupports = value => /^(?:#[\da-f]{6}|rgba?\()/i.test(value)) {
    const context = { CSS: { supports: (_property, value) => cssSupports(value) } };
    context.window = context;
    vm.runInNewContext(source, context);
    return context.VectoriaDOM;
}

test('DOM safety escapes text for both element and attribute contexts', () => {
    const safety = loadSafety();
    assert.equal(
        safety.escapeHTML(`<img src=x onerror="alert('x')"> & text`),
        '&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt; &amp; text'
    );
    assert.equal(safety.escapeHTML(null), '');
});

test('DOM safety accepts real colors and rejects style or attribute injection', () => {
    const safety = loadSafety();
    assert.equal(safety.safeColor('#3B82F6'), '#3B82F6');
    assert.equal(safety.safeColor('rgba(1, 2, 3, 0.5)'), 'rgba(1, 2, 3, 0.5)');
    assert.equal(safety.safeColor('red; background:url(javascript:alert(1))', '#000000'), '#000000');
    assert.equal(safety.safeColor('" onmouseover="alert(1)', null), null);
});

test('DOM safety has a strict color fallback when CSS.supports is unavailable', () => {
    const context = {};
    context.window = context;
    vm.runInNewContext(source, context);
    assert.equal(context.VectoriaDOM.safeColor('#abc'), '#abc');
    assert.equal(context.VectoriaDOM.safeColor('color-mix(in srgb, red, blue)', 'gray'), 'gray');
});
