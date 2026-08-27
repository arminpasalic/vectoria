import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [css, mlCss, index, vectoria, visualization, webgl, fastSearch, capabilities, browserIntegration] = await Promise.all([
    readFile(new URL('../web_interface/static/css/main.css', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/css/browser-ml.css', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/index.html', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/vectoria.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/viz.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/webgl-renderer.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/fast-search.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/browser-capabilities.js', import.meta.url), 'utf8'),
    readFile(new URL('../web_interface/static/js/browser-integration.js', import.meta.url), 'utf8')
]);

function hexToRgb(hex) {
    const value = hex.replace('#', '');
    const normalized = value.length === 3
        ? value.split('').map(character => character + character).join('')
        : value;
    return [0, 2, 4].map(offset => Number.parseInt(normalized.slice(offset, offset + 2), 16));
}

function luminance(hex) {
    const channels = hexToRgb(hex).map(value => {
        const normalized = value / 255;
        return normalized <= 0.04045
            ? normalized / 12.92
            : ((normalized + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(first, second) {
    const a = luminance(first);
    const b = luminance(second);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

function rootToken(name) {
    const root = css.match(/:root\s*\{([\s\S]*?)\n\}/)?.[1] || '';
    const match = root.match(new RegExp(`${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*:\\s*(#[0-9a-f]{6})`, 'i'));
    assert.ok(match, `missing six-digit light token ${name}`);
    return match[1];
}

test('light theme text, status, control, and focus roles meet WCAG contrast targets', () => {
    const lightSurfaces = ['#FFFFFF', '#F0F0F0'];
    for (const name of ['--text-primary', '--text-secondary', '--text-muted']) {
        for (const surface of lightSurfaces) {
            assert.ok(contrast(rootToken(name), surface) >= 4.5, `${name} must reach 4.5:1 on ${surface}`);
        }
    }

    assert.ok(contrast(rootToken('--text-inverse'), rootToken('--button-primary-bg')) >= 4.5);
    for (const name of ['--success-color', '--warning-color', '--error-color', '--info-color']) {
        assert.ok(contrast(rootToken(name), '#FFFFFF') >= 4.5, `${name} must reach 4.5:1 on white`);
    }
    for (const name of ['--control-border', '--control-border-hover', '--focus-outline']) {
        for (const surface of lightSurfaces) {
            assert.ok(contrast(rootToken(name), surface) >= 3, `${name} must reach 3:1 on ${surface}`);
        }
    }
});

function darkToken(name) {
    const block = css.match(/html\.dark\s*\{([\s\S]*?)\n\}/)?.[1] || '';
    const match = block.match(new RegExp(`${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*:\\s*(#[0-9a-f]{6})`, 'i'));
    assert.ok(match, `missing six-digit dark token ${name}`);
    return match[1];
}

test('dark theme text roles meet WCAG contrast on every dark surface', () => {
    // Chat labels, message metadata, and source scores are rendered in the
    // secondary and muted roles at 11-12px, so they need real AA contrast
    // rather than the 3.2:1 the muted role used to carry.
    const darkSurfaces = ['#000000', '#0A0A0A', '#141414', '#1A1A1A'];
    for (const name of ['--text-primary', '--text-secondary', '--text-muted']) {
        for (const surface of darkSurfaces) {
            const ratio = contrast(darkToken(name), surface);
            assert.ok(ratio >= 4.5, `${name} must reach 4.5:1 on ${surface} (got ${ratio.toFixed(2)}:1)`);
        }
    }

    // The roles must stay visually ordered, or the hierarchy collapses.
    assert.ok(contrast(darkToken('--text-primary'), '#000000') > contrast(darkToken('--text-secondary'), '#000000'));
    assert.ok(contrast(darkToken('--text-secondary'), '#000000') > contrast(darkToken('--text-muted'), '#000000'));

    // Dark status colours are read as text on the page background.
    for (const name of ['--success-color', '--warning-color', '--error-color', '--info-color']) {
        assert.ok(contrast(darkToken(name), '#050505') >= 4.5, `${name} must reach 4.5:1 on the dark page background`);
    }
});

test('filled danger buttons carry white text in both themes', () => {
    // --error-color doubles as error *text*, where it only needs to beat the
    // page background. Behind white button text it fell to 3.76:1 in dark, so
    // filled danger controls use a dedicated deeper red instead.
    for (const [label, token] of [['light', rootToken('--danger-button-bg')], ['dark', darkToken('--danger-button-bg')]]) {
        const ratio = contrast('#FFFFFF', token);
        assert.ok(ratio >= 4.5, `${label} --danger-button-bg needs 4.5:1 with white text (got ${ratio.toFixed(2)}:1)`);
    }
    assert.match(css, /\.btn-danger\s*\{[\s\S]*?background:\s*var\(--danger-button-bg\)/);
    assert.match(css, /\.chat-stop-btn\s*\{[\s\S]*?background:\s*var\(--danger-button-bg\)/);
});

test('content links are themed rather than falling back to browser default blue', () => {
    // Unstyled <a> renders as #0000EE, which is 1.96:1 on the dark background.
    const rule = css.match(/\.settings-about-links a,[\s\S]*?\{([\s\S]*?)\}/)?.[1];
    assert.ok(rule, 'expected a shared rule styling content links');
    const colour = rule.match(/(?:^|[;{\s])color:\s*([^;]+);/)?.[1]?.trim();
    assert.equal(colour, 'var(--text-primary)', 'content links must use a theme token, not a literal colour');
    assert.match(rule, /text-decoration:\s*underline/, 'links need a non-colour affordance');

    // Every external link in the markup must be covered by that rule.
    const externalLinks = (index.match(/<a\s[^>]*href="https?:/g) || []).length;
    assert.ok(externalLinks > 0, 'expected external links in the markup');
    for (const selector of ['.settings-about-links a', '.form-help a', '.info-box a']) {
        assert.ok(css.includes(selector), `${selector} must carry an explicit link colour`);
    }
});

test('all non-fallback CSS variables are defined and retired aliases stay absent', () => {
    const combined = `${css}\n${mlCss}\n${index}`;
    const definitions = new Set([...combined.matchAll(/(--[\w-]+)\s*:/g)].map(match => match[1]));
    const missing = new Set();
    for (const match of combined.matchAll(/var\((--[\w-]+)(\s*,)?/g)) {
        if (!match[2] && !definitions.has(match[1])) missing.add(match[1]);
    }
    assert.deepEqual([...missing].sort(), []);
    assert.doesNotMatch(combined, /var\(--(?:shadow-strong|shadow-card-inner|border-color|danger-color)\)/);
});

test('explicit html theme classes are the only component theme authorities', () => {
    const combined = `${css}\n${mlCss}`;
    assert.doesNotMatch(combined, /body\.theme-(?:light|dark)/);
    assert.doesNotMatch(combined, /prefers-color-scheme/);
    assert.match(combined, /html\.light/);
    assert.match(combined, /html\.dark/);
});

test('light overlays use semantic surfaces while dark overrides remain explicitly scoped', () => {
    assert.match(css, /\n\.modal-overlay\s*\{[\s\S]*?background:\s*var\(--overlay-backdrop\)/);
    assert.match(css, /\n\.modal-container\s*\{[\s\S]*?background:\s*var\(--bg-glass-strong\)/);
    assert.match(css, /#advanced-settings-modal \.modal-container\s*\{[\s\S]*?background:\s*var\(--card-bg\)/);
    assert.match(css, /\.processing-summary-card\s*\{[\s\S]*?background:\s*var\(--card-bg\)/);
    assert.match(css, /\.ai-answer-container\s*\{[\s\S]*?background:\s*var\(--card-bg\)/);
    assert.match(mlCss, /\.ml-modal-content\s*\{[\s\S]*?background:\s*var\(--surface-overlay\)/);
    assert.doesNotMatch(css, /Force modal palette to dark|Always dark theme|Force dark appearance/);
    assert.doesNotMatch(mlCss, /Force ML modals to use the dark neutral palette in all themes/);
});

test('custom actions, visualization controls, and disclosures expose complete focus and disabled states', () => {
    assert.match(index, /id="quick-analysis-btn"[^>]*class="btn btn-secondary btn-sm btn-cluster-toolbar"/);
    assert.match(index, /id="cluster-label-all-quick"[^>]*class="btn btn-primary btn-sm btn-cluster-toolbar"/);
    assert.match(css, /\.btn:focus-visible/);
    assert.match(css, /\.viz-control-btn:focus-visible/);
    assert.match(css, /\.viz-control-btn:disabled/);
    assert.match(css, /\.cluster-summarizer-quick:focus-visible/);
    assert.match(css, /\.cluster-summarizer-quick:disabled\s*\{[\s\S]*?opacity:\s*1/);
    assert.match(css, /\.settings-disclosure > summary:hover/);
    assert.match(css, /summary:focus-visible/);
    assert.match(css, /\.rag-toggle-input:focus-visible\+\.rag-toggle-slider/);

    const customInputRule = css.match(/\.checkbox-option input\[type="checkbox"\],[\s\S]*?\.radio-label input\[type="radio"\]\s*\{([\s\S]*?)\}/)?.[1] || '';
    assert.match(customInputRule, /position:\s*absolute/);
    assert.match(customInputRule, /clip-path:\s*inset\(50%\)/);
    assert.doesNotMatch(customInputRule, /display:\s*none/);
});

test('secondary light-mode controls avoid opacity-based contrast loss', () => {
    assert.match(css, /\.workspace-view-tab \.text-count\s*\{[\s\S]*?color:\s*var\(--text-muted\)[\s\S]*?opacity:\s*1/);
    assert.match(css, /\.chat-message-actions\s*\{[\s\S]*?opacity:\s*0\.72/);
    assert.match(css, /\.hyde-viewer-icon\s*\{[\s\S]*?color:\s*var\(--text-muted\)[\s\S]*?opacity:\s*1/);
    assert.match(css, /\.rag-context-chip\.active\s*\{[\s\S]*?color:\s*var\(--info-color\)/);
    assert.match(mlCss, /\.model-loading-icon\.complete\s*\{[\s\S]*?color:\s*var\(--success-color\)/);
    assert.match(mlCss, /\.model-size-badge\.downloading\s*\{[\s\S]*?color:\s*var\(--info-color\)/);
    assert.doesNotMatch(index, /#888(?:888)?|#ff1493/i);
    assert.match(index, /class="footer-heart" aria-hidden="true"/);
    assert.match(css, /\.footer-heart\s*\{[\s\S]*?color:\s*var\(--error-color\)/);
});

test('range filters and every modal family expose accessible names and dialog semantics', () => {
    assert.match(vectoria, /aria-label="\$\{escapedDisplayName\} minimum"/);
    assert.match(vectoria, /aria-label="\$\{escapedDisplayName\} maximum"/);
    assert.match(vectoria, /aria-label="\$\{escapedDisplayName\} start date"/);
    assert.match(vectoria, /aria-label="\$\{escapedDisplayName\} end date"/);

    assert.match(index, /id="processing-summary-modal"[^>]*role="dialog"[^>]*aria-modal="true"[^>]*aria-labelledby="processing-summary-title"/);
    assert.match(index, /id="quick-analysis-modal"[^>]*role="dialog"[^>]*aria-modal="true"[^>]*aria-labelledby="quick-analysis-title"/);
    assert.match(capabilities, /modal\.setAttribute\('role', 'dialog'\)[\s\S]*?modal\.setAttribute\('aria-modal', 'true'\)[\s\S]*?modal\.setAttribute\('aria-labelledby', 'capability-modal-title'\)/);
    assert.match(browserIntegration, /modal\.id = 'processing-modal'[\s\S]*?modal\.setAttribute\('role', 'dialog'\)[\s\S]*?modal\.setAttribute\('aria-labelledby', 'processing-modal-title'\)/);
});

test('cluster hues are accents rather than arbitrary text colors', () => {
    assert.match(vectoria, /--cluster-color: \$\{clusterColor\}; --cluster-tint: \$\{clusterBadgeColor\}/);
    assert.doesNotMatch(vectoria, /--cluster-tint:[^"`]*;\s*color:/);
    assert.match(css, /\.text-item-cluster\s*\{[\s\S]*?color:\s*var\(--text-primary\)/);
    assert.match(css, /\.tooltip-cluster\s*\{[\s\S]*?color:\s*var\(--text-primary\)/);
    assert.doesNotMatch(visualization, /tooltip-cluster[^`]*color:\s*(?:white|#fff)/i);
    assert.doesNotMatch(fastSearch, /cluster-indicator[^`]*box-shadow:/);
});

test('dark visualization point tooltips keep every text layer white', () => {
    assert.match(css, /html\.dark \.tooltip-content,[\s\S]*?html\.dark \.tooltip-confidence-value\s*\{\s*color:\s*#ffffff;/);
});

test('Canvas and WebGL outlines respond to the active theme and redraw without recentering', () => {
    assert.match(visualization, /darkMode\s*\?\s*'rgba\(255, 255, 255, 0\.34\)'\s*:\s*'rgba\(30, 30, 30, 0\.48\)'/);
    assert.match(webgl, /if \(isDark\)[\s\S]*?outlineColor[\s\S]*?0\.34[\s\S]*?neutralOutline[\s\S]*?0\.48/);
    assert.match(vectoria, /themeColor\.setAttribute\('content', normalized === 'dark' \? '#050505' : '#F0F0F0'\)/);
    const applyTheme = vectoria.slice(vectoria.indexOf('function applyThemePreference'), vectoria.indexOf('function updateLogoForTheme'));
    assert.match(applyTheme, /mainVisualization\.requestRender\(\)/);
    assert.doesNotMatch(applyTheme, /centerView\(/);
});

test('visualization tooltips stay below overlays and are cleared for every modal family', () => {
    assert.match(css, /--z-tooltip:\s*3000/);
    assert.match(css, /--z-modal-backdrop:\s*9990/);
    assert.match(css, /body\.modal-open \.tooltip,[\s\S]*?body\.ml-modal-open \.tooltip[\s\S]*?display:\s*none\s*!important/);
    assert.match(visualization, /document\.body\?\.classList\?\.contains\('modal-open'\)/);
    assert.match(vectoria, /function clearVisualizationTransientState[\s\S]*?hideTooltip\?\.\(\)[\s\S]*?clearChatPreview\?\.\(\)/);
    assert.match(index, /window\.clearVisualizationTransientState\?\.\(\)[\s\S]*?modelSetupModal\.style\.display = 'flex'/);
});
