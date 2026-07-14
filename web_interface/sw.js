/**
 * Service Worker for Vectoria
 * Enables offline support and faster loading through intelligent caching
 */

// BUILD_ID is auto-stamped by `npm run stamp` (scripts/stamp-version.js) from
// the git date + short SHA, and kept in sync with the ?v= query strings in
// index.html. Do NOT edit by hand — run `npm run stamp` before committing a
// release. The SW uses it for the cache name (so old caches get evicted in the
// activate handler) and appends it as a ?v= query string to every precached
// URL (so a stale Vercel/CDN immutable copy is not served after a deploy).
const BUILD_ID = '2026-07-14-6707bdd';
const CACHE_VERSION = `vectoria-${BUILD_ID}`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;

const v = (path) => `${path}?v=${BUILD_ID}`;

// Only these small app-code asset types are eligible for dynamic caching.
// Large media (gif/png), JSON data, WASM and model files are deliberately
// excluded so the service worker can never exhaust the origin storage quota.
const CACHEABLE_ASSET = /^\/static\/.*\.(js|css|woff2?|ttf|svg|ico)$/;

// Assets to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    v('/static/css/main.css'),
    v('/static/css/browser-ml.css'),
    v('/static/js/viz.js'),
    v('/static/js/webgl-renderer.js'),
    v('/static/js/hyde-handler.js'),
    v('/static/js/fast-search.js'),
    v('/static/js/search-enhancement.js'),
    v('/static/js/browser-capabilities.js'),
    v('/static/js/config-manager.js'),
    v('/static/js/model-constraints.js'),
    v('/static/js/export-import.js'),
    v('/static/js/vectoria.js'),
    v('/static/js/browser-ml/index.js'),
    v('/static/js/browser-ml/embeddings.js'),
    v('/static/js/browser-ml/vector-search.js'),
    v('/static/js/browser-ml/llm-rag.js'),
    v('/static/js/browser-ml/file-processor.js'),
    v('/static/js/browser-ml/clustering.js'),
    v('/static/js/browser-ml/storage.js'),
    v('/static/js/browser-integration.js'),
    '/static/img/favicon.svg'
];

// Install event - cache static assets.
// Cache each asset individually (not cache.addAll) so a single missing/404
// URL can't reject the whole install and silently disable the service worker.
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => Promise.all(
                STATIC_ASSETS.map((asset) =>
                    cache.add(asset).catch((err) =>
                        console.warn(`[SW] Skipped precaching ${asset}:`, err.message)
                    )
                )
            ))
            .then(() => self.skipWaiting())
            .catch((err) => console.error('[SW] Cache failed:', err))
    );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => {
                return Promise.all(
                    keys
                        .filter((key) => key.startsWith('vectoria-') && key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
                        .map((key) => {
                            return caches.delete(key);
                        })
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }

    // Skip external CDN resources (they have their own caching)
    if (url.origin !== location.origin) {
        return;
    }

    // Never cache the sample dataset: it's large (14MB) and a stale/broken
    // copy silently breaks the "Load sample" button. Always go to network.
    if (url.pathname.startsWith('/static/samples/')) {
        return;
    }

    event.respondWith(
        caches.match(request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    // Return cached version and update in background
                    return cachedResponse;
                }

                // Not in cache, fetch from network
                return fetch(request)
                    .then((response) => {
                        // Only cache successful responses
                        if (!response || response.status !== 200 || response.type === 'error') {
                            return response;
                        }

                        // Only dynamic-cache small app code (js/css/fonts/icons)
                        // under /static/. Never cache large media, JSON data,
                        // WASM or model files — they can exhaust the origin
                        // storage quota, which makes Cache.put() fail and
                        // cascades into other fetches failing (ERR_FAILED).
                        if (CACHEABLE_ASSET.test(url.pathname)) {
                            const responseToCache = response.clone();
                            caches.open(DYNAMIC_CACHE)
                                .then((cache) => cache.put(request, responseToCache))
                                .catch((err) => console.warn('[SW] Dynamic cache skipped:', err.message));
                        }

                        return response;
                    })
                    .catch((err) => {
                        console.error('[SW] Fetch failed:', err);
                        // Could return a custom offline page here
                        throw err;
                    });
            })
    );
});

// Handle messages from clients
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }

    if (event.data && event.data.type === 'CLEAR_CACHE') {
        event.waitUntil(
            caches.keys().then((keys) => {
                return Promise.all(
                    keys.map((key) => caches.delete(key))
                );
            })
        );
    }
});

