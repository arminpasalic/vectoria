/**
 * Service Worker for Vectoria
 * Enables offline support and faster loading through intelligent caching
 */

// Bump BUILD_ID on every deploy. The service worker uses it for the cache
// name (so old caches get evicted in the activate handler) and appends it as
// a ?v= query string to every precached URL (so a stale Vercel/CDN copy is
// not served from the previous deploy's immutable cache).
const BUILD_ID = '2026-05-18-01';
const CACHE_VERSION = `vectoria-${BUILD_ID}`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;

const v = (path) => `${path}?v=${BUILD_ID}`;

// Assets to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    v('/static/css/main.css'),
    v('/static/css/browser-ml.css'),
    v('/static/js/viz.js'),
    v('/static/js/webgl-renderer.js'),
    v('/static/js/fast-search.js'),
    v('/static/js/search-enhancement.js'),
    v('/static/js/vectoria.js'),
    v('/static/js/browser-ml/index.js'),
    v('/static/js/browser-ml/embeddings.js'),
    v('/static/js/browser-ml/vector-search.js'),
    v('/static/js/browser-ml/llm-rag.js'),
    v('/static/js/browser-ml/file-processor.js'),
    v('/static/js/browser-ml/clustering.js'),
    v('/static/js/browser-ml/storage.js'),
    v('/static/js/browser-integration.js'),
    '/static/img/favicon.svg',
    '/static/img/icon.ico'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                return cache.addAll(STATIC_ASSETS);
            })
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

                        // Clone the response (can only be consumed once)
                        const responseToCache = response.clone();

                        // Cache dynamic content
                        caches.open(DYNAMIC_CACHE)
                            .then((cache) => {
                                cache.put(request, responseToCache);
                            });

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

