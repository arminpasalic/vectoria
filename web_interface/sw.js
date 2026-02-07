/**
 * Service Worker for Vectoria
 * Enables offline support and faster loading through intelligent caching
 */

const CACHE_VERSION = 'vectoria-v3';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;

// Assets to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/static/css/main.css?v=20241016',
    '/static/css/browser-ml.css',
    '/static/js/viz.js',
    '/static/js/webgl-renderer.js',
    '/static/js/fast-search.js',
    '/static/js/search-enhancement.js',
    '/static/js/vectoria.js?v=20241016',
    '/static/js/browser-ml/index.js',
    '/static/js/browser-ml/embeddings.js',
    '/static/js/browser-ml/vector-search.js',
    '/static/js/browser-ml/llm-rag.js',
    '/static/js/browser-ml/file-processor.js',
    '/static/js/browser-ml/clustering.js',
    '/static/js/browser-ml/storage.js',
    '/static/js/browser-integration.js',
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

