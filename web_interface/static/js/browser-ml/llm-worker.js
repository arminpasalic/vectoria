import { WebWorkerMLCEngineHandler } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.83/+esm";

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (event) => {
    handler.onmessage(event);
};

