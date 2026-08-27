(function installGenerationModeController(root) {
    'use strict';

    class GenerationModeController {
        constructor({ storage, confirmChange, alertBlocked, activeOperation, unloadLocal, reload, publish } = {}) {
            this.storage = storage || root.localStorage;
            this.confirmChange = confirmChange || (() => true);
            this.alertBlocked = alertBlocked || (() => {});
            this.activeOperation = activeOperation || (() => null);
            this.unloadLocal = unloadLocal || (() => true);
            this.reload = reload || (() => root.location?.reload?.());
            this.publish = publish || ((mode) => root.document?.dispatchEvent?.(
                new CustomEvent('vectoria:generation-mode-changed', { detail: { mode } })
            ));
            this.listeners = new Set();
            this.mode = this._load();
            this._applyGlobals();
        }

        _load() {
            let mode = this.storage?.getItem?.('vectoria_generation_mode');
            if (mode !== 'local' && mode !== 'external') {
                mode = this.storage?.getItem?.('vectoria_mcp_llm_unloaded') === 'true' ? 'external' : 'local';
                this.storage?.setItem?.('vectoria_generation_mode', mode);
            }
            return mode;
        }

        _applyGlobals() {
            root.__vectoriaGenerationMode = this.mode;
            root.__vectoriaLLMUnloaded = this.mode === 'external';
        }

        subscribe(listener) {
            this.listeners.add(listener);
            listener(this.mode);
            return () => this.listeners.delete(listener);
        }

        _notify() {
            for (const listener of this.listeners) listener(this.mode);
        }

        request(nextMode) {
            if (nextMode !== 'local' && nextMode !== 'external') return false;
            if (nextMode === this.mode) {
                this._notify();
                return true;
            }
            const operation = this.activeOperation();
            if (operation) {
                this.alertBlocked(operation);
                this._notify();
                return false;
            }
            if (!this.confirmChange(nextMode)) {
                this._notify();
                return false;
            }
            this.mode = nextMode;
            this.storage?.setItem?.('vectoria_generation_mode', nextMode);
            this.storage?.setItem?.('vectoria_mcp_llm_unloaded', nextMode === 'external' ? 'true' : 'false');
            this._applyGlobals();
            this._notify();
            // Every transition publishes the authoritative state immediately.
            // Local mode still reloads to initialize WebLLM, but the UI must not
            // remain in the external state if that navigation is delayed.
            this.publish(nextMode);
            if (nextMode === 'external') {
                this.unloadLocal();
            } else {
                this.reload();
            }
            return true;
        }
    }

    root.VectoriaGenerationModeController = GenerationModeController;
})(typeof window !== 'undefined' ? window : globalThis);
