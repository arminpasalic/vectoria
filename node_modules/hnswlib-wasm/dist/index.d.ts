/***************** GENERATED FILE ********************/
import type * as module from './hnswlib-wasm';
import type factory from './hnswlib-wasm';
export type HierarchicalNSW = module.HierarchicalNSW;
export type BruteforceSearch = module.BruteforceSearch;
export type EmscriptenFileSystemManager = module.EmscriptenFileSystemManager;
export type L2Space = module.L2Space;
export type InnerProductSpace = module.InnerProductSpace;
export type HnswModuleFactory = typeof factory;
export type normalizePoint = HnswlibModule['normalizePoint'];
export declare const IDBFS_STORE_NAME = "FILE_DATA";
export * from './constants';
export interface HnswlibModule extends Omit<EmscriptenModule, '_malloc' | '_free'> {
    normalizePoint(vec: number[]): number[];
    L2Space: typeof module.L2Space;
    InnerProductSpace: typeof module.InnerProductSpace;
    BruteforceSearch: typeof module.BruteforceSearch;
    HierarchicalNSW: typeof module.HierarchicalNSW;
    EmscriptenFileSystemManager: typeof module.EmscriptenFileSystemManager;
    asm: {
        malloc(size: number): number;
        free(ptr: number): void;
    };
}
type InputFsType = 'IDBFS' | undefined;
export declare const syncFileSystem: (action: 'read' | 'write') => Promise<void>;
export declare const waitForFileSystemInitalized: () => Promise<void>;
export declare const waitForFileSystemSynced: () => Promise<void>;
/**
 * Load the HNSW library in node or browser
 */
export declare const loadHnswlib: (inputFsType?: InputFsType) => Promise<HnswlibModule>;
/***************** GENERATED FILE ********************/
