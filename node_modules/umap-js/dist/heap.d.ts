import { RandomFn } from './umap';
export declare type Heap = number[][][];
export declare function makeHeap(nPoints: number, size: number): Heap;
export declare function rejectionSample(nSamples: number, poolSize: number, random: RandomFn): number[];
export declare function heapPush(heap: Heap, row: number, weight: number, index: number, flag: number): number;
export declare function uncheckedHeapPush(heap: Heap, row: number, weight: number, index: number, flag: number): number;
export declare function buildCandidates(currentGraph: Heap, nVertices: number, nNeighbors: number, maxCandidates: number, random: RandomFn): Heap;
export declare function deheapSort(heap: Heap): {
    indices: number[][];
    weights: number[][];
};
export declare function smallestFlagged(heap: Heap, row: number): number;
