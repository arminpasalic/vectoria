import { RandomFn, Vector, Vectors } from './umap';
export declare class FlatTree {
    hyperplanes: number[][];
    offsets: number[];
    children: number[][];
    indices: number[][];
    constructor(hyperplanes: number[][], offsets: number[], children: number[][], indices: number[][]);
}
export declare function makeForest(data: Vectors, nNeighbors: number, nTrees: number, random: RandomFn): FlatTree[];
export declare function makeLeafArray(rpForest: FlatTree[]): number[][];
export declare function searchFlatTree(point: Vector, tree: FlatTree, random: RandomFn): number[];
