import * as matrix from './matrix';
export declare type DistanceFn = (x: Vector, y: Vector) => number;
export declare type RandomFn = () => number;
export declare type EpochCallback = (epoch: number) => boolean | void;
export declare type Vector = number[];
export declare type Vectors = Vector[];
export declare const enum TargetMetric {
    categorical = "categorical",
    l1 = "l1",
    l2 = "l2"
}
export interface UMAPParameters {
    distanceFn?: DistanceFn;
    learningRate?: number;
    localConnectivity?: number;
    minDist?: number;
    nComponents?: number;
    nEpochs?: number;
    nNeighbors?: number;
    negativeSampleRate?: number;
    repulsionStrength?: number;
    random?: RandomFn;
    setOpMixRatio?: number;
    spread?: number;
    transformQueueSize?: number;
}
export interface UMAPSupervisedParams {
    targetMetric?: TargetMetric;
    targetWeight?: number;
    targetNNeighbors?: number;
}
export declare class UMAP {
    private learningRate;
    private localConnectivity;
    private minDist;
    private nComponents;
    private nEpochs;
    private nNeighbors;
    private negativeSampleRate;
    private random;
    private repulsionStrength;
    private setOpMixRatio;
    private spread;
    private transformQueueSize;
    private targetMetric;
    private targetWeight;
    private targetNNeighbors;
    private distanceFn;
    private knnIndices?;
    private knnDistances?;
    private graph;
    private X;
    private isInitialized;
    private rpForest;
    private initFromRandom;
    private initFromTree;
    private search;
    private searchGraph;
    private Y?;
    private embedding;
    private optimizationState;
    constructor(params?: UMAPParameters);
    fit(X: Vectors): number[][];
    fitAsync(X: Vectors, callback?: (epochNumber: number) => void | boolean): Promise<number[][]>;
    setSupervisedProjection(Y: number[], params?: UMAPSupervisedParams): void;
    setPrecomputedKNN(knnIndices: number[][], knnDistances: number[][]): void;
    initializeFit(X: Vectors): number;
    private makeSearchFns;
    private makeSearchGraph;
    transform(toTransform: Vectors): Vectors;
    private processGraphForSupervisedProjection;
    step(): number;
    getEmbedding(): number[][];
    private nearestNeighbors;
    private fuzzySimplicialSet;
    private categoricalSimplicialSetIntersection;
    private smoothKNNDistance;
    private computeMembershipStrengths;
    private initializeSimplicialSetEmbedding;
    private makeEpochsPerSample;
    private assignOptimizationStateParameters;
    private prepareForOptimizationLoop;
    private initializeOptimization;
    private optimizeLayoutStep;
    private optimizeLayoutAsync;
    private optimizeLayout;
    private getNEpochs;
}
export declare function euclidean(x: Vector, y: Vector): number;
export declare function cosine(x: Vector, y: Vector): number;
export declare function findABParams(spread: number, minDist: number): {
    a: number;
    b: number;
};
export declare function fastIntersection(graph: matrix.SparseMatrix, target: number[], unknownDist?: number, farDist?: number): matrix.SparseMatrix;
export declare function resetLocalConnectivity(simplicialSet: matrix.SparseMatrix): matrix.SparseMatrix;
export declare function initTransform(indices: number[][], weights: number[][], embedding: Vectors): number[][];
