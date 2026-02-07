/***************** GENERATED FILE ********************/
export declare const defaultParams: {
    /**
     * Default parameters for the HNSW index.
     * @param {number} m The maximum number of outgoing connections on the graph (default: 16).
     * @param {number} efConstruction The parameter that controls speed/accuracy trade-off during the index construction (default: 200).
     * @param {number} randomSeed The seed value of random number generator (default: 100).
     */
    readonly initIndex: readonly [32, 128, 100];
};
export type defaultParamtersTypes = keyof typeof defaultParams;
export declare const hnswParamsForAda: {
    readonly m: 32;
    readonly efSearch: 128;
    readonly efConstruction: 128;
    readonly numNeighbors: 8;
    readonly dimensions: 1538;
};
/***************** GENERATED FILE ********************/
