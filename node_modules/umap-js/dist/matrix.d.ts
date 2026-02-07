export declare class SparseMatrix {
    private entries;
    readonly nRows: number;
    readonly nCols: number;
    constructor(rows: number[], cols: number[], values: number[], dims: number[]);
    private makeKey;
    private checkDims;
    set(row: number, col: number, value: number): void;
    get(row: number, col: number, defaultValue?: number): number;
    getAll(ordered?: boolean): {
        value: number;
        row: number;
        col: number;
    }[];
    getDims(): number[];
    getRows(): number[];
    getCols(): number[];
    getValues(): number[];
    forEach(fn: (value: number, row: number, col: number) => void): void;
    map(fn: (value: number, row: number, col: number) => number): SparseMatrix;
    toArray(): number[][];
}
export declare function transpose(matrix: SparseMatrix): SparseMatrix;
export declare function identity(size: number[]): SparseMatrix;
export declare function pairwiseMultiply(a: SparseMatrix, b: SparseMatrix): SparseMatrix;
export declare function add(a: SparseMatrix, b: SparseMatrix): SparseMatrix;
export declare function subtract(a: SparseMatrix, b: SparseMatrix): SparseMatrix;
export declare function maximum(a: SparseMatrix, b: SparseMatrix): SparseMatrix;
export declare function multiplyScalar(a: SparseMatrix, scalar: number): SparseMatrix;
export declare function eliminateZeros(m: SparseMatrix): SparseMatrix;
export declare function normalize(m: SparseMatrix, normType?: NormType): SparseMatrix;
export declare const enum NormType {
    max = "max",
    l1 = "l1",
    l2 = "l2"
}
export declare function getCSR(x: SparseMatrix): {
    indices: number[];
    values: number[];
    indptr: number[];
};
