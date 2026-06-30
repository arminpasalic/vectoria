/**
 * Pure statistical helpers for advanced analysis tools.
 * No external deps — all approximations live here.
 */

export function quantile(sortedAsc, q) {
    if (!sortedAsc.length) return null;
    const pos = (sortedAsc.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sortedAsc[base + 1] !== undefined) {
        return sortedAsc[base] + rest * (sortedAsc[base + 1] - sortedAsc[base]);
    }
    return sortedAsc[base];
}

export function chiSquare(observed) {
    const rows = observed.length;
    const cols = observed[0]?.length ?? 0;
    if (!rows || !cols) {
        return { chi2: 0, dof: 0, p_value: 1, n: 0, expected: [] };
    }

    const rowTotals = observed.map(r => r.reduce((s, v) => s + v, 0));
    const colTotals = new Array(cols).fill(0);
    for (const row of observed) for (let j = 0; j < cols; j++) colTotals[j] += row[j];
    const n = rowTotals.reduce((s, v) => s + v, 0);

    if (n === 0) return { chi2: 0, dof: 0, p_value: 1, n: 0, expected: [] };

    const expected = observed.map((row, i) =>
        row.map((_, j) => (rowTotals[i] * colTotals[j]) / n)
    );

    let chi2 = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const e = expected[i][j];
            if (e > 0) {
                const diff = observed[i][j] - e;
                chi2 += (diff * diff) / e;
            }
        }
    }

    const dof = (rows - 1) * (cols - 1);
    const p_value = chiSquarePValue(chi2, dof);
    return { chi2, dof, p_value, n, expected };
}

export function cramersV(chi2, n, rows, cols) {
    const k = Math.min(rows - 1, cols - 1);
    if (n === 0 || k === 0) return 0;
    return Math.sqrt(chi2 / (n * k));
}

/**
 * Wilson-Hilferty approximation for chi-square p-value (upper tail).
 * Good for dof >= 1; matches standard tables to ~3 decimals for typical values.
 */
export function chiSquarePValue(chi2, dof) {
    if (dof <= 0 || chi2 <= 0) return 1;
    const z = Math.pow(chi2 / dof, 1 / 3);
    const mu = 1 - 2 / (9 * dof);
    const sigma = Math.sqrt(2 / (9 * dof));
    const t = (z - mu) / sigma;
    return 1 - standardNormalCDF(t);
}

function standardNormalCDF(x) {
    // Abramowitz & Stegun 7.1.26 approximation
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x) / Math.SQRT2;
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
    return 0.5 * (1.0 + sign * y);
}

export function jsDivergence(p, q) {
    // Symmetric KL: 0.5*KL(p||m) + 0.5*KL(q||m) where m = 0.5*(p+q)
    const keys = new Set([...Object.keys(p), ...Object.keys(q)]);
    const pSum = Object.values(p).reduce((a, b) => a + b, 0) || 1;
    const qSum = Object.values(q).reduce((a, b) => a + b, 0) || 1;
    let div = 0;
    for (const k of keys) {
        const pi = (p[k] || 0) / pSum;
        const qi = (q[k] || 0) / qSum;
        const mi = 0.5 * (pi + qi);
        if (pi > 0 && mi > 0) div += 0.5 * pi * Math.log2(pi / mi);
        if (qi > 0 && mi > 0) div += 0.5 * qi * Math.log2(qi / mi);
    }
    return div;
}

export function welchT(arrA, arrB) {
    if (!arrA.length || !arrB.length) return { t: 0, df: 0, p_value: 1, mean_a: 0, mean_b: 0 };
    const meanA = mean(arrA);
    const meanB = mean(arrB);
    const varA = variance(arrA, meanA);
    const varB = variance(arrB, meanB);
    const nA = arrA.length;
    const nB = arrB.length;
    const seSq = varA / nA + varB / nB;
    if (seSq === 0) return { t: 0, df: 0, p_value: 1, mean_a: meanA, mean_b: meanB };
    const t = (meanA - meanB) / Math.sqrt(seSq);
    const df = Math.pow(seSq, 2) /
        (Math.pow(varA / nA, 2) / (nA - 1 || 1) + Math.pow(varB / nB, 2) / (nB - 1 || 1));
    // Approximate two-tailed p via normal for df large; for small df it's still rough.
    const p_value = 2 * (1 - standardNormalCDF(Math.abs(t)));
    return { t, df, p_value, mean_a: meanA, mean_b: meanB };
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function variance(arr, mu) {
    if (arr.length <= 1) return 0;
    let s = 0;
    for (const v of arr) s += (v - mu) * (v - mu);
    return s / (arr.length - 1);
}

export function buildContingency(rowValues, colValues) {
    const rowKeys = [...new Set(rowValues)].map(String);
    const colKeys = [...new Set(colValues)].map(String);
    const rowIdx = new Map(rowKeys.map((k, i) => [k, i]));
    const colIdx = new Map(colKeys.map((k, i) => [k, i]));
    const matrix = rowKeys.map(() => new Array(colKeys.length).fill(0));
    for (let i = 0; i < rowValues.length; i++) {
        const r = rowIdx.get(String(rowValues[i]));
        const c = colIdx.get(String(colValues[i]));
        if (r !== undefined && c !== undefined) matrix[r][c]++;
    }
    return { matrix, rowKeys, colKeys };
}
