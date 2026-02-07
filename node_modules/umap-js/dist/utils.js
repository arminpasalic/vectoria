"use strict";
var __values = (this && this.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.reshape2d = exports.rejectionSample = exports.max2d = exports.max = exports.mean = exports.sum = exports.linear = exports.ones = exports.zeros = exports.filled = exports.range = exports.empty = exports.norm = exports.tauRand = exports.tauRandInt = void 0;
function tauRandInt(n, random) {
    return Math.floor(random() * n);
}
exports.tauRandInt = tauRandInt;
function tauRand(random) {
    return random();
}
exports.tauRand = tauRand;
function norm(vec) {
    var e_1, _a;
    var result = 0;
    try {
        for (var vec_1 = __values(vec), vec_1_1 = vec_1.next(); !vec_1_1.done; vec_1_1 = vec_1.next()) {
            var item = vec_1_1.value;
            result += Math.pow(item, 2);
        }
    }
    catch (e_1_1) { e_1 = { error: e_1_1 }; }
    finally {
        try {
            if (vec_1_1 && !vec_1_1.done && (_a = vec_1.return)) _a.call(vec_1);
        }
        finally { if (e_1) throw e_1.error; }
    }
    return Math.sqrt(result);
}
exports.norm = norm;
function empty(n) {
    var output = [];
    for (var i = 0; i < n; i++) {
        output.push(undefined);
    }
    return output;
}
exports.empty = empty;
function range(n) {
    return empty(n).map(function (_, i) { return i; });
}
exports.range = range;
function filled(n, v) {
    return empty(n).map(function () { return v; });
}
exports.filled = filled;
function zeros(n) {
    return filled(n, 0);
}
exports.zeros = zeros;
function ones(n) {
    return filled(n, 1);
}
exports.ones = ones;
function linear(a, b, len) {
    return empty(len).map(function (_, i) {
        return a + i * ((b - a) / (len - 1));
    });
}
exports.linear = linear;
function sum(input) {
    return input.reduce(function (sum, val) { return sum + val; });
}
exports.sum = sum;
function mean(input) {
    return sum(input) / input.length;
}
exports.mean = mean;
function max(input) {
    var max = 0;
    for (var i = 0; i < input.length; i++) {
        max = input[i] > max ? input[i] : max;
    }
    return max;
}
exports.max = max;
function max2d(input) {
    var max = 0;
    for (var i = 0; i < input.length; i++) {
        for (var j = 0; j < input[i].length; j++) {
            max = input[i][j] > max ? input[i][j] : max;
        }
    }
    return max;
}
exports.max2d = max2d;
function rejectionSample(nSamples, poolSize, random) {
    var result = zeros(nSamples);
    for (var i = 0; i < nSamples; i++) {
        var rejectSample = true;
        while (rejectSample) {
            var j = tauRandInt(poolSize, random);
            var broken = false;
            for (var k = 0; k < i; k++) {
                if (j === result[k]) {
                    broken = true;
                    break;
                }
            }
            if (!broken) {
                rejectSample = false;
            }
            result[i] = j;
        }
    }
    return result;
}
exports.rejectionSample = rejectionSample;
function reshape2d(x, a, b) {
    var rows = [];
    var count = 0;
    var index = 0;
    if (x.length !== a * b) {
        throw new Error('Array dimensions must match input length.');
    }
    for (var i = 0; i < a; i++) {
        var col = [];
        for (var j = 0; j < b; j++) {
            col.push(x[index]);
            index += 1;
        }
        rows.push(col);
        count += 1;
    }
    return rows;
}
exports.reshape2d = reshape2d;
