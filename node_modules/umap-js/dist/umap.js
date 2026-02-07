"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spread = (this && this.__spread) || function () {
    for (var ar = [], i = 0; i < arguments.length; i++) ar = ar.concat(__read(arguments[i]));
    return ar;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.initTransform = exports.resetLocalConnectivity = exports.fastIntersection = exports.findABParams = exports.cosine = exports.euclidean = exports.UMAP = void 0;
var heap = __importStar(require("./heap"));
var matrix = __importStar(require("./matrix"));
var nnDescent = __importStar(require("./nn_descent"));
var tree = __importStar(require("./tree"));
var utils = __importStar(require("./utils"));
var ml_levenberg_marquardt_1 = __importDefault(require("ml-levenberg-marquardt"));
var SMOOTH_K_TOLERANCE = 1e-5;
var MIN_K_DIST_SCALE = 1e-3;
var UMAP = (function () {
    function UMAP(params) {
        var _this = this;
        if (params === void 0) { params = {}; }
        this.learningRate = 1.0;
        this.localConnectivity = 1.0;
        this.minDist = 0.1;
        this.nComponents = 2;
        this.nEpochs = 0;
        this.nNeighbors = 15;
        this.negativeSampleRate = 5;
        this.random = Math.random;
        this.repulsionStrength = 1.0;
        this.setOpMixRatio = 1.0;
        this.spread = 1.0;
        this.transformQueueSize = 4.0;
        this.targetMetric = "categorical";
        this.targetWeight = 0.5;
        this.targetNNeighbors = this.nNeighbors;
        this.distanceFn = euclidean;
        this.isInitialized = false;
        this.rpForest = [];
        this.embedding = [];
        this.optimizationState = new OptimizationState();
        var setParam = function (key) {
            if (params[key] !== undefined)
                _this[key] = params[key];
        };
        setParam('distanceFn');
        setParam('learningRate');
        setParam('localConnectivity');
        setParam('minDist');
        setParam('nComponents');
        setParam('nEpochs');
        setParam('nNeighbors');
        setParam('negativeSampleRate');
        setParam('random');
        setParam('repulsionStrength');
        setParam('setOpMixRatio');
        setParam('spread');
        setParam('transformQueueSize');
    }
    UMAP.prototype.fit = function (X) {
        this.initializeFit(X);
        this.optimizeLayout();
        return this.embedding;
    };
    UMAP.prototype.fitAsync = function (X, callback) {
        if (callback === void 0) { callback = function () { return true; }; }
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        this.initializeFit(X);
                        return [4, this.optimizeLayoutAsync(callback)];
                    case 1:
                        _a.sent();
                        return [2, this.embedding];
                }
            });
        });
    };
    UMAP.prototype.setSupervisedProjection = function (Y, params) {
        if (params === void 0) { params = {}; }
        this.Y = Y;
        this.targetMetric = params.targetMetric || this.targetMetric;
        this.targetWeight = params.targetWeight || this.targetWeight;
        this.targetNNeighbors = params.targetNNeighbors || this.targetNNeighbors;
    };
    UMAP.prototype.setPrecomputedKNN = function (knnIndices, knnDistances) {
        this.knnIndices = knnIndices;
        this.knnDistances = knnDistances;
    };
    UMAP.prototype.initializeFit = function (X) {
        if (X.length <= this.nNeighbors) {
            throw new Error("Not enough data points (" + X.length + ") to create nNeighbors: " + this.nNeighbors + ".  Add more data points or adjust the configuration.");
        }
        if (this.X === X && this.isInitialized) {
            return this.getNEpochs();
        }
        this.X = X;
        if (!this.knnIndices && !this.knnDistances) {
            var knnResults = this.nearestNeighbors(X);
            this.knnIndices = knnResults.knnIndices;
            this.knnDistances = knnResults.knnDistances;
        }
        this.graph = this.fuzzySimplicialSet(X, this.nNeighbors, this.setOpMixRatio);
        this.makeSearchFns();
        this.searchGraph = this.makeSearchGraph(X);
        this.processGraphForSupervisedProjection();
        var _a = this.initializeSimplicialSetEmbedding(), head = _a.head, tail = _a.tail, epochsPerSample = _a.epochsPerSample;
        this.optimizationState.head = head;
        this.optimizationState.tail = tail;
        this.optimizationState.epochsPerSample = epochsPerSample;
        this.initializeOptimization();
        this.prepareForOptimizationLoop();
        this.isInitialized = true;
        return this.getNEpochs();
    };
    UMAP.prototype.makeSearchFns = function () {
        var _a = nnDescent.makeInitializations(this.distanceFn), initFromTree = _a.initFromTree, initFromRandom = _a.initFromRandom;
        this.initFromTree = initFromTree;
        this.initFromRandom = initFromRandom;
        this.search = nnDescent.makeInitializedNNSearch(this.distanceFn);
    };
    UMAP.prototype.makeSearchGraph = function (X) {
        var knnIndices = this.knnIndices;
        var knnDistances = this.knnDistances;
        var dims = [X.length, X.length];
        var searchGraph = new matrix.SparseMatrix([], [], [], dims);
        for (var i = 0; i < knnIndices.length; i++) {
            var knn = knnIndices[i];
            var distances = knnDistances[i];
            for (var j = 0; j < knn.length; j++) {
                var neighbor = knn[j];
                var distance = distances[j];
                if (distance > 0) {
                    searchGraph.set(i, neighbor, distance);
                }
            }
        }
        var transpose = matrix.transpose(searchGraph);
        return matrix.maximum(searchGraph, transpose);
    };
    UMAP.prototype.transform = function (toTransform) {
        var _this = this;
        var rawData = this.X;
        if (rawData === undefined || rawData.length === 0) {
            throw new Error('No data has been fit.');
        }
        var nNeighbors = Math.floor(this.nNeighbors * this.transformQueueSize);
        nNeighbors = Math.min(rawData.length, nNeighbors);
        var init = nnDescent.initializeSearch(this.rpForest, rawData, toTransform, nNeighbors, this.initFromRandom, this.initFromTree, this.random);
        var result = this.search(rawData, this.searchGraph, init, toTransform);
        var _a = heap.deheapSort(result), indices = _a.indices, distances = _a.weights;
        indices = indices.map(function (x) { return x.slice(0, _this.nNeighbors); });
        distances = distances.map(function (x) { return x.slice(0, _this.nNeighbors); });
        var adjustedLocalConnectivity = Math.max(0, this.localConnectivity - 1);
        var _b = this.smoothKNNDistance(distances, this.nNeighbors, adjustedLocalConnectivity), sigmas = _b.sigmas, rhos = _b.rhos;
        var _c = this.computeMembershipStrengths(indices, distances, sigmas, rhos), rows = _c.rows, cols = _c.cols, vals = _c.vals;
        var size = [toTransform.length, rawData.length];
        var graph = new matrix.SparseMatrix(rows, cols, vals, size);
        var normed = matrix.normalize(graph, "l1");
        var csrMatrix = matrix.getCSR(normed);
        var nPoints = toTransform.length;
        var eIndices = utils.reshape2d(csrMatrix.indices, nPoints, this.nNeighbors);
        var eWeights = utils.reshape2d(csrMatrix.values, nPoints, this.nNeighbors);
        var embedding = initTransform(eIndices, eWeights, this.embedding);
        var nEpochs = this.nEpochs
            ? this.nEpochs / 3
            : graph.nRows <= 10000
                ? 100
                : 30;
        var graphMax = graph
            .getValues()
            .reduce(function (max, val) { return (val > max ? val : max); }, 0);
        graph = graph.map(function (value) { return (value < graphMax / nEpochs ? 0 : value); });
        graph = matrix.eliminateZeros(graph);
        var epochsPerSample = this.makeEpochsPerSample(graph.getValues(), nEpochs);
        var head = graph.getRows();
        var tail = graph.getCols();
        this.assignOptimizationStateParameters({
            headEmbedding: embedding,
            tailEmbedding: this.embedding,
            head: head,
            tail: tail,
            currentEpoch: 0,
            nEpochs: nEpochs,
            nVertices: graph.getDims()[1],
            epochsPerSample: epochsPerSample,
        });
        this.prepareForOptimizationLoop();
        return this.optimizeLayout();
    };
    UMAP.prototype.processGraphForSupervisedProjection = function () {
        var _a = this, Y = _a.Y, X = _a.X;
        if (Y) {
            if (Y.length !== X.length) {
                throw new Error('Length of X and y must be equal');
            }
            if (this.targetMetric === "categorical") {
                var lt = this.targetWeight < 1.0;
                var farDist = lt ? 2.5 * (1.0 / (1.0 - this.targetWeight)) : 1.0e12;
                this.graph = this.categoricalSimplicialSetIntersection(this.graph, Y, farDist);
            }
        }
    };
    UMAP.prototype.step = function () {
        var currentEpoch = this.optimizationState.currentEpoch;
        if (currentEpoch < this.getNEpochs()) {
            this.optimizeLayoutStep(currentEpoch);
        }
        return this.optimizationState.currentEpoch;
    };
    UMAP.prototype.getEmbedding = function () {
        return this.embedding;
    };
    UMAP.prototype.nearestNeighbors = function (X) {
        var _a = this, distanceFn = _a.distanceFn, nNeighbors = _a.nNeighbors;
        var log2 = function (n) { return Math.log(n) / Math.log(2); };
        var metricNNDescent = nnDescent.makeNNDescent(distanceFn, this.random);
        var round = function (n) {
            return n === 0.5 ? 0 : Math.round(n);
        };
        var nTrees = 5 + Math.floor(round(Math.pow(X.length, 0.5) / 20.0));
        var nIters = Math.max(5, Math.floor(Math.round(log2(X.length))));
        this.rpForest = tree.makeForest(X, nNeighbors, nTrees, this.random);
        var leafArray = tree.makeLeafArray(this.rpForest);
        var _b = metricNNDescent(X, leafArray, nNeighbors, nIters), indices = _b.indices, weights = _b.weights;
        return { knnIndices: indices, knnDistances: weights };
    };
    UMAP.prototype.fuzzySimplicialSet = function (X, nNeighbors, setOpMixRatio) {
        if (setOpMixRatio === void 0) { setOpMixRatio = 1.0; }
        var _a = this, _b = _a.knnIndices, knnIndices = _b === void 0 ? [] : _b, _c = _a.knnDistances, knnDistances = _c === void 0 ? [] : _c, localConnectivity = _a.localConnectivity;
        var _d = this.smoothKNNDistance(knnDistances, nNeighbors, localConnectivity), sigmas = _d.sigmas, rhos = _d.rhos;
        var _e = this.computeMembershipStrengths(knnIndices, knnDistances, sigmas, rhos), rows = _e.rows, cols = _e.cols, vals = _e.vals;
        var size = [X.length, X.length];
        var sparseMatrix = new matrix.SparseMatrix(rows, cols, vals, size);
        var transpose = matrix.transpose(sparseMatrix);
        var prodMatrix = matrix.pairwiseMultiply(sparseMatrix, transpose);
        var a = matrix.subtract(matrix.add(sparseMatrix, transpose), prodMatrix);
        var b = matrix.multiplyScalar(a, setOpMixRatio);
        var c = matrix.multiplyScalar(prodMatrix, 1.0 - setOpMixRatio);
        var result = matrix.add(b, c);
        return result;
    };
    UMAP.prototype.categoricalSimplicialSetIntersection = function (simplicialSet, target, farDist, unknownDist) {
        if (unknownDist === void 0) { unknownDist = 1.0; }
        var intersection = fastIntersection(simplicialSet, target, unknownDist, farDist);
        intersection = matrix.eliminateZeros(intersection);
        return resetLocalConnectivity(intersection);
    };
    UMAP.prototype.smoothKNNDistance = function (distances, k, localConnectivity, nIter, bandwidth) {
        if (localConnectivity === void 0) { localConnectivity = 1.0; }
        if (nIter === void 0) { nIter = 64; }
        if (bandwidth === void 0) { bandwidth = 1.0; }
        var target = (Math.log(k) / Math.log(2)) * bandwidth;
        var rho = utils.zeros(distances.length);
        var result = utils.zeros(distances.length);
        for (var i = 0; i < distances.length; i++) {
            var lo = 0.0;
            var hi = Infinity;
            var mid = 1.0;
            var ithDistances = distances[i];
            var nonZeroDists = ithDistances.filter(function (d) { return d > 0.0; });
            if (nonZeroDists.length >= localConnectivity) {
                var index = Math.floor(localConnectivity);
                var interpolation = localConnectivity - index;
                if (index > 0) {
                    rho[i] = nonZeroDists[index - 1];
                    if (interpolation > SMOOTH_K_TOLERANCE) {
                        rho[i] +=
                            interpolation * (nonZeroDists[index] - nonZeroDists[index - 1]);
                    }
                }
                else {
                    rho[i] = interpolation * nonZeroDists[0];
                }
            }
            else if (nonZeroDists.length > 0) {
                rho[i] = utils.max(nonZeroDists);
            }
            for (var n = 0; n < nIter; n++) {
                var psum = 0.0;
                for (var j = 1; j < distances[i].length; j++) {
                    var d = distances[i][j] - rho[i];
                    if (d > 0) {
                        psum += Math.exp(-(d / mid));
                    }
                    else {
                        psum += 1.0;
                    }
                }
                if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) {
                    break;
                }
                if (psum > target) {
                    hi = mid;
                    mid = (lo + hi) / 2.0;
                }
                else {
                    lo = mid;
                    if (hi === Infinity) {
                        mid *= 2;
                    }
                    else {
                        mid = (lo + hi) / 2.0;
                    }
                }
            }
            result[i] = mid;
            if (rho[i] > 0.0) {
                var meanIthDistances = utils.mean(ithDistances);
                if (result[i] < MIN_K_DIST_SCALE * meanIthDistances) {
                    result[i] = MIN_K_DIST_SCALE * meanIthDistances;
                }
            }
            else {
                var meanDistances = utils.mean(distances.map(utils.mean));
                if (result[i] < MIN_K_DIST_SCALE * meanDistances) {
                    result[i] = MIN_K_DIST_SCALE * meanDistances;
                }
            }
        }
        return { sigmas: result, rhos: rho };
    };
    UMAP.prototype.computeMembershipStrengths = function (knnIndices, knnDistances, sigmas, rhos) {
        var nSamples = knnIndices.length;
        var nNeighbors = knnIndices[0].length;
        var rows = utils.zeros(nSamples * nNeighbors);
        var cols = utils.zeros(nSamples * nNeighbors);
        var vals = utils.zeros(nSamples * nNeighbors);
        for (var i = 0; i < nSamples; i++) {
            for (var j = 0; j < nNeighbors; j++) {
                var val = 0;
                if (knnIndices[i][j] === -1) {
                    continue;
                }
                if (knnIndices[i][j] === i) {
                    val = 0.0;
                }
                else if (knnDistances[i][j] - rhos[i] <= 0.0) {
                    val = 1.0;
                }
                else {
                    val = Math.exp(-((knnDistances[i][j] - rhos[i]) / sigmas[i]));
                }
                rows[i * nNeighbors + j] = i;
                cols[i * nNeighbors + j] = knnIndices[i][j];
                vals[i * nNeighbors + j] = val;
            }
        }
        return { rows: rows, cols: cols, vals: vals };
    };
    UMAP.prototype.initializeSimplicialSetEmbedding = function () {
        var _this = this;
        var nEpochs = this.getNEpochs();
        var nComponents = this.nComponents;
        var graphValues = this.graph.getValues();
        var graphMax = 0;
        for (var i = 0; i < graphValues.length; i++) {
            var value = graphValues[i];
            if (graphMax < graphValues[i]) {
                graphMax = value;
            }
        }
        var graph = this.graph.map(function (value) {
            if (value < graphMax / nEpochs) {
                return 0;
            }
            else {
                return value;
            }
        });
        this.embedding = utils.zeros(graph.nRows).map(function () {
            return utils.zeros(nComponents).map(function () {
                return utils.tauRand(_this.random) * 20 + -10;
            });
        });
        var weights = [];
        var head = [];
        var tail = [];
        var rowColValues = graph.getAll();
        for (var i = 0; i < rowColValues.length; i++) {
            var entry = rowColValues[i];
            if (entry.value) {
                weights.push(entry.value);
                tail.push(entry.row);
                head.push(entry.col);
            }
        }
        var epochsPerSample = this.makeEpochsPerSample(weights, nEpochs);
        return { head: head, tail: tail, epochsPerSample: epochsPerSample };
    };
    UMAP.prototype.makeEpochsPerSample = function (weights, nEpochs) {
        var result = utils.filled(weights.length, -1.0);
        var max = utils.max(weights);
        var nSamples = weights.map(function (w) { return (w / max) * nEpochs; });
        nSamples.forEach(function (n, i) {
            if (n > 0)
                result[i] = nEpochs / nSamples[i];
        });
        return result;
    };
    UMAP.prototype.assignOptimizationStateParameters = function (state) {
        Object.assign(this.optimizationState, state);
    };
    UMAP.prototype.prepareForOptimizationLoop = function () {
        var _a = this, repulsionStrength = _a.repulsionStrength, learningRate = _a.learningRate, negativeSampleRate = _a.negativeSampleRate;
        var _b = this.optimizationState, epochsPerSample = _b.epochsPerSample, headEmbedding = _b.headEmbedding, tailEmbedding = _b.tailEmbedding;
        var dim = headEmbedding[0].length;
        var moveOther = headEmbedding.length === tailEmbedding.length;
        var epochsPerNegativeSample = epochsPerSample.map(function (e) { return e / negativeSampleRate; });
        var epochOfNextNegativeSample = __spread(epochsPerNegativeSample);
        var epochOfNextSample = __spread(epochsPerSample);
        this.assignOptimizationStateParameters({
            epochOfNextSample: epochOfNextSample,
            epochOfNextNegativeSample: epochOfNextNegativeSample,
            epochsPerNegativeSample: epochsPerNegativeSample,
            moveOther: moveOther,
            initialAlpha: learningRate,
            alpha: learningRate,
            gamma: repulsionStrength,
            dim: dim,
        });
    };
    UMAP.prototype.initializeOptimization = function () {
        var headEmbedding = this.embedding;
        var tailEmbedding = this.embedding;
        var _a = this.optimizationState, head = _a.head, tail = _a.tail, epochsPerSample = _a.epochsPerSample;
        var nEpochs = this.getNEpochs();
        var nVertices = this.graph.nCols;
        var _b = findABParams(this.spread, this.minDist), a = _b.a, b = _b.b;
        this.assignOptimizationStateParameters({
            headEmbedding: headEmbedding,
            tailEmbedding: tailEmbedding,
            head: head,
            tail: tail,
            epochsPerSample: epochsPerSample,
            a: a,
            b: b,
            nEpochs: nEpochs,
            nVertices: nVertices,
        });
    };
    UMAP.prototype.optimizeLayoutStep = function (n) {
        var optimizationState = this.optimizationState;
        var head = optimizationState.head, tail = optimizationState.tail, headEmbedding = optimizationState.headEmbedding, tailEmbedding = optimizationState.tailEmbedding, epochsPerSample = optimizationState.epochsPerSample, epochOfNextSample = optimizationState.epochOfNextSample, epochOfNextNegativeSample = optimizationState.epochOfNextNegativeSample, epochsPerNegativeSample = optimizationState.epochsPerNegativeSample, moveOther = optimizationState.moveOther, initialAlpha = optimizationState.initialAlpha, alpha = optimizationState.alpha, gamma = optimizationState.gamma, a = optimizationState.a, b = optimizationState.b, dim = optimizationState.dim, nEpochs = optimizationState.nEpochs, nVertices = optimizationState.nVertices;
        var clipValue = 4.0;
        for (var i = 0; i < epochsPerSample.length; i++) {
            if (epochOfNextSample[i] > n) {
                continue;
            }
            var j = head[i];
            var k = tail[i];
            var current = headEmbedding[j];
            var other = tailEmbedding[k];
            var distSquared = rDist(current, other);
            var gradCoeff = 0;
            if (distSquared > 0) {
                gradCoeff = -2.0 * a * b * Math.pow(distSquared, b - 1.0);
                gradCoeff /= a * Math.pow(distSquared, b) + 1.0;
            }
            for (var d = 0; d < dim; d++) {
                var gradD = clip(gradCoeff * (current[d] - other[d]), clipValue);
                current[d] += gradD * alpha;
                if (moveOther) {
                    other[d] += -gradD * alpha;
                }
            }
            epochOfNextSample[i] += epochsPerSample[i];
            var nNegSamples = Math.floor((n - epochOfNextNegativeSample[i]) / epochsPerNegativeSample[i]);
            for (var p = 0; p < nNegSamples; p++) {
                var k_1 = utils.tauRandInt(nVertices, this.random);
                var other_1 = tailEmbedding[k_1];
                var distSquared_1 = rDist(current, other_1);
                var gradCoeff_1 = 0.0;
                if (distSquared_1 > 0.0) {
                    gradCoeff_1 = 2.0 * gamma * b;
                    gradCoeff_1 /=
                        (0.001 + distSquared_1) * (a * Math.pow(distSquared_1, b) + 1);
                }
                else if (j === k_1) {
                    continue;
                }
                for (var d = 0; d < dim; d++) {
                    var gradD = 4.0;
                    if (gradCoeff_1 > 0.0) {
                        gradD = clip(gradCoeff_1 * (current[d] - other_1[d]), clipValue);
                    }
                    current[d] += gradD * alpha;
                }
            }
            epochOfNextNegativeSample[i] += nNegSamples * epochsPerNegativeSample[i];
        }
        optimizationState.alpha = initialAlpha * (1.0 - n / nEpochs);
        optimizationState.currentEpoch += 1;
        return headEmbedding;
    };
    UMAP.prototype.optimizeLayoutAsync = function (epochCallback) {
        var _this = this;
        if (epochCallback === void 0) { epochCallback = function () { return true; }; }
        return new Promise(function (resolve, reject) {
            var step = function () { return __awaiter(_this, void 0, void 0, function () {
                var _a, nEpochs, currentEpoch, epochCompleted, shouldStop, isFinished;
                return __generator(this, function (_b) {
                    try {
                        _a = this.optimizationState, nEpochs = _a.nEpochs, currentEpoch = _a.currentEpoch;
                        this.embedding = this.optimizeLayoutStep(currentEpoch);
                        epochCompleted = this.optimizationState.currentEpoch;
                        shouldStop = epochCallback(epochCompleted) === false;
                        isFinished = epochCompleted === nEpochs;
                        if (!shouldStop && !isFinished) {
                            setTimeout(function () { return step(); }, 0);
                        }
                        else {
                            return [2, resolve(isFinished)];
                        }
                    }
                    catch (err) {
                        reject(err);
                    }
                    return [2];
                });
            }); };
            setTimeout(function () { return step(); }, 0);
        });
    };
    UMAP.prototype.optimizeLayout = function (epochCallback) {
        if (epochCallback === void 0) { epochCallback = function () { return true; }; }
        var isFinished = false;
        var embedding = [];
        while (!isFinished) {
            var _a = this.optimizationState, nEpochs = _a.nEpochs, currentEpoch = _a.currentEpoch;
            embedding = this.optimizeLayoutStep(currentEpoch);
            var epochCompleted = this.optimizationState.currentEpoch;
            var shouldStop = epochCallback(epochCompleted) === false;
            isFinished = epochCompleted === nEpochs || shouldStop;
        }
        return embedding;
    };
    UMAP.prototype.getNEpochs = function () {
        var graph = this.graph;
        if (this.nEpochs > 0) {
            return this.nEpochs;
        }
        var length = graph.nRows;
        if (length <= 2500) {
            return 500;
        }
        else if (length <= 5000) {
            return 400;
        }
        else if (length <= 7500) {
            return 300;
        }
        else {
            return 200;
        }
    };
    return UMAP;
}());
exports.UMAP = UMAP;
function euclidean(x, y) {
    var result = 0;
    for (var i = 0; i < x.length; i++) {
        result += Math.pow((x[i] - y[i]), 2);
    }
    return Math.sqrt(result);
}
exports.euclidean = euclidean;
function cosine(x, y) {
    var result = 0.0;
    var normX = 0.0;
    var normY = 0.0;
    for (var i = 0; i < x.length; i++) {
        result += x[i] * y[i];
        normX += Math.pow(x[i], 2);
        normY += Math.pow(y[i], 2);
    }
    if (normX === 0 && normY === 0) {
        return 0;
    }
    else if (normX === 0 || normY === 0) {
        return 1.0;
    }
    else {
        return 1.0 - result / Math.sqrt(normX * normY);
    }
}
exports.cosine = cosine;
var OptimizationState = (function () {
    function OptimizationState() {
        this.currentEpoch = 0;
        this.headEmbedding = [];
        this.tailEmbedding = [];
        this.head = [];
        this.tail = [];
        this.epochsPerSample = [];
        this.epochOfNextSample = [];
        this.epochOfNextNegativeSample = [];
        this.epochsPerNegativeSample = [];
        this.moveOther = true;
        this.initialAlpha = 1.0;
        this.alpha = 1.0;
        this.gamma = 1.0;
        this.a = 1.5769434603113077;
        this.b = 0.8950608779109733;
        this.dim = 2;
        this.nEpochs = 500;
        this.nVertices = 0;
    }
    return OptimizationState;
}());
function clip(x, clipValue) {
    if (x > clipValue)
        return clipValue;
    else if (x < -clipValue)
        return -clipValue;
    else
        return x;
}
function rDist(x, y) {
    var result = 0.0;
    for (var i = 0; i < x.length; i++) {
        result += Math.pow(x[i] - y[i], 2);
    }
    return result;
}
function findABParams(spread, minDist) {
    var curve = function (_a) {
        var _b = __read(_a, 2), a = _b[0], b = _b[1];
        return function (x) {
            return 1.0 / (1.0 + a * Math.pow(x, (2 * b)));
        };
    };
    var xv = utils
        .linear(0, spread * 3, 300)
        .map(function (val) { return (val < minDist ? 1.0 : val); });
    var yv = utils.zeros(xv.length).map(function (val, index) {
        var gte = xv[index] >= minDist;
        return gte ? Math.exp(-(xv[index] - minDist) / spread) : val;
    });
    var initialValues = [0.5, 0.5];
    var data = { x: xv, y: yv };
    var options = {
        damping: 1.5,
        initialValues: initialValues,
        gradientDifference: 10e-2,
        maxIterations: 100,
        errorTolerance: 10e-3,
    };
    var parameterValues = ml_levenberg_marquardt_1.default(data, curve, options).parameterValues;
    var _a = __read(parameterValues, 2), a = _a[0], b = _a[1];
    return { a: a, b: b };
}
exports.findABParams = findABParams;
function fastIntersection(graph, target, unknownDist, farDist) {
    if (unknownDist === void 0) { unknownDist = 1.0; }
    if (farDist === void 0) { farDist = 5.0; }
    return graph.map(function (value, row, col) {
        if (target[row] === -1 || target[col] === -1) {
            return value * Math.exp(-unknownDist);
        }
        else if (target[row] !== target[col]) {
            return value * Math.exp(-farDist);
        }
        else {
            return value;
        }
    });
}
exports.fastIntersection = fastIntersection;
function resetLocalConnectivity(simplicialSet) {
    simplicialSet = matrix.normalize(simplicialSet, "max");
    var transpose = matrix.transpose(simplicialSet);
    var prodMatrix = matrix.pairwiseMultiply(transpose, simplicialSet);
    simplicialSet = matrix.add(simplicialSet, matrix.subtract(transpose, prodMatrix));
    return matrix.eliminateZeros(simplicialSet);
}
exports.resetLocalConnectivity = resetLocalConnectivity;
function initTransform(indices, weights, embedding) {
    var result = utils
        .zeros(indices.length)
        .map(function (z) { return utils.zeros(embedding[0].length); });
    for (var i = 0; i < indices.length; i++) {
        for (var j = 0; j < indices[0].length; j++) {
            for (var d = 0; d < embedding[0].length; d++) {
                var a = indices[i][j];
                result[i][d] += weights[i][j] * embedding[a][d];
            }
        }
    }
    return result;
}
exports.initTransform = initTransform;
