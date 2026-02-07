# hdbscan-ts

A TypeScript implementation of **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise), based on [Campello et al. 2017](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14).

HDBSCAN is particularly effective at:

- Detecting clusters of varying densities
- Identifying noise points
- Handling clusters of different shapes
- Providing cluster membership probabilities

## Installation

```bash
npm install hdbscan-ts
```

## Usage

```ts
import { HDBSCAN } from "hdbscan-ts";
const data = [
  [1.1, 2.1],
  [2.1, 1.1],
  [1.1, 1.1],
  [0.1, 1.1],
  [10.1, 11.1],
  [11.1, 10.1],
  [10.1, 10.1]
];
const hdbscan = new HDBSCAN({
  minClusterSize: 2
});
const labels = hdbscan.fit(data);
console.log(labels);
// [0, 0, 0, 0, 1, 1, 1]
```

## API

### HDBSCAN

#### Constructor Options

- `minClusterSize` (default: 5): Minimum size of clusters
- `minSamples` (default: 5): Minimum number of samples in neighborhood
- `debugMode` (default: false): Enable debug logging

#### Methods

- `fit(data: number[][]): HDBSCAN`
- `labels_: number[]`
- `probabilities_: number[]`

## License

MIT
