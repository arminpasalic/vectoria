// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#include "knncolle/knncolle.hpp"
#include "knncolle_hnsw/knncolle_hnsw.hpp"
#include "knncolle_nndescent.hpp"

#include "common.hpp"
#include "distances.hpp"
#include "knn.hpp"

class KNNContextBase {
  public:
    virtual int search(int i, int k, int *output_indices,
                       float *output_distances) = 0;
    virtual int search(const float *query, int k, int *output_indices,
                       float *output_distances) = 0;
    virtual ~KNNContextBase() {}
};

template <typename KNNAlgorithm> class KNNContext : public KNNContextBase {
  private:
    std::unique_ptr<KNNAlgorithm> algorithm;

  public:
    KNNContext(std::unique_ptr<KNNAlgorithm> &&algo)
        : algorithm(std::move(algo)) {}

    virtual int search(int i, int k, int *output_indices,
                       float *output_distances) {
        std::vector<int> idx;
        std::vector<float> distances;

        auto searcher = algorithm->initialize();

        searcher->search(i, k, &idx, &distances);

        int p = 0;
        for (; p < idx.size() && p < k; p++) {
            if (output_indices) {
                output_indices[p] = idx[p];
            }
            if (output_distances) {
                output_distances[p] = distances[p];
            }
        }

        return p;
    }
    virtual int search(const float *query, int k, int *output_indices,
                       float *output_distances) {
        std::vector<int> idx;
        std::vector<float> distances;

        auto searcher = algorithm->initialize();

        searcher->search(query, k, &idx, &distances);

        int p = 0;
        for (; p < idx.size() && p < k; p++) {
            if (output_indices) {
                output_indices[p] = idx[p];
            }
            if (output_distances) {
                output_distances[p] = distances[p];
            }
        }

        return p;
    }
};

class KNNOptions {
  public:
    int method;
    int metric;
    knncolle_hnsw::HnswOptions<int, float> hnsw_options;
    knncolle_nndescent::NNDescentOptions nndescent_options;

    KNNOptions() {
        method = kKNN_HNSW;
        metric = kMETRIC_EUCLIDEAN;
        // Default to fixed seed for nndescent.
        nndescent_options.seed = 42;
    }
};

knn_options_t knn_options_create() { return new KNNOptions(); }

void knn_options_destroy(knn_options_t ptr) { delete ptr; }

int knn_options_number(knn_options_t ptr, const char *name_cstr, double value) {
    std::string name(name_cstr);
    if (name == "hnsw_n_links") {
        ptr->hnsw_options.num_links = value;
    } else if (name == "hnsw_ef_construction") {
        ptr->hnsw_options.ef_construction = value;
    } else if (name == "hnsw_ef_search") {
        ptr->hnsw_options.ef_search = value;
    } else if (name == "nndescent_n_neighbors") {
        ptr->nndescent_options.n_neighbors = value;
    } else if (name == "nndescent_n_trees") {
        ptr->nndescent_options.n_trees = value;
    } else if (name == "nndescent_n_iters") {
        ptr->nndescent_options.n_iters = value;
    } else if (name == "nndescent_seed") {
        ptr->nndescent_options.seed = value;
    } else {
        return -1;
    }
    return 0;
}

int knn_options_string(knn_options_t ptr, const char *name_cstr,
                       const char *value_cstr) {
    std::string name(name_cstr);
    std::string value(value_cstr);
    if (name == "metric") {
        if (value == "euclidean") {
            ptr->metric = kMETRIC_EUCLIDEAN;
        } else if (value == "cosine") {
            ptr->metric = kMETRIC_COSINE;
        } else {
            return -1;
        }
    } else if (name == "method") {
        if (value == "hnsw") {
            ptr->method = kKNN_HNSW;
        } else if (value == "nndescent") {
            ptr->method = kKNN_NNDESCENT;
        } else if (value == "vptree") {
            ptr->method = kKNN_VPTREE;
        } else {
            return -1;
        }
    } else {
        return -1;
    }
    return 0;
}

knn_context_t knn_context_create_f32(int count, int input_dim, float *data,
                                     knn_options_t options_ptr) {
    knncolle::SimpleMatrix<int, int, float> mat(input_dim, count, data);
    int metric = options_ptr->metric;
    if (metric == kMETRIC_COSINE) {
        // For cosine metric, we convert it to euclidean metrics by normalizing
        // all vectors. After normalization, cosine distance is equal to squared
        // euclidean distance.
        for (int i = 0; i < count; i++) {
            normalize_vector(data + i * input_dim, input_dim);
        }
    }
    switch (options_ptr->method) {
    case kKNN_HNSW: {
        knncolle_hnsw::HnswBuilder<knncolle::SimpleMatrix<int, int, float>,
                                   float>
            builder(options_ptr->hnsw_options);
        switch (metric) {
        case kMETRIC_COSINE: {
            builder.get_options().distance_options.create =
                [](int dim) -> hnswlib::SpaceInterface<float> * {
                return new HNSWHalfSquaredEuclideanDistance<float>(dim);
            };
        }
        }
        return new KNNContext(builder.build_unique(mat));
    }
    case kKNN_NNDESCENT: {
        knncolle_nndescent::NNDescentOptions options =
            options_ptr->nndescent_options;
        switch (metric) {
        case kMETRIC_COSINE: {
            options.metric = "cosine";
        }
        }
        knncolle_nndescent::NNDescentBuilder builder(options);
        return new KNNContext(builder.build_unique(mat));
    }
    case kKNN_VPTREE:
    default: {
        switch (metric) {
        case kMETRIC_COSINE: {
            knncolle::VptreeBuilder<HalfSquaredEuclideanDistance,
                                    knncolle::SimpleMatrix<int, int, float>,
                                    float>
                builder;
            return new KNNContext(builder.build_unique(mat));
        }
        case kMETRIC_EUCLIDEAN:
        default: {
            knncolle::VptreeBuilder<knncolle::EuclideanDistance,
                                    knncolle::SimpleMatrix<int, int, float>,
                                    float>
                builder;
            return new KNNContext(builder.build_unique(mat));
        }
        }
    }
    }
}

int knn_context_query_by_index(knn_context_t ctx, int index, int k,
                               int *output_indices, float *output_distances) {
    return ctx->search(index, k, output_indices, output_distances);
}

int knn_context_query_by_vector(knn_context_t ctx, const float *data, int k,
                                int *output_indices, float *output_distances) {
    return ctx->search(data, k, output_indices, output_distances);
}

void knn_context_destroy(knn_context_t ctx) { delete ctx; }
