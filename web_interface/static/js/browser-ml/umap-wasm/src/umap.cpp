// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#include <cmath>

#include "knncolle/knncolle.hpp"
#include "knncolle_hnsw/knncolle_hnsw.hpp"
#include "knncolle_nndescent.hpp"
#include "umappp/initialize.hpp"

#include "common.hpp"
#include "distances.hpp"
#include "umap.hpp"

class UMAPContextBase {
  public:
    virtual void run(int epoch_limit) = 0;
    virtual int n_epochs() = 0;
    virtual int epoch() = 0;
    virtual ~UMAPContextBase() {}
};

template <typename Status_> class UMAPContext : public UMAPContextBase {
  public:
    UMAPContext(Status_ &&status) : status_(std::move(status)) {}
    virtual int n_epochs() { return status_.num_epochs(); }
    virtual int epoch() { return status_.epoch(); }
    void run(int epoch_limit) { status_.run(epoch_limit); }

  private:
    Status_ status_;
};

class UMAPOptions {
  public:
    int knn_method;
    int metric;
    umappp::Options options;
    knncolle_hnsw::HnswOptions<int, float> hnsw_options;
    knncolle_nndescent::NNDescentOptions nndescent_options;

    UMAPOptions() {
        knn_method = kKNN_HNSW;
        metric = kMETRIC_EUCLIDEAN;
        // Default to fixed seed for nndescent.
        nndescent_options.seed = 42;
    }
};

umap_options_t umap_options_create() { return new UMAPOptions(); }

void umap_options_destroy(umap_options_t ptr) { delete ptr; }

int umap_options_number(umap_options_t ptr, const char *name_cstr,
                        double value) {
    std::string name(name_cstr);
    if (name == "local_connectivity") {
        ptr->options.local_connectivity = value;
    } else if (name == "bandwidth") {
        ptr->options.bandwidth = value;
    } else if (name == "mix_ratio") {
        ptr->options.mix_ratio = value;
    } else if (name == "spread") {
        ptr->options.spread = value;
    } else if (name == "min_dist") {
        ptr->options.min_dist = value;
    } else if (name == "a") {
        ptr->options.a = value;
    } else if (name == "b") {
        ptr->options.b = value;
    } else if (name == "repulsion_strength") {
        ptr->options.repulsion_strength = value;
    } else if (name == "learning_rate") {
        ptr->options.learning_rate = value;
    } else if (name == "negative_sample_rate") {
        ptr->options.negative_sample_rate = value;
    } else if (name == "n_epochs") {
        ptr->options.num_epochs = value;
    } else if (name == "n_neighbors") {
        ptr->options.num_neighbors = value;
    } else if (name == "seed") {
        ptr->options.seed = value;
    } else if (name == "hnsw_n_links") {
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

int umap_options_string(umap_options_t ptr, const char *name_cstr,
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
    } else if (name == "knn_method") {
        if (value == "hnsw") {
            ptr->knn_method = kKNN_HNSW;
        } else if (value == "nndescent") {
            ptr->knn_method = kKNN_NNDESCENT;
        } else if (value == "vptree") {
            ptr->knn_method = kKNN_VPTREE;
        } else {
            return -1;
        }
    } else if (name == "initialize_method") {
        if (value == "spectral") {
            ptr->options.initialize = umappp::InitializeMethod::SPECTRAL;
        } else if (value == "random") {
            ptr->options.initialize = umappp::InitializeMethod::RANDOM;
        } else if (value == "none") {
            ptr->options.initialize = umappp::InitializeMethod::NONE;
        } else {
            return -1;
        }
    } else {
        return -1;
    }
    return 0;
}

void fill_non_finite(float *data, int count, float value) {
    for (int i = 0; i < count; i++) {
        if (!std::isfinite(data[i])) {
            data[i] = value;
        }
    }
}

umap_context_t umap_context_create_f32(int count, int input_dim, int output_dim,
                                       float *data, float *embedding,
                                       umap_options_t options_ptr) {
    umappp::Options options = options_ptr->options;
    int knn_method = options_ptr->knn_method;
    int metric = options_ptr->metric;

    fill_non_finite(data, count * input_dim, 0);

    if (metric == kMETRIC_COSINE) {
        // For cosine metric, we convert it to euclidean metrics by normalizing
        // all vectors. After normalization, cosine distance is equal to squared
        // euclidean distance.
        for (int i = 0; i < count; i++) {
            normalize_vector(data + i * input_dim, input_dim);
        }
    }

    switch (knn_method) {
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
        return new UMAPContext(umappp::initialize(
            input_dim, count, data, builder, output_dim, embedding, options));
    }
    case kKNN_NNDESCENT: {
        knncolle_nndescent::NNDescentOptions nnoptions =
            options_ptr->nndescent_options;
        switch (metric) {
        case kMETRIC_COSINE: {
            nnoptions.metric = "cosine";
        }
        }
        if (nnoptions.n_neighbors < options.num_neighbors + 1) {
            nnoptions.n_neighbors = options.num_neighbors + 1;
        }
        knncolle_nndescent::NNDescentBuilder builder(nnoptions);
        return new UMAPContext(umappp::initialize(
            input_dim, count, data, builder, output_dim, embedding, options));
    }
    case kKNN_VPTREE:
    default: {
        switch (metric) {
        case kMETRIC_COSINE: {
            knncolle::VptreeBuilder<HalfSquaredEuclideanDistance,
                                    knncolle::SimpleMatrix<int, int, float>,
                                    float>
                builder;
            return new UMAPContext(umappp::initialize(input_dim, count, data,
                                                      builder, output_dim,
                                                      embedding, options));
        }
        case kMETRIC_EUCLIDEAN:
        default: {
            knncolle::VptreeBuilder<knncolle::EuclideanDistance,
                                    knncolle::SimpleMatrix<int, int, float>,
                                    float>
                builder;
            return new UMAPContext(umappp::initialize(input_dim, count, data,
                                                      builder, output_dim,
                                                      embedding, options));
        }
        }
    }
    }
}

void umap_context_run(umap_context_t context, int iter) { context->run(iter); }

int umap_context_n_epochs(umap_context_t context) {
    return context->n_epochs();
}

int umap_context_epoch(umap_context_t context) { return context->epoch(); }

void umap_context_destroy(umap_context_t context) { delete context; }
