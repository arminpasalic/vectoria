// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#ifndef UMAPWASM_KNN_H
#define UMAPWASM_KNN_H

#include "common.hpp"

static const int kKNN_VPTREE = 0;
static const int kKNN_HNSW = 2;
static const int kKNN_NNDESCENT = 3;

static const int kMETRIC_EUCLIDEAN = 0;
static const int kMETRIC_COSINE = 1;

class KNNContextBase;
class KNNOptions;

typedef KNNContextBase *knn_context_t;
typedef KNNOptions *knn_options_t;

extern "C" {
// # knn_options_t
APIFUNC knn_options_t knn_options_create();
APIFUNC void knn_options_destroy(knn_options_t ptr);
APIFUNC int knn_options_number(knn_options_t ptr, const char *name,
                               double value);
APIFUNC int knn_options_string(knn_options_t ptr, const char *name,
                               const char *value);

// # knn_context_t
APIFUNC knn_context_t knn_context_create_f32(int count, int input_dim,
                                             float *data,
                                             knn_options_t options_ptr);
APIFUNC void knn_context_destroy(knn_context_t ctx);

APIFUNC int knn_context_query_by_index(knn_context_t ctx, int index, int k,
                                       int *output_indices,
                                       float *output_distances);

APIFUNC int knn_context_query_by_vector(knn_context_t ctx, const float *data,
                                        int k, int *output_indices,
                                        float *output_distances);
}
#endif