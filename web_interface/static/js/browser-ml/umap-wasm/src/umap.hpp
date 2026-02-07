// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#ifndef UMAPWASM_UMAP_H
#define UMAPWASM_UMAP_H

#include "common.hpp"
#include "knn.hpp"

static const int kINITIALIZE_METHOD_SPECTRAL = 0;
static const int kINITIALIZE_METHOD_RANDOM = 1;
static const int kINITIALIZE_METHOD_NONE = 2;

class UMAPContextBase;
class UMAPOptions;

typedef UMAPContextBase *umap_context_t;
typedef UMAPOptions *umap_options_t;

extern "C" {
// # umap_options_t
APIFUNC umap_options_t umap_options_create();
APIFUNC void umap_options_destroy(umap_options_t ptr);
APIFUNC int umap_options_number(umap_options_t ptr, const char *name,
                                double value);
APIFUNC int umap_options_string(umap_options_t ptr, const char *name,
                                const char *value);

APIFUNC umap_context_t umap_context_create_f32(int count, int input_dim,
                                               int output_dim, float *data,
                                               float *embedding,
                                               umap_options_t options_ptr);

APIFUNC void umap_context_destroy(umap_context_t context);

APIFUNC void umap_context_run(umap_context_t context, int iter);
APIFUNC int umap_context_n_epochs(umap_context_t context);
APIFUNC int umap_context_epoch(umap_context_t context);
}

#endif