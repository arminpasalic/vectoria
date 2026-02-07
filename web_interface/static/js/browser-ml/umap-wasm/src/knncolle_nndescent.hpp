// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#ifndef KNNCOLLE_NNDESCENT_HPP
#define KNNCOLLE_NNDESCENT_HPP

/**
 * @file knncolle_nndescent.hpp
 * @brief Approximate nearest neighbor search with nndescent
 * (https://github.com/brj0/nndescent)
 */

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include "knncolle/knncolle.hpp"
#include "nnd.h"

/**
 * @namespace knncolle_nndescent
 * @brief Approximate nearest neighbor search with nndescent.
 *
 * The nndescent implementation is https://github.com/brj0/nndescent, by brj0.
 */
namespace knncolle_nndescent {

typedef knncolle::SimpleMatrix<int, int, float> InputMatrix;
typedef nndescent::Parms NNDescentOptions;

class NNDescentSearcher : public knncolle::Searcher<int, float> {
  private:
    nndescent::NNDescent *my_index;
    int my_dim;

  public:
    NNDescentSearcher(nndescent::NNDescent *index, int my_dim)
        : my_index(index), my_dim(my_dim) {}

  public:
    void search(int i, int k, std::vector<int> *output_indices,
                std::vector<float> *output_distances) {
        fill_outputs(k, i, i, my_index->neighbor_indices,
                     my_index->neighbor_distances, output_indices,
                     output_distances);
    }

    void search(const float *query, int k, std::vector<int> *output_indices,
                std::vector<float> *output_distances) {
        nndescent::Matrix data(1, my_dim, const_cast<float *>(query));
        my_index->query(data, k);
        fill_outputs(k, -1, 0, my_index->query_indices,
                     my_index->query_distances, output_indices,
                     output_distances);
    }

  private:
    void fill_outputs(int k, int self, int row, nndescent::Matrix<int> &indices,
                      nndescent::Matrix<float> &distances,
                      std::vector<int> *output_indices,
                      std::vector<float> *output_distances) {
        int ncols = indices.ncols();
        int count = ncols > k ? k : ncols;
        if (output_indices) {
            output_indices->resize(count);
            int p = 0;
            for (int j = 0; j < ncols && p < count; j++) {
                int idx = indices(row, j);
                if (idx != self) {
                    (*output_indices)[p++] = idx;
                }
            }
            output_indices->resize(p);
        }
        if (output_distances) {
            output_distances->resize(count);
            int p = 0;
            for (int j = 0; j < ncols && p < count; j++) {
                int idx = indices(row, j);
                if (idx != self) {
                    (*output_distances)[p++] = distances(row, j);
                }
            }
            output_distances->resize(p);
        }
    }
};

class NNDescentPrebuilt : public knncolle::Prebuilt<int, int, float> {
  public:
    NNDescentPrebuilt(const InputMatrix &data, const NNDescentOptions &options)
        : my_dim(data.num_dimensions()), my_obs(data.num_observations()) {
        NNDescentOptions opts = options;
        auto workspace = data.create_workspace();
        data_ptr = const_cast<float *>(data.get_observation(workspace));
        my_matrix.reset(new nndescent::Matrix(my_obs, my_dim, data_ptr));
        my_index.reset(new nndescent::NNDescent(*my_matrix.get(), opts));
    }

  private:
    int my_dim;
    int my_obs;
    float *data_ptr;
    std::shared_ptr<nndescent::NNDescent> my_index;
    std::shared_ptr<nndescent::Matrix<float>> my_matrix;

    friend class NNDescentSearcher;

  public:
    int num_dimensions() const { return my_dim; }
    int num_observations() const { return my_obs; }

    std::unique_ptr<knncolle::Searcher<int, float>> initialize() const {
        return std::make_unique<NNDescentSearcher>(my_index.get(), my_dim);
    }
};

class NNDescentBuilder : public knncolle::Builder<InputMatrix, float> {
  private:
    NNDescentOptions my_options;

  public:
    NNDescentBuilder(NNDescentOptions options)
        : my_options(std::move(options)) {}

    NNDescentBuilder() = default;

    NNDescentOptions &get_options() { return my_options; }

  public:
    knncolle::Prebuilt<int, int, float> *
    build_raw(const InputMatrix &data) const {
        return new NNDescentPrebuilt(data, my_options);
    }
};

} // namespace knncolle_nndescent

#endif