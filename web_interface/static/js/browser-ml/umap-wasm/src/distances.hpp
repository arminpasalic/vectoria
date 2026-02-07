// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

#include "hnswlib/hnswlib.h"

struct HalfSquaredEuclideanDistance {
    template <typename Output_, typename DataX_, typename DataY_, typename Dim_>
    static Output_ raw_distance(const DataX_ *x, const DataY_ *y,
                                Dim_ num_dimensions) {
        Output_ output = 0;
        for (Dim_ d = 0; d < num_dimensions; ++d, ++x, ++y) {
            auto delta = static_cast<Output_>(*x) - static_cast<Output_>(*y);
            output += delta * delta;
        }
        return output / 2.0;
    }

    template <typename Output_> static Output_ normalize(Output_ raw) {
        return raw;
    }

    template <typename Output_> static Output_ denormalize(Output_ norm) {
        return norm;
    }
};

template <typename InternalData_>
class HNSWHalfSquaredEuclideanDistance
    : public hnswlib::SpaceInterface<InternalData_> {
  private:
    size_t my_data_size;
    size_t my_dim;

  public:
    /**
     * @param dim Number of dimensions over which to compute the distance.
     */
    HNSWHalfSquaredEuclideanDistance(size_t dim)
        : my_data_size(dim * sizeof(InternalData_)), my_dim(dim) {}

    /**
     * @cond
     */
  public:
    size_t get_data_size() { return my_data_size; }

    hnswlib::DISTFUNC<InternalData_> get_dist_func() { return L2; }

    void *get_dist_func_param() { return &my_dim; }

  private:
    static InternalData_ L2(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
        const InternalData_ *pVect1 =
            static_cast<const InternalData_ *>(pVect1v);
        const InternalData_ *pVect2 =
            static_cast<const InternalData_ *>(pVect2v);
        size_t qty = *((size_t *)qty_ptr);
        InternalData_ res = 0;
        for (; qty > 0; --qty, ++pVect1, ++pVect2) {
            auto delta = *pVect1 - *pVect2;
            res += delta * delta;
        }
        return res / 2.0;
    }
    /**
     * @endcond
     */
};

template <typename T> void normalize_vector(T *vector, int length) {
    T sum_squared = 0;
    for (int i = 0; i < length; i++) {
        T v = vector[i];
        sum_squared += v * v;
    }
    if (sum_squared <= 0) {
        return;
    }
    T scaler = 1.0 / std::sqrt(sum_squared);
    for (int i = 0; i < length; i++) {
        vector[i] *= scaler;
    }
}