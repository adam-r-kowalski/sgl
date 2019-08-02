#pragma once

#include <array>
#include <numeric>
#include <iterator>

#include <sgl/traits.hh>
#include <sgl/concepts.hh>

namespace sgl {

template <size_t N> struct row_major {
  using shape_t = std::array<size_t, N>;

  explicit row_major(shape_t const &shape) {
    stride_[size(shape) - 1] = 1;
    std::partial_sum(rbegin(shape), rend(shape) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(shape_t const &cartesian_index) const -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  shape_t stride_;
};

template <size_t N> struct traits::shape<row_major<N>> {
  using type = typename row_major<N>::shape_t;
};

template <size_t N> struct column_major {
  using shape_t = std::array<size_t, N>;

  explicit column_major(shape_t const &shape) {
    stride_[0] = 1;
    std::partial_sum(begin(shape), end(shape) - 1, begin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(shape_t const &cartesian_index) const -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  shape_t stride_;
};

template <size_t N> struct traits::shape<column_major<N>> {
  using type = typename column_major<N>::shape_t;
};

} // namespace sgl
