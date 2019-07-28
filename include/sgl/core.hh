#pragma once

#include <array>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

static constexpr size_t dynamic = -1;

template <size_t N> concept Dynamic = N == dynamic;

template <class T> constexpr size_t size_v = size(T{});
template <class T> constexpr auto shape_v = shape(T{});

// clang-format off
template <class D>
concept Dimensions = requires() {
  { size_v<D> } -> size_t;
  { shape_v<D> };
};
// clang-format on

template <class T> constexpr size_t rank_v = rank(T{});
template <class T>
constexpr size_t dynamic_dimensions_v = dynamic_dimensions(T{});

template <size_t... Ns> struct dimensions {};

template <size_t... Ns> constexpr auto size(dimensions<Ns...>) -> size_t {
  return (Dynamic<Ns> || ...) ? dynamic : (Ns * ...);
}

template <size_t... Ns>
constexpr auto shape(dimensions<Ns...>) -> std::array<size_t, sizeof...(Ns)> {
  return {Ns...};
}

constexpr auto rank(Dimensions auto d) -> size_t { return size(shape(d)); }

constexpr auto dynamic_dimensions(Dimensions auto d) -> size_t {
  size_t count = 0;
  for (auto &s : shape(d))
    if (s == dynamic)
      ++count;
  return count;
}

// clang-format off
template <class S> concept Storage = requires(S s, size_t index) {
  { s[index] };
};
// clang-format on

template <class T, size_t N> struct storage {
  using storage_type =
      std::conditional_t<Dynamic<N>, std::vector<T>, std::array<T, N>>;

  explicit storage(size_t n) requires Dynamic<N> : data_(n) {}
  storage() requires(!Dynamic<N>) = default;

  auto operator[](size_t index) -> T & { return data_[index]; }
  auto operator[](size_t index) const -> const T & { return data_[index]; }

private:
  storage_type data_;
};

// clang-format off
template <class L> concept Layout = true;
// clang-format on

template <size_t N> struct row_major {
  explicit row_major(const std::array<size_t, N> &shape) {
    stride_[size(shape) - 1] = 1;
    std::partial_sum(rbegin(shape), rend(shape) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const std::array<size_t, N> &cartesian_index) -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  std::array<size_t, N> stride_;
};

template <size_t N> struct column_major {
  explicit column_major(const std::array<size_t, N> &shape) {
    stride_[0] = 1;
    std::partial_sum(begin(shape), end(shape) - 1, begin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const std::array<size_t, N> &cartesian_index) -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  std::array<size_t, N> stride_;
};

template <class T, Dimensions D, Storage S = storage<T, size_v<D>>,
          Layout L = row_major<rank_v<D>>>
struct basic_tensor {
  basic_tensor() requires(!Dynamic<size_v<D>>)
      : shape_{shape_v<D>}, layout_{shape_} {}

  friend constexpr auto shape(const basic_tensor &)
      -> std::array<size_t, rank_v<D>> requires(!Dynamic<size_v<D>>) {
    return shape_v<D>;
  }

private:
  std::array<size_t, rank_v<D>> shape_;
  S storage_;
  L layout_;
};

template <class T, size_t... Ns>
using tensor = basic_tensor<T, dimensions<Ns...>>;

} // namespace v0

} // namespace sgl
