#pragma once

#include <array>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

// clang-format off
template <class T, class U>
concept Same = std::is_same_v<T, U> && std::is_same_v<U, T>;
// clang-format on

static constexpr size_t dynamic = -1;

template <size_t N> concept Dynamic = N == dynamic;

template <class T> constexpr size_t size_v = size(T{});

template <class T> constexpr auto shape_v = shape(T{});
template <class T> using shape_t = decltype(shape_v<T>);

// clang-format off
template <class D>
concept Dimensions = requires() {
  { size_v<D> } -> size_t;
  { shape_v<D> } -> shape_t<D>;
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

template <class T> struct value;
template <class T> using value_t = typename value<T>::type;

template <class T> struct reference { using type = value_t<T> &; };
template <class T> using reference_t = typename reference<T>::type;

template <class T> struct const_reference { using type = const value_t<T> &; };
template <class T> using const_reference_t = typename const_reference<T>::type;

// clang-format off
template <class I>
concept Iterator =
	requires(I i) {
	  { *i } -> typename std::iterator_traits<I>::reference;
		{ ++i } -> I&;
	};
// clang-format on

template <Iterator I> struct value<I> {
  using type = typename std::iterator_traits<I>::value_type;
};

// clang-format off
template <class R> concept Range =
	requires(R&& r) {
		{ begin(r) } -> Iterator;
		{ end(r) } -> Iterator;
	};
// clang-format on

template <Range R> struct value<R> { using type = typename R::value_type; };

// clang-format off
template <class F, class Result, class... Args> concept Callable =
  requires(F&& f, Args&&... args) {
		{ f(std::forward<Args>(args)...) } -> Result;
	};
// clang-format on

template <Range R, class A, class E = const_reference_t<R>>
constexpr auto reduce(R range, A accumulator, Callable<A, A, E> auto op) -> A {
  for (auto const &element : range)
    accumulator = op(std::move(accumulator), element);
  return accumulator;
}

constexpr auto dynamic_dimensions(Dimensions auto d) -> size_t {
  return reduce(shape(d), 0, [](const auto &a, const auto &v) {
    return v == dynamic ? a + 1 : a;
  });
}

// clang-format off
template <class S> concept Storage =
  requires(S s, const S cs, size_t index) {
    { s[index] } -> reference_t<S>;
    { cs[index] } -> const_reference_t<S>;
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

template <class T, size_t N> struct value<storage<T, N>> {
  using type = value_t<typename storage<T, N>::storage_type>;
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

  auto linear_index(const std::array<size_t, N> &cartesian_index) const
      -> size_t {
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

  auto linear_index(const std::array<size_t, N> &cartesian_index) const
      -> size_t {
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

  template <class... Ds>
  explicit basic_tensor(Ds... ds) requires(Dynamic<size_v<D>> &&
                                           (sizeof...(Ds) ==
                                            dynamic_dimensions_v<D>))
      : shape_{full_shape_(ds...)}, storage_{storage_size_()}, layout_{shape_} {
  }

  constexpr auto shape() const
      -> std::array<size_t, rank_v<D>> requires(!Dynamic<size_v<D>>) {
    return shape_v<D>;
  }

  auto shape() const
      -> std::array<size_t, rank_v<D>> requires(Dynamic<size_v<D>>) {
    return shape_;
  }

private:
  // TODO: no raw loops
  template <class... Ds>
  auto full_shape_(Ds... ds) const -> std::array<size_t, rank_v<D>> {
    auto static_dims = shape_v<D>;
    const auto dynamic_dims = std::array{ds...};
    size_t j = 0;
    for (size_t i = 0; i < size(static_dims); ++i)
      if (static_dims[i] == dynamic) {
        static_dims[i] = dynamic_dims[j];
        ++j;
      }

    return static_dims;
  }

  auto storage_size_() const -> size_t {
    return reduce(shape_, 1, std::multiplies{});
  }

  std::array<size_t, rank_v<D>> shape_;
  S storage_;
  L layout_;
};

template <class T, Dimensions D, Storage S, Layout L>
constexpr auto shape(const basic_tensor<T, D, S, L> &t)
    -> std::array<size_t, rank_v<D>> {
  return t.shape();
}

template <class T, size_t... Ns>
using tensor = basic_tensor<T, dimensions<Ns...>>;

} // namespace v0

} // namespace sgl
