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
concept same_as = std::is_same_v<T, U> && std::is_same_v<U, T>;
// clang-format on

constexpr size_t dynamic = -1;
template <size_t N> concept dynamic_v = N == dynamic;

template <class T> constexpr auto size_v = size(T{});

template <class T> constexpr auto shape_v = shape(T{});
template <class T> using shape_t = decltype(shape_v<T>);

// clang-format off
template <class D>
concept dimensions = requires() {
  { size_v<D> } -> size_t;
  { shape_v<D> } -> shape_t<D>;
};
// clang-format on

template <class T> constexpr auto rank_v = rank(T{});

template <class T>
constexpr auto dynamic_dimensions_v = dynamic_dimensions(T{});

template <size_t... Ns> struct dimension_list {};

template <size_t... Ns> constexpr auto size(dimension_list<Ns...>) -> size_t {
  return (dynamic_v<Ns> || ...) ? dynamic : (Ns * ...);
}

template <size_t... Ns>
constexpr auto shape(dimension_list<Ns...>)
    -> std::array<size_t, sizeof...(Ns)> {
  return {Ns...};
}

constexpr auto rank(dimensions auto d) -> size_t { return size(shape(d)); }

// TODO: no raw loops
template <dimensions D, class... Ds>
auto runtime_shape(Ds... ds)
    -> std::array<size_t, rank_v<D>> requires(sizeof...(Ds) ==
                                              dynamic_dimensions_v<D>) {
  auto static_shape = shape_v<D>;
  if constexpr (sizeof...(Ds) != 0) {
    const auto dynamic_dimensions = std::array{ds...};
    size_t j = 0;
    for (size_t i = 0; i < size(static_shape); ++i)
      if (static_shape[i] == dynamic) {
        static_shape[i] = dynamic_dimensions[j];
        ++j;
      }
  }
  return static_shape;
}

template <class T> struct value;
template <class T> using value_t = typename value<T>::type;

template <class T> struct reference { using type = value_t<T> &; };
template <class T> using reference_t = typename reference<T>::type;

template <class T> struct const_reference { using type = value_t<T> const &; };
template <class T> using const_reference_t = typename const_reference<T>::type;

// clang-format off
template <class I>
concept iterator_type =
  requires(I i) {
    { *i } -> typename std::iterator_traits<I>::reference;
    { ++i } -> I&;
  };
// clang-format on

template <iterator_type I> struct value<I> {
  using type = typename std::iterator_traits<I>::value_type;
};

// clang-format off
template <class R> concept range =
  requires(R r) {
    { begin(r) } -> iterator_type;
    { end(r) } -> iterator_type;
  };
// clang-format on

template <range R> struct value<R> { using type = typename R::value_type; };

// clang-format off
template <class F, class Result, class... Args> concept callable =
  requires(F f, Args&&... args) {
    { f(std::forward<Args>(args)...) } -> Result;
  };
// clang-format on

template <range R, class A, class E = const_reference_t<R>>
constexpr auto reduce(R const &range, A accumulator, callable<A, A, E> auto op)
    -> A {
  for (auto const &element : range)
    accumulator = op(std::move(accumulator), element);
  return accumulator;
}

constexpr auto dynamic_dimensions(dimensions auto d) -> size_t {
  return reduce(shape(d), 0, [](auto const &a, auto const &v) {
    return v == dynamic ? a + 1 : a;
  });
}

// clang-format off
template <class S> concept storage =
  requires(S s, S const cs, size_t index) {
    { s[index] } -> reference_t<S>;
    { cs[index] } -> const_reference_t<S>;
  };
// clang-format on

template <class T, size_t N> struct default_storage {
  using storage_type =
      std::conditional_t<dynamic_v<N>, std::vector<T>, std::array<T, N>>;

  explicit default_storage(size_t n) requires dynamic_v<N> : data_(n) {}
  default_storage() requires(!dynamic_v<N>) = default;

  auto operator[](size_t index) -> T & { return data_[index]; }
  auto operator[](size_t index) const -> T const & { return data_[index]; }

private:
  storage_type data_;
};

template <class T, size_t N> struct value<default_storage<T, N>> {
  using type = value_t<typename default_storage<T, N>::storage_type>;
};

// clang-format off
template <class L, class Shape = std::array<size_t, 3>> concept layout =
  requires(L const cl, Shape shape, Shape cartesian_index) {
    { L{shape} } -> L;
    { cl.linear_index(cartesian_index) } -> size_t;
  };
// clang-format on

template <size_t N> struct row_major {
  explicit row_major(std::array<size_t, N> const &shape) {
    stride_[size(shape) - 1] = 1;
    std::partial_sum(rbegin(shape), rend(shape) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(std::array<size_t, N> const &cartesian_index) const
      -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  std::array<size_t, N> stride_;
};

template <size_t N> struct column_major {
  explicit column_major(std::array<size_t, N> const &shape) {
    stride_[0] = 1;
    std::partial_sum(begin(shape), end(shape) - 1, begin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(std::array<size_t, N> const &cartesian_index) const
      -> size_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

private:
  std::array<size_t, N> stride_;
};

template <range R, class T = value_t<R>>
constexpr auto product(R const &range) -> T {
  return reduce(range, T{1}, std::multiplies{});
}

template <class T, dimensions D, storage S = default_storage<T, size_v<D>>,
          layout L = row_major<rank_v<D>>>
struct basic_tensor {
  basic_tensor() requires(!dynamic_v<size_v<D>>)
      : shape_{shape_v<D>}, layout_{shape_} {}

  template <class... Ds>
  explicit basic_tensor(Ds... ds) requires(dynamic_v<size_v<D>>)
      : shape_{runtime_shape<D>(std::forward<Ds>(ds)...)},
        storage_{product(shape_)}, layout_{shape_} {}

  constexpr auto shape() const
      -> std::array<size_t, rank_v<D>> requires(!dynamic_v<size_v<D>>) {
    return shape_v<D>;
  }

  auto shape() const
      -> std::array<size_t, rank_v<D>> requires(dynamic_v<size_v<D>>) {
    return std::is_constant_evaluated() ? shape_v<D> : shape_;
  }

  template <class... Is>
  friend auto index(basic_tensor &t, Is... is)
      -> T &requires(sizeof...(Is) == rank_v<D>) {
    const auto cartesian_index = std::array<size_t, sizeof...(Is)>{is...};
    return t.storage_[t.layout_.linear_index(cartesian_index)];
  }

  template <class... Is>
  friend auto index(basic_tensor const &t, Is... is)
      -> T const &requires(sizeof...(Is) == rank_v<D>) {
    const auto cartesian_index = std::array<size_t, sizeof...(Is)>{is...};
    return t.storage_[t.layout_.linear_index(cartesian_index)];
  }

private:
  std::array<size_t, rank_v<D>> shape_;
  S storage_;
  L layout_;
};

template <class T, dimensions D, storage S, layout L>
constexpr auto shape(basic_tensor<T, D, S, L> const &t)
    -> std::array<size_t, rank_v<D>> {
  return t.shape();
}

template <class T, size_t... Ns>
using tensor = basic_tensor<T, dimension_list<Ns...>>;

} // namespace v0

} // namespace sgl
