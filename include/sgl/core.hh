#pragma once

#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

static constexpr size_t dynamic = -1;

template <size_t N> concept Dynamic = (N == dynamic);

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

template <size_t... Ns> struct dimensions {};

template <size_t... Ns> constexpr auto size(dimensions<Ns...>) -> size_t {
  return (Dynamic<Ns> || ...) ? dynamic : (Ns * ...);
}

template <size_t... Ns>
constexpr auto shape(dimensions<Ns...>) -> std::array<size_t, sizeof...(Ns)> {
  return {Ns...};
}

template <Dimensions D> constexpr auto rank(D d) -> size_t {
  return shape(d).size();
}

template <class T, size_t N> struct storage {
  using storage_type =
      std::conditional_t<Dynamic<N>, std::vector<T>, std::array<T, N>>;

  explicit storage(size_t n) requires Dynamic<N> : data_(n) {}
  storage() requires(!Dynamic<N>) = default;

  constexpr friend auto size(const storage &s) -> size_t {
    return s.data_.size();
  }

  auto operator[](size_t index) -> T & { return data_[index]; }
  auto operator[](size_t index) const -> const T & { return data_[index]; }

private:
  storage_type data_;
};

} // namespace v0

} // namespace sgl
