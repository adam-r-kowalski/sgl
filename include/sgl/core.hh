#pragma once

#include <array>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

static constexpr size_t dynamic = -1;

template <size_t N> concept Dynamic = (N == dynamic);

template <class T> struct dimension_traits;

template <size_t... Ns> struct dimensions {};

template <size_t... Ns> struct dimension_traits<dimensions<Ns...>> {
  static constexpr size_t rank = sizeof...(Ns);
  static constexpr size_t size = (Dynamic<Ns> || ...) ? dynamic : (Ns * ...);
  static constexpr std::array<size_t, rank> shape = {Ns...};
};

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
