#pragma once

#include <sgl/dimensions.hh>
#include <sgl/traits.hh>
#include <sgl/concepts.hh>

namespace sgl {

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

template <class T, size_t N> struct traits::value<default_storage<T, N>> {
  using type = value_t<typename default_storage<T, N>::storage_type>;
};

} // namespace sgl
