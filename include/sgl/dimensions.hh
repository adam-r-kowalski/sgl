#pragma once

#include <array>
#include <iterator>

#include <sgl/traits.hh>
#include <sgl/concepts.hh>
#include <sgl/algorithms.hh>

namespace sgl {

template <size_t... Ns> struct dimension_list {};

template <size_t... Ns> struct traits::shape<dimension_list<Ns...>> {
  using type = std::array<size_t, sizeof...(Ns)>;
  static constexpr type value = {Ns...};
};

template <dimensions D>
constexpr auto dynamic_dimensions() -> size_t {
  return fold(shape_v<D>, 0, [](auto const &a, auto const &v) {
    return v == dynamic ? a + 1 : a;
  });
}

template <dimensions D>
constexpr size_t dynamic_dimensions_v = dynamic_dimensions<D>();

template <dimensions D> struct traits::size<D> {
  static constexpr size_t value = dynamic_dimensions_v<D> > 0 ?
    dynamic :
    product(shape_v<D>);
};

template <dimensions D> struct traits::rank<D> {
  static constexpr size_t value = std::size(shape_v<D>);
};

// TODO: no raw loops
template <dimensions D, class... Is>
auto runtime_shape(Is... is) 
    -> std::array<size_t, rank_v<D>> requires(sizeof...(Is) == dynamic_dimensions<D>()) {
  auto static_shape = shape_v<D>;
  auto runtime_shape = std::array<size_t, sizeof...(Is)>{static_cast<size_t>(is)...};
  size_t j = 0;
  for (size_t i = 0; i < size(static_shape); ++i)
    if (static_shape[i] == dynamic) {
      static_shape[i] = runtime_shape[j];
      ++j;
    }
  return static_shape;
}

// TODO: bounds check tensor index if in constexpr context
template <dimensions D>
constexpr auto in_bounds(std::array<size_t, rank_v<D>> const & cartesian_index) -> bool {
  auto inside = std::array<bool, rank_v<D>>{};
  std::transform(begin(cartesian_index), end(cartesian_index),
                 begin(shape_v<D>), begin(inside),
                 [](auto i, auto s) { return i < s; });
  return std::all_of(begin(inside), end(inside), [](auto b) { return b; });
}

} // namespace sgl
