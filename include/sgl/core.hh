#pragma once

#include <array>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include <sgl/dimensions.hh>
#include <sgl/storage.hh>
#include <sgl/layout.hh>
#include <sgl/traits.hh>
#include <sgl/concepts.hh>

namespace sgl {

template <class T, dimensions D, storage S = default_storage<T, size_v<D>>,
          layout L = row_major<rank_v<D>>>
struct basic_tensor {
  basic_tensor() requires(!dynamic_v<size_v<D>>)
      : shape_{shape_v<D>}, storage_{}, layout_{shape_} {}

  template <class... Ds>
  explicit basic_tensor(Ds... ds) requires(dynamic_v<size_v<D>>)
      : shape_{runtime_shape<D>(std::forward<Ds>(ds)...)},
        storage_{product(shape_)}, layout_{shape_} {}

  friend constexpr auto shape(basic_tensor const &t) -> std::array<size_t, rank_v<D>> {
    return std::is_constant_evaluated() ? shape_v<D> : t.shape_;
  }

  friend auto index(basic_tensor &t,
                    std::array<size_t, rank_v<D>> const &cartesian_index)
      -> T & {
    return t.storage_[t.layout_.linear_index(cartesian_index)];
  }

  friend auto index(basic_tensor const &t,
                    std::array<size_t, rank_v<D>> const &cartesian_index)
      -> const T & {
    return t.storage_[t.layout_.linear_index(cartesian_index)];
  }

private:
  std::array<size_t, rank_v<D>> shape_;
  S storage_;
  L layout_;
};

template <class T, size_t... Ns>
using cpu_tensor = basic_tensor<T, dimension_list<Ns...>>;

// clang-format off
template <class T>
concept tensor = true;
// clang-format on

} // namespace sgl
