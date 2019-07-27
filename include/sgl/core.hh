#pragma once

#include <array>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

static constexpr size_t dynamic = -1;

template <size_t N>
struct is_dynamic
    : std::conditional_t<(N == dynamic), std::true_type, std::false_type> {};

template <size_t N> static constexpr bool is_dynamic_v = is_dynamic<N>::value;

template <class T> struct dimension_traits;

template <size_t... Ns> struct dimensions {};

template <size_t... Ns> struct dimension_traits<dimensions<Ns...>> {
  static constexpr size_t rank = sizeof...(Ns);
  static constexpr size_t size =
      (is_dynamic_v<Ns> || ...) ? dynamic : (Ns * ...);
  static constexpr std::array<size_t, rank> shape = {Ns...};
};

template <class T, size_t N, bool = is_dynamic_v<N>> struct storage;

template <class T, size_t N> struct storage<T, N, true> {
  using storage_type = std::vector<T>;

  storage(size_t n) : data_(n) {}

  auto operator[](size_t index) -> T & { return data_[index]; }
  auto operator[](size_t index) const -> const T & { return data_[index]; }

  friend auto size(const storage &s) -> size_t { return s.data_.size(); }

private:
  storage_type data_;
};

template <class T, size_t N> struct storage<T, N, false> {
  using storage_type = std::array<T, N>;

  auto operator[](size_t index) -> T & { return data_[index]; }
  auto operator[](size_t index) const -> const T & { return data_[index]; }

private:
  storage_type data_;
};

template <class T, size_t N>
constexpr auto size(const storage<T, N, false> &) -> size_t {
  return N;
}

} // namespace v0

} // namespace sgl
