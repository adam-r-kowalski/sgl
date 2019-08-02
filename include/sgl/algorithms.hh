#pragma once

namespace sgl {

template <range R, class A, class E = const_reference_t<R>>
constexpr auto fold(R const &range, A accumulator, function<A, A, E> auto op)
    -> A {
  for (auto const &element : range)
    accumulator = op(std::move(accumulator), element);
  return accumulator;
}

template <range R, class T = value_t<R>>
constexpr auto product(R const &range) -> T {
  return fold(range, T{1}, std::multiplies{});
}

} 
