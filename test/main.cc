#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <sgl/core.hh>

using namespace sgl;

TEST_CASE("dimension model the Dimensions concept") {
  static_assert(Dimensions<dimensions<3, 5, 7>>);
}

TEST_CASE("dimension have a rank") {
  using traits = dimension_traits<dimensions<3, 5, 7>>;
  static_assert(traits::rank == 3);
}

TEST_CASE("dimension have a size") {
  using traits = dimension_traits<dimensions<3, 5, 7>>;
  static_assert(traits::size == 3 * 5 * 7);
}

TEST_CASE("dimension have a shape") {
  using traits = dimension_traits<dimensions<3, 5, 7>>;
  static_assert(traits::shape == std::array<size_t, 3>{3, 5, 7});
}

TEST_CASE("storage with known dimensions is array") {
  using storage_type = storage<int, 3>::storage_type;
  static_assert(std::is_same_v<storage_type, std::array<int, 3>>);
}

TEST_CASE("storage with unknown dimensions is vector") {
  using storage_type = storage<int, dynamic>::storage_type;
  static_assert(std::is_same_v<storage_type, std::vector<int>>);
}

TEST_CASE("storage with dynamic size can be created with a runtime capacity") {
  auto s = storage<int, dynamic>{5};
  REQUIRE(size(s) == 5);
}

TEST_CASE("storage with static size must be default constructed") {
  auto s = storage<int, 5>{};
  REQUIRE(size(s) == 5);
  static_assert(size(s) == 5);
}
