#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <sgl/core.hh>

using namespace sgl;

/*
TEST_CASE("dimension model the Dimensions concept") {
  static_assert(Dimensions<dimensions<3, 5, 7>>);
}
*/

TEST_CASE("dimension have a rank") {
  using traits = dimension_traits<dimensions<3, 5, 7>>;
  static_assert(traits::rank == 3);
}

TEST_CASE("dynamic dimensions have a rank") {
  using traits = dimension_traits<dimensions<dynamic, 5, 7>>;
  static_assert(traits::rank == 3);
}

TEST_CASE("multiple dimensions can be dynamic") {
  using traits = dimension_traits<dimensions<dynamic, 5, dynamic>>;
  static_assert(traits::rank == 3);
}

TEST_CASE("dimension have a size") {
  using traits = dimension_traits<dimensions<3, 5, 7>>;
  static_assert(traits::size == 3 * 5 * 7);
}

TEST_CASE("dynamic dimensions have a dynamic size") {
  using traits = dimension_traits<dimensions<dynamic, 5, 7>>;
  static_assert(traits::size == dynamic);
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

TEST_CASE("storage with dynamic size can be indexed") {
  auto s = storage<int, dynamic>{5};

  REQUIRE(s[0] == 0);
  REQUIRE(s[1] == 0);
  REQUIRE(s[2] == 0);
  REQUIRE(s[3] == 0);
  REQUIRE(s[4] == 0);

  s[0] = 0;
  s[1] = 1;
  s[2] = 2;
  s[3] = 3;
  s[4] = 4;

  REQUIRE(s[0] == 0);
  REQUIRE(s[1] == 1);
  REQUIRE(s[2] == 2);
  REQUIRE(s[3] == 3);
  REQUIRE(s[4] == 4);
}

TEST_CASE("storage with static size must be default constructed") {
  auto s = storage<int, 5>{};
  static_assert(size(s) == 5);
}

TEST_CASE("storage with static size can be indexed") {
  auto s = storage<int, 5>{};

  REQUIRE(s[0] == 0);
  REQUIRE(s[1] == 0);
  REQUIRE(s[2] == 0);
  REQUIRE(s[3] == 0);
  REQUIRE(s[4] == 0);

  s[0] = 0;
  s[1] = 1;
  s[2] = 2;
  s[3] = 3;
  s[4] = 4;

  REQUIRE(s[0] == 0);
  REQUIRE(s[1] == 1);
  REQUIRE(s[2] == 2);
  REQUIRE(s[3] == 3);
  REQUIRE(s[4] == 4);
}
