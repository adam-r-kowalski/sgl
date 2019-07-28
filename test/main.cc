#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <sgl/core.hh>

using namespace sgl;

TEST_CASE("dimension model the Dimensions concept") {
  static_assert(Dimensions<dimensions<3, 5, 7>>);
}

TEST_CASE("static dimension have a rank") {
  static_assert(rank_v<dimensions<3, 5, 7>> == 3);
}

TEST_CASE("dynamic dimensions have a rank") {
  static_assert(rank_v<dimensions<dynamic, 5, 7>> == 3);
}

TEST_CASE("multiple dimensions can be dynamic") {
  static_assert(rank_v<dimensions<dynamic, 5, dynamic>> == 3);
}

TEST_CASE("static dimension have a compile time size") {
  static_assert(size_v<dimensions<3, 5, 7>> == 3 * 5 * 7);
}

TEST_CASE("dynamic dimensions have a dynamic size") {
  static_assert(size_v<dimensions<dynamic, 5, 7>> == dynamic);
}

TEST_CASE("static dimension have a static shape") {
  static_assert(shape_v<dimensions<3, 5, 7>> == std::array<size_t, 3>{3, 5, 7});
}

TEST_CASE("dynamic dimension have a dynamic shape") {
  static_assert(shape_v<dimensions<dynamic, 5, 7>> == std::array<size_t, 3>{dynamic, 5, 7});
}

TEST_CASE("storage with known dimensions is array") {
  static_assert(std::is_same_v<storage<int, 3>::storage_type, std::array<int, 3>>);
}

TEST_CASE("storage with unknown dimensions is vector") {
  static_assert(std::is_same_v<storage<int, dynamic>::storage_type, std::vector<int>>);
}

TEST_CASE("storage with static size can be default constructed") {
  static_assert(size(storage<int, 5>{}) == 5);
}

TEST_CASE("storage with dynamic size can be created with a runtime capacity") {
  REQUIRE(size(storage<int, dynamic>{5}) == 5);
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
