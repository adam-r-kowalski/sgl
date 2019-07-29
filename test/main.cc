#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <sgl/core.hh>

using namespace sgl;

TEST_CASE("dimension model the Dimensions concept") {
  static_assert(Dimensions<dimensions<3, 5, 7>>);
}

TEST_CASE("dimension shape model the Range concept") {
  static_assert(Range<decltype(shape_v<dimensions<3, 5, 7>>)>);
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

TEST_CASE("static dimension have a rank") {
  static_assert(rank_v<dimensions<3, 5, 7>> == 3);
}

TEST_CASE("dynamic dimensions have a rank") {
  static_assert(rank_v<dimensions<dynamic, 5, 7>> == 3);
}

TEST_CASE("multiple dimensions can be dynamic") {
  static_assert(rank_v<dimensions<dynamic, 5, dynamic>> == 3);
}

TEST_CASE("dynamic_dimensions_v computes the number of dynamic dimensions") {
  static_assert(dynamic_dimensions_v<dimensions<1, 2, 3>> == 0);
  static_assert(dynamic_dimensions_v<dimensions<dynamic, 2, 3>> == 1);
  static_assert(dynamic_dimensions_v<dimensions<dynamic, 2, dynamic>> == 2);
  static_assert(dynamic_dimensions_v<dimensions<dynamic, dynamic, dynamic>> == 3);
}

TEST_CASE("static dimension have a compile time size") {
  static_assert(size_v<dimensions<3, 5, 7>> == 3 * 5 * 7);
}

TEST_CASE("storage with known dimensions is array") {
  static_assert(std::is_same_v<storage<int, 3>::storage_type, std::array<int, 3>>);
}

TEST_CASE("storage with unknown dimensions is vector") {
  static_assert(std::is_same_v<storage<int, dynamic>::storage_type, std::vector<int>>);
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

TEST_CASE("row major maps cartesian to linear index") {
  auto l = row_major{std::array<size_t, 3>{2, 3, 4}};
  REQUIRE(l.linear_index({0, 0, 0}) == 0);
  REQUIRE(l.linear_index({0, 0, 1}) == 1);
  REQUIRE(l.linear_index({0, 0, 2}) == 2);
  REQUIRE(l.linear_index({0, 0, 3}) == 3);
  REQUIRE(l.linear_index({0, 1, 0}) == 4);
  REQUIRE(l.linear_index({0, 1, 1}) == 5);
  REQUIRE(l.linear_index({0, 1, 2}) == 6);
  REQUIRE(l.linear_index({0, 1, 3}) == 7);
  REQUIRE(l.linear_index({0, 2, 0}) == 8);
  REQUIRE(l.linear_index({0, 2, 1}) == 9);
  REQUIRE(l.linear_index({0, 2, 2}) == 10);
  REQUIRE(l.linear_index({0, 2, 3}) == 11);
  REQUIRE(l.linear_index({1, 0, 0}) == 12);
  REQUIRE(l.linear_index({1, 0, 1}) == 13);
  REQUIRE(l.linear_index({1, 0, 2}) == 14);
  REQUIRE(l.linear_index({1, 0, 3}) == 15);
  REQUIRE(l.linear_index({1, 1, 0}) == 16);
  REQUIRE(l.linear_index({1, 1, 1}) == 17);
  REQUIRE(l.linear_index({1, 1, 2}) == 18);
  REQUIRE(l.linear_index({1, 1, 3}) == 19);
  REQUIRE(l.linear_index({1, 2, 0}) == 20);
  REQUIRE(l.linear_index({1, 2, 1}) == 21);
  REQUIRE(l.linear_index({1, 2, 2}) == 22);
  REQUIRE(l.linear_index({1, 2, 3}) == 23);
}

TEST_CASE("column major maps cartesian to linear index") {
  auto l = column_major{std::array<size_t, 3>{2, 3, 4}};
  REQUIRE(l.linear_index({0, 0, 0}) == 0);
  REQUIRE(l.linear_index({1, 0, 0}) == 1);
  REQUIRE(l.linear_index({0, 1, 0}) == 2);
  REQUIRE(l.linear_index({1, 1, 0}) == 3);
  REQUIRE(l.linear_index({0, 2, 0}) == 4);
  REQUIRE(l.linear_index({1, 2, 0}) == 5);
  REQUIRE(l.linear_index({0, 0, 1}) == 6);
  REQUIRE(l.linear_index({1, 0, 1}) == 7);
  REQUIRE(l.linear_index({0, 1, 1}) == 8);
  REQUIRE(l.linear_index({1, 1, 1}) == 9);
  REQUIRE(l.linear_index({0, 2, 1}) == 10);
  REQUIRE(l.linear_index({1, 2, 1}) == 11);
  REQUIRE(l.linear_index({0, 0, 2}) == 12);
  REQUIRE(l.linear_index({1, 0, 2}) == 13);
  REQUIRE(l.linear_index({0, 1, 2}) == 14);
  REQUIRE(l.linear_index({1, 1, 2}) == 15);
  REQUIRE(l.linear_index({0, 2, 2}) == 16);
  REQUIRE(l.linear_index({1, 2, 2}) == 17);
  REQUIRE(l.linear_index({0, 0, 3}) == 18);
  REQUIRE(l.linear_index({1, 0, 3}) == 19);
  REQUIRE(l.linear_index({0, 1, 3}) == 20);
  REQUIRE(l.linear_index({1, 1, 3}) == 21);
  REQUIRE(l.linear_index({0, 2, 3}) == 22);
  REQUIRE(l.linear_index({1, 2, 3}) == 23);
}

TEST_CASE("tensors can be default constructed if they have known dimensions") {
  auto t = tensor<int, 2, 3>{};
  static_assert(shape(t) == std::array<size_t, 2>{2, 3});
}

TEST_CASE("ranges have a value type") {
	static_assert(Same<value_t<std::array<int, 3>>, int>);
}

TEST_CASE("ranges have a reference type") {
	static_assert(Same<reference_t<std::array<int, 3>>, int&>);
}

TEST_CASE("iterators have a value type") {
	static_assert(Same<value_t<std::array<int, 3>::iterator>, int>);
}

TEST_CASE("iterators have a reference type") {
	static_assert(Same<reference_t<std::array<int, 3>::iterator>, int&>);
}
