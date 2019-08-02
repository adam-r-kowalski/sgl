#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <sgl/core.hh>

using namespace sgl;

TEST_CASE("dimension model the Dimensions concept") {
  static_assert(dimensions<dimension_list<3, 5, 7>>);
}

TEST_CASE("dimension shape model the Range concept") {
  static_assert(range<decltype(shape_v<dimension_list<3, 5, 7>>)>);
}

TEST_CASE("dynamic dimensions have a dynamic size") {
  static_assert(size_v<dimension_list<dynamic, 5, 7>> == dynamic);
}

TEST_CASE("static dimension have a static shape") {
  static_assert(shape_v<dimension_list<3, 5, 7>> == std::array<size_t, 3>{3, 5, 7});
}

TEST_CASE("dynamic dimension have a dynamic shape") {
  static_assert(shape_v<dimension_list<dynamic, 5, 7>> == std::array<size_t, 3>{dynamic, 5, 7});
}

TEST_CASE("static dimension have a rank") {
  static_assert(rank_v<dimension_list<3, 5, 7>> == 3);
}

TEST_CASE("dynamic dimensions have a rank") {
  static_assert(rank_v<dimension_list<dynamic, 5, 7>> == 3);
}

TEST_CASE("multiple dimensions can be dynamic") {
  static_assert(rank_v<dimension_list<dynamic, 5, dynamic>> == 3);
}

TEST_CASE("dynamic_dimensions_v computes the number of dynamic dimensions") {
  static_assert(dynamic_dimensions_v<dimension_list<1, 2, 3>> == 0);
  static_assert(dynamic_dimensions_v<dimension_list<dynamic, 2, 3>> == 1);
  static_assert(dynamic_dimensions_v<dimension_list<dynamic, 2, dynamic>> == 2);
  static_assert(dynamic_dimensions_v<dimension_list<dynamic, dynamic, dynamic>> == 3);
}

TEST_CASE("static dimension have a compile time size") {
  static_assert(size_v<dimension_list<3, 5, 7>> == 3 * 5 * 7);
}

TEST_CASE("storage with known dimensions is array") {
  static_assert(same_as<default_storage<size_t, 3>::storage_type, std::array<size_t, 3>>);
}

TEST_CASE("storage with unknown dimensions is vector") {
  static_assert(same_as<default_storage<int, dynamic>::storage_type, std::vector<int>>);
}

TEST_CASE("storage with dynamic size can be indexed") {
  auto s = default_storage<int, dynamic>{5};

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
  auto s = default_storage<int, 5>{};

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

TEST_CASE("storage has a value type") {
  static_assert(same_as<value_t<default_storage<int, 3>>, int>);
  static_assert(same_as<value_t<default_storage<int, dynamic>>, int>);
}

TEST_CASE("storage has a reference type") {
	static_assert(same_as<reference_t<default_storage<int, 3>>, int&>);
	static_assert(same_as<reference_t<default_storage<int, dynamic>>, int&>);
}

TEST_CASE("storage has a const reference type") {
	static_assert(same_as<const_reference_t<default_storage<int, 3>>, const int&>);
	static_assert(same_as<const_reference_t<default_storage<int, dynamic>>, const int&>);
}


TEST_CASE("row major maps cartesian to linear index") {
  const auto l = row_major{std::array<size_t, 3>{2, 3, 4}};
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
  const auto l = column_major{std::array<size_t, 3>{2, 3, 4}};
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

TEST_CASE("cpu_tensors can be default constructed if they have known dimensions") {
  const auto t = cpu_tensor<int, 3, 5, 7>{};
  static_assert(shape(t) == std::array<size_t, 3>{3, 5, 7});
}

TEST_CASE("runtime shape fills in missing dimensions at runtime") {
  REQUIRE(runtime_shape<dimension_list<3, 5, 7>>() == std::array<size_t, 3>{3, 5, 7});
  REQUIRE(runtime_shape<dimension_list<dynamic, 5, 7>>(3) == std::array<size_t, 3>{3, 5, 7});
  REQUIRE(runtime_shape<dimension_list<dynamic, 5, dynamic>>(3, 7) == std::array<size_t, 3>{3, 5, 7});
  REQUIRE(runtime_shape<dimension_list<dynamic, dynamic, dynamic>>(3, 5, 7) == std::array<size_t, 3>{3, 5, 7});
}

TEST_CASE("cpu_tensors can be constructed by providing dynamic dimensions") {
  const auto t1 = cpu_tensor<int, dynamic, 3, 4>{2};
  const auto t2 = cpu_tensor<int, 2, dynamic, 4>{3};
  const auto t3 = cpu_tensor<int, 2, 3, dynamic>{4};
  const auto t4 = cpu_tensor<int, dynamic, dynamic, 4>{2, 3};
  const auto t5 = cpu_tensor<int, dynamic, 3, dynamic>{2, 4};
  const auto t6 = cpu_tensor<int, 2, dynamic, dynamic>{3, 4};
  const auto t7 = cpu_tensor<int, dynamic, dynamic, dynamic>{2, 3, 4};
  REQUIRE(shape(t1) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t2) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t3) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t4) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t5) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t6) == std::array<size_t, 3>{2, 3, 4});
  REQUIRE(shape(t7) == std::array<size_t, 3>{2, 3, 4});
}


TEST_CASE("cpu_tensors can be indexed with an array") {
  auto t = cpu_tensor<int, 1, 2, 3>{};

  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 0}) == 0);
  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 1}) == 0);
  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 2}) == 0);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 0}) == 0);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 1}) == 0);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 2}) == 0);

  index(t, std::array<size_t, 3>{0, 0, 0}) = 1;
  index(t, std::array<size_t, 3>{0, 0, 1}) = 2;
  index(t, std::array<size_t, 3>{0, 0, 2}) = 3;
  index(t, std::array<size_t, 3>{0, 1, 0}) = 4;
  index(t, std::array<size_t, 3>{0, 1, 1}) = 5;
  index(t, std::array<size_t, 3>{0, 1, 2}) = 6;

  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 0}) == 1);
  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 1}) == 2);
  REQUIRE(index(t, std::array<size_t, 3>{0, 0, 2}) == 3);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 0}) == 4);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 1}) == 5);
  REQUIRE(index(t, std::array<size_t, 3>{0, 1, 2}) == 6);
}

TEST_CASE("cpu_tensors model the tensor concept") {
  static_assert(tensor<cpu_tensor<int, 1, 2, 3>>);
}

TEST_CASE("ranges have a value type") {
  static_assert(same_as<value_t<std::array<int, 3>>, int>);
}

TEST_CASE("ranges have a reference type") {
  static_assert(same_as<reference_t<std::array<int, 3>>, int&>);
}

TEST_CASE("ranges have a const reference type") {
  static_assert(same_as<const_reference_t<std::array<int, 3>>, const int&>);
}

TEST_CASE("iterators have a value type") {
  static_assert(same_as<value_t<std::array<int, 3>::iterator>, int>);
}

TEST_CASE("iterators have a reference type") {
  static_assert(same_as<reference_t<std::array<int, 3>::iterator>, int&>);
}

TEST_CASE("iterators have a const reference type") {
  static_assert(same_as<const_reference_t<std::array<int, 3>::iterator>, const int&>);
}

