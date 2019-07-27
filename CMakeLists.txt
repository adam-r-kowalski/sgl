cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
project(sgl)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(${CMAKE_PROJECT_NAME} test/main.cc)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CONAN_LIBS})

target_compile_options(${CMAKE_PROJECT_NAME}
  PRIVATE
  -std=c++2a
  -Xclang
  -fconcepts-ts
  -Wall
  -Wextra
  -Wpedantic)
