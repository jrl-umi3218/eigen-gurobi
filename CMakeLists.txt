cmake_minimum_required(VERSION 2.8)

include(cmake/base.cmake)
include(cmake/boost.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

set(PROJECT_NAME eigen-gurobi)
set(PROJECT_DESCRIPTION "Gurobi QP solver through Eigen3 library.")
set(PROJECT_URL "https://github.com/haudren/eigen-gurobi")

project(${PROJECT_NAME} CXX)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(DOXYGEN_USE_MATHJAX "YES")

setup_project()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

find_package(GUROBI REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${GUROBI_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})

add_definitions("-DEIGEN_GUROBI_EXPORT")

add_subdirectory(src)
add_subdirectory(tests)

setup_project_finalize()
