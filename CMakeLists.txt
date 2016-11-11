cmake_minimum_required(VERSION 2.8)

include(cmake/base.cmake)
include(cmake/boost.cmake)
include(cmake/eigen.cmake)

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

#########################
# External dependencies
if(PKG_CONFIG_FOUND)
  search_for_eigen()
else()
  find_path(Eigen_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES eigen3 eigen
  )
  include_directories(SYSTEM ${Eigen_INCLUDE_DIR})
endif()

#########################

find_package(GUROBI)
if(NOT GUROBI_FOUND)
  message("Unable to find Gurobi. Make sure that the env GUROBI_HOME is set")
else()
  include_directories(${GUROBI_INCLUDE_DIRS})
endif()

add_definitions("-DEIGEN_GUROBI_EXPORT")

add_subdirectory(src)
add_subdirectory(tests)

# Add dependency towards the library in the pkg-config file.
pkg_config_append_libs(${PROJECT_NAME})

setup_project_finalize()
