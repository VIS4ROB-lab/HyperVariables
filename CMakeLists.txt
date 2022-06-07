set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++2a -O3 -march=native -fconcepts -fopenmp)
add_compile_options(-Wall -Wextra -Werror -Wnon-virtual-dtor -pedantic)

# Setup.
cmake_minimum_required(VERSION 3.16)
project(HyperVariables VERSION 22.06.14 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Includes.
include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)

# Fetch (external) content.
FetchContent_Declare(gtest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.11.0)
FetchContent_Declare(glog GIT_REPOSITORY https://github.com/google/glog.git GIT_TAG v0.6.0)

# Make (external) content available.
set(WITH_GTEST FALSE)
set(WITH_UNWIND FALSE)
set(BUILD_TESTING FALSE)
FetchContent_MakeAvailable(gtest glog)
set(HAVE_LIB_GTEST TRUE)
set(HAVE_LIB_GMOCK TRUE)

# Find Eigen.
if (NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED NO_MODULE)
    add_definitions(-DEIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")
endif ()

# Add library and alias.
add_library(${PROJECT_NAME} INTERFACE)
add_library(Hyper::Variables ALIAS ${PROJECT_NAME})

# Add features and includes, and link libraries.
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen glog::glog GTest::gtest)

# Build tests.
option(HYPER_BUILD_TESTING "Build tests." TRUE)
if (HYPER_BUILD_TESTING)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif (HYPER_BUILD_TESTING)
