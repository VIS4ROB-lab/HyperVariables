# Check CMake version.
cmake_minimum_required(VERSION 3.16)
enable_language(CXX)

# Define projects.
project(HyperVariables VERSION 22.06.14 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Define libraries and aliases.
add_library(${PROJECT_NAME} INTERFACE)
add_library(Hyper::Variables ALIAS ${PROJECT_NAME})

# Set properties.
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_compile_options(${PROJECT_NAME} INTERFACE -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_compile_options(${PROJECT_NAME} INTERFACE -O3 -march=native)

# Find Eigen target if not present.
if (NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_compile_definitions(${PROJECT_NAME} INTERFACE EIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")
endif ()

# Find gtest target if not present.
if (NOT TARGET GTest::gtest)
    find_package(GTest REQUIRED)
endif ()

# Find glog library.
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

# Add includes and link libraries.
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen glog GTest::gtest)

# Build tests.
option(HYPER_BUILD_TESTING "Build tests." TRUE)
if (HYPER_BUILD_TESTING)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif (HYPER_BUILD_TESTING)
