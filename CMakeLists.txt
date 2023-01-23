# Check CMake version.
cmake_minimum_required(VERSION 3.16)

# Define projects.
project(HyperVariables VERSION 23.01.19 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Set properties.
if (PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED YES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wnon-virtual-dtor -pedantic")
    option(HYPER_BUILD_TESTS "Build tests." TRUE)
endif ()

# Define libraries and aliases.
file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/internal/*.cpp)
add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_FILES})
add_library(Hyper::Variables ALIAS ${PROJECT_NAME})

# Add compile definitions.
# target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_CERES=true)

# Find Eigen target if not present.
if (NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED NO_MODULE)
endif ()

# Find GTest target if not present.
if (NOT TARGET GTest::GTest)
    find_package(GTest REQUIRED)
endif ()

# Find GTest target if not present.
if (NOT TARGET glog::glog)
    find_package(glog REQUIRED)
endif ()

# Add includes and link libraries.
target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen PUBLIC glog::glog PUBLIC GTest::GTest)

# Build tests.
if (HYPER_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/apps/tests)
endif (HYPER_BUILD_TESTS)
