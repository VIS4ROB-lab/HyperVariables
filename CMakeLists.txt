cmake_minimum_required(VERSION 3.16)
project(hyper_variables VERSION 23.05.09 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

if (PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED YES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wnon-virtual-dtor -pedantic")
endif ()

option(HYPER_BUILD_WITH_GLOBAL_LIE_GROUP_DERIVATIVES "Build with global Lie group derivatives." FALSE)
option(HYPER_BUILD_WITH_CERES "Build with Ceres." FALSE)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/internal/*.cpp)
add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_FILES})
ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (HYPER_BUILD_WITH_GLOBAL_LIE_GROUP_DERIVATIVES)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES)
endif ()

if (HYPER_BUILD_WITH_CERES)
    find_package(Ceres REQUIRED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_CERES EIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")
    target_link_libraries(${PROJECT_NAME} Ceres::ceres Eigen3::Eigen glog::glog)
    ament_export_dependencies(Ceres Eigen3 glog)
else ()
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen glog::glog)
    ament_export_dependencies(Eigen3 glog)
endif ()

install(
        DIRECTORY include/
        DESTINATION include
)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

if (BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    file(GLOB_RECURSE PROJECT_TEST_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/*.hpp ${PROJECT_SOURCE_DIR}/tests/*.cpp)
    ament_add_gtest(${PROJECT_NAME}_tests ${PROJECT_TEST_FILES})
    target_include_directories(${PROJECT_NAME}_tests PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests/include>
            $<INSTALL_INTERFACE:include>
            )
    target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})
endif ()

ament_package()

