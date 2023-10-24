cmake_minimum_required(VERSION 3.16)
project(hyper_variables VERSION 23.05.09 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(HYPER_BUILD_WITH_GLOBAL_LIE_GROUP_DERIVATIVES "Build with global Lie group derivatives." OFF)
option(HYPER_BUILD_WITH_CERES "Build with Ceres." OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/internal/*.cpp)
add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_FILES})
ament_export_targets(${PROJECT_NAME}_targets HAS_LIBRARY_TARGET)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_MATRIXBASE_PLUGIN="${PROJECT_SOURCE_DIR}/plugins/matrix_base_plugin.hpp")

if (HYPER_BUILD_WITH_GLOBAL_LIE_GROUP_DERIVATIVES)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES=1)
else ()
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES=0)
endif ()

if (HYPER_BUILD_WITH_CERES)
    find_package(Ceres REQUIRED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_CERES=1)
    target_link_libraries(${PROJECT_NAME} Ceres::ceres Eigen3::Eigen glog::glog)
    ament_export_dependencies(Ceres Eigen3 glog)
else ()
    target_compile_definitions(${PROJECT_NAME} PUBLIC HYPER_COMPILE_WITH_CERES=0)
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen glog::glog)
    ament_export_dependencies(Eigen3 glog)
endif ()

install(
        DIRECTORY include/
        DESTINATION include
)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

MACRO(SUBDIRLIST result dir)
    FILE(GLOB children RELATIVE ${dir} ${dir}/*)
    SET(dirlist "")
    FOREACH (child ${children})
        IF (IS_DIRECTORY ${dir}/${child})
            LIST(APPEND dirlist ${dir}/${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(SUBDIRS ${PROJECT_SOURCE_DIR}/apps)
FOREACH (subdir ${SUBDIRS})
    add_subdirectory(${subdir})
ENDFOREACH ()

if (BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    file(GLOB_RECURSE PROJECT_TEST_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/*.hpp ${PROJECT_SOURCE_DIR}/tests/*.cpp)

    if (PROJECT_TEST_FILES)
        ament_add_gtest(${PROJECT_NAME}_tests ${PROJECT_TEST_FILES})
        target_include_directories(${PROJECT_NAME}_tests PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests/include>
                $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})
    endif ()
endif ()

if (BUILD_BENCHMARKS)
    #find_package(... REQUIRED)
    file(GLOB_RECURSE PROJECT_BENCHMARK_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/benchmarks/*.hpp ${PROJECT_SOURCE_DIR}/benchmarks/*.cpp)

    if (PROJECT_BENCHMARK_FILES)
        add_executable(${PROJECT_NAME}_benchmarks ${PROJECT_BENCHMARK_FILES})
        target_include_directories(${PROJECT_NAME}_benchmarks PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/benchmarks/include>
                $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(${PROJECT_NAME}_benchmarks ${PROJECT_NAME})
    endif ()
endif ()

ament_package()

