# Setup.
cmake_minimum_required(VERSION 3.16)
project(HyperVariablesTests VERSION 22.06.14 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Collect files.
set(SOURCE_FILES "${PROJECT_NAME}_SOURCE_FILES")
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/plugins/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add executable.
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Add features and includes, and link libraries.
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} Hyper::Variables)
