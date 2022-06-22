# Define projects.
project(HyperVariablesTests VERSION 22.06.14 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Collect files and add executable.
set(SOURCE_FILES "${PROJECT_NAME}_SOURCE_FILES")
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/plugins/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set properties.
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)

# Add includes and link libraries.
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} Hyper::Variables)
