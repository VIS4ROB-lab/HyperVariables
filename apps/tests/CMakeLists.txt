# Define projects.
project(HyperVariablesTests VERSION 22.06.14 DESCRIPTION "" HOMEPAGE_URL https://github.com/VIS4ROB-lab/HyperVariables LANGUAGES CXX)

# Collect files and add executable.
get_filename_component(APPS_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(PARENT_DIR ${APPS_DIR} DIRECTORY)
file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS ${PARENT_DIR}/tests/*.hpp ${PARENT_DIR}/tests/*.cpp)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_FILES})

# Set properties.
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)

# Add includes and link libraries.
target_include_directories(${PROJECT_NAME} PRIVATE ${PARENT_DIR}/tests/include)
target_link_libraries(${PROJECT_NAME} Hyper::Variables)
