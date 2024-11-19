project(scope_guard)

add_library(${PROJECT_NAME} INTERFACE include/scope_guard.hpp)
target_include_directories(${PROJECT_NAME} INTERFACE include)
