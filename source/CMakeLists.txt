project(M4Image)
set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} STATIC "")

target_sources(${PROJECT_NAME}
	PRIVATE
		M4Image.cpp
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/../include/M4Image.h
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/pixman-1/pixman)
target_link_libraries(${PROJECT_NAME} scope_guard mango pixman-1_static)
