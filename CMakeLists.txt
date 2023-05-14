cmake_minimum_required(VERSION 3.13)
project(pybind_demo)
set(CMAKE_CXX_STANDARD 14)	
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2" CACHE STRING "Comment" FORCE)

add_definitions(${MYCOMPILE_FLAGS})

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(Filter Filter.cpp)
add_link_options("-mavx2")

target_compile_features(Filter PUBLIC cxx_std_14)
set_target_properties(Filter PROPERTIES SUFFIX ".so")