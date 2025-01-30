cmake_minimum_required(VERSION 3.28)
project(PDproject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(EIGEN3_INCLUDE_DIR "/home/cyberbazana/eigen")
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(PDproject main.cpp
        numpy_funcs.cpp
        numpy_funcs.h
        distributions.cpp
        distributions.h
        moments_impl.cpp
        moments_impl.h
        spacial_moments.cpp
        spacial_moments.h
        RK4.cpp
        RK4.h
)
