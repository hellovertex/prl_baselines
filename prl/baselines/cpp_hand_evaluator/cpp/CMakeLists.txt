cmake_minimum_required(VERSION 3.16)
project(HandEvaluator)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD 20)  # random sample without replacement not available in c++14
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /Zc:__cplusplus")
add_subdirectory(ext/pybind11)
pybind11_add_module(hand_eval run_monte_carlo.cpp)