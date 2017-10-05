cmake_minimum_required(VERSION 2.8.7)

project(ConvBase)

set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(ConvBase_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(ConvBase_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${ConvBase_INCLUDE_DIR})
include_directories(${ConvBase_SRC_DIR})

add_executable(ConvBase
               ${ConvBase_SRC_DIR}/main.cpp
               ${ConvBase_SRC_DIR}/net.cpp
               ${ConvBase_SRC_DIR}/operators/convolution.cpp
               ${ConvBase_SRC_DIR}/operators/pooling.cpp
               ${ConvBase_SRC_DIR}/operators/relu.cpp
               ${ConvBase_SRC_DIR}/operators/reshape.cpp
               ${ConvBase_SRC_DIR}/operators/im2col.cpp)