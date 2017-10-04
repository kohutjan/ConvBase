cmake_minimum_required(VERSION 2.8.7)

project(ConvBase)

set(ConvBase_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(ConvBase_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${ConvBase_INCLUDE_DIR})
include_directories(${ConvBase_SRC_DIR})

add_executable(ConvBase
               ${ConvBase_SRC_DIR}/main.cpp)


