cmake_minimum_required(VERSION 3.12)
project(ZZKG_OpenCL)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

configure_file(calculateLCS.cl calculateLCS.cl COPYONLY)
add_executable(ZZKG_OpenCL main.cpp calculateLCS.cl)

target_link_libraries(ZZKG_OpenCL OpenCL::OpenCL)
