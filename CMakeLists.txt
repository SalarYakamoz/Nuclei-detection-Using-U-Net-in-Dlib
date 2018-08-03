cmake_minimum_required(VERSION 2.8.12)
project(first_project)

add_subdirectory(/home/salar/Downloads/dlib-19.15/dlib dlib_build)
  
add_executable(u-net u-net.cpp)
target_link_libraries(u-net dlib::dlib)

add_executable(u-net2 u-net2.cpp)
target_link_libraries(u-net2 dlib::dlib)

add_executable(u-net3 u-net3.cpp)
target_link_libraries(u-net3 dlib::dlib)



