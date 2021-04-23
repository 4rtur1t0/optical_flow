cmake_minimum_required(VERSION 2.8)
find_package( OpenCV 3.4 REQUIRED )
LINK_LIBRARIES( ${OpenCV_LIBS} )

add_executable( of of.cpp )
