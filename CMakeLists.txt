cmake_minimum_required(VERSION 3.0)
project(joelnet)

find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})

find_package(xtensor-blas REQUIRED)
include_directories(${xtensor-blas_INCLUDE_DIRS})

add_definitions(-DHAVE_CBLAS=1)

if (WIN32)
  find_package(OpenBLAS REQUIRED)
  set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})


include_directories( ${joelnet_SOURCE_DIR}/include )

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory( tests )
endif()
