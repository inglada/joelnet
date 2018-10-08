add_executable(xor xor.cxx )
target_link_libraries(xor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_test(xor ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/xor)


