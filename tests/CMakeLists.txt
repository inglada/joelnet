add_executable(xor xor.cxx )
target_link_libraries(xor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_test(xor ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/xor)

add_executable(fizzbuzz fizzbuzz.cxx )
target_link_libraries(fizzbuzz ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_test(fizzbuzz ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fizzbuzz)


