add_executable(joelnet joelnet.cxx )
target_link_libraries(joelnet ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_test(joelnet ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/joelnet)


