# Node-local GPU copy daemon for the nvlink_proxy transport. Links the CUDA
# runtime statically so the binary only needs the driver (libcuda.so.1) that
# the container runtime injects.
add_executable(mooncake_nvlink_proxy mooncake_nvlink_proxy.cpp)
target_include_directories(
  mooncake_nvlink_proxy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(mooncake_nvlink_proxy PRIVATE CUDA::cudart_static
                                                    pthread)
install(TARGETS mooncake_nvlink_proxy DESTINATION bin)
