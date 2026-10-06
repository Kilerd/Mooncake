# Node-local GPU copy daemon for the nvlink_proxy transport. Links the CUDA
# runtime statically so the binary only needs the driver (libcuda.so.1) that
# the container runtime injects.
if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
endif()
enable_language(CUDA)

# SASS for the common data-center / workstation generations plus sm_80 PTX
# so newer GPUs can JIT the gather kernel. Override with
# -DNVLINK_PROXY_CUDA_ARCHITECTURES=...
if(NOT NVLINK_PROXY_CUDA_ARCHITECTURES)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
    set(NVLINK_PROXY_CUDA_ARCHITECTURES
        80-real 90-real 100-real 103-real 120-real 80-virtual)
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    set(NVLINK_PROXY_CUDA_ARCHITECTURES 80-real 90-real 100-real 120-real
                                        80-virtual)
  else()
    set(NVLINK_PROXY_CUDA_ARCHITECTURES 80-real 90-real 80-virtual)
  endif()
endif()

add_executable(mooncake_nvlink_proxy mooncake_nvlink_proxy.cpp gather_copy.cu)
set_target_properties(
  mooncake_nvlink_proxy
  PROPERTIES CUDA_ARCHITECTURES "${NVLINK_PROXY_CUDA_ARCHITECTURES}"
             CUDA_RUNTIME_LIBRARY Static)
target_include_directories(
  mooncake_nvlink_proxy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(mooncake_nvlink_proxy PRIVATE CUDA::cudart_static
                                                    pthread)
install(TARGETS mooncake_nvlink_proxy DESTINATION bin)
