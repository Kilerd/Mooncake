file(GLOB NVLINK_PROXY_SOURCES "*.cpp")

add_library(nvlink_proxy_transport OBJECT ${NVLINK_PROXY_SOURCES})
target_link_libraries(nvlink_proxy_transport PRIVATE glog::glog)
target_include_directories(nvlink_proxy_transport PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
