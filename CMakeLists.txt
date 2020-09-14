cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ir)

set(ENV{Torch_DIR} /home/workspace/libtorch)

find_package( OpenCV REQUIRED )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)


file(GLOB project_SRCS src/*.cpp)

add_executable(ir ${project_SRCS})

target_link_libraries(ir "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ir PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
