cmake_minimum_required(VERSION 3.20)
project(surfel_vulkan_minimal LANGUAGES CXX)
find_package(Vulkan REQUIRED COMPONENTS glslc)

add_executable(surfel_vulkan main.cpp)
target_compile_features(surfel_vulkan PRIVATE cxx_std_17)
target_link_libraries(surfel_vulkan PRIVATE Vulkan::Vulkan)
target_compile_definitions(surfel_vulkan PRIVATE SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
if(MSVC)
  target_compile_options(surfel_vulkan PRIVATE /W4 /utf-8)
endif()

# GLSL -> SPIR-V；程序运行时读取这些二进制着色器。
foreach(name integrate resolve)
  set(spv "${CMAKE_CURRENT_BINARY_DIR}/shaders/${name}.spv")
  add_custom_command(OUTPUT "${spv}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.comp" -o "${spv}"
    DEPENDS shaders/${name}.comp shaders/common.glsl VERBATIM)
  list(APPEND shader_outputs "${spv}")
endforeach()
add_custom_target(shaders ALL DEPENDS ${shader_outputs})
add_dependencies(surfel_vulkan shaders)
