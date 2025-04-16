cmake_minimum_required(VERSION 3.16)
project(bevy_signed_distance_mesh_generation CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(cuda/includes)
include_directories(cuda/kernels)

add_executable(
    bevy_signed_distance_mesh_generation
    cuda/modules/compute_render.cu
    cuda/modules/compute_mesh_generation.cu
)

set_target_properties(bevy_signed_distance_mesh_generation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bevy_signed_distance_mesh_generation PROPERTIES CUDA_ARCHITECTURES OFF)
