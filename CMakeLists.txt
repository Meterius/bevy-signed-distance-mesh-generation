cmake_minimum_required(VERSION 3.16)
project(bevy_signed_distance_mesh_generation CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(cuda/includes)
include_directories(cuda/kernels)

add_executable(
    bevy_signed_distance_mesh_generation
    cuda/includes/signed_distance.cu
    cuda/includes/ray_marching.cu
    cuda/modules/compute_render.cu
    cuda/includes/types.cu
        cuda/includes/utils.cu
        cuda/includes/color.cu
        cuda/modules/compute_mesh_generation.cu
        cuda/includes/bindings.h
        cuda/modules/common.cu
)

set_target_properties(bevy_signed_distance_mesh_generation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bevy_signed_distance_mesh_generation PROPERTIES CUDA_ARCHITECTURES OFF)
