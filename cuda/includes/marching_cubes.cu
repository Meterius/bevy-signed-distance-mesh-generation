#pragma once

#include "./utils.cu"
#include "./marching_cubes_constants.cu"

using namespace glm;

struct McCube {
    vec3 vertices[8];
    float values[8];
};

__device__ vec3 edge_vertex(const McCube &cube, const int i0, const int i1) {
    float v = cube.values[i0] / (cube.values[i1] - cube.values[i0]);
    return mix(cube.vertices[i0], cube.vertices[i1], 0.5f);
}

__device__ void write_triangle(const McCube &cube, const int edge_indices[3], Point *const triangle) {
    for (int i = 0; i < 3; i++) {
        triangle[i] = to_point(
            edge_vertex(cube, MC_EDGE_TABLE[edge_indices[i]][0], MC_EDGE_TABLE[edge_indices[i]][1])
        );
    }
}

__device__ int march_cube(const McCube &cube, Point *const triangles) {
    unsigned char cube_index = 0;

    for (int i = 0; i < 8; i++) {
        cube_index |= ((unsigned char) (cube.values[i] <= 0.0f)) << i;
    }

    auto cube_triangles = MC_TRIANGLE_TABLE[cube_index];

    int triangle_count = 0;
    for (int i = 0; cube_triangles[i] != -1; i += 3) {
        int edge_indices[3] = { cube_triangles[i], cube_triangles[i + 1], cube_triangles[i + 2] };
        write_triangle(cube, edge_indices, &triangles[i]);
        triangle_count++;
    }

    return triangle_count;
}