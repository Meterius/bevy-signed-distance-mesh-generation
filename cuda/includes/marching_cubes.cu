#pragma once

#include "./utils.cu"
#include "./marching_cubes_constants.cu"

using namespace glm;

struct McCube {
    vec3 vertices[8];
    float values[8];
};

__device__ vec3 edge_vertex(const McCube &cube, const int i0, const int i1) {
    float v = 0.5f; // cube.values[i0] / (cube.values[i0] - cube.values[i1]);
    return mix(cube.vertices[i0], cube.vertices[i1], v);
}

__device__ unsigned int march_cube(const McCube &cube, Triangle *const triangles) {
    unsigned char cube_index = 0;

    for (int i = 0; i < 8; i++) {
        cube_index |= ((unsigned char) (cube.values[i] <= 0.0f)) << i;
    }

    auto cube_triangles = MC_TRIANGLE_TABLE[cube_index];

    unsigned int triangle_count = 0;
    for (int i = 0; cube_triangles[i] != -1; i += 3) {
        int edge_indices[3] = { cube_triangles[i], cube_triangles[i + 1], cube_triangles[i + 2] };

        Vertex vertices[3];
        for (int j = 0; j < 3; j++) {
            vertices[j].position = to_point(
                edge_vertex(cube, MC_EDGE_TABLE[edge_indices[j]][0], MC_EDGE_TABLE[edge_indices[j]][1])
            );
        }

        triangles[triangle_count] = Triangle {  { vertices[0], vertices[1], vertices[2] } };
        triangle_count++;
    }

    return triangle_count;
}