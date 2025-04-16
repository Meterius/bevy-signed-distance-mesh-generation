#include "../includes/marching_cubes.cu"
#include "./common.cu"

#include <algorithm>
#include <assert.h>
#include <stdio.h>

__device__ bool obj_contains(const vec3 p) {
    return sd_obj(p) <= 0.0f;
}

extern "C" __global__ void compute_refine_voxel_field_by_sdf(
    const VoxelField input_field,
    VoxelField output_field
) {
    const int increase_fac = 2;
    const int increase_fac2 = increase_fac * increase_fac;
    const int increase_fac3 = increase_fac2 * increase_fac;

    const vec3 output_voxel_size = from_point(input_field.voxel_size) / (float) increase_fac;
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id == 0) {
        output_field.voxel_size = to_point(output_voxel_size);
    }

    if (id < input_field.voxel_count) {
        const vec3 base = from_point(input_field.voxels[id]);

        for (int i = 0; i < increase_fac; i++) {
            for (int j = 0; j < increase_fac; j++) {
                for (int k = 0; k < increase_fac; k++) {
                    vec3 lower = base + vec3 { i, j, k } * output_voxel_size;
                    vec3 upper = base + vec3 { i + 1, j + 1, k + 1 } * output_voxel_size;

                    bool is_border = false;
                    bool prev = obj_contains(lower);
                    for (int c = 1; c < 8; c++) {
                        if (prev != obj_contains(
                            {
                                c & 1 ? upper[0] : lower[0],
                                c & 2 ? upper[1] : lower[1],
                                c & 4 ? upper[2] : lower[2]
                            }
                        )) {
                            is_border = true;
                            break;
                        }
                    }

                    const unsigned int n_id = id * increase_fac3 + i * increase_fac2 + j * increase_fac + k;

                    if (n_id < output_field.voxel_count) {
                        output_field.voxels[n_id] = {
                            is_border ? lower.x : INFINITY, is_border ? lower.y : INFINITY, is_border ? lower.z : INFINITY
                        };
                    }
                }
            }
        }
    }
}

extern "C" __global__ void compute_mesh_from_voxel_field_by_sdf(
    VoxelField field,
    NaiveTriMesh tri_mesh
) {
    const vec3 voxel_size = from_point(field.voxel_size);
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int triangle_start = 3 * 5 * id;

    if (id < field.voxel_count) {
        const vec3 base = from_point(field.voxels[id]);

        McCube cube;
        for (int c = 0; c < 8; c++) {
            vec3 v = base;
            v[0] += (c % 4) == 1 || (c % 4) == 2 ? voxel_size.x : 0.0f;
            v[1] += (c % 4) >= 2 ? voxel_size.y : 0.0f;
            v[2] += c >= 4 ? voxel_size.z : 0.0f;

            cube.vertices[c] = v;
            cube.values[c] = sd_obj(v);
        }

        int tr_count = march_cube(cube, &tri_mesh.vertices[triangle_start]);

        for (int k = 0; k < tr_count; k++) {
            vec3 v0 = from_point(tri_mesh.vertices[triangle_start + 3 * k].position);
            vec3 v1 = from_point(tri_mesh.vertices[triangle_start + 3 * k + 1].position);
            vec3 v2 = from_point(tri_mesh.vertices[triangle_start + 3 * k + 2].position);

            v0 = closest_surface_point(sd_obj, v0);
            v1 = closest_surface_point(sd_obj, v1);
            v2 = closest_surface_point(sd_obj, v2);

            vec3 n0 = empirical_normal(sd_obj, v0);
            vec3 n1 = empirical_normal(sd_obj, v1);
            vec3 n2 = empirical_normal(sd_obj, v2);

            const vec3 triangle_normal = normalize(cross(v1 - v0, v2 - v0));
            const vec3 actual_normal = empirical_normal(sd_obj, (v0 + v1 + v2) / 3.0f);
            const bool change_orientation = dot(triangle_normal, actual_normal) <= 0.0f;

            tri_mesh.vertices[triangle_start + 3 * k] = { to_point(change_orientation ? v2 : v0), to_point(change_orientation ? n2 : n0) };
            tri_mesh.vertices[triangle_start + 3 * k + 1] = { to_point(v1), to_point(n1) };
            tri_mesh.vertices[triangle_start + 3 * k + 2] = { to_point(change_orientation ? v0 : v2), to_point(change_orientation ? n0 : n2) };
        }

        for (int i = 3 * tr_count; i < 3 * 5; i++) {
            tri_mesh.vertices[triangle_start + i] = { INFINITY, INFINITY, INFINITY };
        }
    }
}
