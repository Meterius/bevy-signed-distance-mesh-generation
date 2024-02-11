#include "./common.cu"
#include "../includes/marching_cubes.cu"

#include <assert.h>
#include <stdio.h>

__device__ bool obj_contains(const vec3 p) {
    return sd_obj(p) <= 0.0f;
}

extern "C" __global__ void compute_mesh_block_generation(
    BlockPartition prev_partition,
    BlockPartition partition,
    bool is_initial
) {
    const int increase_fac = is_initial ? 1 : partition.factor / prev_partition.factor;
    const int increase_fac2 = increase_fac * increase_fac;
    const int increase_fac3 = increase_fac2 * increase_fac;

    const vec3 block_size = (MESH_GENERATION_BB_MAX - MESH_GENERATION_BB_MIN) / (float) partition.factor;
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < partition.base_length) {
        const vec3 base = is_initial ? MESH_GENERATION_BB_MIN + block_size * vec3(
            (float) (id / (partition.factor * partition.factor)),
            fmod((float) (id / partition.factor), (float) partition.factor),
            fmod((float) id, (float) partition.factor)
        ) : from_point(prev_partition.bases[id]);

        for (int i = 0; i < increase_fac; i++) {
            for (int j = 0; j < increase_fac; j++) {
                for (int k = 0; k < increase_fac; k++) {
                    vec3 lower = base + vec3 { i, j, k } * block_size;
                    vec3 upper = base + vec3 { i + 1, j + 1, k + 1 } * block_size;

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
                    partition.bases[n_id] = {
                        is_border ? lower.x : INFINITY, is_border ? lower.y : INFINITY, is_border ? lower.z : INFINITY
                    };
                }
            }
        }
    }
}

extern "C" __global__ void compute_mesh_block_projected_marching_cube_mesh(
    BlockPartition partition,
    NaiveTriMesh tri_mesh
) {
    const vec3 block_size = (MESH_GENERATION_BB_MAX - MESH_GENERATION_BB_MIN) / (float) partition.factor;
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int triangle_start = 3 * 5 * id;

    if (id < partition.base_length) {
        const vec3 base = from_point(partition.bases[id]);

        McCube cube;
        for (int c = 0; c < 8; c++) {
            vec3 v = base;
            v[0] += (c % 4) == 1 || (c % 4) == 2 ? block_size[0] : 0.0f;
            v[1] += (c % 4) >= 2 ? block_size[1] : 0.0f;
            v[2] += c >= 4 ? block_size[2] : 0.0f;

            cube.vertices[c] = v;
            cube.values[c] = sd_obj(v);
        }

        int tr_count = march_cube(cube, &tri_mesh.vertices[triangle_start]);

        for (int k = 0; k < tr_count; k++) {
            vec3 v0 = from_point(tri_mesh.vertices[triangle_start + 3 * k]);
            vec3 v1 = from_point(tri_mesh.vertices[triangle_start + 3 * k + 1]);
            vec3 v2 = from_point(tri_mesh.vertices[triangle_start + 3 * k + 2]);

            vec3 triangle_normal = normalize(cross(v1 - v0, v2 - v0));
            vec3 actual_normal = empirical_normal(sd_obj, (v0 + v1 + v2) / 3.0f);

            if (dot(triangle_normal, actual_normal) <= 0.0f) {
                tri_mesh.vertices[triangle_start + 3 * k] = to_point(v2);
                tri_mesh.vertices[triangle_start + 3 * k + 2] = to_point(v0);
            }
        }

        for (int i = 3 * tr_count; i < 3 * 5; i++) {
            tri_mesh.vertices[triangle_start + i] = { INFINITY, INFINITY, INFINITY };
        }
    }
}
