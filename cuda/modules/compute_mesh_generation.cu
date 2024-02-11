#include "./common.cu"

__device__ bool obj_contains(const vec3 p) {
    return sd_obj(p) <= 0.0f;
}

extern "C" __global__ void compute_mesh_block_generation(
    BlockPartition prev_partition,
    BlockPartition partition
) {
    const int increase_fac = partition.factor / prev_partition.factor;
    const int increase_fac2 = increase_fac * increase_fac;
    const int increase_fac3 = increase_fac2 * increase_fac;

    const vec3 block_size = (MESH_GENERATION_BB_MAX - MESH_GENERATION_BB_MIN) / (float) partition.factor;

    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < partition.base_length) {
        const vec3 base = from_point(prev_partition.bases[id]);

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
                                c & 1 ? lower[0] : upper[0],
                                c & 2 ? lower[1] : upper[1],
                                c & 4 ? lower[2] : upper[2]
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

extern "C" __global__ void compute_mesh_generation() {
}
