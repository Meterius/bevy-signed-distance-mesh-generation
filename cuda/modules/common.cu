#define GLM_SWIZZLE

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/color.cu"
#include "../includes/utils.cu"
#include "../includes/signed_distance.cu"
#include "../includes/ray_marching.cu"

using namespace glm;

// coordinate system conversion

__device__ vec2 texture_to_ndc(const vec2 p, const vec2 texture_size) {
    return (p + vec2(0.5f, 0.5f)) / texture_size;
}

__device__ uvec2 ndc_to_texture(const vec2 p, const vec2 texture_size) {
    return uvec2(round((p * texture_size) - vec2(0.5f, 0.5f)));
}

template<typename Func, typename Texture>
__device__ auto fetch_2d(const ivec2 p, const Texture &texture, Func map) {
    return map(
        texture.texture
        [min(max(p.x, 0), texture.size[0] - 1) +
         min(max(p.y, 0), texture.size[1] - 1) * texture.size[0]]
    );
}

template<typename Texture>
__device__ auto index_2d(const ivec2 p, const Texture &texture) {
    return min(max(p.x, 0), texture.size[0] - 1) + min(max(p.y, 0), texture.size[1] - 1) * texture.size[0];
}


__device__ float
cubic_interpolate(const float y0, const float y1, const float y2, const float y3, const float rx1) {
    return y1 + 0.5f * rx1 *
                (y2 - y0 +
                 rx1 * (2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3 +
                        rx1 * (3.0f * (y1 - y2) + y3 - y0)));
}

template<typename Func, typename Texture>
__device__ auto ndc_to_interpolated_value(const vec2 p, const Texture &texture, const Func map) {
    vec2 t = (p * vec2((float) texture.size[0], (float) texture.size[1])) -
             vec2(0.5f, 0.5f);
    ivec2 tc = ivec2(floor(t));

    float interps[4];
    for (int i = 0; i < 4; i++) {
        interps[i] = cubic_interpolate(
            fetch_2d(ivec2(tc.x - 1, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x + 1, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x + 2, tc.y + i - 1), texture, map),
            t.x - (float) tc.x
        );
    }

    return cubic_interpolate(
        interps[0], interps[1], interps[2], interps[3], t.y - (float) tc.y
    );
}

__device__ vec2 ndc_to_camera(const vec2 p, const vec2 render_screen_size) {
    return {
        (2 * p.x - 1) * (render_screen_size.x / render_screen_size.y),
        1 - 2 * p.y
    };
}

__device__ vec3 camera_to_ray(
    const vec2 p, const CameraBuffer CAMERA, const vec2 screen_size, const vec2 texture_size
) {
    float width_factor =
        (screen_size.x / texture_size.x) * (texture_size.y / screen_size.y);

    float fov_fac = tan(CAMERA.fov / 2);
    return normalize(
        vec3(CAMERA.forward[0], CAMERA.forward[1], CAMERA.forward[2]) +
        p.y * fov_fac * vec3(CAMERA.up[0], CAMERA.up[1], CAMERA.up[2]) +
        p.x * fov_fac * width_factor *
        vec3(CAMERA.right[0], CAMERA.right[1], CAMERA.right[2])
    );
}

// ray-marching

#include <cuda_runtime.h>

template<typename Texture>
__device__ float get_pixel_cone_radius(
    const uvec2 texture_coord,
    const CameraBuffer &camera,
    const Texture &texture,
    const GlobalsBuffer &globals
) {
    const auto texture_to_dir = [&camera, &texture, &globals](
        vec2 p
    ) {
        vec2 ndc_coord = texture_to_ndc(
            p,
            {
                texture.size[0],
                texture.size[1]
            }
        );
        vec2 cam_coord = ndc_to_camera(
            ndc_coord,
            {
                texture.size[0],
                texture.size[1]
            }
        );
        return camera_to_ray(
            cam_coord,
            camera,
            from_array(globals.render_screen_size),
            vec2(globals.render_texture_size[0], globals.render_texture_size[1])
        );
    };

    vec2 ndc_coord = texture_to_ndc(
        texture_coord,
        {
            texture.size[0],
            texture.size[1]
        }
    );
    vec2 cam_coord = ndc_to_camera(
        ndc_coord,
        {
            texture.size[0],
            texture.size[1]
        }
    );
    Ray ray {
        { camera.position[0], camera.position[1], camera.position[2] },
        camera_to_ray(
            cam_coord,
            camera,
            from_array(globals.render_screen_size),
            vec2(globals.render_texture_size[0], globals.render_texture_size[1])
        )
    };

    vec3 border_dirs[4] = {
        texture_to_dir(
            {
                (float) texture_coord.x - SQRT_INV,
                (float) texture_coord.y - SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x - SQRT_INV,
                (float) texture_coord.y + SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x + SQRT_INV,
                (float) texture_coord.y - SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x + SQRT_INV,
                (float) texture_coord.y + SQRT_INV
            }
        ),
    };

    return max(
        max(
            length(ray.direction - border_dirs[0]),
            length(ray.direction - border_dirs[1])),
        max(
            length(ray.direction - border_dirs[2]),
            length(ray.direction - border_dirs[3])));
}

__device__ ivec2 render_texture_coord(const ivec2 render_texture_size) {
    const int WARP_H = 8;
    const int WARP_W = 4;

    int warp_local_id = threadIdx.x % 32;
    ivec2 warp_local_coord = { warp_local_id % WARP_W, warp_local_id / WARP_W };

    const int BLOCK_WARP_H = 2;
    const int BLOCK_WARP_W = (BLOCK_SIZE / BLOCK_WARP_H) / 32;

    int warp_id = threadIdx.x / 32;
    ivec2 warp_coord = { warp_id % BLOCK_WARP_W, warp_id / BLOCK_WARP_W };

    ivec2 block_local_texture_coord = {
        warp_local_coord.x + WARP_W * warp_coord.x,
        warp_local_coord.y + WARP_H * warp_coord.y
    };

    ivec2 block_count = {
        render_texture_size.x / (WARP_W * BLOCK_WARP_W),
        render_texture_size.x / (WARP_H * BLOCK_WARP_H),
    };

    ivec2 block_texture_coord = {
        (WARP_W * BLOCK_WARP_W) * (blockIdx.x % block_count.x),
        (WARP_H * BLOCK_WARP_H) * (blockIdx.x / block_count.x)
    };

    return block_texture_coord + block_local_texture_coord;
}

// signed-distance

#define MESH_GENERATION_BB_MIN (vec3 { -(MESH_GENERATION_BB_SIZE / 2.0f) })
#define MESH_GENERATION_BB_MAX (vec3 { (MESH_GENERATION_BB_SIZE / 2.0f) })

__device__ float sd_obj(vec3 p) {
    float a1 = sd_box_skeleton(p, vec3(0.0f), vec3(3.0f, 1.0f, 0.5f), 0.1f);
    float a2 = length(p) - 1.0f;
    return smooth_min(a1, a2, 0.5f);
}
