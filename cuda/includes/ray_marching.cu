#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"
#include "../includes/signed_distance.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 256
#define RAY_MARCH_DEPTH_LIMIT 500.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

template<typename SdFunc>
__device__ RayMarchHit ray_march(
    const SdFunc sd_scene,
    const Ray ray,
    const float cone_radius_at_unit = 0.0f
) {
    RayMarchHit hit {
        (int) 0,
        ray.origin,
        0.0f,
        StepLimit,
        clock64()
    };

    for (; hit.steps < RAY_MARCH_STEP_LIMIT; hit.steps++) {
        float collision_distance = cone_radius_at_unit * hit.depth;
        float d = sd_scene(hit.position);

        if (d <= collision_distance + RAY_MARCH_COLLISION_DISTANCE) {
            hit.outcome = Collision;
            break;
        }

        hit.depth += (d - collision_distance);
        hit.position += (d - collision_distance) * ray.direction;

        if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    hit.cycles = clock64() - hit.cycles;

    return hit;
}
