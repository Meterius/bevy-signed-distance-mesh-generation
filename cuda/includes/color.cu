#pragma once

#include "./libraries/glm/glm.hpp"

using namespace glm;

__device__ vec3 hdr_map_aces_tone(const vec3 hdr) {
    auto m1 = mat3x3(
        0.59719f, 0.07600f, 0.02840f,
        0.35458f, 0.90834f, 0.13383f,
        0.04823f, 0.01566f, 0.83777f
    );
    auto m2 = mat3x3(
        1.60475f, -0.10208f, -0.00327f,
        -0.53108f, 1.10813f, -0.07276f,
        -0.07367f, -0.00605f, 1.07602f
    );
    vec3 v = m1 * hdr;
    auto a = v * (v + 0.0245786f) - 0.000090537f;
    auto b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return min(max(m2 * (a / b), vec3(0.0f)), vec3(1.0f));
}