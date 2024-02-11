#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/types.cu"
#include "../includes/utils.cu"

using namespace glm;

__device__ float wrap(const float x, const float lower, const float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ vec3 wrap(const vec3 p, const vec3 lower, const vec3 higher) {
    return {
        wrap(p.x, lower.x, higher.x), wrap(p.y, lower.y, higher.y),
        wrap(p.z, lower.z, higher.z)
    };
}

__device__ float smooth_min(const float a, const float b, const float k) {
    float h = max(k - abs(a - b), 0.0f) / k;
    return min(a, b) - h * h * h * k * (1.0f / 6.0f);
}

// fractals

#define POWER 7.0f

__device__ float sd_mandelbulb(const vec3 p, const float time) {
    vec3 z = p;
    float dr = 1.0f;
    float r;

    float power = POWER * (1.0f + time * 0.001f);

    for (int i = 0; i < 25; i++) {
        r = length(z);
        if (r > 2.0f) {
            break;
        }

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        float zr = pow(r, power);
        dr = pow(r, power - 1.0f) * power * dr + 1.0f;

        float s_theta = sin(theta);
        z = zr * vec3(s_theta * cos(phi), sin(phi) * s_theta, cos(theta));
        z += p;
    }

    return 0.5f * log(r) * r / dr;
}

__device__ float sd_unit_mandelbulb(const vec3 p) {
    return sd_mandelbulb(p / 0.4f, 0.0f) * 0.4f;
}

// primitives

__device__ float sd_ray(const vec3 p, const vec3 bl, const vec3 bd) {
    return distance(bl + bd * dot(p - bl, bd), p);
}

__device__ float sd_ray(const vec3 p, const vec3 bl, const vec3 bd, float len) {
    float d = dot(p - bl, bd);

    if (d < 0) {
        return distance(bl, p);
    } else if (d > len) {
        return distance(bl + len * bd, p);
    }

    return distance(bl + bd * d, p);
}

__device__ float sd_line(const vec3 p, const vec3 b0, const vec3 b1) {
    float len = length(b1 - b0);
    return sd_ray(p, b0, (b1 - b0) / len, len);
}

__device__ float sd_unit_sphere(const vec3 p) {
    return length(p) - 0.5f;
}

__device__ float sd_box(const vec3 p, const vec3 bp, const vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    float udst = length(max(q, vec3(0.0f)));
    float idst = maximum(min(q, vec3(0.0f)));
    return udst + idst;
}

__device__ float sd_box_skeleton(const vec3 p, const vec3 bp, const vec3 bs, const float lw) {
    vec3 bpl = bp - bs / 2.0f;
    float sd = MAX_POSITIVE_F32;

    for (int dir = 0; dir < 3; dir++) {
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 2; c1++) {
                vec3 m0 = bpl;
                m0[(dir + 1) % 3] += c0 ? bs[(dir + 1) % 2] : 0.0f;
                m0[(dir + 2) % 3] += c1 ? bs[(dir + 2) % 3] : 0.0f;

                vec3 m1 = m0;
                m1[dir] += bs[dir];

                sd = min(sd, sd_line(p, m0, m1) - lw);
            }
        }
    }

    return sd;
}

__device__ float sd_simple_box(const vec3 p, const vec3 bp, const vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    return maximum(min(q, vec3(0.0f)));
}

__device__ float sd_simple_bounding_box(const vec3 p, const vec3 bb_min, const vec3 bb_max) {
    return max(
        max(
            bb_min.x - p.x,
            max(bb_min.y - p.y, bb_min.z - p.z)
        ),
        max(
            p.x - bb_max.x,
            max(p.y - bb_max.y, p.z - bb_max.z)
        )
    );
}

__device__ float sd_unit_cube(const vec3 p) {
    return sd_box(p, vec3(0.0f), vec3(1.0f));
}

bool inside_aabb(const vec3 p, const vec3 bb_min, const vec3 bb_max) {
    return bb_min.x <= p.x && p.x <= bb_max.x && bb_min.y <= p.y && p.y <= bb_max.y && bb_min.z <= p.z &&
           p.z <= bb_max.z;
}

float ray_distance_to_bb(const Ray &ray, const vec3 &bb_min, const vec3 &bb_max) {
    if (inside_aabb(ray.origin, bb_min, bb_max)) {
        return 0.0f;
    }

    float tmin = std::numeric_limits<float>::lowest();
    float tmax = std::numeric_limits<float>::max();

    for (int i = 0; i < 3; ++i) {
        if (abs(ray.direction[i]) < std::numeric_limits<float>::epsilon()) {
            // Ray is parallel to the slab. No hit if origin not within slab
            if (ray.origin[i] < bb_min[i] || ray.origin[i] > bb_max[i])
                return std::numeric_limits<float>::max();
        } else {
            // Compute intersection t value of ray with near and far plane of slab
            float ood = 1.0f / ray.direction[i];
            float t1 = (bb_min[i] - ray.origin[i]) * ood;
            float t2 = (bb_max[i] - ray.origin[i]) * ood;

            // Make t1 be intersection with near plane, t2 with far plane
            if (t1 > t2) std::swap(t1, t2);

            // Compute the intersection of slab intersection intervals
            tmin = max(tmin, t1);
            tmax = min(tmax, t2);

            // Exit with no collision as soon as slab intersection becomes empty
            if (tmin > tmax) return std::numeric_limits<float>::max();
        }
    }

    // Ray intersects all 3 slabs. Return distance to first hit
    return tmin > 0 ? tmin : tmax;
}

// normals

#define NORMAL_EPSILON 0.001f

template<typename SFunc>
__device__ vec3 empirical_normal(
    const SFunc sd_func,
    const vec3 p
) {
    float dx = (-sd_func(p + vec3(2.0f * NORMAL_EPSILON, 0.0f, 0.0f)) +
                8.0f * sd_func(p + vec3(NORMAL_EPSILON, 0.0f, 0.0f)) -
                8.0f * sd_func(p + vec3(-NORMAL_EPSILON, 0.0f, 0.0f)) +
                sd_func(p + vec3(-2.0f * NORMAL_EPSILON, 0.0f, 0.0f)));

    float dy = (-sd_func(p + vec3(0.0f, 2.0f * NORMAL_EPSILON, 0.0f)) +
                8.0f * sd_func(p + vec3(0.0f, NORMAL_EPSILON, 0.0f)) -
                8.0f * sd_func(p + vec3(0.0f, -NORMAL_EPSILON, 0.0f)) +
                sd_func(p + vec3(0.0f, -2.0f * NORMAL_EPSILON, 0.0f)));

    float dz = (-sd_func(p + vec3(0.0f, 0.0f, 2.0f * NORMAL_EPSILON)) +
                8.0f * sd_func(p + vec3(0.0f, 0.0f, NORMAL_EPSILON)) -
                8.0f * sd_func(p + vec3(0.0f, 0.0f, -NORMAL_EPSILON)) +
                sd_func(p + vec3(0.0f, 0.0f, -2.0f * NORMAL_EPSILON)));

    return normalize(vec3(dx, dy, dz));
}

struct NormalPlane {
    vec3 up;
    vec3 forward;
    vec3 right;
};

template<typename SFunc>
__device__ NormalPlane empirical_normal_plane(
    const SFunc sd_func,
    const vec3 p
) {
    NormalPlane plane;

    plane.up = empirical_normal(sd_func, p);
    plane.right = cross(
        plane.up,
        abs(dot(plane.up, vec3 { 0.0f, 0.0f, 1.0f })) < 0.5f ? vec3 { 0.0f, 0.0f, 1.0f } : vec3 { 0.0f, 1.0f, 0.0f }
    );
    plane.forward = cross(plane.up, plane.right);

    return plane;
}

template<typename SFunc>
__device__ vec3 closest_surface_point(const SFunc sd_func, const vec3 p) {
    vec3 g = p;

    bool collision = false;
    for (int i = 0; !collision && i < 10000; i++) {
        float sd = sd_func(g);
        vec3 n = empirical_normal(sd_func, g);
        g -= sd * n;
        collision = abs(sd) <= 0.001f;
    }

    return g;
}
