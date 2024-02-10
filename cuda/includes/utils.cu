#pragma once

#include "./libraries/glm/glm.hpp"
#include "./libraries/glm/gtx/quaternion.hpp"

using namespace glm;

#define MAX_POSITIVE_F32 3.40282347E+38
#define PI 3.14159265358979323846264338327950288f
#define PI_HALF 1.5707963267948966f
#define SQRT 1.41421356237309504880168872420969808f
#define SQRT_INV 0.7071067811865475f

__device__ float minimum(const vec3 p) { return min(min(p.x, p.y), p.z); }

__device__ float minimum(const vec2 p) { return min(p.x, p.y); }

__device__ float maximum(const vec3 p) { return max(max(p.x, p.y), p.z); }

__device__ float maximum(const vec2 p) { return max(p.x, p.y); }

__device__ vec3 from_array(const float p[3]) { return {p[0], p[1], p[2]}; }

__device__ quat from_quat_array(const float p[4]) { return {p[0], p[1], p[2], p[3]}; }

template<uint32_t N>
class BitSet {
    static_assert(N > 0, "N must be positive");
    uint32_t bits[N / 32 + (N % 32 != 0)] = {};

public:
    __forceinline__
    __device__ bool get(uint32_t index) {
        return (bits[index / 32] >> (index % 32)) & 1;
    }

    __forceinline__
    __device__ void set(uint32_t index, bool value) {
        bits[index / 32] = (bits[index / 32] & ~(1 << (index % 32))) | (value << (index % 32));
    }
};

template<>
class BitSet<32> {
    uint32_t bits = 0;

public:
    __forceinline__
    __device__ bool get(const uint32_t index) const {
        return (bits >> index) & 1;
    }

    __forceinline__
    __device__ void set(const uint32_t index, const bool value) {
        bits = (bits & ~(1 << index)) | (value << index);
    }

    __forceinline__
    __device__ void set(const uint32_t index, const bool value, const bool condition) {
        bits = condition ? (bits & ~(1 << index)) | (value << index) : bits;
    }
};