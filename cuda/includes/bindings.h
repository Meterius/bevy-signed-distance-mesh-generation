#pragma once

#include <stdbool.h>

#define CUDA_DEBUG false

#define BLOCK_SIZE 128

enum RayMarchHitOutcome {
    Collision, StepLimit, DepthLimit
};

struct GlobalsBuffer {
    unsigned long long tick;
    float time;
    unsigned int render_texture_size[2];
    float render_screen_size[2];
};

struct CameraBuffer {
    float position[3];
    float forward[3];
    float up[3];
    float right[3];
    float fov;
};

struct Rgba {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

struct RenderTexture {
    unsigned int size[2];
    struct Rgba* data;
};
