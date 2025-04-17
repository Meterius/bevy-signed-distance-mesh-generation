#pragma once

#include <stdbool.h>

#define CUDA_DEBUG false

#define BLOCK_SIZE 128

#define MESH_GENERATION_INIT_FACTOR 32
#define MESH_GENERATION_BB_SIZE 5.0f

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

struct Point {
    float x;
    float y;
    float z;
};

#define POINT_NAN Point { NAN, NAN, NAN }

struct VoxelField {
    struct Point voxel_size;
    struct Point* voxels;
    unsigned int voxel_count;
};

struct Vertex {
    struct Point position;
    struct Point normal;
};

struct Triangle {
    struct Vertex vertices[3];
};

struct IndexTriangle {
    unsigned int indices[3];
};

struct Mesh {
    struct Vertex* vertices;
    unsigned int vertex_count;

    struct IndexTriangle* triangles;
};
