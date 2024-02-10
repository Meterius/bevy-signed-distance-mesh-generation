#pragma once

#include "./libraries/glm/glm.hpp"
#include "../includes/bindings.h"

using namespace glm;

struct __align__(32) RayMarchHit {
int steps;
vec3 position;
float depth;
RayMarchHitOutcome outcome;
long long cycles;
};

struct __align__(32) Ray {
vec3 origin;
vec3 direction;
};

struct RayRender {
    struct RayMarchHit hit;
    vec3 color;
    float light;
};

struct RenderSurfaceData {
    vec3 color;
};
