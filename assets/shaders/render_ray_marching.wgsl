// #import "shaders/compiled/render_scene_sd.wgsl"::{sd_scene, sd_scene_normal, SdSceneData}
#import "shaders/compiled/utils.wgsl"::{log_b, min_comp3} 

// Ray Marching

const RAY_MARCHING_MAX_STEP_DEPTH = 500;
const RAY_MARCHER_COLLISION_DISTANCE = 0.001;
const RAY_MARCHER_MAX_DEPTH = 10000.0;

const CUTOFF_REASON_NONE = 0u;
const CUTOFF_REASON_DISTANCE = 1u;
const CUTOFF_REASON_STEPS = 2u;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

const APPROX_AO_SAMPLE_COUNT = 10;
const APPROX_AO_SAMPLE_STEP = 0.1;

fn ray_march_hit_approx_soft_shadow(ray: Ray, hit: RayMarchHit, sun_direction: vec3<f32>) -> f32 {
    let light_hit = ray_march(Ray(hit.origin + sun_direction * 0.01, sun_direction), RayMarchOptions(RAY_MARCHER_MAX_DEPTH, RAY_MARCHING_MAX_STEP_DEPTH));
    return f32(light_hit.cutoff_reason != CUTOFF_REASON_NONE) * clamp(
        pow(light_hit.weighted_shortest_distance, 0.8), 0.0, 1.0,
    );
}

fn ray_march_hit_approx_ao(ray: Ray, hit: RayMarchHit) -> f32 {
    if (hit.cutoff_reason != CUTOFF_REASON_NONE) {
        return 0.0;
    }

    let normal = sd_scene_normal(hit.origin);

    var total = 0.0;
    var collision = false;

    for (var i = 1; i < APPROX_AO_SAMPLE_COUNT; i += 1) {
        let delta = f32(i) * APPROX_AO_SAMPLE_STEP;
        var sd: f32 = 0.0;

        if (!collision) {
            sd = sd_scene(hit.origin + normal * delta);

            if (sd <= RAY_MARCHER_COLLISION_DISTANCE) {
                collision = true;
                sd = 0.0;
            }
        }

        total += pow(2.0, f32(-i)) * (delta - sd);
    }

    return 1.0 - clamp(5.0 * total, 0.0, 1.0);
}

struct RayMarchOptions {
    depth_limit: f32,
    step_depth_limit: i32,
}

struct RayMarchHit {
    origin: vec3<f32>,
    depth: f32,
    weighted_shortest_distance: f32,
    step_depth: i32,
    cutoff_reason: u32,
}

fn steps_until_depth_limit(initial_sd: f32, depth: f32, depth_limit: f32, direction: f32) -> f32 {
    return log_b(
        1.0 + (depth_limit - depth) * direction / ((direction + 1.0) * initial_sd), (direction + 1.0)
    );
}

var<private> rs_compute_node_offset: i32 = 0;
var<workgroup> rs_compute_nodes: array<RsComputeNode, 2048> = array<RsComputeNode, 2048>();

fn ray_march(ray: Ray, options: RayMarchOptions) -> RayMarchHit {
    var hit = RayMarchHit();
    hit.weighted_shortest_distance = 3.40282346638528859812e+38f;
    hit.origin = ray.origin;

    for (; hit.step_depth < options.step_depth_limit; hit.step_depth += 1) {
        var sd = sd_scene(hit.origin);

        //

        if (false && length(hit.origin) <= 20.0) {
            for (var i = 0; i < SD_SCENE.pre_count; i += 1) {
                rs_compute_nodes[rs_compute_node_offset + i].origin = (hit.origin - SD_SCENE.pre[i].translation) * SD_SCENE.pre[i].scale;
            }

            for (var i = 0; i < SD_SCENE.primitive_count; i += 1) {
                let dist = length(rs_compute_nodes[rs_compute_node_offset + i].origin);

                rs_compute_nodes[rs_compute_node_offset + i].sd = select(
                    min_comp3(abs(rs_compute_nodes[rs_compute_node_offset + i].origin)) - 0.5,
                    length(rs_compute_nodes[rs_compute_node_offset + i].origin) - 0.5,
                    SD_SCENE.primitive[i].is_sphere != 0,
                );
            }

            for (var i = 0; i < SD_SCENE.pre_count; i += 1) {
                sd = min(sd, rs_compute_nodes[rs_compute_node_offset + i].sd * SD_SCENE.pre[i].min_scale);
            }
        }

        //

        let cutoff_plane = 100.0;
        if (false && hit.origin.y > cutoff_plane) {
            if (ray.direction.y < 0.0) {
                sd = (-(hit.origin.y - cutoff_plane) / ray.direction.y) + 0.1;
            } else {
                hit.step_depth = min(i32(f32(hit.step_depth) + steps_until_depth_limit(sd, hit.depth, options.depth_limit, ray.direction.y)), options.step_depth_limit);
                hit.origin += (options.depth_limit - hit.depth) * ray.direction;
                hit.cutoff_reason = CUTOFF_REASON_DISTANCE;
                hit.depth = options.depth_limit;
                break;
            }
        }

        if (sd <= RAY_MARCHER_COLLISION_DISTANCE + max(0.0, hit.depth - 100.0) * (0.01 / 100.0)) {
            hit.cutoff_reason = CUTOFF_REASON_NONE;
            break;
        }

        if (hit.depth >= 0.1) {
            hit.weighted_shortest_distance = min(
                hit.weighted_shortest_distance,
                sd / hit.depth,
            );
        }

        hit.depth += sd;
        hit.origin += sd * ray.direction;

        if (hit.depth >= options.depth_limit) {
            hit.cutoff_reason = CUTOFF_REASON_DISTANCE;
            break;
        }
    }

    if (hit.step_depth == options.step_depth_limit) {
        hit.cutoff_reason = CUTOFF_REASON_STEPS;
    }

    return hit;
}
