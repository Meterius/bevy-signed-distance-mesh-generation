// #import "shaders/compiled/render_scene_sd.wgsl"::{sd_scene_material}
// #import "shaders/compiled/render_ray_marching.wgsl"::{Ray, RayMarchHit, ray_march_with, ray_march, ray_march_hit_approx_ao, ray_march_hit_approx_soft_shadow}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionLight, phong_reflect_light}
#import "shaders/compiled/color.wgsl"::{color_map_default, hdr_map_aces_tone, color_map_temp}
#import "shaders/compiled/utils.wgsl"::{PI, random_state, rand}

var<private> is_main_invocation: bool;
var<private> pixel_color_override: vec3<f32> = vec3<f32>(-1.0, -1.0, -1.0);

//

fn invocation_id_to_texture_coord(invocation_id: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
}

fn texture_coord_to_viewport_coord(texture_coord: vec2<f32>) -> vec2<f32> {
    let flipped = vec2<f32>(2.0) * texture_coord / vec2<f32>(GLOBALS.render_texture_size) - vec2<f32>(1.0);
    return flipped * vec2<f32>(1.0, -1.0);
}

fn viewport_coord_to_ray_dir(viewport_coord: vec2<f32>) -> vec3<f32> {
    return normalize(
        CAMERA.forward * CAMERA.unit_plane_distance
        + CAMERA.right * viewport_coord.x * 0.5 * CAMERA.aspect_ratio
        + CAMERA.up * viewport_coord.y * 0.5
    );
}

// Rendering

const PIXEL_SAMPLING_RATE = 1;
const PIXEL_SAMPLING_BORDER = 0.4;

fn render_skybox(ray: Ray, hit: RayMarchHit) -> vec3<f32> {
    let sun_proj = dot(ray.direction, SCENE.sun_direction);
    let sun_angle = acos(sun_proj);

    let SUN_DISK_ANGLE = (PI / 180.0) * 0.5;
    let SUN_FADE_ANGLE = (PI / 180.0) * 1.0;

    let sun_color = vec3<f32>(20.0);

    let sky_fac = pow(f32(hit.step_depth) * 0.002, 2.0);
    let sky_color =
        0.0 * vec3<f32>(0.7, 0.7, 1.0) * vec3<f32>(max(0.0, 1.0 - sky_fac))
        + vec3<f32>(30.0, 30.0, 30.0) * vec3<f32>(sky_fac);

    if (sun_angle <= SUN_DISK_ANGLE) {
        return sun_color;
    } else if (sun_angle <= SUN_FADE_ANGLE) {
        let fac = 1.0 - (sun_angle - SUN_DISK_ANGLE) / (SUN_FADE_ANGLE - SUN_DISK_ANGLE);
        return mix(sky_color, sun_color, pow(fac, 3.0));
    } else {
        return sky_color;
    }
}

fn render_hit(ray: Ray, hit: RayMarchHit) -> vec3<f32> {
    let fogColor = render_skybox(ray, hit);
    // let fogColor = mix(vec3<f32>(0.7, 0.7, 1.0), normalize(render_skybox(ray, hit)), 0.9);

    var color: vec3<f32>;
    if (hit.cutoff_reason == CUTOFF_REASON_DISTANCE) {
        color = render_skybox(ray, hit);
    } else if (hit.cutoff_reason == CUTOFF_REASON_STEPS) {
        color = fogColor;
    } else {
        let normal = sd_scene_normal(hit.origin);

        // Shading And Coloring

        let scene_light = PhongReflectionLight(
            SCENE.sun_direction * 10000.0,
            1.0,
            1.0,
        );

        let base_material_color = color_map_default(
            0.0 * length(
                hit.origin + vec3<f32>(30.0 * sin(GLOBALS.time * 0.1 + 20.75), 40.0 * sin(GLOBALS.time * 0.25 + 10.75), 50.0 * sin(GLOBALS.time * 0.5 + 100.75))
            ) * 0.001 + (f32(hit.step_depth) / f32(RAY_MARCHING_MAX_STEP_DEPTH)) * 0.1
        ) * 5.0;

        let material = sd_scene_material(hit.origin, base_material_color);

        let phong_light = phong_reflect_light(
            CAMERA.origin, hit.origin, normal,
            material,
            scene_light,
        );

        let shadow = ray_march_hit_approx_soft_shadow(ray, hit, SCENE.sun_direction);
        let ao = ray_march_hit_approx_ao(ray, hit);

        let shading_light = shadow * phong_light;
        let light = (shading_light * 0.99 + 0.01 * ao);

        color = material.color * 1.0 * pow(light, 0.8);

        // Fog

        let fogFac = 1.0 - exp(-pow(hit.depth * 0.0001, 2.0));
        color = mix(color, fogColor, clamp(fogFac, 0.0, 1.0));
    }

    return color;
}

fn render_ray(ray: Ray) -> vec3<f32> {
    let hit = ray_march(ray, RayMarchOptions(RAY_MARCHER_MAX_DEPTH, RAY_MARCHING_MAX_STEP_DEPTH));
    return render_hit(ray, hit);
}

fn render_pixel(texture_coord: vec2<i32>) -> vec3<f32> {
    if (PIXEL_SAMPLING_RATE == 1) {
        return render_ray(Ray(CAMERA.origin, viewport_coord_to_ray_dir(texture_coord_to_viewport_coord(vec2<f32>(texture_coord)))));
    } else {
        var color = vec3<f32>(0.0);

        for (var i: i32 = 0; i < PIXEL_SAMPLING_RATE; i += 1) {
            for (var j: i32 = 0; j < PIXEL_SAMPLING_RATE; j += 1) {
                let offset = vec2<f32>(
                    -0.5 + PIXEL_SAMPLING_BORDER / 2.0 + (1.0 - PIXEL_SAMPLING_BORDER) * (f32(i) / (f32(PIXEL_SAMPLING_RATE - 1))),
                    -0.5 + PIXEL_SAMPLING_BORDER / 2.0 + (1.0 - PIXEL_SAMPLING_BORDER) * (f32(j) / (f32(PIXEL_SAMPLING_RATE - 1))),
                );

                let sub_pixel_viewport_coord = texture_coord_to_viewport_coord(vec2<f32>(texture_coord) + offset);
                let sub_pixel_ray_dir = viewport_coord_to_ray_dir(sub_pixel_viewport_coord);
                color += render_ray(
                    Ray(CAMERA.origin, sub_pixel_ray_dir)
                ) / vec3<f32>(f32(PIXEL_SAMPLING_RATE * PIXEL_SAMPLING_RATE));
            }
        }

        return color;
    }
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

@compute @workgroup_size(8, 8, 1)
fn update(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>
) {
    random_state = invocation_id.x + u32(GLOBALS.render_texture_size.x) * invocation_id.y + u32(GLOBALS.render_texture_size.x) * u32(GLOBALS.render_texture_size.y) * GLOBALS.seed;

    let texture_coord = invocation_id_to_texture_coord(invocation_id);
    let viewport_coord = texture_coord_to_viewport_coord(vec2<f32>(texture_coord));

    rs_compute_node_offset = 32 * i32(local_invocation_id.x + local_invocation_id.y * 8u);

    is_main_invocation = invocation_id.x == 4u && local_invocation_id.y == 4u;

    var color: vec3<f32>;

    color = render_pixel(texture_coord);

    if (pixel_color_override.x != -1.0) {
        color = pixel_color_override;
    }

    textureStore(TEXTURE, texture_coord, vec4<f32>(color, 1.0));
}