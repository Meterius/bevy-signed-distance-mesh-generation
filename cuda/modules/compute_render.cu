#include "./common.cu"

__device__ float sd_scene(vec3 p) {
    return min(
        length(p - vec3 { 0.0f, 2.0f, 0.0f }) - 1.0f,
        sd_box(p, vec3 { 0.0f, -0.5f, 0.0f }, vec3 { 10.0f, 1.0f, 10.0f })
    );
}

extern "C" __global__ void compute_render(
    const RenderTexture render_texture,
    const GlobalsBuffer globals,
    const CameraBuffer camera
) {
    // calculate ray

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = render_texture_coord({ render_texture.size[0], render_texture.size[1] });

    if (texture_coord.y >= render_texture.size[1] || texture_coord.x >= render_texture.size[0]) {
        return;
    }

    u32 texture_index = index_2d(texture_coord, render_texture);

    vec2 ndc_coord = texture_to_ndc(
        texture_coord,
        { render_texture.size[0], render_texture.size[1] }
    );
    vec2 cam_coord = ndc_to_camera(
        ndc_coord, { render_texture.size[0], render_texture.size[1] }
    );

    // ray marching

    Ray ray {
        { camera.position[0], camera.position[1], camera.position[2] },
        camera_to_ray(
            cam_coord,
            camera,
            from_array(globals.render_screen_size),
            vec2(globals.render_texture_size[0], globals.render_texture_size[1])
        )
    };

    float cone_radius_at_unit = get_pixel_cone_radius(
        texture_coord, camera, render_texture,
        globals
    );

    RayMarchHit hit = ray_march(sd_scene, ray, cone_radius_at_unit);

    vec3 light_dir = normalize(vec3(1.0f, 1.0f, 1.0f));

    vec3 color { 0.0f };
    switch (hit.outcome) {
        case RayMarchHitOutcome::Collision: {
            vec3 normal = emperical_normal(sd_scene, hit.position);
            color = mix(
                vec3 { 19.0f, 9.0f, 130.0f } / 255.0f,
                vec3 { 240.0f, 103.0f, 24.0f } / 255.0f,
                (dot(normal, light_dir) + 1.0f) / 2.0f
            );
            break;
        }

        case RayMarchHitOutcome::StepLimit:
            color = vec3 { 1.0f };
            break;

        case RayMarchHitOutcome::DepthLimit:
            break;
    }

    color = hdr_map_aces_tone(color);

    render_texture.data[texture_index] = {
        (unsigned char) (clamp(color.x, 0.0f, 1.0f) * 255.0f),
        (unsigned char) (clamp(color.y, 0.0f, 1.0f) * 255.0f),
        (unsigned char) (clamp(color.z, 0.0f, 1.0f) * 255.0f),
        0xFF,
    };
}
