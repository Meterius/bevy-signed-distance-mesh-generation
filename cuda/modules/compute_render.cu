#include "./common.cu"

__device__ float sd_scene(vec3 p) {
    float sd = MAX_POSITIVE_F32;

    sd = min(sd, sd_obj(p));
    sd = min(sd, sd_box(p, vec3 { 0.0f, MESH_GENERATION_BB_MIN[1] - 0.5f, 0.0f }, vec3 { 10.0f, 1.0f, 10.0f }));

    for (int dir = 0; dir < 3; dir++) {
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 2; c1++) {
                vec3 m0 = MESH_GENERATION_BB_MIN;

                if (c0) {
                    m0[(dir + 1) % 3] = MESH_GENERATION_BB_MAX[(dir + 1) % 3];
                }

                if (c1) {
                    m0[(dir + 2) % 3] = MESH_GENERATION_BB_MAX[(dir + 2) % 3];
                }

                vec3 m1 = m0;
                m1[dir] = MESH_GENERATION_BB_MAX[dir];

                sd = min(sd, sd_line(p, m0, m1) - 0.05f);
            }
        }
    }

    return sd;
}

extern "C" __global__ void compute_render(
    const RenderTexture render_texture,
    const GlobalsBuffer globals,
    const CameraBuffer camera,
    const BlockPartition partition
) {
    // calculate ray

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

    auto sd_scene_with_partition = [&](vec3 p) {
        float sd = sd_scene(p);

        if (globals.show_partition) {
            vec3 block_size = (MESH_GENERATION_BB_MAX - MESH_GENERATION_BB_MIN) / (float) partition.factor;

            for (int i = 0; i < partition.base_length; i++) {
                sd = min(sd, sd_box(p, from_point(partition.bases[i]) + block_size / 2.0f, block_size));
            }
        }

        return sd;
    };

    RayMarchHit hit = ray_march(sd_scene_with_partition, ray, cone_radius_at_unit);

    vec3 light_dir = normalize(vec3(1.0f, 1.0f, 1.0f));

    vec3 color { 0.0f };
    switch (hit.outcome) {
        case RayMarchHitOutcome::Collision: {
            vec3 normal = empirical_normal(sd_scene_with_partition, hit.position);
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
