#import "shaders/compiled/utils.wgsl"::{min3, min4, min5, wrap, wrap_cell, min_comp3, MAX_POSITIVE_F32}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionMaterial}
#import "shaders/compiled/signed_distance.wgsl"::{sdSphere, sdUnion, sdPostSmoothUnion, sdRecursiveTetrahedron, sdBox, sdPreCheapBend, sdPreMirror, sdPreMirrorB}
#import "shaders/compiled/fractals.wgsl"::{de}

// Runtime Scene

const SD_RUNTIME_SCENE_NODES = 32;

struct RsComputeNode {
    origin: vec3<f32>,
    sd: f32,
}

// Scene

fn sd_scene_axes(p: vec3<f32>) -> f32 {
    return min(
        // sdSphere(vec3<f32>(p.x, wrap(p.y, -0.5, 0.5), p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(wrap(p.x, -0.5, 0.5), p.y, p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(p.x, p.y, wrap(p.z, -0.5, 0.5)), vec3<f32>(0.0), 0.1),
    );
}

fn sd_scene_column(p: vec3<f32>, cell: vec3<f32>) -> f32 {
    return sdPostSmoothUnion(
        min3(
            sdBox(p, vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.5, 3.0, 0.5)),
            sdBox(p, vec3<f32>(0.0, 5.0, 0.0), vec3<f32>(0.75, 0.15, 0.75)),
            sdBox(p, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.7, 1.0, 0.7)),
        ),
        sdSphere(p, vec3<f32>(0.0, 10.0 + (
            2.0 * (
                sin(2.0 * PI * (cell.x * 0.1 + GLOBALS.time) / 10.0)
                + sin(2.0 * PI * (cell.z * 0.1 + GLOBALS.time * 4.0) / 15.0)
            )
        ), 0.0), 0.5),
        1.1,
    );
}

fn sd_scene_column_pattern(p: vec3<f32>, grid_gap: vec2<f32>) -> f32 {
    let q = p;
    let columns_cell = vec3<f32>(
        wrap_cell(q.x, -grid_gap.x, grid_gap.x),
        q.y,
        wrap_cell(q.z, -grid_gap.y, grid_gap.y),
    );

    let columns_wrapped_pos = vec3<f32>(
        wrap(q.x, -grid_gap.x, grid_gap.x),
        q.y,
        wrap(q.z, -grid_gap.y, grid_gap.y),
    );

    return sdPostSmoothUnion(sdPostSmoothUnion(
        sd_scene_column(columns_wrapped_pos + vec3<f32>(2.0 * grid_gap.x, 0.0, 0.0), vec3<f32>(grid_gap, 1.0) * (columns_cell + vec3<f32>(1.0, 0.0, 0.0))),
        sd_scene_column(columns_wrapped_pos - vec3<f32>(2.0 * grid_gap.x, 0.0, 0.0), vec3<f32>(grid_gap, 1.0) * (columns_cell - vec3<f32>(1.0, 0.0, 0.0))),
        2.0,
    ), sd_scene_column(columns_wrapped_pos, columns_cell * vec3<f32>(grid_gap, 1.0)), 4.0);
}

fn rotateZ(vec: vec3<f32>, angle: f32) -> vec3<f32> {
    let cosA = cos(angle);
    let sinA = sin(angle);
    
    // Rotation matrix for Z-axis
    let rotMat = mat3x3<f32>(
        cosA, -sinA, 0.0,
        sinA, cosA, 0.0,
        0.0,  0.0,  1.0
    );

    return rotMat * vec;
}

fn sd_fractal(p: vec3<f32>) -> f32 {
    var q = p / 25.0;
    let pre = q.x;
    q.x = wrap(q.x, -4.0, 4.0);

    q.y += select(0.0, cos(GLOBALS.time * pre * 0.01) * 0.15, abs(pre) >= 2.0);
    // q.y += pow(abs(p.x) * cos(p.x) * 0.001 * (1.1 * (sin(GLOBALS.time * 2.0) + 1.2 * cos(GLOBALS.time * 3.0))), 3.0);

    let t = abs(pow((cos(GLOBALS.time * 2.0) + sin(GLOBALS.time * 2.0 * 3.0 + 0.1)), 3.0) / 4.0);

    q = rotateZ(q, GLOBALS.time * (1.0 + sin(GLOBALS.time * 2.0 * 0.005)) * 0.05);

    return sdPostSmoothUnion(
        de(q, GLOBALS.time), length(q) - (0.1 + 0.5 * sin(GLOBALS.time * 2.0)), 0.6 * t,
     ) * 25.0;
}

fn sd_scene(p: vec3<f32>) -> f32 {
    var q = p; // sdPreMirror(p, normalize(vec3<f32>(0.0, -0.25, 1.0)), -200.0);

    // Tetrahedron
    let tetrahedron_scale = 400.0;
    let tetrahedron_wrapped_pos = mix(vec3<f32>(
        wrap(q.x, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
        q.y,
        wrap(q.z, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
    ), q, 1.0);

    let tetrahedron_translated_pos = tetrahedron_wrapped_pos - vec3<f32>(0.0, tetrahedron_scale, 0.0);
    let tetrahedron_scaled_pos = tetrahedron_translated_pos / tetrahedron_scale;

    let p2 = (q / tetrahedron_scale + vec3<f32>(-0.2, -0.3, 1.9));
    let sd_tetrahedron_data = sdRecursiveTetrahedron(p2);
    let sd_tetrahedron = sd_tetrahedron_data.x * tetrahedron_scale;

    // Columns

    let sd_columns = sd_scene_column_pattern(q, vec2<f32>(1.5, 1.5));
    let sd_large_columns = sd_scene_column_pattern(q / 30.0 + vec3<f32>(00.0, 0.0, 30.0), vec2<f32>(5.0 + sin(GLOBALS.time * 0.1) * 6.0, 10.0)) * 30.0;

    // Octa

    let sd_octa = sd_fractal(q);

    // Axes

    let sd_axes = MAX_POSITIVE_F32; // sd_scene_axes(q);

    let sd_runtime = MAX_POSITIVE_F32; // sd_runtime_scene(q);

    return min(
        min(sd_axes, min(sd_tetrahedron, sd_runtime)),
        sdPostSmoothUnion(
            MAX_POSITIVE_F32, //p.y,
            sdPostSmoothUnion(
                sd_octa,
                sdSphere(q, vec3<f32>(0.0, 0.0, 50.0 * sin(GLOBALS.time * 0.1)), 5.0),
                4.0,
            ),
            4.0
        )
    );
}

fn sd_scene_material(p: vec3<f32>, base_color: vec3<f32>) -> PhongReflectionMaterial {
    let sd =         sdPostSmoothUnion(
            p.y,
            sdPostSmoothUnion(
                sd_fractal(p),
                sdSphere(p, vec3<f32>(0.0, 0.0, 50.0 * sin(GLOBALS.time * 0.1)), 5.0),
                4.0,
            ),
            3.9
        );

    if (true || (sd < 0.01 && p.y > 0.2)) {
        return PhongReflectionMaterial(vec3<f32>(1.0, 0.0, 0.0), 5.0, 0.7, 0.05, 30.0);
    }

    return PhongReflectionMaterial(base_color, 0.01, 0.7, 0.05, 30.0);
}

const NORMAL_EPSILON = 0.0001;

fn sd_scene_normal(p: vec3<f32>) -> vec3<f32> {
    let epsilon = NORMAL_EPSILON;

    let dx = (
        - sd_scene(vec3<f32>(p.x + 2.0 * epsilon, p.y, p.z))
        + 8.0 * sd_scene(vec3<f32>(p.x + epsilon, p.y, p.z))
        - 8.0 * sd_scene(vec3<f32>(p.x - epsilon, p.y, p.z))
        + sd_scene(vec3<f32>(p.x - 2.0 * epsilon, p.y, p.z))
    );

    let dy = (
        - sd_scene(vec3<f32>(p.x, p.y + 2.0 * epsilon, p.z))
        + 8.0 * sd_scene(vec3<f32>(p.x, p.y + epsilon, p.z))
        - 8.0 * sd_scene(vec3<f32>(p.x, p.y - epsilon, p.z))
        + sd_scene(vec3<f32>(p.x, p.y - 2.0 * epsilon, p.z))
    );

    let dz = (
        - sd_scene(vec3<f32>(p.x, p.y, p.z + 2.0 * epsilon))
        + 8.0 * sd_scene(vec3<f32>(p.x, p.y, p.z + epsilon))
        - 8.0 * sd_scene(vec3<f32>(p.x, p.y, p.z - epsilon))
        + sd_scene(vec3<f32>(p.x, p.y, p.z - 2.0 * epsilon))
    );

    return normalize(vec3(dx, dy, dz));
}
