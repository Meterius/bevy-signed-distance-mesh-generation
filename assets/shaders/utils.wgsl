const PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

const MAX_POSITIVE_F32 = 3.40282346638528859812e+38f;
const MIN_NORMAL_POSITIVE_F32 = 1.17549435082228750797e-38f;

const SQRT_2 = 1.41421356237309504880168872420969808;
const SQRT_2_INVERSE = 0.7071067811865475;

const SQRT_3 = 1.732050807568877293527446341505872367;
const SQRT_3_INVERSE = 0.5773502691896258;

var<private> random_state: u32;

fn rand() -> f32 {
    random_state = random_state ^ 2747636419u;
    random_state = random_state * 2654435769u;
    random_state = random_state ^ (random_state >> 16u);
    random_state = random_state * 2654435769u;
    random_state = random_state ^ (random_state >> 16u);
    random_state = random_state * 2654435769u;
    return f32(random_state) / 4294967296.0;
}

fn rand_unit_vec() -> vec3<f32> {
    return normalize(vec3<f32>(rand(), rand(), rand()));
}

fn max3(v1: f32, v2: f32, v3: f32) -> f32 {
    return max(v1, max(v2, v3));
}

fn max4(v1: f32, v2: f32, v3: f32, v4: f32) -> f32 {
    return max(v1, max(v2, max(v3, v4)));
}

fn max5(v1: f32, v2: f32, v3: f32, v4: f32, v5: f32) -> f32 {
    return max(v1, max(v2, max(v3, max(v4, v5))));
}

fn min3(v1: f32, v2: f32, v3: f32) -> f32 {
    return min(v1, min(v2, v3));
}

fn min4(v1: f32, v2: f32, v3: f32, v4: f32) -> f32 {
    return min(v1, min(v2, min(v3, v4)));
}

fn min5(v1: f32, v2: f32, v3: f32, v4: f32, v5: f32) -> f32 {
    return min(v1, min(v2, min(v3, min(v4, v5))));
}

fn max_comp3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn min_comp3(v: vec3<f32>) -> f32 {
    return min(v.x, min(v.y, v.z));
}

fn smooth_min(x1: f32, x2: f32, k: f32) -> f32 {
    let h = max(k - abs(x1 - x2), 0.0) / k;
    return min(x1, x2) - h * h * h * k * (1.0 / 6.0);
}

fn euclid_div(x: f32, n: f32) -> f32 {
    return floor(x / n) * n;
}

fn euclid_mod(x: f32, n: f32) -> f32 {
    return x - floor(x / n) * n;
}

fn wrap_cell(x: f32, lower: f32, upper: f32) -> f32 {
    return euclid_div(x - lower, (upper - lower));
}

fn wrap(x: f32, lower: f32, upper: f32) -> f32 {
    let offset = euclid_mod(x - lower, (upper - lower));
    return lower + offset;
}

fn wrap_reflect(x: f32, lower: f32, upper: f32) -> f32 {
    let diff = upper - lower;
    let offset = euclid_mod(x - lower, 2.0 * diff);

    if (offset > diff) {
        return lower + 2.0 * diff - offset;
    } else {
        return lower + offset;
    }
}

fn log_b(x: f32, b: f32) -> f32 {
    return log2(x) / log2(b);
}
