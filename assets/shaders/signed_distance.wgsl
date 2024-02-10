#import "shaders/compiled/utils.wgsl"::{max_comp3, smooth_min, euclid_mod, max3, max4, max5, min3, min4, min5, SQRT_2_INVERSE, SQRT_2, SQRT_3, SQRT_3_INVERSE}

// SD Primitives

fn sdOctahedron(p: vec3<f32>, op: vec3<f32>, s: f32) -> f32 {
    let q = abs(p - op);
    let m = q.x + q.y + q.z - s;

    var t = vec3<f32>(0.0);

    if (3.0 * q.x < m) {
        t = vec3<f32>(q.x, q.y, q.z);
    } else if (3.0 * q.y < m) {
        t = vec3<f32>(q.y, q.z, q.x);
    } else if (3.0 * q.z < m) {
        t = vec3<f32>(q.z, q.x, q.y);
    } else {
        return m * 0.57735027;
    }

    let k = clamp(0.5 * (t.z - t.y + s), 0.0, s);
    return length(vec3<f32>(t.x, t.y - s + k, t.z - k));
}

fn sdOctahedronApprox(p: vec3<f32>, op: vec3<f32>, s: f32) -> f32 {
    let q = abs(p - op);
    return (q.x+q.y+q.z-s)*0.57735027;
}

fn sdSphere(p: vec3<f32>, sp: vec3<f32>, r: f32) -> f32 {
    return length(p - sp) - r;
}

fn sdBox(p: vec3<f32>, bp: vec3<f32>, bs: vec3<f32>) -> f32 {
    let q = abs(p - bp) - bs;
    let udst = length(max(q, vec3<f32>(0.0)));
    let idst = max_comp3(min(q, vec3<f32>(0.0)));
    return udst + idst;
}

fn sdUnitSphere(p: vec3<f32>) -> f32 {
    return length(p) - 1.0;
}

fn sdVertexPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32 {
    return dot(p, n) - d;
}

fn sdVertexPlaneB(p: vec3<f32>, n: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(p, n) - dot(b, n);
}

// SD Complexes

const REC_TETR_ITER: i32 = 10;
const REC_TETR_SCALE: f32 = 1.0;
const REC_TETR_OFFSET: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

fn sdTetrahedron(p: vec3<f32>) -> f32 {
    return SQRT_3_INVERSE * max(
        abs(p.x - p.z) + p.y,
        abs(p.x + p.z) - p.y
    ) - SQRT_3_INVERSE;
}

fn sdRecursiveTetrahedron(p: vec3<f32>) -> vec2<f32> {
    var q = p;
    var adjustment = 1.0;

    var i = 1;
    for (; i <= REC_TETR_ITER; i += 1) {
        let sd_temp = sdTetrahedron(q);
        if (sdTetrahedron(q) >= 0.1) {
            return vec2<f32>(sd_temp * adjustment, f32(i));
        }

        adjustment *= 0.5;

        q.x = 2.0 * q.x - 1.0;
        q.y = 2.0 * q.y - 1.0;
        q.z = 2.0 * q.z - 1.0;

        var diff = q.x - q.z;
        if (diff <= 0.0) {
            let t = q.x;
            q.x = q.z;
            q.z = t;
        }

        diff = q.y - q.x;
        if (diff <= 0.0) {
            let t = q.x;
            q.x = q.y;
            q.y = t;
        }

        diff = q.x + q.z + 2.0;
        if (diff <= 0.0) {
            q.x -= diff;
            q.z -= diff;
        }
    }

    let sd = sdTetrahedron(q) * adjustment;

    return vec2<f32>(sd, f32(i));
}

// SD Operators

fn sdPreTranslate(p: vec3<f32>, translation: vec3<f32>) -> vec3<f32> {
    return p - translation;
}

fn sdPreScale(p: vec3<f32>, scale_origin: vec3<f32>, scale: f32) -> vec3<f32> {
    return (p - scale_origin) * vec3<f32>(scale) + scale_origin;
}

fn sdPostScale(sd: f32, scale_origin: vec3<f32>, scale: f32) -> f32 {
    return sd / vec3<f32>(scale);
}

fn sdPostUnion(sd1: f32, sd2: f32) -> f32 {
    return min(sd1, sd2);
}

fn sdPostSmoothUnion(sd1: f32, sd2: f32, k: f32) -> f32 {
    return smooth_min(sd1, sd2, k);
}

fn sdPostIntersect(sd1: f32, sd2: f32) -> f32 {
    return max(sd1, sd2);
}

fn sdPostInverse(sd1: f32) -> f32 {
    return -sd1;
}

fn sdPostDifference(sd1: f32, sd2: f32) -> f32 {
    return max(sd1, -sd2);
}

fn sdPreMirror(p: vec3<f32>, n: vec3<f32>, d: f32) -> vec3<f32> {
    let dist = sdVertexPlane(p, n, d);
    if (dist <= 0.0) {
        return p - 2.0 * dist * n;
    } else {
        return p;
    }
}

fn sdPreMirrorB(p: vec3<f32>, n: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let dist = sdVertexPlaneB(p, n, b);
    if (dist <= 0.0) {
        return p - 2.0 * dist * n;
    } else {
        return p;
    }
}

fn sdPreCheapBend(p: vec3<f32>) -> vec3<f32> {
    let k = 10.0;
    let c = cos(k * p.x);
    let s = sin(k * p.x);
    let m = mat2x2(c, -s, s, c);
    return vec3<f32>(m * vec2<f32>(p.x, p.y), p.z);
}

