fn de(pos: vec3<f32>, time: f32) -> f32 {
    let Iterations = 16;
    let Power = 8.0 + sin(time * 6.28 / 4.0) * 2.0;
    let Bailout = 20.0;

    var z: vec3<f32> = pos;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < Iterations; i = i + 1) {
        r = length(z);
        if (r > Bailout) {
            break;
        }

        // convert to polar coordinates
        let theta: f32 = acos(z.z / r);
        let phi: f32 = atan2(z.y, z.x);
        dr = pow(r, Power - 1.0) * Power * dr + 1.0;

        // scale and rotate the point
        let zr: f32 = pow(r, Power);
        let new_theta: f32 = theta * Power;
        let new_phi: f32 = phi * Power;

        // convert back to cartesian coordinates
        z = zr * vec3<f32>(sin(new_theta) * cos(new_phi), sin(new_phi) * sin(new_theta), cos(new_theta));
        z = z + pos;
    }

    return 0.5 * log(r) * r / dr;
}
