use crate::renderer::{RenderCameraTarget, RenderMeshGenPreviewMesh};
use bevy::prelude::*;
use bevy_flycam::FlyCam;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                is_active: false,
                ..default()
            },
            transform: Transform::from_xyz(5.0, 2.0, -5.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 0.0f32 })),
            material: materials.add(Color::rgb_u8(124, 144, 255).into()),
            ..default()
        },
        RenderMeshGenPreviewMesh {},
    ));

    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(25.0f32).into()),
        material: materials.add(Color::WHITE.into()),
        transform: Transform::from_xyz(0.0f32, -2.5f32, 0.0f32),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
}

#[derive(Debug, Clone, Component)]
pub struct RotateAxisMotion {
    axis: Vec3,
    cycle_duration: f32,
}

#[derive(Debug, Clone, Component)]
pub struct SphericCyclicMotion {
    center: Option<Vec3>,
    distances: Vec3,
    cycle_durations: Vec3,
}

impl Default for SphericCyclicMotion {
    fn default() -> Self {
        Self {
            center: None,
            distances: Vec3::ONE,
            cycle_durations: Vec3::ONE * 5.0,
        }
    }
}

#[derive(Debug, Clone, Component)]
pub struct AxisCyclicMotion {
    center: Option<Vec3>,
    direction: Vec3,
    cycle_duration: f32,
}

impl Default for AxisCyclicMotion {
    fn default() -> Self {
        Self {
            center: None,
            direction: Vec3::Y,
            cycle_duration: 5.0,
        }
    }
}

fn set_center(
    mut motions: Query<(&Transform, &mut AxisCyclicMotion), Added<AxisCyclicMotion>>,
    mut sphere_motions: Query<(&Transform, &mut SphericCyclicMotion), Added<SphericCyclicMotion>>,
) {
    for (trn, mut mot) in motions.iter_mut() {
        if mot.center.is_none() {
            mot.center = Some(trn.translation);
        }
    }

    for (trn, mut mot) in sphere_motions.iter_mut() {
        if mot.center.is_none() {
            mot.center = Some(trn.translation);
        }
    }
}

fn apply_motion(
    settings: Res<ExampleSceneSettings>,
    time: Res<Time>,
    mut motions: Query<(
        &mut Transform,
        Option<&AxisCyclicMotion>,
        Option<&SphericCyclicMotion>,
        Option<&RotateAxisMotion>,
    )>,
) {
    if settings.enable_movement {
        for (mut trn, ax_mot, sp_mot, rot_mot) in motions.iter_mut() {
            if let Some(ax_mot) = ax_mot {
                trn.translation = ax_mot.center.unwrap_or_default()
                    + ax_mot.direction
                        * (2.0 * std::f32::consts::PI * time.elapsed_seconds()
                            / ax_mot.cycle_duration)
                            .sin();
            } else if let Some(sp_mot) = sp_mot {
                let d = Vec3::ONE * 2.0 * std::f32::consts::PI * time.elapsed_seconds()
                    / sp_mot.cycle_durations;

                trn.translation = sp_mot.center.unwrap_or_default()
                    + sp_mot.distances * Vec3::new(d.x.sin(), d.y.sin(), d.z.sin());
            }

            if let Some(rot_mot) = rot_mot {
                trn.rotation = Quat::from_axis_angle(
                    rot_mot.axis,
                    2.0 * std::f32::consts::PI * (time.elapsed_seconds() / rot_mot.cycle_duration),
                );
            }
        }
    }
}

#[derive(Debug, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct ExampleSceneSettings {
    pub enable_movement: bool,
}

#[derive(Default)]
pub struct ExampleScenePlugin {}

impl Plugin for ExampleScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ExampleSceneSettings::default())
            .add_systems(Update, (set_center, apply_motion.after(set_center)));
    }
}
