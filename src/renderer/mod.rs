use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use bevy::render::mesh::{Indices, Mesh, PrimitiveTopology};
use obj::ObjData;
use crate::cuda::{CudaHandler, CudaVoxelField};

const RENDER_IMAGE_SIZE: (usize, usize) = (2560, 1440);
const MESH_GENERATION_OUTPUT_FILEPATH: &'static str = "generated_mesh.obj";

#[derive(Debug, Clone, Default, Event)]
pub struct AdvanceMeshGenerationEvent {}

#[derive(Debug, Clone, Default, Event)]
pub struct FinalizeMeshGenerationEvent {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderMeshTarget {}

#[derive(Debug, Clone, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSettings {
    pub show_partition: bool,
}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderRelayCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

struct RenderCudaContext {
    pub handler: CudaHandler,
}

#[derive(Resource)]
struct RenderCudaVoxelField {
    pub field: CudaVoxelField,
}

// App Systems

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: RENDER_IMAGE_SIZE.0 as u32,
            height: RENDER_IMAGE_SIZE.1 as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::rgba(1.0, 1.0, 1.0, 1.0),
                custom_size: Some(Vec2::new(
                    RENDER_IMAGE_SIZE.0 as f32,
                    RENDER_IMAGE_SIZE.1 as f32,
                )),
                ..default()
            },
            texture: image.clone(),
            ..default()
        },
        RenderTargetSprite::default(),
    ));

    commands.spawn((Camera2dBundle {
        camera_2d: Camera2d {
            clear_color: ClearColorConfig::None,
            ..default()
        },
        camera: Camera {
            order: 1,
            ..default()
        },
        ..default()
    }, RenderRelayCameraTarget {}));

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 0.0f32 })),
            material: materials.add(Color::rgb_u8(124, 144, 255).into()),
            ..default()
        },
        RenderMeshTarget {},
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

    commands.insert_resource(RenderTargetImage(image));
}

// Mesh Loading

fn obj_to_bevy_mesh(obj_data: &ObjData) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, obj_data.position.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, obj_data.normal.clone());
    mesh.set_indices(Some(Indices::U32(
        obj_data.objects[0].groups[0].polys
            .iter().flat_map(|poly| {
            return poly.clone().0.into_iter().map(|vert| {
                let p_idx = vert.0 as u32;
                let n_idx = vert.2.unwrap_or(0) as u32;
                if p_idx != n_idx { panic!("Position and normal index must be identical."); }
                return p_idx;
            });
        }).collect()
    )));

    mesh
}

// Mesh Generation

fn advance_mesh_generation(
    mut ev: EventReader<AdvanceMeshGenerationEvent>,
    mut target_handles: Query<&mut Handle<Mesh>, With<RenderMeshTarget>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_context: NonSendMut<RenderCudaContext>,
    mut render_voxel_field: ResMut<RenderCudaVoxelField>,
) {
    if ev.is_empty() {
        return;
    }

    for _ in ev.read() {
        render_context.handler.refine_voxel_field(&mut render_voxel_field.field);
        let obj = render_context.handler.voxel_field_to_mesh(&mut render_voxel_field.field);

        let handle = meshes.add(obj_to_bevy_mesh(&obj));

        for mut target_handle in target_handles.iter_mut() {
            *target_handle = handle.clone();
        }
    }
}

fn finalize_mesh_generation(
    mut ev: EventReader<FinalizeMeshGenerationEvent>,
    mut target_handles: Query<&mut Handle<Mesh>, With<RenderMeshTarget>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_context: NonSendMut<RenderCudaContext>,
    mut render_voxel_field: ResMut<RenderCudaVoxelField>,
) {
    if ev.is_empty() {
        return;
    }

    for _ in ev.read() {
        let obj = render_context.handler.voxel_field_to_mesh(&mut render_voxel_field.field);

        let handle = meshes.add(obj_to_bevy_mesh(&obj));

        for mut target_handle in target_handles.iter_mut() {
            *target_handle = handle.clone();
        }

        obj.save(MESH_GENERATION_OUTPUT_FILEPATH).unwrap();
        info!("Stored generated mesh at {MESH_GENERATION_OUTPUT_FILEPATH};");
    }
}

// Render Systems

fn setup_cuda(world: &mut World) {
    let handler = CudaHandler::new();
    let field = CudaHandler::create_cuda_voxel_field();
    world.insert_non_send_resource(RenderCudaContext { handler });
    world.insert_resource(RenderCudaVoxelField { field });
}

fn render(
    time: Res<Time>,
    mut camera: Query<(&mut Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    mut relay_camera: Query<(&mut Camera), (With<RenderRelayCameraTarget>, Without<RenderCameraTarget>)>,
    mut render_context: NonSendMut<RenderCudaContext>,
    render_settings: Res<RenderSettings>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,
    mut tick: Local<u64>,
) {
    let (mut cam, cam_projection, cam_transform) = camera.single_mut();
    let (mut relay_cam) = relay_camera.single_mut();

    relay_cam.is_active = !render_settings.show_partition;
    cam.is_active = render_settings.show_partition;

    if relay_cam.is_active {
        let globals = crate::bindings::cuda::GlobalsBuffer {
            time: time.elapsed_seconds(),
            tick: tick.clone(),
            render_texture_size: [RENDER_IMAGE_SIZE.0 as u32, RENDER_IMAGE_SIZE.1 as u32],
            render_screen_size: [
                cam.logical_viewport_size().map(|s| s.x).unwrap_or(1.0) as _,
                cam.logical_viewport_size().map(|s| s.y).unwrap_or(1.0) as _,
            ],
        };

        let camera = crate::bindings::cuda::CameraBuffer {
            position: cam_transform.translation().as_ref().clone(),
            forward: cam_transform.forward().as_ref().clone(),
            up: cam_transform.up().as_ref().clone(),
            right: cam_transform.right().as_ref().clone(),
            fov: match cam_projection {
                Projection::Perspective(perspective) => perspective.fov,
                Projection::Orthographic(_) => 1.0,
            },
        };

        render_context.handler.render(
            &mut images.get_mut(&render_target_image.0).unwrap().data,
            globals,
            camera,
        );

        *tick += 1;
    }
}

// Synchronization

fn synchronize_target_sprite(
    mut target: Query<&mut Transform, With<RenderTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    let mut target_transform = target.single_mut();

    target_transform.scale = Vec2::new(
        window.single().width() / (RENDER_IMAGE_SIZE.0 as f32),
        window.single().height() / (RENDER_IMAGE_SIZE.1 as f32),
    ).extend(1.0);
}

// Render Systems

// Render Pipeline

// Plugin

#[derive(Default)]
pub struct RayMarcherRenderPlugin {}

impl Plugin for RayMarcherRenderPlugin {
    fn build(&self, app: &mut App) {
        // Main App Build
        app.insert_resource(RenderSettings::default())
            .add_event::<AdvanceMeshGenerationEvent>()
            .add_event::<FinalizeMeshGenerationEvent>()
            .add_systems(Startup, (setup, setup_cuda))
            .add_systems(Last, render)
            .add_systems(
                PostUpdate,
                (
                    advance_mesh_generation,
                    finalize_mesh_generation.after(advance_mesh_generation),
                    synchronize_target_sprite,
                ),
            );
    }
}
