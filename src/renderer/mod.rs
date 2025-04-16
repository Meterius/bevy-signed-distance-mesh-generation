use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use crate::cuda::{CudaHandler, CudaMeshGenState};

const RENDER_IMAGE_SIZE: (usize, usize) = (2560, 1440);

#[derive(Debug, Clone, Default, Event)]
pub struct AdvanceMeshGenerationEvent {}

#[derive(Debug, Clone, Default, Event)]
pub struct FinalizeMeshGenerationEvent {}

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
struct RenderCudaMeshGenState {
    pub state: CudaMeshGenState,
}

// App Systems

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                order: 1,
                ..default()
            },
            camera_2d: Camera2d {
                clear_color: ClearColorConfig::None,
                ..default()
            },
            ..default()
        },
        RenderRelayCameraTarget::default(),
    ));

    commands.insert_resource(RenderTargetImage(image));
}

// Mesh Generation

fn advance_mesh_generation(
    mut ev: EventReader<AdvanceMeshGenerationEvent>,
    mut render_context: NonSendMut<RenderCudaContext>,
    mut render_mesh_gen_state: ResMut<RenderCudaMeshGenState>,
) {
    if ev.is_empty() {
        return;
    }

    for _ in ev.read() {
        render_context.handler.advance_mesh_generation(&mut render_mesh_gen_state.state)
    }
}

fn finalize_mesh_generation(
    mut ev: EventReader<FinalizeMeshGenerationEvent>,
    mut render_context: NonSendMut<RenderCudaContext>,
    mut render_mesh_gen_state: ResMut<RenderCudaMeshGenState>,
) {
    if ev.is_empty() {
        return;
    }

    for _ in ev.read() {
        render_context.handler.finalize_mesh_generation(&mut render_mesh_gen_state.state);
    }
}

// Render Systems

fn setup_cuda(world: &mut World) {
    let handler = CudaHandler::new();
    let state = CudaHandler::create_mesh_gen_state();
    world.insert_non_send_resource(RenderCudaContext { handler });
    world.insert_resource(RenderCudaMeshGenState { state });
}

fn render(
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    mut render_context: NonSendMut<RenderCudaContext>,
    render_target_image: Res<RenderTargetImage>,
    render_settings: Res<RenderSettings>,
    mut render_mesh_gen_state: ResMut<RenderCudaMeshGenState>,
    mut images: ResMut<Assets<Image>>,
    mut tick: Local<u64>,
) {
    let (cam, cam_projection, cam_transform) = camera.single();

    let globals = crate::bindings::cuda::GlobalsBuffer {
        time: time.elapsed_seconds(),
        tick: tick.clone(),
        render_texture_size: [RENDER_IMAGE_SIZE.0 as u32, RENDER_IMAGE_SIZE.1 as u32],
        render_screen_size: [
            cam.logical_viewport_size().map(|s| s.x).unwrap_or(1.0) as _,
            cam.logical_viewport_size().map(|s| s.y).unwrap_or(1.0) as _,
        ],
        show_partition: render_settings.show_partition,
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
        &mut render_mesh_gen_state.state,
        globals,
        camera,
    );

    *tick += 1;
}

// Synchronization

fn synchronize_target_sprite(
    mut sprite: Query<&mut Transform, With<RenderTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    sprite.single_mut().scale = Vec2::new(
        window.single().width() / (RENDER_IMAGE_SIZE.0 as f32),
        window.single().height() / (RENDER_IMAGE_SIZE.1 as f32),
    )
    .extend(1.0);
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
