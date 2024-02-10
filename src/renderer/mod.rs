use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

#[derive(Debug, Clone, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSettings {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderRelayCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

struct RenderCudaContext {
    #[allow(dead_code)]
    pub device: Arc<CudaDevice>,
    pub func_compute_render: CudaFunction,
}

struct RenderCudaStreams {
    render_stream: CudaStream,
}

struct RenderCudaBuffers {
    render_texture_buffer: CudaSlice<crate::bindings::cuda::Rgba>,
}

// App Systems

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: RENDER_TEXTURE_SIZE.0 as u32,
            height: RENDER_TEXTURE_SIZE.1 as u32,
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
                    RENDER_TEXTURE_SIZE.0 as f32,
                    RENDER_TEXTURE_SIZE.1 as f32,
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

// Render Systems

fn setup_cuda(world: &mut World) {
    let start = std::time::Instant::now();

    let device = CudaDevice::new(0).unwrap();

    info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

    let start = std::time::Instant::now();

    device
        .load_ptx(
            Ptx::from_src(include_str!(
                "../../assets/cuda/compiled/compute_render.ptx"
            )),
            "compute_render",
            &["compute_render"],
        )
        .unwrap();

    let func_compute_render = device.get_func("compute_render", "compute_render").unwrap();

    info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

    let render_texture_buffer = unsafe {
        device
            .alloc::<crate::bindings::cuda::Rgba>(RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    let render_stream = device.fork_default_stream().unwrap();

    world.insert_resource(RenderSettings::default());
    world.insert_non_send_resource(RenderCudaContext {
        device,
        func_compute_render,
    });
    world.insert_non_send_resource(RenderCudaStreams { render_stream });
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
    });
}

fn render(
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_context: NonSend<RenderCudaContext>,
    render_streams: NonSendMut<RenderCudaStreams>,
    render_buffers: NonSendMut<RenderCudaBuffers>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,

    mut tick: Local<u64>,
) {
    let range_id = nvtx::range_start!("Render System Wait For Previous Frame");

    let image = images.get_mut(&render_target_image.0).unwrap();
    let (cam, cam_projection, cam_transform) = camera.single();

    if *tick == 0 {
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                image.data.as_mut_ptr() as *mut _,
                image.data.as_mut_slice().len(),
                0,
            )
            .result()
            .unwrap()
        };
    }

    unsafe {
        cudarc::driver::result::stream::synchronize(render_streams.render_stream.stream).unwrap()
    };

    nvtx::range_end!(range_id);

    let range_id = nvtx::range_start!("Render System Invoke");

    // Render Parameters

    let globals = crate::bindings::cuda::GlobalsBuffer {
        time: time.elapsed_seconds(),
        tick: tick.clone(),
        render_texture_size: [RENDER_TEXTURE_SIZE.0 as u32, RENDER_TEXTURE_SIZE.1 as u32],
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

    let render_texture = crate::bindings::cuda::RenderTexture {
        data: unsafe {
            std::mem::transmute(*(&render_buffers.render_texture_buffer).device_ptr())
        },
        size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
    };

    unsafe {
        render_context
            .func_compute_render
            .clone()
            .launch_on_stream(
                &render_streams.render_stream,
                LaunchConfig {
                    block_dim: (crate::bindings::cuda::BLOCK_SIZE as usize as u32, 1, 1),
                    grid_dim: (
                        (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32)
                            / (crate::bindings::cuda::BLOCK_SIZE as usize as u32),
                        1,
                        1,
                    ),
                    shared_mem_bytes: 0,
                },
                (
                    render_texture.clone(),
                    globals.clone(),
                    camera.clone(),
                ),
            )
            .unwrap()
    };

    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(
            image.data.as_mut_slice(),
            *render_buffers.render_texture_buffer.device_ptr(),
            render_streams.render_stream.stream,
        )
        .unwrap()
    };

    *tick += 1;

    nvtx::range_end!(range_id);
}

// Synchronization

fn synchronize_target_sprite(
    mut sprite: Query<&mut Transform, With<RenderTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    sprite.single_mut().scale = Vec2::new(
        window.single().width() / (RENDER_TEXTURE_SIZE.0 as f32),
        window.single().height() / (RENDER_TEXTURE_SIZE.1 as f32),
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
        app.add_systems(Startup, (setup, setup_cuda))
            .add_systems(Last, render)
            .add_systems(PostUpdate, (synchronize_target_sprite,));
    }
}
