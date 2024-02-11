use crate::bindings::cuda::BLOCK_SIZE;
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::Render;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{
    result, CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

const MESH_GENERATION_POINT_BUFFER_SIZE: usize = 65536;

#[derive(Debug, Clone, Default, Event)]
pub struct AdvanceMeshGenerationEvent {}

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
    #[allow(dead_code)]
    pub device: Arc<CudaDevice>,
    pub func_compute_render: CudaFunction,
    pub func_compute_mesh_block_generation: CudaFunction,
}

struct RenderCudaStreams {
    render_stream: CudaStream,
}

struct RenderCudaBuffers {
    render_texture_buffer: CudaSlice<crate::bindings::cuda::Rgba>,
    mesh_gen_point_buffers: (
        CudaSlice<crate::bindings::cuda::Point>,
        CudaSlice<crate::bindings::cuda::Point>,
    ),
}

#[derive(Resource, Default, Clone)]
struct RenderCudaMeshGenState {
    partition_points: Vec<crate::bindings::cuda::Point>,
    partition_factor: usize,
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

// Mesh Generation

fn advance_border_block_partition(
    mut ev: EventReader<AdvanceMeshGenerationEvent>,
    render_context: NonSend<RenderCudaContext>,
    mut render_buffers: NonSendMut<RenderCudaBuffers>,
    mut render_mesh_gen_state: ResMut<RenderCudaMeshGenState>,
) {
    if ev.is_empty() {
        return;
    }

    for _ in ev.read() {}

    let block_factor_increase: usize = 2;

    let curr_point_count = render_mesh_gen_state.partition_points.len();
    let worst_case_next_point_count = curr_point_count * block_factor_increase.pow(3);

    info!(
        "Advancing Mesh Generation; Current Point Count: {}; Worst Case Point Count: {};",
        render_mesh_gen_state.partition_points.len(),
        worst_case_next_point_count
    );

    if worst_case_next_point_count > MESH_GENERATION_POINT_BUFFER_SIZE {
        panic!(
            "Advancing block border partitions may require {worst_case_next_point_count} points, \
           as we currently have {curr_point_count} points but buffer can only store {MESH_GENERATION_POINT_BUFFER_SIZE}"
        );
    }

    unsafe {
        render_context
            .func_compute_mesh_block_generation
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (
                        (render_mesh_gen_state.partition_points.len() as f32 / BLOCK_SIZE as f32)
                            .ceil() as u32,
                        1,
                        1,
                    ),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    crate::bindings::cuda::BlockPartition {
                        bases: std::mem::transmute(
                            *(&render_buffers.mesh_gen_point_buffers.0).device_ptr(),
                        ),
                        base_length: render_mesh_gen_state.partition_points.len() as _,
                        factor: render_mesh_gen_state.partition_factor as _,
                    },
                    crate::bindings::cuda::BlockPartition {
                        bases: std::mem::transmute(
                            *(&render_buffers.mesh_gen_point_buffers.1).device_ptr(),
                        ),
                        base_length: worst_case_next_point_count as _,
                        factor: (render_mesh_gen_state.partition_factor * block_factor_increase)
                            as _,
                    },
                ),
            )
            .unwrap()
    };

    render_mesh_gen_state.partition_points.resize(
        worst_case_next_point_count,
        crate::bindings::cuda::Point::default(),
    );

    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(
            render_mesh_gen_state.partition_points.as_mut_slice(),
            *render_buffers.mesh_gen_point_buffers.1.device_ptr(),
        )
        .unwrap()
    };

    render_mesh_gen_state
        .partition_points
        .retain(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());

    unsafe {
        cudarc::driver::result::memcpy_htod_sync(
            *render_buffers.mesh_gen_point_buffers.0.device_ptr(),
            render_mesh_gen_state.partition_points.as_slice(),
        )
        .unwrap()
    };
    render_mesh_gen_state.partition_factor *= block_factor_increase;

    info!(
        "Advanced Mesh Generation; Point Count: {}",
        render_mesh_gen_state.partition_points.len()
    );
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

    device
        .load_ptx(
            Ptx::from_src(include_str!(
                "../../assets/cuda/compiled/compute_mesh_generation.ptx"
            )),
            "compute_mesh_generation",
            &["compute_mesh_block_generation"],
        )
        .unwrap();

    let func_compute_render = device.get_func("compute_render", "compute_render").unwrap();
    let func_compute_mesh_block_generation = device
        .get_func("compute_mesh_generation", "compute_mesh_block_generation")
        .unwrap();

    info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

    let render_texture_buffer = unsafe {
        device
            .alloc::<crate::bindings::cuda::Rgba>(RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    let mut partition_points =
        Vec::with_capacity(crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR.pow(3) as _);

    {
        let block_size = crate::bindings::cuda::MESH_GENERATION_BB_SIZE as f32
            / crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR as f32;
        for i in 0..crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR {
            for j in 0..crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR {
                for k in 0..crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR {
                    partition_points.push(crate::bindings::cuda::Point {
                        x: -crate::bindings::cuda::MESH_GENERATION_BB_SIZE as f32 / 2.0
                            + i as f32 * block_size,
                        y: -crate::bindings::cuda::MESH_GENERATION_BB_SIZE as f32 / 2.0
                            + j as f32 * block_size,
                        z: -crate::bindings::cuda::MESH_GENERATION_BB_SIZE as f32 / 2.0
                            + k as f32 * block_size,
                    })
                }
            }
        }
    }

    let mut mesh_gen_point_buffers = unsafe {
        (
            device
                .alloc::<crate::bindings::cuda::Point>(MESH_GENERATION_POINT_BUFFER_SIZE)
                .unwrap(),
            device
                .alloc::<crate::bindings::cuda::Point>(MESH_GENERATION_POINT_BUFFER_SIZE)
                .unwrap(),
        )
    };

    unsafe {
        cudarc::driver::result::memcpy_htod_sync(
            *mesh_gen_point_buffers.0.device_ptr(),
            partition_points.as_slice(),
        )
        .unwrap()
    };

    let render_stream = device.fork_default_stream().unwrap();

    world.insert_non_send_resource(RenderCudaContext {
        device,
        func_compute_render,
        func_compute_mesh_block_generation,
    });
    world.insert_non_send_resource(RenderCudaStreams { render_stream });
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
        mesh_gen_point_buffers,
    });
    world.insert_resource(RenderCudaMeshGenState {
        partition_points,
        partition_factor: crate::bindings::cuda::MESH_GENERATION_INIT_FACTOR as _,
    });
}

fn render(
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_context: NonSend<RenderCudaContext>,
    render_streams: NonSendMut<RenderCudaStreams>,
    render_buffers: NonSendMut<RenderCudaBuffers>,
    render_target_image: Res<RenderTargetImage>,
    render_settings: Res<RenderSettings>,
    render_mesh_gen_state: Res<RenderCudaMeshGenState>,
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

    let partition = crate::bindings::cuda::BlockPartition {
        bases: unsafe {
            std::mem::transmute(*(&render_buffers.mesh_gen_point_buffers.0).device_ptr())
        },
        base_length: render_mesh_gen_state.partition_points.len() as _,
        factor: render_mesh_gen_state.partition_factor as _,
    };

    let render_texture = crate::bindings::cuda::RenderTexture {
        data: unsafe { std::mem::transmute(*(&render_buffers.render_texture_buffer).device_ptr()) },
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
                    partition.clone(),
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
        app.insert_resource(RenderSettings::default())
            .add_event::<AdvanceMeshGenerationEvent>()
            .add_systems(Startup, (setup, setup_cuda))
            .add_systems(Last, render)
            .add_systems(
                PostUpdate,
                (advance_border_block_partition, synchronize_target_sprite),
            );
    }
}
