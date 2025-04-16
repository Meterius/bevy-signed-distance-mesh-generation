use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use bevy::log::{error, info};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::nvrtc::Ptx;
use itertools::Itertools;
use crate::bindings::cuda::BLOCK_SIZE;

const MESH_GENERATION_OUTPUT_FILEPATH: &'static str = "generated_mesh.obj";

struct DynamicCudaSlice<T> {
    device: Arc<CudaDevice>,
    data: Option<CudaSlice<T>>,
}

impl<T: DeviceRepr> DynamicCudaSlice<T> {
    unsafe fn get_or_alloc_sync(&mut self, length: usize) -> CUdeviceptr {
        if self.data.as_ref().is_none_or(|data| data.len() < length) {
            self.data = Some(self.device.alloc(length).unwrap());
        }

        return std::mem::transmute(*(&self.data.as_ref().unwrap()).device_ptr());
    }
}

pub struct CudaHandler {
    device: Arc<CudaDevice>,

    func_compute_render: CudaFunction,

    func_compute_mesh_block_generation: CudaFunction,
    func_compute_mesh_block_projected_marching_cube_mesh: CudaFunction,

    render_texture_buffer: DynamicCudaSlice<crate::bindings::cuda::Rgba>,
    mesh_gen_triangle_buffer: DynamicCudaSlice<crate::bindings::cuda::Vertex>,
    mesh_gen_point_input_buffer: DynamicCudaSlice<crate::bindings::cuda::Point>,
    mesh_gen_point_output_buffer: DynamicCudaSlice<crate::bindings::cuda::Point>,

    _marker: PhantomData<bool>
}

pub struct CudaMeshGenState {
    pub initialized: bool,
    pub partition_points: Vec<crate::bindings::cuda::Point>,
    pub partition_factor: usize,

    _marker: PhantomData<bool>
}

impl CudaHandler {
    pub fn new() -> CudaHandler {
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
                &[
                    "compute_mesh_block_generation",
                    "compute_mesh_block_projected_marching_cube_mesh",
                ],
            )
            .unwrap();

        let func_compute_render = device.get_func("compute_render", "compute_render").unwrap();
        let func_compute_mesh_block_generation = device
            .get_func("compute_mesh_generation", "compute_mesh_block_generation")
            .unwrap();
        let func_compute_mesh_block_projected_marching_cube_mesh = device
            .get_func(
                "compute_mesh_generation",
                "compute_mesh_block_projected_marching_cube_mesh",
            )
            .unwrap();

        info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

        let render_texture_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let mesh_gen_point_input_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let mesh_gen_point_output_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let mesh_gen_triangle_buffer = DynamicCudaSlice { device: device.clone(), data: None };

        return Self {
            device, func_compute_mesh_block_projected_marching_cube_mesh, func_compute_render, func_compute_mesh_block_generation,
            mesh_gen_point_output_buffer, mesh_gen_point_input_buffer, mesh_gen_triangle_buffer, render_texture_buffer, _marker: PhantomData {},
        }
    }

    pub fn create_mesh_gen_state() -> CudaMeshGenState {
        return CudaMeshGenState {
            initialized: false,
            partition_factor: 0,
            partition_points: Vec::new(),
            _marker: PhantomData {},
        };
    }

    pub fn advance_mesh_generation(&mut self, mesh_gen_state: &mut CudaMeshGenState) {
        let block_factor_increase: usize = if mesh_gen_state.initialized {
            2
        } else {
            1
        };

        if !mesh_gen_state.initialized {
            mesh_gen_state.partition_factor = (8.0f32 * 1048576.0f32)
                .powf(1.0 / 3.0)
                .floor() as _;
        }

        let curr_point_count = if mesh_gen_state.initialized {
            mesh_gen_state.partition_points.len()
        } else {
            mesh_gen_state.partition_factor.pow(3)
        };
        let worst_case_next_point_count = curr_point_count * block_factor_increase.pow(3);

        info!(
            "Advancing Mesh Generation; Current Point Count: {}; Worst Case Point Count: {};",
            curr_point_count, worst_case_next_point_count
        );

        if curr_point_count != 0 {
            let mesh_gen_point_input_buffer_ptr = unsafe { self.mesh_gen_point_input_buffer.get_or_alloc_sync(mesh_gen_state.partition_points.len()) };
            let mesh_gen_point_output_buffer_ptr = unsafe { self.mesh_gen_point_output_buffer.get_or_alloc_sync(worst_case_next_point_count) };

            unsafe {
                cudarc::driver::result::memcpy_htod_sync(
                    mesh_gen_point_input_buffer_ptr,
                    mesh_gen_state.partition_points.as_slice(),
                )
                    .unwrap()
            };
            
            unsafe {
                self
                    .func_compute_mesh_block_generation
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (
                                (curr_point_count as f32 / BLOCK_SIZE as f32).ceil() as u32,
                                1,
                                1,
                            ),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            crate::bindings::cuda::BlockPartition {
                                bases: std::mem::transmute(mesh_gen_point_input_buffer_ptr),
                                base_length: curr_point_count as _,
                                factor: mesh_gen_state.partition_factor as _,
                            },
                            crate::bindings::cuda::BlockPartition {
                                bases: std::mem::transmute(mesh_gen_point_output_buffer_ptr),
                                base_length: worst_case_next_point_count as _,
                                factor: (mesh_gen_state.partition_factor
                                    * block_factor_increase)
                                    as _,
                            },
                            !mesh_gen_state.initialized,
                        ),
                    )
                    .unwrap()
            };

            mesh_gen_state.partition_points.resize(
                worst_case_next_point_count,
                crate::bindings::cuda::Point::default(),
            );

            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(
                    mesh_gen_state.partition_points.as_mut_slice(),
                    mesh_gen_point_output_buffer_ptr
                )
                    .unwrap()
            };

            mesh_gen_state
                .partition_points
                .retain(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }

        mesh_gen_state.partition_factor *= block_factor_increase;
        mesh_gen_state.initialized = true;

        info!(
            "Advanced Mesh Generation; Point Count: {}",
            mesh_gen_state.partition_points.len()
        );
    }

    pub fn finalize_mesh_generation(&mut self, mesh_gen_state: &mut CudaMeshGenState) {
        let worst_case_triangle_count = mesh_gen_state.partition_points.len() * 5;

        info!(
            "Finalizing Mesh Generation; Current Point Count: {}; Worst Case Triangle Count: {};",
            mesh_gen_state.partition_points.len(),
            worst_case_triangle_count
        );

        if !mesh_gen_state.initialized {
            error!("Cannot finalize mesh as mesh generation has not been initialized yet.");
        }

        let mesh_gen_point_input_buffer_ptr = unsafe { self.mesh_gen_point_input_buffer.get_or_alloc_sync(mesh_gen_state.partition_points.len()) };
        let mesh_gen_triangle_buffer_ptr = unsafe { self.mesh_gen_triangle_buffer.get_or_alloc_sync(3 * worst_case_triangle_count) };

        if mesh_gen_state.partition_points.len() != 0 {
            unsafe {
                self
                    .func_compute_mesh_block_projected_marching_cube_mesh
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (
                                (mesh_gen_state.partition_points.len() as f32
                                    / BLOCK_SIZE as f32)
                                    .ceil() as u32,
                                1,
                                1,
                            ),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            crate::bindings::cuda::BlockPartition {
                                bases: std::mem::transmute(mesh_gen_point_input_buffer_ptr),
                                base_length: mesh_gen_state.partition_points.len() as _,
                                factor: mesh_gen_state.partition_factor as _,
                            },
                            crate::bindings::cuda::NaiveTriMesh {
                                vertices: std::mem::transmute(mesh_gen_triangle_buffer_ptr),
                            },
                        ),
                    )
                    .unwrap()
            };

            let mut raw_triangles = Vec::with_capacity(worst_case_triangle_count * 3);

            for _ in 0..worst_case_triangle_count * 3 {
                raw_triangles.push(crate::bindings::cuda::Vertex::default());
            }

            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(
                    raw_triangles.as_mut_slice(),
                    mesh_gen_triangle_buffer_ptr,
                )
                    .unwrap()
            };

            self.device.synchronize().unwrap();

            let mut indices = vec![];

            let mut vertices = vec![];
            let mut normals = vec![];

            let mut vertex_index_map = HashMap::<[i64; 3], usize>::default();

            let f32_discrete = |x: f32| (x * 10e4f32).round() as i64;

            let map_vertex_pos = |v: crate::bindings::cuda::Point| {
                [f32_discrete(v.x), f32_discrete(v.y), f32_discrete(v.z)]
            };

            let mut add_or_get_vertex = |v: crate::bindings::cuda::Vertex| {
                vertex_index_map
                    .entry(map_vertex_pos(v.position))
                    .or_insert_with(|| {
                        let new_index = vertices.len();
                        vertices.push([v.position.x, v.position.y, v.position.z]);
                        normals.push([v.normal.x, v.normal.y, v.normal.z]);
                        new_index
                    })
                    .clone()
            };

            for (v0, v1, v2) in raw_triangles.into_iter().tuples() {
                if v0.position.x.is_finite() {
                    indices.push([
                        add_or_get_vertex(v0),
                        add_or_get_vertex(v1),
                        add_or_get_vertex(v2),
                    ]);
                }
            }

            let vertex_count = vertices.len();
            let triangle_count = indices.len() / 3;

            let data = obj::ObjData {
                position: vertices,
                normal: normals,
                texture: vec![[0.0, 0.0]],
                material_libs: Vec::new(),
                objects: vec![obj::Object {
                    name: String::from("default"),
                    groups: vec![obj::Group {
                        name: String::from("default"),
                        index: 0,
                        material: None,
                        polys: indices
                            .into_iter()
                            .map(|t| {
                                obj::SimplePolygon(
                                    t.into_iter()
                                        .map(|idx| obj::IndexTuple(idx, Some(0), Some(idx)))
                                        .collect(),
                                )
                            })
                            .collect(),
                    }],
                }],
            };

            data.save(MESH_GENERATION_OUTPUT_FILEPATH).unwrap();

            info!("Finalized Mesh Generation; Vertex Count: {vertex_count}; Triangle Count: {triangle_count}; Stored generated mesh at {MESH_GENERATION_OUTPUT_FILEPATH};");
        } else {
            info!("Finalized Mesh Generation; Empty Mesh; Stored generated mesh at {MESH_GENERATION_OUTPUT_FILEPATH};");
        }
    }

    pub fn render(
        &mut self,
        target_image: &mut [u8],
        mesh_gen_state: &mut CudaMeshGenState,
        globals: crate::bindings::cuda::GlobalsBuffer,
        camera: crate::bindings::cuda::CameraBuffer
    ) {
        let range_id = nvtx::range_start!("Render System Wait For Previous Frame");

        let render_width = globals.render_texture_size[0] as usize;
        let render_height = globals.render_texture_size[1] as usize;
        let render_buffer_size = 4 * render_width * render_height;

        let mesh_gen_point_buffer_ptr = unsafe {
            self.mesh_gen_point_input_buffer.get_or_alloc_sync(mesh_gen_state.partition_points.len())
        };

        if target_image.len() != render_buffer_size {
            error!("Target image has size {render_buffer_size}, but should have size {render_buffer_size}.");
        }

        let render_texture_buffer_ptr = unsafe {
            self.render_texture_buffer.get_or_alloc_sync(render_buffer_size)
        };

        unsafe {
            cudarc::driver::result::memcpy_htod_sync(
                mesh_gen_point_buffer_ptr,
                mesh_gen_state.partition_points.as_slice(),
            )
                .unwrap()
        };

        nvtx::range_end!(range_id);

        let range_id = nvtx::range_start!("Render System Invoke");

        let partition = crate::bindings::cuda::BlockPartition {
            bases: unsafe { std::mem::transmute(mesh_gen_point_buffer_ptr) },
            base_length: mesh_gen_state.partition_points.len() as _,
            factor: mesh_gen_state.partition_factor as _,
        };

        let render_texture = crate::bindings::cuda::RenderTexture {
            data: unsafe { std::mem::transmute(render_texture_buffer_ptr) },
            size: [globals.render_texture_size[0] as _, globals.render_texture_size[1] as _],
        };

        unsafe {
            self
                .func_compute_render
                .clone()
                .launch(
                    LaunchConfig {
                        block_dim: (crate::bindings::cuda::BLOCK_SIZE as usize as u32, 1, 1),
                        grid_dim: (
                            (render_width * render_height) as u32
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
            cudarc::driver::result::memcpy_dtoh_sync(
                target_image,
                render_texture_buffer_ptr,
            ).unwrap()
        };

        nvtx::range_end!(range_id);
    }
}
